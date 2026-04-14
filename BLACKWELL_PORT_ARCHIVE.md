# Blackwell (chips_a / r000 GSP) Port Archive

## Quick Restore

```bash
# Branch with all Blackwell work:
git checkout zhi/vgpu-m2-staging-r000

# Remote copy (on GitHub):
#   repo: zhiwang-nvidia/nova-core
#   branch: agent/zhi/vgpu-m2-staging-r000

# Base commit (pre-Blackwell):
#   2b03a5aa9c1d  (parent of first Blackwell commit)

# First Blackwell commit:
#   e459c08ac12a  nova-core/gsp: update message queue protocol to MCTP/NVDM framing for r000 GSP

# Latest Blackwell commit:
#   fb755b085393  fix: use read32 instead of try_read32 on Bar1

# Build on col-blackwell:
ssh col-blackwell
cd /root/agent-vgpu-linux
git fetch origin && git checkout origin/agent/zhi/vgpu-m2-staging-r000 --force
export PATH="$HOME/.cargo/bin:$PATH"
make CC=clang LD=ld.lld LLVM=1 -j$(nproc) M=drivers/gpu/nova-core

# Load:
sudo rmmod nova_core 2>/dev/null
echo 1 | sudo tee /sys/bus/pci/devices/0000:c1:00.0/reset
sleep 3
sudo insmod drivers/gpu/nova-core/nova_core.ko
```

## Target Hardware

- **GPU**: NVIDIA RTX PRO 6000 Blackwell Server Edition (GB202, rev a.1)
- **Machine**: `col-blackwell` (SSH)
- **PCI**: `0000:c1:00.0`
- **BAR1**: 128 GB (`0x2000000000`)
- **FB region**: `0x0..0x1776770000` (~100 GB)
- **MMU**: V3 (6-level page tables)
- **GSP FW**: `nvidia/gb202/gsp/gsp-000.bin` (internal version 615.02, chips_a / r000)
- **Host**: AMD IOMMU enabled (IO_PAGE_FAULT events are benign pre-driver noise)
- **Kernel built with Clang** → must use `CC=clang LD=ld.lld LLVM=1`

## Commit History (30 commits, chronological)

### Phase 1: GSP/RPC Protocol for r000 firmware
| Commit | Description |
|--------|------------|
| `e459c08ac12a` | Update message queue protocol to MCTP/NVDM framing for r000 GSP |
| `2e390959dc00` | Fix off-by-one in message queue read pointer advance |
| `59e24ce3424d` | Gracefully handle GET_DEVICE_INFO_TABLE RM RPC failure |
| `f76169775a6e` | Use engineCaps from GetGspStaticInfo for engine bitmap |
| `2c3e336c75f3` | Bump ctrl_buff version to 2 for chips_a GSP plugin |

### Phase 2: MMU V3 & Blackwell Hardware Support
| Commit | Description |
|--------|------------|
| `db5632c4b12b` | Support MMU v3 (Blackwell) and fix BAR1 index for 64-bit BAR0 |
| `3aa6bfd56f4f` | Force mock_bootload to run regardless of vgpu_enabled |
| `d0b84fbf87e4` | Force vgpu_requested through FSP mode check for testing |
| `3611a2351807` | Use Blackwell DC-3Q vGPU type blob for mock_bootload |
| `bfc8b471acff` | Fix vgpu type name overflow (35 > 32 bytes) |
| `c78852ec5205` | Expand VGPU_TYPE_NAME_MAX to 64 to match RM struct |

### Phase 3: ENOMEM Bug — V3 level_index Fix
| Commit | Description |
|--------|------------|
| `f3c9d56a5c8e` | Cap BarUser VA space to 256MB (temporary, later removed) |
| `7d0c729ea65a` | Add ENOMEM tracing in VMM map path, remove 256MB cap |
| `ebd63435c08f` | Add tracing in execute_map to find ENOMEM source |
| `578b380ca4de` | Fix: RBTree has no len() |
| `8979b80575d0` | More granular tracing in PDE drain loop |
| `04fad814d323` | Fix: use .raw() for VramAddress LowerHex |
| `f5acd2310b44` | **ROOT CAUSE FIX**: Correct V3 page table level_index mapping for 6-level hierarchy |

### Phase 4: BAR1 Timeout Debugging (current)
| Commit | Description |
|--------|------------|
| `9e09018f6959` | Trace wait_plugin_ready ctrl buffer values |
| `153f652671f6` | Dump PDB entries at bar1_pde_base |
| `f4f169d9d678` | Fix: usize type for pramin read |
| `0d1869d37815` | Fix: remove duplicate brace |
| `54f48c407eaa` | Readback PDB after execute_map to verify writes |
| `1317348af314` | Print child page VRAM addr and alignment check |
| `083b9144950a` | Fix: add parens for bitwise precedence |
| `b3ab0c18c69b` | Fix: compute alignment check outside macro |
| `8aad71cc12a1` | **KEY TEST**: PRAMIN→BAR1 round-trip test in wait_plugin_ready |
| `4fd202d2d093` | Fix: move comparison outside pr_info macro |
| `3a86b54231c6` | Raw BAR1 read at offset 0 before BarUser init |
| `fb755b085393` | Fix: use read32 instead of try_read32 on Bar1 (STILL BUILD-BROKEN) |

## Key Files Changed

| File | Changes |
|------|---------|
| `mm.rs` | Added `MmuVersion` param to `VirtualAddress::level_index()`, added `pd3_v3_index` bitfield for V3 L1 9-bit index |
| `mm/vmm.rs` | V3-aware VMM: `entries_per_page()`, `pt_pages_upper_bound()`, debug prints throughout `prepare_map`/`execute_map` |
| `mm/pagetable.rs` | `MmuVersion` enum, V3 dispatch in `Pde`/`DualPde`/`Pte` wrappers, `entry_size()` for dual PDE |
| `mm/pagetable/ver3.rs` | Full V3 PDE/PTE/DualPde bitfield definitions with PCF |
| `mm/pagetable/walk.rs` | Pass `mmu_version` to `level_index()` calls |
| `mm/bar_user.rs` | Self-test skip for V3 |
| `gpu.rs` | PDB dump, raw BAR1 probe, force vgpu_requested |
| `vgpu.rs` | PRAMIN→BAR1 round-trip test, ctrl_buf polling debug, version bumps |
| `gsp/cmdq.rs` | MCTP/NVDM message framing |
| `gsp/fw.rs` | r000 firmware loading |
| `gsp/fw/r000_00/bindings.rs` | Additional RPC struct fields (bar1PdeBase, engineCaps, etc.) |
| `driver.rs` | Dynamic BAR1 region detection for 64-bit BAR0 |
| `blackwell_3q.bin` | Blackwell DC-3Q vGPU type binary blob |

## Root Cause: ENOMEM Bug (FIXED)

**Problem**: `probe with driver NovaCore failed with error -12` during `bar_user.map()`.

**Root cause**: `VirtualAddress::level_index()` did not differentiate between MMU V2 and V3. For V3 (6-level hierarchy), Level 4 (dual PDE) was incorrectly mapped to `pt_index [20:12]` instead of `pd0_index [28:21]`. This caused `ensure_pte_path` to generate excessive page table pages (512 instead of ~2), exceeding the pre-reserved capacity in `execute_map`.

**Fix** (commit `f5acd2310b44`):
- Added `pd3_v3_index` bitfield (bits [55:47]) for V3's L1 level
- Made `level_index()` take `MmuVersion` parameter
- V3 mapping: PDB→0, L1→pd3_v3_index, L2→pd2_index, L3→pd1_index, L4→pd0_index, L5→pt_index

### V2 vs V3 Page Table Layout

```
V2 (5 levels, Turing/Ampere/Ada):
  PDB:  bits [48:47]  2 bits   4 entries
  L1:   bits [46:38]  9 bits   512 entries
  L2:   bits [37:29]  9 bits   512 entries
  L3:   bits [28:21]  8 bits   256 entries  (dual PDE)
  L4:   bits [20:12]  9 bits   512 entries  (PTE)

V3 (6 levels, Hopper/Blackwell):
  PDB:  bits [56:47]  10 bits  (only low bits used)
  L1:   bits [55:47]  9 bits   512 entries
  L2:   bits [46:38]  9 bits   512 entries
  L3:   bits [37:29]  9 bits   512 entries
  L4:   bits [28:21]  8 bits   256 entries  (dual PDE, 16-byte entries)
  L5:   bits [20:12]  9 bits   512 entries  (PTE)
```

## Current Blocker: BAR1 Returns Poison (`0xbad0ac00`)

**Status**: Page table writes are verified correct via PRAMIN readback, but BAR1 reads return `0xbad0ac00` (GPU access fault poison).

### Evidence
```
PRAMIN->BAR1 test: wrote 0xdeadbeef to VRAM 0x1774000000, BAR1 read=0xbad0ac00 FAIL
PRAMIN readback=0xdeadbeef
```

### What Works
- GSP boots and responds (615.02 firmware)
- `mock_bootload` RPC succeeds (instance created, bootload sent)
- Page table chain is correct: PDB[0] → L1[0] → L2[0] → L3[255] → L4[254,255] → PTEs
- All allocated page table pages are 4K-aligned
- PRAMIN readback confirms PDE writes persisted
- TLB flush completes without error

### What Doesn't Work
- BAR1 reads at mapped VA (`0x1FFFC00000`, near end of 128GB range) return `0xbad0ac00`
- Every 4-byte offset in the control buffer reads as `0xbad0ac00`

### Leading Hypotheses (Untested)
1. **BAR1 instance block not bound to our PDB**: GSP-RM owns the instance block. Maybe on Blackwell, it uses a different PDB or mechanism.
2. **TLB flush register incompatible with V3**: The flush registers at `0x00b830a0-b0` may need different format/address for Blackwell.
3. **BAR1 iomap doesn't cover 128GB**: The ioremap for a 128GB BAR might fail silently or the kernel might limit the mapping.
4. **V3 PDE needs additional bits/flags**: Maybe V3 PDEs need PCF bits set or other fields we're not populating.
5. **VFN allocation at top of range**: Buddy allocator allocates from the end (VFN `0x1FFFC00`). GSP may have instance block configured for a smaller VA range.

### Next Steps to Try
- Read raw BAR1 at offset 0 to check if BAR1 works at all (build was broken, needs `try_read32` API fix)
- Try mapping at a LOW VA instead of top-of-range (constrain buddy allocation)
- Check if GSP's instance block PDB matches our `bar1_pde_base`
- Compare TLB flush mechanism with nouveau's Blackwell support
- Check Joel Fernandes's patches for any BAR1-specific V3 handling

## Debug Prints Currently Active

Many `pr_info!` / `pr_err!` debug prints are active in:
- `mm/vmm.rs`: VMM::new, prepare_map, execute_map (PDE drain, PTE write, TLB flush, readback)
- `gpu.rs`: PDB dump, raw BAR1 probe
- `vgpu.rs`: wait_plugin_ready (PRAMIN→BAR1 test, ctrl_buf dump, polling values)

These should be removed before any production use.

## Reference: Joel's nouveau Patches

Joel Fernandes's patches for nouveau on Blackwell were reviewed during debugging. Key takeaway: his implementation uses separate `VirtualAddressV2`/`VirtualAddressV3` bitfield types and version-aware `level_index()` logic, confirming our fix approach was correct.
