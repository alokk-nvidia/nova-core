// SPDX-License-Identifier: GPL-2.0

//! Memory management subsystems for nova-core.

#![expect(dead_code)]

pub(crate) mod bar_user;
pub(crate) mod pagetable;
pub(crate) mod pramin;
pub(crate) mod tlb;
pub(crate) mod vmm;

use kernel::{
    devres::Devres,
    gpu::buddy::{
        AllocatedBlocks,
        BuddyFlags,
        GpuBuddy,
        GpuBuddyAllocParams,
        GpuBuddyParams, //
    },
    prelude::*,
    sizes::SZ_4K,
    sync::Arc, //
};

use crate::{
    driver::Bar0,
    mm::pagetable::MmuVersion,
    num::{u64_as_usize, IntoSafeCast}, //
};

pub(crate) use tlb::Tlb;

/// GPU Memory Manager - owns all core MM components.
///
/// Provides centralized ownership of memory management resources:
/// - [`GpuBuddy`] allocator for VRAM page table allocation.
/// - [`pramin::Pramin`] for direct VRAM access.
/// - [`Tlb`] manager for translation buffer flush operations.
#[pin_data]
pub(crate) struct GpuMm {
    buddy: GpuBuddy,
    #[pin]
    pramin: pramin::Pramin,
    #[pin]
    tlb: Tlb,
}

impl GpuMm {
    /// Create a pin-initializer for `GpuMm`.
    pub(crate) fn new(
        bar: Arc<Devres<Bar0>>,
        buddy_params: GpuBuddyParams,
    ) -> Result<impl PinInit<Self>> {
        let buddy = GpuBuddy::new(buddy_params)?;
        let tlb_init = Tlb::new(bar.clone());
        let pramin_init = pramin::Pramin::new(bar)?;

        Ok(pin_init!(Self {
            buddy,
            pramin <- pramin_init,
            tlb <- tlb_init,
        }))
    }

    /// Access the [`GpuBuddy`] allocator.
    pub(crate) fn buddy(&self) -> &GpuBuddy {
        &self.buddy
    }

    /// Access the [`pramin::Pramin`].
    pub(crate) fn pramin(&self) -> &pramin::Pramin {
        &self.pramin
    }

    /// Access the [`Tlb`] manager.
    pub(crate) fn tlb(&self) -> &Tlb {
        &self.tlb
    }
}

/// Page size in bytes (4 KiB).
pub(crate) const PAGE_SIZE: usize = SZ_4K;

bitfield! {
    pub(crate) struct VramAddress(u64), "Physical VRAM address in GPU video memory" {
        11:0    offset          as u64, "Offset within 4KB page";
        63:12   frame_number    as u64 => Pfn, "Physical frame number";
    }
}

impl VramAddress {
    /// Create a new VRAM address from a raw value.
    pub(crate) const fn new(addr: u64) -> Self {
        Self(addr)
    }

    /// Get the raw address value as `usize` (useful for MMIO offsets).
    pub(crate) const fn raw(&self) -> usize {
        u64_as_usize(self.0)
    }

    /// Get the raw address value as `u64`.
    pub(crate) const fn raw_u64(&self) -> u64 {
        self.0
    }
}

impl PartialEq for VramAddress {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for VramAddress {}

impl PartialOrd for VramAddress {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VramAddress {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl From<Pfn> for VramAddress {
    fn from(pfn: Pfn) -> Self {
        Self::default().set_frame_number(pfn)
    }
}

/// A block of allocated VRAM. Freed when dropped.
pub(crate) struct VramBlock {
    blocks: Pin<KBox<AllocatedBlocks>>,
    addr: u64,
    size: u64,
}

impl VramBlock {
    /// The physical VRAM address of this block.
    pub(crate) fn addr(&self) -> u64 {
        self.addr
    }

    /// The size of this block in bytes.
    pub(crate) fn size(&self) -> u64 {
        self.size
    }

    /// The physical frame number for this block.
    pub(crate) fn pfn(&self) -> Pfn {
        Pfn::from(VramAddress::new(self.addr))
    }
}

/// Allocate a contiguous block of VRAM.
pub(crate) fn alloc_vram(mm: &GpuMm, size: u64, align: u64) -> Result<VramBlock> {
    let params = GpuBuddyAllocParams {
        start_range_address: 0,
        end_range_address: 0,
        size,
        min_block_size: align.max(PAGE_SIZE.into_safe_cast()),
        buddy_flags: BuddyFlags::empty(),
    };
    let blocks = KBox::pin_init(mm.buddy().alloc_blocks(params), GFP_KERNEL)?;
    let first = blocks.iter().next().ok_or(ENOMEM)?;
    Ok(VramBlock {
        addr: first.offset(),
        size: first.size(),
        blocks,
    })
}

// GPU virtual address decomposed into MMU v2 page table level indices.
//
// Bit widths match the hardware layout from `kern_gmmu_fmt_gp10x.c`:
//   V2 (5 levels):
//     PD3 (root): [48:47] = 2 bits,  4 entries
//     PD2:        [46:38] = 9 bits, 512 entries
//     PD1:        [37:29] = 9 bits, 512 entries
//     PD0 (dual): [28:21] = 8 bits, 256 entries
//     PT:         [20:12] = 9 bits, 512 entries
//
//   V3 (6 levels) — same bit positions, but level assignment shifts by 1:
//     PDB:        [56:47]  (10 bits, but only low bits matter for small VAs)
//     L1:         [55:47] = 9 bits, 512 entries
//     L2:         [46:38] = 9 bits, 512 entries  (same bits as V2 PD2/L1)
//     L3:         [37:29] = 9 bits, 512 entries  (same bits as V2 PD1/L2)
//     L4 (dual):  [28:21] = 8 bits, 256 entries  (same bits as V2 PD0/L3)
//     L5 (PTE):   [20:12] = 9 bits, 512 entries  (same bits as V2 PT/L4)
bitfield! {
    pub(crate) struct VirtualAddress(u64), "Virtual address in GPU address space" {
        11:0    offset          as u64, "Offset within 4KB page";
        20:12   pt_index        as u64, "PT index (PTE, 9 bits)";
        28:21   pd0_index       as u64, "PD0 index (Dual PDE, 8 bits)";
        37:29   pd1_index       as u64, "PD1 index (9 bits)";
        46:38   pd2_index       as u64, "PD2 index (9 bits)";
        48:47   pd3_index       as u64, "PD3 index (root PDB v2, 2 bits)";
        55:47   pd3_v3_index    as u64, "PD3 index (L1 for v3, 9 bits)";
        63:12   frame_number    as u64 => Vfn, "Virtual frame number";
    }
}

impl VirtualAddress {
    /// Create a new virtual address from a raw value.
    #[expect(dead_code)]
    pub(crate) const fn new(addr: u64) -> Self {
        Self(addr)
    }

    /// Get the page table index for a given level (0-5).
    ///
    /// V2 (5 levels): PDB=pd3, L1=pd2, L2=pd1, L3=pd0(dual), L4=pt
    /// V3 (6 levels): PDB=0, L1=pd3_v3, L2=pd2, L3=pd1, L4=pd0(dual), L5=pt
    pub(crate) fn level_index(&self, level: u64, mmu_version: MmuVersion) -> u64 {
        match mmu_version {
            MmuVersion::V2 => match level {
                0 => self.pd3_index(),
                1 => self.pd2_index(),
                2 => self.pd1_index(),
                3 => self.pd0_index(),
                4 => self.pt_index(),
                _ => 0,
            },
            MmuVersion::V3 => match level {
                0 => 0,
                1 => self.pd3_v3_index(),
                2 => self.pd2_index(),
                3 => self.pd1_index(),
                4 => self.pd0_index(),
                5 => self.pt_index(),
                _ => 0,
            },
        }
    }
}

impl From<Vfn> for VirtualAddress {
    fn from(vfn: Vfn) -> Self {
        Self::default().set_frame_number(vfn)
    }
}

/// Physical Frame Number.
///
/// Represents a physical page in VRAM.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Pfn(u64);

impl Pfn {
    /// Create a new PFN from a frame number.
    pub(crate) const fn new(frame_number: u64) -> Self {
        Self(frame_number)
    }

    /// Get the raw frame number.
    pub(crate) const fn raw(self) -> u64 {
        self.0
    }
}

impl From<VramAddress> for Pfn {
    fn from(addr: VramAddress) -> Self {
        addr.frame_number()
    }
}

impl From<u64> for Pfn {
    fn from(val: u64) -> Self {
        Self(val)
    }
}

impl From<Pfn> for u64 {
    fn from(pfn: Pfn) -> Self {
        pfn.0
    }
}

/// Virtual Frame Number.
///
/// Represents a virtual page in GPU address space.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Vfn(u64);

impl Vfn {
    /// Create a new VFN from a frame number.
    pub(crate) const fn new(frame_number: u64) -> Self {
        Self(frame_number)
    }

    /// Get the raw frame number.
    pub(crate) const fn raw(self) -> u64 {
        self.0
    }
}

impl From<VirtualAddress> for Vfn {
    fn from(addr: VirtualAddress) -> Self {
        addr.frame_number()
    }
}

impl From<u64> for Vfn {
    fn from(val: u64) -> Self {
        Self(val)
    }
}

impl From<Vfn> for u64 {
    fn from(vfn: Vfn) -> Self {
        vfn.0
    }
}
