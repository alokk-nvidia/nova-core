// SPDX-License-Identifier: GPL-2.0

//! BAR1 user interface for CPU access to GPU virtual memory. Used for USERD
//! for GPU work submission, and applications to access GPU buffers via mmap().

use kernel::{
    device,
    io::Io,
    prelude::*, //
};

use crate::{
    driver::Bar1,
    gpu::Chipset,
    mm::{
        vmm::{
            MappedRange,
            Vmm, //
        },
        GpuMm,
        Pfn,
        Vfn,
        VirtualAddress,
        VramAddress,
        PAGE_SIZE, //
    },
    num::IntoSafeCast,
};

/// BAR1 user interface for virtual memory mappings.
///
/// Owns a [`Vmm`] instance with virtual address tracking and provides
/// BAR1-specific mapping and cleanup operations.
pub(crate) struct BarUser {
    vmm: Vmm,
}

impl BarUser {
    /// Create a new [`BarUser`] with virtual address tracking.
    pub(crate) fn new(pdb_addr: VramAddress, chipset: Chipset, va_size: u64) -> Result<Self> {
        Ok(Self {
            vmm: Vmm::new(pdb_addr, chipset.mmu_version(), va_size)?,
        })
    }

    /// Map physical pages to a contiguous BAR1 virtual range.
    pub(crate) fn map<'a>(
        &'a mut self,
        dev: &'a device::Device<device::Bound>,
        mm: &'a GpuMm,
        bar: &'a Bar1,
        pfns: &[Pfn],
        writable: bool,
    ) -> Result<BarUserAccess<'a>> {
        if pfns.is_empty() {
            return Err(EINVAL);
        }

        let mapped = self.vmm.map_pages(dev, mm, pfns, None, writable)?;

        Ok(BarUserAccess {
            vmm: &mut self.vmm,
            dev,
            mm,
            bar,
            mapped: Some(mapped),
        })
    }
}

/// Access object for a mapped BAR1 region.
///
/// Wraps a [`MappedRange`] and provides BAR1 access. When dropped,
/// unmaps pages and releases the VA range (by passing the range to
/// [`Vmm::unmap_pages()`], which consumes it).
pub(crate) struct BarUserAccess<'a> {
    vmm: &'a mut Vmm,
    dev: &'a device::Device<device::Bound>,
    mm: &'a GpuMm,
    bar: &'a Bar1,
    /// Needs to be an `Option` so that we can `take()` it and call `Drop`
    /// on it in [`Vmm::unmap_pages()`].
    mapped: Option<MappedRange>,
}

impl<'a> BarUserAccess<'a> {
    /// Returns the active mapping.
    fn mapped(&self) -> &MappedRange {
        // `mapped` is only `None` after `take()` in `Drop`; accessors are
        // never called from within `Drop`, so `unwrap()` never panics.
        self.mapped.as_ref().unwrap()
    }

    /// Get the base virtual address of this mapping.
    pub(crate) fn base(&self) -> VirtualAddress {
        VirtualAddress::from(self.mapped().vfn_start)
    }

    /// Get the total size of the mapped region in bytes.
    pub(crate) fn size(&self) -> usize {
        self.mapped().num_pages * PAGE_SIZE
    }

    /// Get the starting virtual frame number.
    pub(crate) fn vfn_start(&self) -> Vfn {
        self.mapped().vfn_start
    }

    /// Get the number of pages in this mapping.
    pub(crate) fn num_pages(&self) -> usize {
        self.mapped().num_pages
    }

    /// Translate an offset within this mapping to a BAR1 aperture offset.
    fn bar_offset(&self, offset: usize) -> Result<usize> {
        if offset >= self.size() {
            return Err(EINVAL);
        }

        let base_vfn: usize = self.mapped().vfn_start.raw().into_safe_cast();
        let base = base_vfn.checked_mul(PAGE_SIZE).ok_or(EOVERFLOW)?;
        base.checked_add(offset).ok_or(EOVERFLOW)
    }

    // Fallible accessors with runtime bounds checking.

    /// Read a 32-bit value at the given offset.
    pub(crate) fn try_read32(&self, offset: usize) -> Result<u32> {
        self.bar.try_read32(self.bar_offset(offset)?)
    }

    /// Write a 32-bit value at the given offset.
    pub(crate) fn try_write32(&self, value: u32, offset: usize) -> Result {
        self.bar.try_write32(value, self.bar_offset(offset)?)
    }

    /// Read a 64-bit value at the given offset.
    pub(crate) fn try_read64(&self, offset: usize) -> Result<u64> {
        self.bar.try_read64(self.bar_offset(offset)?)
    }

    /// Write a 64-bit value at the given offset.
    pub(crate) fn try_write64(&self, value: u64, offset: usize) -> Result {
        self.bar.try_write64(value, self.bar_offset(offset)?)
    }
}

impl Drop for BarUserAccess<'_> {
    fn drop(&mut self) {
        if let Some(mapped) = self.mapped.take() {
            if self.vmm.unmap_pages(self.dev, self.mm, mapped).is_err() {
                kernel::pr_warn_once!("BarUserAccess: unmap_pages failed.\n");
            }
        }
    }
}

/// Run MM subsystem self-tests during probe.
///
/// Tests page table infrastructure and `BAR1` MMIO access using the `BAR1`
/// address space. Uses the `GpuMm`'s buddy allocator to allocate page tables
/// and test pages as needed.
#[cfg(CONFIG_NOVA_MM_SELFTESTS)]
pub(crate) fn run_self_test(
    pdev: &device::Device<device::Bound>,
    mm: &GpuMm,
    bar1: &Bar1,
    bar1_pdb: u64,
    chipset: Chipset,
) -> Result {
    use kernel::gpu::buddy::{
        GpuBuddyAllocFlags,
        GpuBuddyAllocMode, //
    };
    use kernel::ptr::Alignment;
    use kernel::sizes::{
        SZ_16K,
        SZ_32K,
        SZ_4K,
        SZ_64K, //
    };

    // Test patterns.
    const PATTERN_PRAMIN: u32 = 0xDEAD_BEEF;
    const PATTERN_BAR1: u32 = 0xCAFE_BABE;

    let dev = pdev;
    dev_info!(dev, "MM: Starting self-test...\n");

    let pdb_addr = VramAddress::new(bar1_pdb);

    // Check if initial page tables are in VRAM.
    if crate::mm::pagetable::check_pdb_valid(pdev, mm.pramin(), pdb_addr, chipset).is_err() {
        dev_info!(dev, "MM: Self-test SKIPPED - no valid VRAM page tables\n");
        return Ok(());
    }

    // Set up a test page from the buddy allocator.
    let test_page_blocks = KBox::pin_init(
        mm.buddy().alloc_blocks(
            GpuBuddyAllocMode::Simple,
            SZ_4K.into_safe_cast(),
            Alignment::new::<SZ_4K>(),
            GpuBuddyAllocFlags::default(),
        ),
        GFP_KERNEL,
    )?;
    let test_vram_offset = test_page_blocks.iter().next().ok_or(ENOMEM)?.offset();
    let test_vram = VramAddress::new(test_vram_offset);
    let test_pfn = Pfn::from(test_vram);

    // Create a VMM of size 64K to track virtual memory mappings.
    let mut vmm = Vmm::new(pdb_addr, chipset.mmu_version(), SZ_64K.into_safe_cast())?;

    // Create a test mapping.
    let mapped = vmm.map_pages(pdev, mm, &[test_pfn], None, true)?;
    let test_vfn = mapped.vfn_start;

    // Pre-compute test addresses for the PRAMIN to BAR1 read test.
    let vfn_offset: usize = test_vfn.raw().into_safe_cast();
    let bar1_base_offset = vfn_offset.checked_mul(PAGE_SIZE).ok_or(EOVERFLOW)?;
    let bar1_read_offset: usize = bar1_base_offset + 0x100;
    let vram_read_addr: usize = test_vram.raw() + 0x100;

    // Test 1: Write via PRAMIN, read via BAR1.
    {
        let mut window = mm.pramin().get_window(pdev)?;
        window.try_write32(vram_read_addr, PATTERN_PRAMIN)?;
    }

    // Read back via BAR1 aperture.
    let bar1_value = bar1.try_read32(bar1_read_offset)?;

    let test1_passed = if bar1_value == PATTERN_PRAMIN {
        true
    } else {
        dev_err!(
            dev,
            "MM: Test 1 FAILED - Expected {:#010x}, got {:#010x}\n",
            PATTERN_PRAMIN,
            bar1_value
        );
        false
    };

    // Cleanup - invalidate PTE.
    vmm.unmap_pages(pdev, mm, mapped)?;

    // Test 2: Two-phase prepare/execute API.
    let prepared = vmm.prepare_map(pdev, mm, 1, None)?;
    let mapped2 = vmm.execute_map(pdev, mm, prepared, &[test_pfn], true)?;
    let readback = vmm.read_mapping(pdev, mm, mapped2.vfn_start)?;
    let test2_passed = if readback == Some(test_pfn) {
        true
    } else {
        dev_err!(dev, "MM: Test 2 FAILED - Two-phase map readback mismatch\n");
        false
    };
    vmm.unmap_pages(pdev, mm, mapped2)?;

    // Test 3: Range-constrained allocation with a hole — exercises block.size()-driven
    // BAR1 mapping. A 4K hole is punched at base+16K, then a single 32K allocation
    // is requested within [base, base+36K). The buddy allocator must split around the
    // hole, returning multiple blocks (expected: {16K, 4K, 8K, 4K} = 32K total).
    // Each block is mapped into BAR1 and verified via PRAMIN read-back.
    //
    // Address layout (base = 0x10000):
    //   [    16K    ] [HOLE 4K] [4K] [ 8K ] [4K]
    //   0x10000       0x14000  0x15000 0x16000 0x18000 0x19000
    let range_base: u64 = SZ_64K.into_safe_cast();
    let sz_4k: u64 = SZ_4K.into_safe_cast();
    let sz_16k: u64 = SZ_16K.into_safe_cast();
    let sz_32k_4k: u64 = (SZ_32K + SZ_4K).into_safe_cast();

    // Punch a 4K hole at base+16K so the subsequent 32K allocation must split.
    let _hole = KBox::pin_init(
        mm.buddy().alloc_blocks(
            GpuBuddyAllocMode::Range(range_base + sz_16k..range_base + sz_16k + sz_4k),
            SZ_4K.into_safe_cast(),
            Alignment::new::<SZ_4K>(),
            GpuBuddyAllocFlags::default(),
        ),
        GFP_KERNEL,
    )?;

    // Allocate 32K within [base, base+36K). The hole forces the allocator to return
    // split blocks whose sizes are determined by buddy alignment.
    let blocks = KBox::pin_init(
        mm.buddy().alloc_blocks(
            GpuBuddyAllocMode::Range(range_base..range_base + sz_32k_4k),
            SZ_32K.into_safe_cast(),
            Alignment::new::<SZ_4K>(),
            GpuBuddyAllocFlags::default(),
        ),
        GFP_KERNEL,
    )?;

    let mut test3_passed = true;
    let mut total_size = 0usize;

    for block in blocks.iter() {
        total_size += IntoSafeCast::<usize>::into_safe_cast(block.size());

        // Map all pages of this block.
        let page_size: u64 = PAGE_SIZE.into_safe_cast();
        let num_pages: usize = (block.size() / page_size).into_safe_cast();

        let mut pfns = KVec::new();
        for j in 0..num_pages {
            let j_u64: u64 = j.into_safe_cast();
            pfns.push(
                Pfn::from(VramAddress::new(
                    block.offset() + j_u64.checked_mul(page_size).ok_or(EOVERFLOW)?,
                )),
                GFP_KERNEL,
            )?;
        }

        let mapped = vmm.map_pages(pdev, mm, &pfns, None, true)?;
        let bar1_base_vfn: usize = mapped.vfn_start.raw().into_safe_cast();
        let bar1_base = bar1_base_vfn.checked_mul(PAGE_SIZE).ok_or(EOVERFLOW)?;

        for j in 0..num_pages {
            let page_bar1_off = bar1_base + j * PAGE_SIZE;
            let j_u64: u64 = j.into_safe_cast();
            let page_phys = block.offset()
                + j_u64
                    .checked_mul(PAGE_SIZE.into_safe_cast())
                    .ok_or(EOVERFLOW)?;

            bar1.try_write32(PATTERN_BAR1, page_bar1_off)?;

            let pramin_val = {
                let mut window = mm.pramin().get_window(pdev)?;
                window.try_read32(page_phys.into_safe_cast())?
            };

            if pramin_val != PATTERN_BAR1 {
                dev_err!(
                    dev,
                    "MM: Test 3 FAILED block offset {:#x} page {} (val={:#x})\n",
                    block.offset(),
                    j,
                    pramin_val
                );
                test3_passed = false;
            }
        }

        vmm.unmap_pages(pdev, mm, mapped)?;
    }

    // Verify aggregate: all returned block sizes must sum to allocation size.
    if total_size != SZ_32K {
        dev_err!(
            dev,
            "MM: Test 3 FAILED - total size {} != expected {}\n",
            total_size,
            SZ_32K
        );
        test3_passed = false;
    }

    // Release Tests 1-3's Vmm before Test 4 constructs a fresh BarUser on
    // the same PDB.
    drop(vmm);

    // Test 4: Exercise `BarUser::map()` end-to-end.
    let mut bar_user = BarUser::new(pdb_addr, chipset, SZ_64K.into_safe_cast())?;
    let test4_passed = {
        let access = bar_user.map(pdev, mm, bar1, &[test_pfn], true)?;

        // Write pattern via PRAMIN, read via BarUserAccess.
        {
            let mut window = mm.pramin().get_window(pdev)?;
            window.try_write32(test_vram.raw(), PATTERN_BAR1)?;
        }

        let readback = access.try_read32(0)?;
        if readback == PATTERN_BAR1 {
            true
        } else {
            dev_err!(
                dev,
                "MM: Test 4 FAILED - Expected {:#010x}, got {:#010x}\n",
                PATTERN_BAR1,
                readback
            );
            false
        }
    };

    if test1_passed && test2_passed && test3_passed && test4_passed {
        dev_info!(dev, "MM: All self-tests PASSED\n");
        Ok(())
    } else {
        dev_err!(dev, "MM: Self-tests FAILED\n");
        Err(EIO)
    }
}
