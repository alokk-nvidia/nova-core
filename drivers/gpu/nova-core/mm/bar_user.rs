// SPDX-License-Identifier: GPL-2.0

//! BAR1 user interface for CPU access to GPU virtual memory. Used for USERD
//! for GPU work submission, and applications to access GPU buffers via mmap().

use kernel::{
    io::Io,
    new_mutex,
    prelude::*,
    ptr::{
        Alignable,
        Alignment, //
    },
    sync::Mutex, //
};

use crate::{
    driver::Bar1,
    gpu::Chipset,
    mm::{
        vmm::{
            MappedRange,
            Vmm, //
        },
        vram::VramRegion,
        GpuMm,
        Pfn,
        Vfn,
        VirtualAddress,
        VramAddress,
        PAGE_SIZE, //
    },
    num::IntoSafeCast,
};

#[cfg(CONFIG_NOVA_CORE_SELFTESTS)]
use kernel::device;

/// BAR1 user interface for virtual memory mappings.
///
/// Owns the [`Vmm`] for the BAR1 address space.
#[pin_data]
pub(crate) struct BarUser<'a> {
    #[pin]
    vmm: Mutex<Vmm>,
    bar1: Bar1<'a>,
}

impl<'a> BarUser<'a> {
    /// Create a pin-initializer for [`BarUser`].
    pub(crate) fn new(
        pdb_addr: VramAddress,
        chipset: Chipset,
        va_size: u64,
        bar1: Bar1<'a>,
    ) -> impl PinInit<Self, Error> + 'a {
        try_pin_init!(Self {
            vmm <- new_mutex!(
                Vmm::new(pdb_addr, chipset.mmu_version(), va_size)?,
                "bar_user_vmm",
            ),
            bar1,
        })
    }

    /// Map physical pages to a contiguous BAR1 virtual range.
    pub(crate) fn map<'access>(
        &'access self,
        mm: &mut GpuMm<'_>,
        pfns: &[Pfn],
        writable: bool,
    ) -> Result<BarUserAccess<'access, 'a>> {
        if pfns.is_empty() {
            return Err(EINVAL);
        }
        let mut vmm = self.vmm.lock();
        let mapped = vmm.map_pages(mm, pfns, None, writable)?;

        Ok(BarUserAccess {
            bar_user: self,
            mapped: Some(mapped),
        })
    }
}

/// Access object for a mapped BAR1 region.
///
/// Borrows the [`BarUser`] managing the mapping for the lifetime of the access object.
pub(crate) struct BarUserAccess<'access, 'gpu> {
    bar_user: &'access BarUser<'gpu>,
    /// Cleared only after PTE invalidation and its TLB flush succeed.
    mapped: Option<MappedRange>,
}

#[expect(dead_code)]
impl BarUserAccess<'_, '_> {
    /// Tear down the BAR1 mapping.
    pub(crate) fn release(mut self, mm: &mut GpuMm<'_>) -> Result {
        self.unmap(mm)
    }

    fn unmap(&mut self, mm: &mut GpuMm<'_>) -> Result {
        let Some(mapped) = self.mapped.as_ref() else {
            return Ok(());
        };
        let mut vmm = self.bar_user.vmm.lock();
        vmm.unmap_pages(mm, mapped)?;
        self.mapped = None;
        Ok(())
    }

    /// Returns the active mapping.
    fn mapped(&self) -> &MappedRange {
        // `release` consumes the access, so no accessor can run after successful release.
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
        let off = self.bar_offset(offset)?;
        self.bar_user.bar1.try_read32(off)
    }

    /// Write a 32-bit value at the given offset.
    pub(crate) fn try_write32(&self, value: u32, offset: usize) -> Result {
        let off = self.bar_offset(offset)?;
        self.bar_user.bar1.try_write32(value, off)
    }

    /// Read a 64-bit value at the given offset.
    pub(crate) fn try_read64(&self, offset: usize) -> Result<u64> {
        let off = self.bar_offset(offset)?;
        self.bar_user.bar1.try_read64(off)
    }

    /// Write a 64-bit value at the given offset.
    pub(crate) fn try_write64(&self, value: u64, offset: usize) -> Result {
        let off = self.bar_offset(offset)?;
        self.bar_user.bar1.try_write64(value, off)
    }
}

impl Drop for BarUserAccess<'_, '_> {
    fn drop(&mut self) {
        if self.mapped.is_some() {
            kernel::pr_warn!(
                "BarUserAccess dropped with a live mapping; BAR1 VA remains reserved.\n"
            );
        }
        // The inner `MappedRange`'s own `MustUnmapGuard` will also fire,
        // identifying the leaked VA range.
    }
}

/// A BAR mapping that retains its backing VRAM region.
#[must_use]
pub(crate) struct BarMapping<'map, 'gpu> {
    access: BarUserAccess<'map, 'gpu>,
    mm: &'map Mutex<GpuMm<'gpu>>,
    region: VramRegion,
    region_offset: usize,
    region_size: usize,
    unmap_error: Option<Error>,
}

#[expect(dead_code)]
impl<'map, 'gpu> BarMapping<'map, 'gpu> {
    /// Maps the containing pages while restricting CPU access to the requested byte range.
    pub(crate) fn new(
        bar_user: &'map BarUser<'gpu>,
        mm: &'map Mutex<GpuMm<'gpu>>,
        region: VramRegion,
        writable: bool,
    ) -> Result<Self> {
        let page_size: u64 = PAGE_SIZE.into_safe_cast();
        let page_align = Alignment::new::<PAGE_SIZE>();

        let region_start = region.address();
        let region_end = region_start.checked_add(region.size()).ok_or(EOVERFLOW)?;

        let map_start = region_start.align_down(page_align);
        let map_end = region_end.align_up(page_align).ok_or(EOVERFLOW)?;
        let map_size = map_end.checked_sub(map_start).ok_or(EINVAL)?;
        let num_pages = usize::try_from(map_size / page_size).map_err(|_| EOVERFLOW)?;
        if num_pages == 0 {
            return Err(EINVAL);
        }

        let region_offset = usize::try_from(region_start - map_start).map_err(|_| EOVERFLOW)?;
        let region_size = usize::try_from(region.size()).map_err(|_| EOVERFLOW)?;

        let mut pfns = KVec::with_capacity(num_pages, GFP_KERNEL)?;
        for page in 0..num_pages {
            let page = u64::try_from(page).map_err(|_| EOVERFLOW)?;
            let byte_offset = page.checked_mul(page_size).ok_or(EOVERFLOW)?;
            let address = map_start.checked_add(byte_offset).ok_or(EOVERFLOW)?;
            pfns.push(Pfn::from(VramAddress::from_raw(address)), GFP_KERNEL)?;
        }

        let access = bar_user.map(&mut mm.lock(), &pfns, writable)?;

        Ok(Self {
            access,
            mm,
            region,
            region_offset,
            region_size,
            unmap_error: None,
        })
    }

    pub(crate) fn region(&self) -> &VramRegion {
        &self.region
    }

    pub(crate) fn gpu_va_addr(&self) -> Result<u64> {
        self.access()?
            .base()
            .into_raw()
            .checked_add(u64::try_from(self.region_offset).map_err(|_| EOVERFLOW)?)
            .ok_or(EOVERFLOW)
    }

    pub(crate) const fn size(&self) -> usize {
        self.region_size
    }

    fn access(&self) -> Result<&BarUserAccess<'map, 'gpu>> {
        if self.access.mapped.is_none() {
            return Err(ENODEV);
        }
        if let Some(error) = self.unmap_error {
            return Err(error);
        }
        Ok(&self.access)
    }

    fn access_offset(&self, offset: usize, width: usize) -> Result<usize> {
        let region_end = offset.checked_add(width).ok_or(EOVERFLOW)?;
        if region_end > self.region_size {
            return Err(EINVAL);
        }

        let access_offset = self.region_offset.checked_add(offset).ok_or(EOVERFLOW)?;
        if !access_offset.is_multiple_of(width) {
            return Err(EINVAL);
        }

        Ok(access_offset)
    }

    pub(crate) fn try_read32(&self, offset: usize) -> Result<u32> {
        self.access()?
            .try_read32(self.access_offset(offset, size_of::<u32>())?)
    }

    pub(crate) fn try_write32(&self, value: u32, offset: usize) -> Result {
        self.access()?
            .try_write32(value, self.access_offset(offset, size_of::<u32>())?)
    }

    pub(crate) fn try_write64(&self, value: u64, offset: usize) -> Result {
        self.access()?
            .try_write64(value, self.access_offset(offset, size_of::<u64>())?)
    }

    /// Unmap while retaining the mapping and its backing storage on failure.
    ///
    /// The first error is retained; later calls do not retry the operation.
    pub(crate) fn unmap(&mut self) -> Result {
        if let Some(error) = self.unmap_error {
            return Err(error);
        }
        let result = self.access.unmap(&mut self.mm.lock());
        if let Err(error) = result {
            self.unmap_error = Some(error);
        }
        result
    }
}

impl Drop for BarMapping<'_, '_> {
    /// Drop unmaps the region and may sleep. Call [`Self::unmap`] to observe errors;
    /// on failure the owner must retain this object until device teardown.
    fn drop(&mut self) {
        if self.unmap_error.is_some() {
            return;
        }
        if let Err(error) = self.unmap() {
            kernel::pr_err!("failed to unmap BAR1 region: {:?}\n", error);
        }
    }
}

/// Run MM subsystem self-tests during probe.
///
/// Tests page table infrastructure and `BAR1` MMIO access using the `BAR1`
/// address space. Uses the `GpuMm`'s buddy allocator to allocate page tables
/// and test pages as needed.
#[cfg(CONFIG_NOVA_CORE_SELFTESTS)]
pub(super) fn run_self_test(
    dev: &device::Device<device::Bound>,
    mm: &mut GpuMm<'_>,
    bar_user: &BarUser<'_>,
    bar1_pdb: u64,
    chipset: Chipset,
) -> Result {
    use kernel::{
        gpu::buddy::{
            GpuBuddyAllocFlags,
            GpuBuddyAllocMode, //
        },
        ptr::Alignment,
        sizes::{
            SZ_16K,
            SZ_32K,
            SZ_4K,
            SZ_64K, //
        },
    };

    // Test patterns.
    const PATTERN_PRAMIN: u32 = 0xDEAD_BEEF;
    const PATTERN_BAR1: u32 = 0xCAFE_BABE;

    let bar1 = bar_user.bar1;
    dev_info!(dev, "MM: Starting self-test...\n");

    let pdb_addr = VramAddress::from_raw(bar1_pdb);

    // Check if initial page tables are in VRAM.
    if crate::mm::pagetable::check_pdb_valid(mm.pramin_mut(), pdb_addr, chipset).is_err() {
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
    let test_vram = VramAddress::from_raw(test_vram_offset);
    let test_pfn = Pfn::from(test_vram);

    // Create a VMM of size 64K to track virtual memory mappings.
    let mut vmm = Vmm::new(pdb_addr, chipset.mmu_version(), SZ_64K.into_safe_cast())?;

    // Create a test mapping.
    let mapped = vmm.map_pages(mm, &[test_pfn], None, true)?;
    let test_vfn = mapped.vfn_start;

    // Pre-compute test addresses for the PRAMIN to BAR1 read test.
    let vfn_offset: usize = test_vfn.raw().into_safe_cast();
    let bar1_base_offset = vfn_offset.checked_mul(PAGE_SIZE).ok_or(EOVERFLOW)?;
    let bar1_read_offset: usize = bar1_base_offset + 0x100;
    let vram_read_addr = test_vram + 0x100;

    // Test 1: Write via PRAMIN, read via BAR1.
    mm.pramin_mut()
        .window_at::<u32>(vram_read_addr)?
        .view()
        .write_val(PATTERN_PRAMIN);

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
    vmm.unmap_pages(mm, &mapped)?;

    // Test 2: Two-phase prepare/execute API.
    let prepared = vmm.prepare_map(mm, 1, None)?;
    let mapped2 = vmm.execute_map(mm, prepared, &[test_pfn], true)?;
    // An old token must not invalidate a new mapping that reused its address.
    let repeated_unmap = vmm.unmap_pages(mm, &mapped);
    let readback = vmm.read_mapping(mm, mapped2.vfn_start)?;
    let test2_passed = if mapped2.vfn_start == test_vfn
        && repeated_unmap == Err(EINVAL)
        && readback == Some(test_pfn)
    {
        true
    } else {
        dev_err!(
            dev,
            "MM: Test 2 FAILED - Remapped page or stale-token rejection mismatch\n"
        );
        false
    };
    vmm.unmap_pages(mm, &mapped2)?;

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
                Pfn::from(VramAddress::from_raw(
                    block.offset() + j_u64.checked_mul(page_size).ok_or(EOVERFLOW)?,
                )),
                GFP_KERNEL,
            )?;
        }

        let mapped = vmm.map_pages(mm, &pfns, None, true)?;
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

            let pramin_val = mm
                .pramin_mut()
                .window_at::<u32>(VramAddress::from_raw(page_phys))?
                .view()
                .read_val();

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

        vmm.unmap_pages(mm, &mapped)?;
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

    // Release Tests 1-3's Vmm before Test 4 uses the GPU's BarUser on the same PDB.
    drop(vmm);

    // Test 4: Exercise `BarUser::map()` end-to-end.
    let access = bar_user.map(mm, &[test_pfn], true)?;

    // Write pattern via PRAMIN, read via BarUserAccess.
    mm.pramin_mut()
        .window_at::<u32>(test_vram)?
        .view()
        .write_val(PATTERN_BAR1);

    let readback = access.try_read32(0)?;
    let test4_passed = if readback == PATTERN_BAR1 {
        true
    } else {
        dev_err!(
            dev,
            "MM: Test 4 FAILED - Expected {:#010x}, got {:#010x}\n",
            PATTERN_BAR1,
            readback
        );
        false
    };
    access.release(mm)?;

    if test1_passed && test2_passed && test3_passed && test4_passed {
        dev_info!(dev, "MM: All self-tests PASSED\n");
        Ok(())
    } else {
        dev_err!(dev, "MM: Self-tests FAILED\n");
        Err(EIO)
    }
}
