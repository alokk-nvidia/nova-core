// SPDX-License-Identifier: GPL-2.0

//! Virtual Memory Manager for NVIDIA GPU page table management.
//!
//! The [`Vmm`] provides high-level page mapping and unmapping operations for GPU
//! virtual address spaces (Channels, BAR1, BAR2). It wraps the page table walker
//! and handles TLB flushing after modifications.

#![allow(dead_code)]

use kernel::{
    gpu::buddy::{
        AllocatedBlocks,
        BuddyFlag,
        BuddyFlags,
        GpuBuddy,
        GpuBuddyAllocParams,
        GpuBuddyParams, //
    },
    prelude::*,
    rbtree::{RBTree, RBTreeNode},
    sizes::SZ_4K, //
};

use core::cell::Cell;
use core::ops::Range;

use crate::{
    mm::{
        pagetable::{
            walk::{
                PtWalk,
                WalkPdeResult,
                WalkResult, //
            },
            DualPde,
            MmuVersion,
            PageTableLevel,
            Pde,
            Pte, //
        },
        GpuMm,
        Pfn,
        Vfn,
        VramAddress,
        PAGE_SIZE, //
    },
    num::{
        IntoSafeCast, //
    },
};

/// Virtual Memory Manager for a GPU address space.
///
/// Each [`Vmm`] instance manages a single address space identified by its Page
/// Directory Base (`PDB`) address. The [`Vmm`] is used for Channel, BAR1 and
/// BAR2 mappings.
pub(crate) struct Vmm {
    pub(crate) pdb_addr: VramAddress,
    pub(crate) mmu_version: MmuVersion,
    /// Page table allocations required for mappings.
    page_table_allocs: KVec<Pin<KBox<AllocatedBlocks>>>,
    /// Buddy allocator for virtual address range tracking.
    virt_buddy: GpuBuddy,
    /// Prepared PT pages pending PDE installation, keyed by `install_addr`.
    ///
    /// Populated by `Vmm` mapping prepare phase and drained in the execute phase.
    /// Shared by all pending maps in the `Vmm`, thus preventing races where 2
    /// maps might be trying to install the same page table/directory entry pointer.
    pt_pages: RBTree<VramAddress, PreparedPtPage>,
}

/// A pre-allocated and zeroed page table page.
///
/// Created during the mapping prepare phase and consumed during the mapping execute phase.
/// Stored in an [`RBTree`] keyed by the PDE slot address (`install_addr`).
struct PreparedPtPage {
    /// The allocated and zeroed page table page.
    alloc: Pin<KBox<AllocatedBlocks>>,
    /// Page table level -- needed to determine if this PT page is for a dual PDE.
    level: PageTableLevel,
}

/// Multi-page prepared mapping -- VA range allocated, ready for execute.
///
/// Produced by [`Vmm::prepare_map()`], consumed by [`Vmm::execute_map()`].
/// The struct owns the VA space allocation between prepare and execute phases.
pub(crate) struct PreparedMapping {
    vfn_start: Vfn,
    num_pages: usize,
    vfn_alloc: Pin<KBox<AllocatedBlocks>>,
}

/// Result of a mapping operation -- tracks the active mapped range.
///
/// Returned by [`Vmm::execute_map()`] and [`Vmm::map_pages()`].
/// Owns the VA allocation; the VA range is freed when this is dropped.
/// Callers must call [`Vmm::unmap_pages()`] before dropping to invalidate
/// PTEs (dropping only frees the VA range, not the PTE entries).
pub(crate) struct MappedRange {
    pub(crate) vfn_start: Vfn,
    pub(crate) num_pages: usize,
    /// VA allocation -- freed when [`MappedRange`] is dropped.
    _vfn_alloc: Pin<KBox<AllocatedBlocks>>,
    /// Logs a warning if dropped without unmapping.
    _drop_guard: MustUnmapGuard,
}

/// Guard that logs a warning once if a [`MappedRange`] is dropped without
/// calling [`Vmm::unmap_pages()`].
struct MustUnmapGuard {
    armed: Cell<bool>,
}

impl MustUnmapGuard {
    const fn new() -> Self {
        Self {
            armed: Cell::new(true),
        }
    }

    fn disarm(&self) {
        self.armed.set(false);
    }
}

impl Drop for MustUnmapGuard {
    fn drop(&mut self) {
        if self.armed.get() {
            kernel::pr_warn!("MappedRange dropped without calling unmap_pages()\n");
        }
    }
}

impl Vmm {
    /// Create a new [`Vmm`] for the given Page Directory Base address.
    ///
    /// The [`Vmm`] will manage a virtual address space of `va_size` bytes.
    pub(crate) fn new(
        pdb_addr: VramAddress,
        mmu_version: MmuVersion,
        va_size: u64,
    ) -> Result<Self> {
        kernel::pr_info!("VMM::new va_size={:#x} mmu={:?}\n", va_size, mmu_version);
        let virt_buddy = GpuBuddy::new(GpuBuddyParams {
            base_offset: 0,
            physical_memory_size: va_size,
            chunk_size: SZ_4K.into_safe_cast(),
        })?;
        kernel::pr_info!("VMM::new buddy ok\n");

        Ok(Self {
            pdb_addr,
            mmu_version,
            page_table_allocs: KVec::new(),
            virt_buddy,
            pt_pages: RBTree::new(),
        })
    }

    /// Allocate a contiguous virtual frame number range.
    ///
    /// # Arguments
    ///
    /// - `num_pages`: Number of pages to allocate.
    /// - `va_range`: `None` = allocate anywhere, `Some(range)` = constrain allocation to the given
    ///   range.
    pub(crate) fn alloc_vfn_range(
        &self,
        num_pages: usize,
        va_range: Option<Range<u64>>,
    ) -> Result<(Vfn, Pin<KBox<AllocatedBlocks>>)> {
        let np: u64 = num_pages.into_safe_cast();
        let size: u64 = np
            .checked_mul(PAGE_SIZE.into_safe_cast())
            .ok_or(EOVERFLOW)?;

        let (start, end) = match va_range {
            Some(r) => {
                let range_size = r.end.checked_sub(r.start).ok_or(EOVERFLOW)?;
                if range_size != size {
                    return Err(EINVAL);
                }
                (r.start, r.end)
            }
            None => (0, 0),
        };

        let params = GpuBuddyAllocParams {
            start_range_address: start,
            end_range_address: end,
            size,
            min_block_size: SZ_4K.into_safe_cast(),
            buddy_flags: BuddyFlag::ContiguousAllocation.into(),
        };

        let alloc = KBox::pin_init(self.virt_buddy.alloc_blocks(params), GFP_KERNEL)?;

        // Get the starting offset of the first block (only block as range is contiguous).
        let offset = alloc.iter().next().ok_or(ENOMEM)?.offset();
        let page_size: u64 = PAGE_SIZE.into_safe_cast();
        let vfn = Vfn::new(offset / page_size);

        Ok((vfn, alloc))
    }

    /// Read the [`Pfn`] for a mapped [`Vfn`] if one is mapped.
    pub(crate) fn read_mapping(&self, mm: &GpuMm, vfn: Vfn) -> Result<Option<Pfn>> {
        let walker = PtWalk::new(self.pdb_addr, self.mmu_version);

        match walker.walk_to_pte_lookup(mm, vfn)? {
            WalkResult::Mapped { pfn, .. } => Ok(Some(pfn)),
            WalkResult::Unmapped { .. } | WalkResult::PageTableMissing => Ok(None),
        }
    }

    /// Allocate and zero a physical page table page for a specific PDE slot.
    /// Called during the map prepare phase.
    fn alloc_and_zero_page_table(
        &mut self,
        mm: &GpuMm,
        level: PageTableLevel,
    ) -> Result<PreparedPtPage> {
        let params = GpuBuddyAllocParams {
            start_range_address: 0,
            end_range_address: 0,
            size: SZ_4K.into_safe_cast(),
            min_block_size: SZ_4K.into_safe_cast(),
            buddy_flags: BuddyFlags::empty(),
        };
        let blocks = KBox::pin_init(mm.buddy().alloc_blocks(params), GFP_KERNEL).inspect_err(|_| {
            kernel::pr_err!("VMM alloc_pt_page: buddy alloc FAILED level={:?}\n", level);
        })?;

        // Get page's VRAM address from the allocation.
        let page_vram = VramAddress::new(blocks.iter().next().ok_or(ENOMEM)?.offset());

        // Zero via PRAMIN.
        let mut window = mm.pramin().window()?;
        let base = page_vram.raw();
        for off in (0..PAGE_SIZE).step_by(8) {
            window.try_write64(base + off, 0)?;
        }

        Ok(PreparedPtPage {
            alloc: blocks,
            level,
        })
    }

    /// Ensure all intermediate page table pages are prepared for a [`Vfn`]. Just
    /// finds out which PDE pages are missing, allocates pages for them, and defers
    /// installation to the execute phase.
    ///
    /// PRAMIN is released before each allocation and re-acquired after. Memory
    /// allocations outside of holding this lock to prevent deadlocks with fence signalling
    /// critical path.
    fn ensure_pte_path(&mut self, mm: &GpuMm, vfn: Vfn) -> Result {
        let walker = PtWalk::new(self.pdb_addr, self.mmu_version);
        let max_iter = 2 * self.mmu_version.pde_level_count();

        // Keep looping until all PDE levels are resolved.
        for _ in 0..max_iter {
            let mut window = mm.pramin().window()?;

            // Walk PDE levels. The closure checks self.pt_pages for prepared-but-uninstalled
            // pages, letting the walker continue through them as if they were installed in HW.
            // The walker keeps calling the closure to get these "prepared but not installed" pages.
            let result = walker.walk_pde_levels(&mut window, vfn, |install_addr| {
                self.pt_pages
                    .get(&install_addr)
                    .and_then(|p| Some(VramAddress::new(p.alloc.iter().next()?.offset())))
            }).inspect_err(|e| {
                kernel::pr_err!("VMM ensure_pte_path: walk_pde_levels FAILED vfn={:#x} {:?}\n", vfn.raw(), e);
            })?;

            match result {
                WalkPdeResult::Complete { .. } => {
                    // All PDE levels resolved.
                    return Ok(());
                }
                WalkPdeResult::Missing {
                    install_addr,
                    level,
                } => {
                    // Drop PRAMIN before allocation.
                    drop(window);
                    let page = self.alloc_and_zero_page_table(mm, level)?;
                    let node = RBTreeNode::new(install_addr, page, GFP_KERNEL)?;
                    let old = self.pt_pages.insert(node);
                    if old.is_some() {
                        kernel::pr_warn_once!(
                            "VMM: duplicate install_addr in pt_pages (internal consistency error)\n"
                        );
                        return Err(EIO);
                    }
                    // Loop: re-acquire PRAMIN and re-walk from root.
                }
            }
        }

        kernel::pr_warn!(
            "VMM: ensure_pte_path: loop exhausted after {} iters (VFN {:?})\n",
            max_iter,
            vfn
        );
        Err(EIO)
    }

    /// Prepare resources for mapping `num_pages` pages.
    ///
    /// Allocates a contiguous VA range, then walks the hierarchy per-VFN to prepare pages
    /// for all missing PDEs. Returns a [`PreparedMapping`] with the VA allocation.
    ///
    /// If `va_range` is not `None`, the VA range is constrained to the given range. Safe
    /// to call outside the fence signalling critical path.
    pub(crate) fn prepare_map(
        &mut self,
        mm: &GpuMm,
        num_pages: usize,
        va_range: Option<Range<u64>>,
    ) -> Result<PreparedMapping> {
        if num_pages == 0 {
            return Err(EINVAL);
        }

        let pt_upper_bound = self.mmu_version.pt_pages_upper_bound(num_pages);
        kernel::pr_info!("VMM prepare_map: num_pages={} pt_upper_bound={}\n", num_pages, pt_upper_bound);

        self.page_table_allocs.reserve(pt_upper_bound, GFP_KERNEL).inspect_err(|_| {
            kernel::pr_err!("VMM prepare_map: reserve({}) FAILED\n", pt_upper_bound);
        })?;

        let (vfn_start, vfn_alloc) = self.alloc_vfn_range(num_pages, va_range).inspect_err(|_| {
            kernel::pr_err!("VMM prepare_map: alloc_vfn_range FAILED\n");
        })?;
        kernel::pr_info!("VMM prepare_map: vfn_start={:#x}\n", vfn_start.raw());

        for i in 0..num_pages {
            let i_u64: u64 = i.into_safe_cast();
            let vfn = Vfn::new(vfn_start.raw() + i_u64);
            self.ensure_pte_path(mm, vfn).inspect_err(|e| {
                kernel::pr_err!("VMM prepare_map: ensure_pte_path vfn={:#x} FAILED {:?}\n", vfn.raw(), e);
            })?;
        }

        Ok(PreparedMapping {
            vfn_start,
            num_pages,
            vfn_alloc,
        })
    }

    /// Execute a prepared multi-page mapping.
    ///
    /// Drain prepared PT pages and install PDEs followed by single TLB flush.
    pub(crate) fn execute_map(
        &mut self,
        mm: &GpuMm,
        prepared: PreparedMapping,
        pfns: &[Pfn],
        writable: bool,
    ) -> Result<MappedRange> {
        if pfns.len() != prepared.num_pages {
            return Err(EINVAL);
        }

        let PreparedMapping {
            vfn_start,
            num_pages,
            vfn_alloc,
        } = prepared;

        let walker = PtWalk::new(self.pdb_addr, self.mmu_version);
        let mut window = mm.pramin().window().inspect_err(|e| {
            kernel::pr_err!("VMM execute_map: pramin window FAILED {:?}\n", e);
        })?;

        kernel::pr_info!("VMM execute_map: draining pt_pages\n");

        let mut cursor = self.pt_pages.cursor_front_mut();
        let mut pde_count = 0u32;
        while let Some(c) = cursor {
            let (next, node) = c.remove_current();
            let (install_addr, page) = node.to_key_value();
            kernel::pr_info!("VMM execute_map: PDE #{} addr={:#x} level={:?}\n", pde_count, install_addr.raw(), page.level);
            let first_block = page.alloc.iter().next();
            if first_block.is_none() {
                kernel::pr_err!("VMM execute_map: alloc has NO blocks!\n");
                return Err(ENOMEM);
            }
            let page_vram = VramAddress::new(first_block.unwrap().offset());

            if page.level == self.mmu_version.dual_pde_level() {
                let new_dpde = DualPde::new_small(self.mmu_version, Pfn::from(page_vram));
                new_dpde.write(&mut window, install_addr).inspect_err(|e| {
                    kernel::pr_err!("VMM execute_map: dpde write FAILED {:?}\n", e);
                })?;
            } else {
                let new_pde = Pde::new_vram(self.mmu_version, Pfn::from(page_vram));
                new_pde.write(&mut window, install_addr).inspect_err(|e| {
                    kernel::pr_err!("VMM execute_map: pde write FAILED {:?}\n", e);
                })?;
            }

            self.page_table_allocs
                .push_within_capacity(page.alloc)
                .map_err(|_| ENOMEM)?;

            pde_count += 1;
            cursor = next;
        }
        kernel::pr_info!("VMM execute_map: {} PDEs installed, writing {} PTEs\n", pde_count, num_pages);

        for (i, &pfn) in pfns.iter().enumerate() {
            let i_u64: u64 = i.into_safe_cast();
            let vfn = Vfn::new(vfn_start.raw() + i_u64);
            let result = walker.walk_to_pte_lookup_with_window(&mut window, vfn).inspect_err(|e| {
                kernel::pr_err!("VMM execute_map: walk_pte vfn={:#x} FAILED {:?}\n", vfn.raw(), e);
            })?;

            match result {
                WalkResult::Unmapped { pte_addr } | WalkResult::Mapped { pte_addr, .. } => {
                    let pte = Pte::new_vram(self.mmu_version, pfn, writable);
                    pte.write(&mut window, pte_addr).inspect_err(|e| {
                        kernel::pr_err!("VMM execute_map: pte write vfn={:#x} FAILED {:?}\n", vfn.raw(), e);
                    })?;
                }
                WalkResult::PageTableMissing => {
                    kernel::pr_err!("VMM execute_map: PT missing vfn={:#x}\n", vfn.raw());
                    return Err(EIO);
                }
            }
        }

        drop(window);
        kernel::pr_info!("VMM execute_map: PTEs done, flushing TLB\n");

        mm.tlb().flush(self.pdb_addr).inspect_err(|e| {
            kernel::pr_err!("VMM execute_map: TLB flush FAILED {:?}\n", e);
        })?;

        Ok(MappedRange {
            vfn_start,
            num_pages,
            _vfn_alloc: vfn_alloc,
            _drop_guard: MustUnmapGuard::new(),
        })
    }

    /// Map pages doing prepare and execute in the same call.
    ///
    /// This is a convenience wrapper for callers outside the fence signalling critical
    /// path (e.g., BAR mappings). For DRM usecases, [`Vmm::prepare_map()`] and
    /// [`Vmm::execute_map()`] will be called separately.
    pub(crate) fn map_pages(
        &mut self,
        mm: &GpuMm,
        pfns: &[Pfn],
        va_range: Option<Range<u64>>,
        writable: bool,
    ) -> Result<MappedRange> {
        if pfns.is_empty() {
            return Err(EINVAL);
        }

        // Check if provided VA range is sufficient (if provided).
        if let Some(ref range) = va_range {
            let required: u64 = pfns
                .len()
                .checked_mul(PAGE_SIZE)
                .ok_or(EOVERFLOW)?
                .into_safe_cast();
            let available = range.end.checked_sub(range.start).ok_or(EINVAL)?;
            if available < required {
                return Err(EINVAL);
            }
        }

        let prepared = self.prepare_map(mm, pfns.len(), va_range)?;
        self.execute_map(mm, prepared, pfns, writable)
    }

    /// Unmap all pages in a [`MappedRange`] with a single TLB flush.
    ///
    /// Takes the range by value (consumes it), then invalidates PTEs for the range,
    /// flushes the TLB, then drops the range (freeing the VA). PRAMIN lock is held.
    pub(crate) fn unmap_pages(&mut self, mm: &GpuMm, range: MappedRange) -> Result {
        let walker = PtWalk::new(self.pdb_addr, self.mmu_version);
        let invalid_pte = Pte::invalid(self.mmu_version);

        let mut window = mm.pramin().window()?;
        for i in 0..range.num_pages {
            let i_u64: u64 = i.into_safe_cast();
            let vfn = Vfn::new(range.vfn_start.raw() + i_u64);
            let result = walker.walk_to_pte_lookup_with_window(&mut window, vfn)?;

            match result {
                WalkResult::Mapped { pte_addr, .. } | WalkResult::Unmapped { pte_addr } => {
                    invalid_pte.write(&mut window, pte_addr)?;
                }
                WalkResult::PageTableMissing => {
                    continue;
                }
            }
        }
        drop(window);

        mm.tlb().flush(self.pdb_addr)?;

        // TODO: Internal page table pages (PDE, PTE pages) are still kept around.
        // This is by design as repeated maps/unmaps will be fast. As a future TODO,
        // we can add a reclaimer here to reclaim if VRAM is short. For now, the PT
        // pages are dropped once the `Vmm` is dropped.

        range._drop_guard.disarm(); // Unmap complete, Ok to drop MappedRange.
        Ok(())
    }
}
