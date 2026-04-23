// SPDX-License-Identifier: GPL-2.0

use kernel::{
    device,
    devres::Devres,
    gpu::buddy::{
        AllocatedBlocks,
        GpuBuddyAllocFlags,
        GpuBuddyAllocMode, //
    },
    io::Io,
    prelude::*,
    ptr::Alignment,
    sync::{
        Arc,
        Mutex, //
    },
};

use crate::{
    driver::Bar1,
    mm::{
        bar_user::BarUser,
        GpuMm,
        Pfn,
        VramAddress,
        PAGE_SIZE, //
    },
};

/// VRAM allocation with RAII lifetime. Drop frees blocks back to buddy allocator.
#[expect(dead_code)]
pub(crate) struct VramBlock {
    _blocks: Pin<KBox<AllocatedBlocks>>,
    pub addr: u64,
    pub size: u64,
}

#[expect(dead_code)]
pub(crate) fn alloc_vram(mm: &GpuMm, size: u64, align: u64) -> Result<VramBlock> {
    let min_block_size =
        Alignment::new_checked(core::cmp::max(align, 4096) as usize).ok_or(EINVAL)?;
    let blocks = KBox::pin_init(
        mm.buddy().alloc_blocks(
            GpuBuddyAllocMode::Simple,
            size,
            min_block_size,
            GpuBuddyAllocFlags::default(),
        ),
        GFP_KERNEL,
    )?;
    let addr = blocks.as_ref().iter().next().ok_or(ENOMEM)?.offset();
    Ok(VramBlock {
        _blocks: blocks,
        addr,
        size,
    })
}

/// BAR1 sub-mapping backed by GPU page tables.
///
/// Maps a contiguous VRAM region into the BAR1 virtual address space via
/// [`BarUser`].  CPU accesses go through `bar1[gpu_va_addr + off]`.
/// Must be explicitly destroyed via [`Bar1Map::destroy()`] to release the
/// GPU page table mapping.
#[expect(dead_code)]
pub(crate) struct Bar1Map {
    bar_user: Arc<Mutex<BarUser>>,
    bar1: Arc<Devres<Bar1>>,
    mapped: crate::mm::vmm::MappedRange,
    /// Physical VRAM address of the mapped region.
    pub fbmem_addr: u64,
    /// Size of the mapped VRAM region in bytes.
    pub fbmem_size: u64,
    /// GPU virtual address within BAR1 aperture (from page table mapping).
    pub gpu_va_addr: u64,
    /// Size of the GPU VA region in bytes.
    pub gpu_va_size: u64,
}

#[expect(dead_code)]
impl Bar1Map {
    pub(crate) fn new(
        bar_user: &Arc<Mutex<BarUser>>,
        bar1: &Arc<Devres<Bar1>>,
        dev: &device::Device<device::Bound>,
        mm: &GpuMm,
        fbmem_addr: u64,
        fbmem_size: u64,
    ) -> Result<Self> {
        let num_pages = (fbmem_size as usize).div_ceil(PAGE_SIZE);
        let mut pfns = KVec::new();
        for i in 0..num_pages {
            let addr = fbmem_addr + (i * PAGE_SIZE) as u64;
            pfns.push(Pfn::from(VramAddress::new(addr)), GFP_KERNEL)?;
        }

        let mapped = bar_user.lock().map_pages(dev, mm, &pfns, true)?;
        let gpu_va_addr = mapped.vfn_start.raw() * PAGE_SIZE as u64;
        let gpu_va_size = mapped.num_pages as u64 * PAGE_SIZE as u64;

        Ok(Self {
            bar_user: bar_user.clone(),
            bar1: bar1.clone(),
            mapped,
            fbmem_addr,
            fbmem_size,
            gpu_va_addr,
            gpu_va_size,
        })
    }

    pub(crate) fn read32(&self, dev: &device::Device<device::Bound>, off: u64) -> Result<u32> {
        let bar = self.bar1.access(dev)?;
        bar.try_read32((self.gpu_va_addr + off) as usize)
    }

    pub(crate) fn write32(
        &self,
        dev: &device::Device<device::Bound>,
        off: u64,
        val: u32,
    ) -> Result {
        let bar = self.bar1.access(dev)?;
        bar.try_write32(val, (self.gpu_va_addr + off) as usize)
    }

    /// Explicitly destroy the mapping, releasing the GPU VA and invalidating PTEs.
    pub(crate) fn destroy(self, dev: &device::Device<device::Bound>, mm: &GpuMm) -> Result {
        self.bar_user.lock().unmap_pages(dev, mm, self.mapped)
    }
}
