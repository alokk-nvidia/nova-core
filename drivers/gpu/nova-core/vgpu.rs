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
    pci,
    prelude::*,
    ptr::Alignment,
    sync::Arc, //
};

use crate::{
    driver::Bar1,
    gpu::Architecture,
    mm::GpuMm, //
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

/// BAR1 sub-mapping with RAII lifetime.
#[expect(dead_code)]
pub(crate) struct Bar1Map {
    bar1: Arc<Devres<Bar1>>,
    pub offset: u64,
    pub size: u64,
}

#[expect(dead_code)]
impl Bar1Map {
    pub(crate) fn new(bar1: &Arc<Devres<Bar1>>, offset: u64, size: u64) -> Result<Self> {
        Ok(Self {
            bar1: bar1.clone(),
            offset,
            size,
        })
    }

    pub(crate) fn read32(&self, dev: &device::Device<device::Bound>, off: u64) -> Result<u32> {
        let bar = self.bar1.access(dev)?;
        bar.try_read32((self.offset + off) as usize)
    }

    pub(crate) fn write32(
        &self,
        dev: &device::Device<device::Bound>,
        off: u64,
        val: u32,
    ) -> Result {
        let bar = self.bar1.access(dev)?;
        bar.try_write32(val, (self.offset + off) as usize)
    }
}

/// Unified vGPU manager.
///
/// On creation, performs platform detection to determine whether vGPU is
/// requested (PRC knob + totalvfs for Blackwell). The `vgpu_requested`
/// flag may be further refined during boot (e.g. FSP PRC knob read).
pub(crate) struct VgpuManager {
    pub(crate) vgpu_requested: bool,
    pub(crate) vgpu_enabled: bool,
    pub(crate) total_vfs: u16,
}

impl VgpuManager {
    pub(crate) fn new(
        pdev: &pci::Device<device::Core>,
        arch: Architecture,
    ) -> Result<VgpuManager> {
        let total_vfs: u16 = if arch.supports_vgpu() {
            pdev.sriov_get_totalvfs()
                .ok()
                .and_then(|n| n.try_into().ok())
                .unwrap_or(0)
        } else {
            0
        };

        Ok(VgpuManager {
            vgpu_requested: total_vfs > 0,
            vgpu_enabled: false,
            total_vfs,
        })
    }

    pub(crate) fn set_vgpu_enabled(&mut self, enabled: bool) {
        self.vgpu_enabled = enabled;
    }
}
