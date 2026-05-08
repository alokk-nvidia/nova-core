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

/// GSP-level configuration for vGPU resource sizing.
#[derive(Default)]
pub(crate) struct GspConfig {
    pub vmmu_segment_size: u64,
    pub total_avail_chids: u32,
    pub total_fbmem_size: u64,
}

const CTRL_SIZE: u64 = 4 * 1024;
const RESPONSE_SIZE: u64 = 4 * 1024;
const MESSAGE_SIZE: u64 = 4 * 1024;
const MIGRATION_SIZE: u64 = 2 * 1024 * 1024;
const ERROR_SIZE: u64 = 4 * 1024;
const INIT_LOG_SIZE: u64 = 128 * 1024;
const VGPU_LOG_SIZE: u64 = 256 * 1024;
const KERNEL_LOG_SIZE: u64 = 64 * 1024;

/// Communication buffer layout within the management heap.
#[expect(dead_code)]
pub(crate) struct CommBuffLayout {
    pub total_size: u64,
    pub init_task_log_offset: u64,
    pub init_task_log_size: u64,
    pub vgpu_task_log_size: u64,
    pub kernel_log_size: u64,
}

impl CommBuffLayout {
    pub(crate) fn calculate() -> Self {
        let init_task_log_offset =
            CTRL_SIZE + RESPONSE_SIZE + MESSAGE_SIZE + MIGRATION_SIZE + ERROR_SIZE;
        let total_size =
            init_task_log_offset + INIT_LOG_SIZE + VGPU_LOG_SIZE + KERNEL_LOG_SIZE;
        Self {
            total_size,
            init_task_log_offset,
            init_task_log_size: INIT_LOG_SIZE,
            vgpu_task_log_size: VGPU_LOG_SIZE,
            kernel_log_size: KERNEL_LOG_SIZE,
        }
    }
}

impl Default for CommBuffLayout {
    fn default() -> Self {
        Self::calculate()
    }
}

use crate::gsp::commands::NVGMC_ENGINE_TYPE_COUNT;

/// Per-GMC-engine-type bitmask of available engine instances.
///
/// Indexed by `NVGMC_ENGINE_TYPE` (GR=1, COPY=2, ..., OFA=19).
/// Each `u64` is a bitmask where bit N means engine instance N exists.
pub(crate) struct GmcEngineMasks {
    #[expect(dead_code)]
    pub masks: [u64; NVGMC_ENGINE_TYPE_COUNT],
}

impl GmcEngineMasks {
    pub(crate) fn new() -> Self {
        Self { masks: [0; NVGMC_ENGINE_TYPE_COUNT] }
    }

    pub(crate) fn from_masks(masks: &[u64; NVGMC_ENGINE_TYPE_COUNT]) -> Self {
        Self { masks: *masks }
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
    pub(crate) gsp_config: GspConfig,
    pub(crate) comm_layout: CommBuffLayout,
    pub(crate) engine_masks: GmcEngineMasks,
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
            gsp_config: GspConfig::default(),
            comm_layout: CommBuffLayout::default(),
            engine_masks: GmcEngineMasks::new(),
        })
    }

    pub(crate) fn set_vgpu_enabled(&mut self, enabled: bool) {
        self.vgpu_enabled = enabled;
    }

    /// One-time initialization after GSP boot completes.
    pub(crate) fn init_post_gsp_boot(
        &mut self,
        gmc_engine_masks: &[u64; NVGMC_ENGINE_TYPE_COUNT],
        total_vram: u64,
        vmmu_segment_size: u64,
    ) -> Result {
        self.gsp_config.vmmu_segment_size = vmmu_segment_size;
        self.gsp_config.total_fbmem_size = total_vram;
        self.gsp_config.total_avail_chids = 2048;
        self.engine_masks = GmcEngineMasks::from_masks(gmc_engine_masks);
        self.comm_layout = CommBuffLayout::calculate();
        Ok(())
    }
}

/// Channel ID allocator using a bitmap over 2048 channels.
#[expect(dead_code)]
pub(crate) struct ChidAllocator {
    bitmap: [u64; 32],
    total: u32,
}

#[expect(dead_code)]
impl ChidAllocator {
    pub(crate) fn new(total: u32) -> Self {
        Self {
            bitmap: [0u64; 32],
            total,
        }
    }

    /// Allocate `count` contiguous channels, aligned to `count` boundary.
    pub(crate) fn alloc(&mut self, count: u32) -> Result<u32> {
        if count == 0 {
            return Err(EINVAL);
        }
        let stride = count as usize;
        let total_bits = self.bitmap.len() * 64;
        let mut offset = 0usize;

        while offset + stride <= total_bits {
            if self.is_range_free(offset, stride) {
                self.set_range(offset, stride);
                return Ok(offset as u32);
            }
            offset += stride;
        }
        Err(ENOSPC)
    }

    /// Free `count` channels starting at `offset`.
    pub(crate) fn free(&mut self, offset: u32, count: u32) {
        let start = offset as usize;
        for i in start..start + count as usize {
            let word = i / 64;
            let bit = i % 64;
            self.bitmap[word] &= !(1u64 << bit);
        }
    }

    fn is_range_free(&self, start: usize, count: usize) -> bool {
        for i in start..start + count {
            let word = i / 64;
            let bit = i % 64;
            if self.bitmap[word] & (1u64 << bit) != 0 {
                return false;
            }
        }
        true
    }

    fn set_range(&mut self, start: usize, count: usize) {
        for i in start..start + count {
            let word = i / 64;
            let bit = i % 64;
            self.bitmap[word] |= 1u64 << bit;
        }
    }
}
