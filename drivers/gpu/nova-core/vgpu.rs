// SPDX-License-Identifier: GPL-2.0

mod chan;
mod mm;

pub(crate) use self::chan::ChidAllocator;

use kernel::{
    device,
    pci,
    prelude::*, //
};

use crate::gpu::Architecture;

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
        Self {
            masks: [0; NVGMC_ENGINE_TYPE_COUNT],
        }
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
    pub(crate) vmmu_segment_size: u64,
    pub(crate) total_avail_chids: u32,
    pub(crate) total_fbmem_size: u64,
    pub(crate) engine_masks: GmcEngineMasks,
}

impl VgpuManager {
    pub(crate) fn new(pdev: &pci::Device<device::Core>, arch: Architecture) -> Result<VgpuManager> {
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
            vmmu_segment_size: 0,
            total_avail_chids: 0,
            total_fbmem_size: 0,
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
        self.vmmu_segment_size = vmmu_segment_size;
        self.total_fbmem_size = total_vram;
        self.total_avail_chids = 2048;
        self.engine_masks = GmcEngineMasks::from_masks(gmc_engine_masks);
        Ok(())
    }
}
