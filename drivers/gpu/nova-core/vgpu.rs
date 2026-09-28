// SPDX-License-Identifier: GPL-2.0

use core::num::NonZero;

use kernel::{
    device,
    new_mutex,
    pci,
    prelude::*,
    sync::Mutex, //
};

use crate::{
    firmware::gsp::BuildId,
    fsp::{
        Fsp,
        VgpuMode, //
    },
    gpu::{
        ChannelIdPool,
        Chipset,
        Spec, //
    },
    gsp::{
        cmdq::Cmdq,
        commands::{
            FifoEngineList,
            GspStaticInfo, //
        }, //
    },
    mm::{
        bar_user::BarUser,
        GpuMm, //
    }, //
};

mod commands;
mod fw;
mod gsp_plugin_comm;
mod gsp_plugin_rpc;
mod hal;
mod instance;
mod log;
mod scrubber;
#[cfg_attr(not(CONFIG_PCI_IOV), expect(dead_code, unreachable_pub))]
pub(crate) mod vgpu_api;
mod vram;

/// vGPU state detected during GPU construction.
#[derive(Debug, Clone, Copy)]
pub(crate) enum VgpuState {
    /// vGPU mode is not enabled for this boot.
    Disabled,
    /// vGPU mode is enabled for this boot.
    Enabled {
        /// Total number of SR-IOV VFs supported by this device.
        total_vfs: NonZero<u16>,
    },
}

impl VgpuState {
    /// Detects the boot mode, falling back to disabled if querying the device fails.
    ///
    /// Call after creating the FSP and before allocating GSP firmware resources.
    pub(crate) fn detect(
        pdev: &pci::Device<device::Core<'_>>,
        chipset: Chipset,
        fsp: Option<&mut Fsp<'_>>,
    ) -> Self {
        let state = Self::query_state(pdev, chipset, fsp).unwrap_or_else(|e| {
            dev_warn!(
                pdev,
                "vGPU state detection failed: {:?}; disabling vGPU\n",
                e
            );
            VgpuState::Disabled
        });
        dev_dbg!(pdev, "vGPU state: {:?}\n", state);
        state
    }

    /// Detects the vGPU state from the chipset, SR-IOV capability and FSP PRC knob.
    fn query_state(
        pdev: &pci::Device<device::Core<'_>>,
        chipset: Chipset,
        fsp: Option<&mut Fsp<'_>>,
    ) -> Result<VgpuState> {
        if !hal::vgpu_hal(chipset).supports_vgpu() {
            return Ok(VgpuState::Disabled);
        }

        let Some(total_vfs) = pdev.sriov_get_totalvfs() else {
            return Ok(VgpuState::Disabled);
        };

        if total_vfs.get() < 2 {
            // The current vGPU path does not support single-VF SR-IOV devices yet.
            // Treat one total VF as vGPU-disabled for now; single-VF support can relax
            // this gate once the manager handles that topology.
            return Ok(VgpuState::Disabled);
        }

        let fsp = fsp.ok_or(ENODEV)?;

        match fsp.read_vgpu_mode(pdev.as_ref())? {
            VgpuMode::Enabled => Ok(VgpuState::Enabled { total_vfs }),
            VgpuMode::Disabled => Ok(VgpuState::Disabled),
        }
    }
}

use self::instance::VgpuInstances;

/// Runtime resources for an enabled vGPU boot.
#[pin_data(PinnedDrop)]
pub(crate) struct VgpuManager<'gpu> {
    #[pin]
    instances: Mutex<VgpuInstances<'gpu>>,
    dev: &'gpu device::Device<device::Bound>,
    cmdq: &'gpu Cmdq<'gpu>,
    spec: Spec,
    build_id: Option<&'gpu BuildId>,
    bar_user: &'gpu BarUser<'gpu>,
    mm: &'gpu Mutex<GpuMm<'gpu>>,
    chid_pool: &'gpu ChannelIdPool,
    /// VMMU segment size in bytes, or zero if GSP-RM omitted it.
    vmmu_segment_size: u64,
    total_channels: u32,
    fifo_engine_list: FifoEngineList,
}

impl<'gpu> VgpuManager<'gpu> {
    /// Retains runtime parameters from a completed vGPU-enabled GSP boot.
    pub(crate) fn new(
        dev: &'gpu device::Device<device::Bound>,
        cmdq: &'gpu Cmdq<'gpu>,
        spec: Spec,
        build_id: Option<&'gpu BuildId>,
        bar_user: &'gpu BarUser<'gpu>,
        mm: &'gpu Mutex<GpuMm<'gpu>>,
        chid_pool: &'gpu ChannelIdPool,
        info: &GspStaticInfo,
        total_channels: u32,
    ) -> impl PinInit<Self> + use<'gpu> {
        let fifo_engine_list = info.fifo_engine_list();
        let vmmu_segment_size = info.vmmu_segment_size;
        pin_init!(Self {
            instances <- new_mutex!(VgpuInstances::new(), "nova-core::vgpu-instances"),
            dev,
            cmdq,
            spec,
            build_id,
            bar_user,
            mm,
            chid_pool,
            vmmu_segment_size,
            total_channels,
            fifo_engine_list,
        })
    }
}

#[pinned_drop]
impl PinnedDrop for VgpuManager<'_> {
    fn drop(self: Pin<&mut Self>) {
        self.instances.lock().release_all(&self);
    }
}
