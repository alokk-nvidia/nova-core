// SPDX-License-Identifier: GPL-2.0

#![allow(dead_code)]

use kernel::{
    device,
    pci,
    prelude::*,
};

use crate::{
    driver::Bar0,
    gpu::Chipset,
    gsp::cmdq::Cmdq,
    mm::VramBlock,
    module_parameters,
};

/// Send a vGPU command to GSP via GMC and check the response status.
///
/// On success, copies the response payload back into `params`.
fn send_vgpu_command(cmdq: &Cmdq, bar: &Bar0, cmd: u32, params: &mut [u8]) -> Result {
    let response = cmdq.send_gmc_and_receive(bar, cmd, params.as_ref(), 8192)?;
    if response.status != 0 {
        kernel::pr_err!(
            "vGPU command {:#x} failed: status={:#x}\n",
            cmd,
            response.status
        );
        return Err(EIO);
    }
    let copy_len = response.payload.len().min(params.len());
    params[..copy_len].copy_from_slice(&response.payload[..copy_len]);
    Ok(())
}

/// Guest Function ID — identifies a VF partition within the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub(crate) struct Gfid(pub u32);

/// Encoded PCI domain/bus/device/function identifier.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub(crate) struct Dbdf(pub u32);

/// NV2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE
const CMD_GET_DEVICE_INFO_TABLE: u32 = 0x2080_1112;

/// NV2080_CTRL_CMD_VGPU_MGR_INTERNAL_BOOTLOAD_GSP_VGPU_PLUGIN_TASK
const CMD_VGPU_BOOTLOAD: u32 = 0x2080_4001;

/// NV2080_CTRL_CMD_VGPU_MGR_INTERNAL_SHUTDOWN_GSP_VGPU_PLUGIN_TASK
const CMD_VGPU_SHUTDOWN: u32 = 0x2080_4002;

/// NV2080_CTRL_CMD_VGPU_MGR_INTERNAL_PGPU_ADD_VGPU_TYPE
const CMD_PGPU_ADD_VGPU_TYPE: u32 = 0x2080_4003;

/// NV2080_CTRL_CMD_VGPU_MGR_INTERNAL_CLEANUP_GSP_VGPU_PLUGIN_TASK
const CMD_VGPU_CLEANUP: u32 = 0x2080_4008;

/// Maximum number of NV2080 engine types (NV2080_ENGINE_TYPE_LAST).
pub(crate) const NV2080_GPU_MAX_ENGINES: usize = 0x54;

/// Number of u64 words needed to store the engine bitmap.
const ENGINE_BITMAP_WORDS: usize = NV2080_GPU_MAX_ENGINES.div_ceil(64);

const ENGINE_DATA_TYPES: usize = 16;
const ENGINE_INFO_TYPE_RM_ENGINE_TYPE: usize = 2;
const DEVICE_INFO_TABLE_MAX_ENTRIES: usize = 32;
const ENGINE_MAX_NAME_LEN: usize = 16;
const ENGINE_MAX_PBDMA: usize = 2;

/// Single device info table entry (NV2080_CTRL_FIFO_DEVICE_ENTRY).
#[repr(C)]
#[derive(Clone, Copy)]
struct DeviceInfoEntry {
    engine_data: [u32; ENGINE_DATA_TYPES],
    pbdma_ids: [u32; ENGINE_MAX_PBDMA],
    pbdma_fault_ids: [u32; ENGINE_MAX_PBDMA],
    num_pbdmas: u32,
    engine_name: [u8; ENGINE_MAX_NAME_LEN],
}

/// Response for GET_DEVICE_INFO_TABLE RM control.
#[repr(C)]
struct DeviceInfoTableParams {
    base_index: u32,
    num_entries: u32,
    b_more: u8,
    _pad: [u8; 3],
    entries: [DeviceInfoEntry; DEVICE_INFO_TABLE_MAX_ENTRIES],
}

const VGPU_TYPE_NAME_MAX: usize = 32;

/// Configuration of a virtual GPU profile (e.g. L40-1Q).
pub(crate) struct VgpuType {
    pub(crate) vgpu_type_id: u32,
    pub(crate) name: [u8; VGPU_TYPE_NAME_MAX],
    pub(crate) vdev_id: u64,
    pub(crate) pdev_id: u64,
    /// Guest framebuffer size in bytes.
    pub(crate) fb_length: u64,
    /// GSP plugin heap size in bytes.
    pub(crate) gsp_heap_size: u64,
    /// BAR1 aperture length in megabytes.
    pub(crate) bar1_length: u64,
    /// Maximum concurrent instances.
    pub(crate) max_instance: u32,
    pub(crate) ecc_supported: u32,
    /// Framebuffer reservation in bytes.
    pub(crate) fb_reservation: u64,
}

/// Per-instance state for a running vGPU.
///
/// Resources attached via [`Option<VramBlock>`] are freed automatically on drop.
pub(crate) struct VgpuInstance {
    pub(crate) id: i32,
    pub(crate) gfid: Gfid,
    pub(crate) dbdf: Dbdf,
    pub(crate) vgpu_type_idx: usize,
    pub(crate) vm_pid: u32,

    pub(crate) chid_offset: u32,
    pub(crate) num_chid: u32,
    pub(crate) num_plugin_channels: u32,

    pub(crate) fbmem_heap: Option<VramBlock>,
    pub(crate) mgmt_heap: Option<VramBlock>,

    pub(crate) active: bool,
}

/// GPU parameters learned from GSP after boot.
pub(crate) struct GspConfig {
    pub(crate) vmmu_segment_size: u64,
    pub(crate) ecc_enabled: bool,
    pub(crate) total_avail_chids: u32,
    pub(crate) total_fbmem_size: u64,
}

/// Fixed layout of GSP plugin communication buffers within the management heap.
pub(crate) struct CommBuffLayout {
    pub(crate) total_size: u64,
    pub(crate) init_task_log_offset: u64,
    pub(crate) init_task_log_size: u64,
    pub(crate) vgpu_task_log_size: u64,
    pub(crate) kernel_log_size: u64,
}

pub(crate) struct Vgpu {
    pub(crate) vgpu_requested: bool,
    pub(crate) vgpu_enabled: bool,
    pub(crate) total_vfs: u16,

    /// Engine bitmap indexed by NV2080 engine type.
    pub(crate) engine_bitmap: [u64; ENGINE_BITMAP_WORDS],

    pub(crate) gsp_config: GspConfig,
    pub(crate) comm_layout: CommBuffLayout,

    pub(crate) vgpu_types: KVec<VgpuType>,
    pub(crate) instances: KVec<VgpuInstance>,
}

impl Vgpu {
    pub(crate) fn new(pdev: &pci::Device<device::Bound>, chipset: Chipset) -> Result<Vgpu> {
        let total_vfs = if chipset.arch().supports_vgpu() {
            match *module_parameters::vgpu_support.value() {
                0 => 0,
                _ => pdev.sriov_get_totalvfs().unwrap_or(0),
            }
        } else {
            0
        };

        Ok(Vgpu {
            vgpu_requested: total_vfs > 0,
            vgpu_enabled: false,
            total_vfs,
            engine_bitmap: [0u64; ENGINE_BITMAP_WORDS],
            gsp_config: GspConfig {
                vmmu_segment_size: 0,
                ecc_enabled: false,
                total_avail_chids: 0,
                total_fbmem_size: 0,
            },
            comm_layout: CommBuffLayout {
                total_size: 0,
                init_task_log_offset: 0,
                init_task_log_size: 0,
                vgpu_task_log_size: 0,
                kernel_log_size: 0,
            },
            vgpu_types: KVec::new(),
            instances: KVec::new(),
        })
    }

    /// Register a vGPU instance, returning its index.
    pub(crate) fn register_instance(&mut self, instance: VgpuInstance) -> Result<usize> {
        for existing in self.instances.iter() {
            if existing.id == instance.id {
                return Err(EBUSY);
            }
        }
        let idx = self.instances.len();
        self.instances.push(instance, GFP_KERNEL)?;
        Ok(idx)
    }

    /// Unregister a vGPU instance by VF id.
    pub(crate) fn unregister_instance(&mut self, vf_id: i32) -> Result {
        let pos = self
            .instances
            .iter()
            .position(|inst| inst.id == vf_id)
            .ok_or(ENODEV)?;
        self.instances.remove(pos)?;
        Ok(())
    }
}
