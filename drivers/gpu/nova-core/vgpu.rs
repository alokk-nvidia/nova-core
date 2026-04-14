// SPDX-License-Identifier: GPL-2.0

#![allow(dead_code)]

use kernel::{
    device,
    io::Io,
    pci,
    prelude::*,
};

use crate::{
    driver::{Bar0, Bar1},
    gpu::Chipset,
    gsp::{
        cmdq::Cmdq,
        rm::commands::send_rmcontrol_with_reply,
    },
    mm::{
        self,
        bar_user::BarUser,
        GpuMm,
        Pfn,
        VramAddress,
        VramBlock,
        PAGE_SIZE,
    },
    num::IntoSafeCast,
    module_parameters,
};

/// Send an RM control command and check the returned NV_STATUS.
fn check_rmcontrol_status(
    cmdq: &Cmdq,
    bar: &Bar0,
    cmd: u32,
    params: &mut [u8],
    h_client: u32,
    h_subdevice: u32,
) -> Result {
    let nv_status = send_rmcontrol_with_reply(cmdq, bar, cmd, params, h_client, h_subdevice)?;
    if nv_status != 0 {
        kernel::pr_err!("RM control {:#x} failed: NV_STATUS={:#x}\n", cmd, nv_status);
        return Err(EIO);
    }
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

/// NV2080_CTRL_CMD_GPU_GET_VMMU_SEGMENT_SIZE
const CMD_GET_VMMU_SEGMENT_SIZE: u32 = 0x2080_017e;

/// NV2080_CTRL_CMD_VGPU_MGR_INTERNAL_BOOTLOAD_GSP_VGPU_PLUGIN_TASK
const CMD_VGPU_BOOTLOAD: u32 = 0x2080_4001;

/// NV2080_CTRL_CMD_VGPU_MGR_INTERNAL_SHUTDOWN_GSP_VGPU_PLUGIN_TASK
const CMD_VGPU_SHUTDOWN: u32 = 0x2080_4002;

/// NV2080_CTRL_CMD_VGPU_MGR_INTERNAL_PGPU_ADD_VGPU_TYPE
const CMD_PGPU_ADD_VGPU_TYPE: u32 = 0x2080_4003;

/// NV2080_CTRL_CMD_VGPU_MGR_INTERNAL_CLEANUP_GSP_VGPU_PLUGIN_TASK
const CMD_VGPU_CLEANUP: u32 = 0x2080_4008;

/// Prebuilt NVA081_CTRL_VGPU_INFO for L40-1Q (type 871).
const L40_1Q_VGPU_INFO: &[u8] = include_bytes!("l40_1q.bin");
const NVA081_CTRL_VGPU_INFO_SIZE: usize = 5424;
const NVA081_MAX_VGPU_TYPES_PER_PGPU: usize = 128;

/// Magic value GSP writes to ctrl buffer on successful plugin bootload.
const GSP_PLUGIN_BOOTLOADED: u32 = 0x4E65_4A6F;

/// Byte offset of message_seq_num within VGPU_CPU_GSP_CTRL_BUFF_REGION.
const CTRL_BUF_MSG_SEQ_NUM_OFFSET: usize = 8;

const PLUGIN_BOOT_TIMEOUT: kernel::time::Delta = kernel::time::Delta::from_secs(10);
const PLUGIN_POLL_INTERVAL: kernel::time::Delta = kernel::time::Delta::from_millis(1);

const VGPU_GSP_CTRL_REGION_SIZE: u64 = 4096;
const VGPU_GSP_RESPONSE_REGION_SIZE: u64 = 4096;
const VGPU_GSP_MESSAGE_REGION_SIZE: u64 = 4096;
const VGPU_GSP_MIGRATION_REGION_SIZE: u64 = 2 * 1024 * 1024;
const VGPU_GSP_ERROR_REGION_SIZE: u64 = 4096;
const VGPU_GSP_INIT_TASK_LOG_SIZE: u64 = 128 * 1024;
const VGPU_GSP_VGPU_TASK_LOG_SIZE: u64 = 256 * 1024;
const VGPU_GSP_KERNEL_LOG_SIZE: u64 = 64 * 1024;

/// Maximum VMMU segments for guest FB.
const NV2080_CTRL_MAX_VMMU_SEGMENTS: usize = 384;

/// Bootload parameters for the GSP vGPU plugin task.
#[repr(C)]
struct BootloadParams {
    dbdf: u32,
    gfid: u32,
    vgpu_type: u32,
    vm_pid: u32,
    swizz_id: u32,
    num_channels: u32,
    num_plugin_channels: u32,
    chid_offset: [u32; NV2080_GPU_MAX_ENGINES],
    b_disable_default_smc_exec_part_restore: u8,
    _pad1: [u8; 3],
    num_guest_fb_segments: u32,
    _pad2: [u8; 4],
    guest_fb_phys_addr_list: [u64; NV2080_CTRL_MAX_VMMU_SEGMENTS],
    guest_fb_length_list: [u64; NV2080_CTRL_MAX_VMMU_SEGMENTS],
    plugin_heap_memory_phys_addr: u64,
    plugin_heap_memory_length: u64,
    ctrl_buff_offset: u64,
    init_task_log_buff_offset: u64,
    init_task_log_buff_size: u64,
    vgpu_task_log_buff_offset: u64,
    vgpu_task_log_buff_size: u64,
    kernel_log_buff_offset: u64,
    kernel_log_buff_size: u64,
    mig_rm_heap_memory_phys_addr: u64,
    mig_rm_heap_memory_length: u64,
    b_device_profiling_enabled: u8,
    _pad3: [u8; 7],
}

const _: () = assert!(size_of::<BootloadParams>() == 6616);

/// Maximum number of NV2080 engine types (NV2080_ENGINE_TYPE_LAST).
pub(crate) const NV2080_GPU_MAX_ENGINES: usize = 0x54;

/// Number of u64 words needed to store the engine bitmap.
const ENGINE_BITMAP_WORDS: usize = (NV2080_GPU_MAX_ENGINES + 63) / 64;

const VGPU_TYPE_NAME_MAX: usize = 32;

/// Configuration of a virtual GPU profile (e.g. L40-1Q).
pub(crate) struct VgpuType {
    pub(crate) vgpu_type_id: u32,
    pub(crate) name: [u8; VGPU_TYPE_NAME_MAX],
    pub(crate) vdev_id: u64,
    pub(crate) pdev_id: u64,
    pub(crate) fb_length: u64,
    pub(crate) gsp_heap_size: u64,
    pub(crate) bar1_length: u64,
    pub(crate) max_instance: u32,
    pub(crate) ecc_supported: u32,
    pub(crate) fb_reservation: u64,
}

/// Per-instance state for a running vGPU.
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

    chid_alloc: ChidAllocator,
}

/// Bitmap-based channel ID allocator for vGPU instances.
struct ChidAllocator {
    bitmap: [u64; 32],
    total: u32,
}

impl ChidAllocator {
    fn new() -> Self {
        Self {
            bitmap: [0u64; 32],
            total: 0,
        }
    }

    fn init(&mut self, total: u32) {
        self.total = total;
        self.bitmap = [0u64; 32];
    }

    /// Allocate a contiguous, aligned block of `count` channel IDs.
    fn alloc(&mut self, count: u32) -> Result<u32> {
        if count == 0 || count > self.total {
            return Err(EINVAL);
        }
        let mut offset: u32 = 0;
        while offset + count <= self.total {
            if self.is_range_free(offset, count) {
                self.set_range(offset, count);
                return Ok(offset);
            }
            offset += count;
        }
        Err(ENOSPC)
    }

    fn free(&mut self, offset: u32, count: u32) {
        for i in offset..offset + count {
            let (word, bit) = ((i / 64) as usize, i % 64);
            if word < self.bitmap.len() {
                self.bitmap[word] &= !(1u64 << bit);
            }
        }
    }

    fn is_range_free(&self, offset: u32, count: u32) -> bool {
        for i in offset..offset + count {
            let (word, bit) = ((i / 64) as usize, i % 64);
            if word >= self.bitmap.len() || self.bitmap[word] & (1u64 << bit) != 0 {
                return false;
            }
        }
        true
    }

    fn set_range(&mut self, offset: u32, count: u32) {
        for i in offset..offset + count {
            let (word, bit) = ((i / 64) as usize, i % 64);
            if word < self.bitmap.len() {
                self.bitmap[word] |= 1u64 << bit;
            }
        }
    }
}

/// Round down to the previous power of 2 (or 0 if input is 0).
fn prev_pow2(x: u32) -> u32 {
    if x == 0 { return 0; }
    1 << (31 - x.leading_zeros())
}

impl Vgpu {
    pub(crate) fn new(pdev: &pci::Device<device::Core>, chipset: Chipset) -> Result<Vgpu> {
        let total_vfs: u16 = if chipset.arch().supports_vgpu() {
            match *module_parameters::vgpu_support.value() {
                0 => 0,
                _ => pdev
                    .sriov_get_totalvfs()
                    .ok()
                    .and_then(|n| n.try_into().ok())
                    .unwrap_or(0),
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
            chid_alloc: ChidAllocator::new(),
        })
    }

    pub(crate) fn set_vgpu_enabled(&mut self, enabled: bool) {
        self.vgpu_enabled = enabled;
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


    /// Allocate VMMU-aligned guest framebuffer memory.
    fn alloc_guest_fb(&self, mm: &GpuMm, vgpu_type: &VgpuType) -> Result<VramBlock> {
        let fb_size = self.compute_fb_size(vgpu_type)?;
        mm::alloc_vram(mm, fb_size, self.gsp_config.vmmu_segment_size.max(4096))
    }

    /// Allocate the management heap for GSP plugin communication buffers.
    fn alloc_plugin_heap(mm: &GpuMm, vgpu_type: &VgpuType) -> Result<VramBlock> {
        mm::alloc_vram(mm, vgpu_type.gsp_heap_size, 4096)
    }

    /// Compute the FB memory size to allocate, accounting for ECC overhead.
    fn compute_fb_size(&self, vgpu_type: &VgpuType) -> Result<u64> {
        if !self.gsp_config.ecc_enabled {
            return Ok(vgpu_type.fb_length);
        }
        if vgpu_type.ecc_supported == 0 {
            return Err(ENODEV);
        }
        let seg = self.gsp_config.vmmu_segment_size;
        if seg == 0 {
            return Err(EINVAL);
        }
        let aligned_total = (self.gsp_config.total_fbmem_size + seg - 1) / seg * seg;
        let per_instance = aligned_total / vgpu_type.max_instance as u64
            - vgpu_type.fb_reservation
            - vgpu_type.gsp_heap_size;
        let fb_length = vgpu_type.fb_length.min(per_instance);
        Ok(fb_length / seg * seg)
    }

    /// Initialize the channel ID allocator with the total available count.
    pub(crate) fn init_chid_allocator(&mut self) {
        self.chid_alloc.init(self.gsp_config.total_avail_chids);
    }

    /// Allocate channel IDs for a vGPU instance.
    pub(crate) fn setup_chids(&mut self, instance: &mut VgpuInstance) -> Result {
        let vgpu_type = self.vgpu_types.get(instance.vgpu_type_idx).ok_or(EINVAL)?;
        let num = prev_pow2(self.gsp_config.total_avail_chids / vgpu_type.max_instance);
        if num == 0 {
            return Err(EINVAL);
        }
        let offset = self.chid_alloc.alloc(num)?;
        instance.chid_offset = offset;
        instance.num_chid = num;
        instance.num_plugin_channels = 1;
        Ok(())
    }

    /// Release channel IDs held by an instance.
    pub(crate) fn release_chids(&mut self, instance: &mut VgpuInstance) {
        if instance.num_chid > 0 {
            self.chid_alloc.free(instance.chid_offset, instance.num_chid);
            instance.chid_offset = 0;
            instance.num_chid = 0;
        }
    }

    /// Release channel IDs by offset and count (used when instance is consumed).
    fn release_chids_by_value(&mut self, offset: u32, count: u32) {
        if count > 0 {
            self.chid_alloc.free(offset, count);
        }
    }

    /// Query the VMMU segment size from GSP via RM control.
    fn query_vmmu_segment_size(
        &mut self,
        cmdq: &Cmdq,
        bar: &Bar0,
        h_client: u32,
        h_subdevice: u32,
    ) -> Result {
        let mut params = [0u8; 8];
        check_rmcontrol_status(
            cmdq, bar, CMD_GET_VMMU_SEGMENT_SIZE, &mut params, h_client, h_subdevice,
        )?;
        self.gsp_config.vmmu_segment_size = u64::from_ne_bytes(params);
        Ok(())
    }

    /// Initialize GSP communication buffer layout from fixed region sizes.
    pub(crate) fn init_comm_layout(&mut self) {
        self.comm_layout.total_size = VGPU_GSP_CTRL_REGION_SIZE
            + VGPU_GSP_RESPONSE_REGION_SIZE
            + VGPU_GSP_MESSAGE_REGION_SIZE
            + VGPU_GSP_MIGRATION_REGION_SIZE
            + VGPU_GSP_ERROR_REGION_SIZE
            + VGPU_GSP_INIT_TASK_LOG_SIZE
            + VGPU_GSP_VGPU_TASK_LOG_SIZE
            + VGPU_GSP_KERNEL_LOG_SIZE;
        self.comm_layout.init_task_log_offset = VGPU_GSP_CTRL_REGION_SIZE
            + VGPU_GSP_RESPONSE_REGION_SIZE
            + VGPU_GSP_MESSAGE_REGION_SIZE
            + VGPU_GSP_MIGRATION_REGION_SIZE
            + VGPU_GSP_ERROR_REGION_SIZE;
        self.comm_layout.init_task_log_size = VGPU_GSP_INIT_TASK_LOG_SIZE;
        self.comm_layout.vgpu_task_log_size = VGPU_GSP_VGPU_TASK_LOG_SIZE;
        self.comm_layout.kernel_log_size = VGPU_GSP_KERNEL_LOG_SIZE;
    }

    /// Initialize post-boot vGPU state.
    ///
    /// Must be called after GSP boot completes. Queries hardware parameters,
    /// builds the engine bitmap, and sets up allocators.
    pub(crate) fn init_post_gsp_boot(
        &mut self,
        total_vram: u64,
        engine_caps: &[u32; 3],
    ) -> Result {
        self.gsp_config.total_fbmem_size = total_vram;
        // TODO: query actual available CHIDs from GSP instead of hardcoding.
        self.gsp_config.total_avail_chids = 2048;
        self.build_engine_bitmap_from_caps(engine_caps);
        self.init_comm_layout();
        self.init_chid_allocator();
        Ok(())
    }

    /// Wait for the GSP vGPU plugin to finish bootloading.
    ///
    /// Maps the management heap ctrl buffer via BAR1 and polls
    /// `message_seq_num` until GSP writes [`GSP_PLUGIN_BOOTLOADED`].
    pub(crate) fn wait_plugin_ready(
        instance: &VgpuInstance,
        bar_user: &mut BarUser,
        mm: &GpuMm,
        bar1: &Bar1,
        comm_size: u64,
    ) -> Result {
        let mgmt = instance.mgmt_heap.as_ref().ok_or(EINVAL)?;
        let base = mgmt.addr();
        let page_size: u64 = PAGE_SIZE.into_safe_cast();
        let num_pages = ((comm_size + page_size - 1) / page_size) as usize;

        let mut pfns = KVec::new();
        for i in 0..num_pages {
            let i_u64: u64 = i.into_safe_cast();
            pfns.push(
                Pfn::from(VramAddress::new(base + i_u64 * page_size)),
                GFP_KERNEL,
            )?;
        }

        let access = bar_user.map(mm, bar1, &pfns, false)?;

        kernel::io::poll::read_poll_timeout(
            || access.try_read32(CTRL_BUF_MSG_SEQ_NUM_OFFSET),
            |val| *val == GSP_PLUGIN_BOOTLOADED,
            PLUGIN_POLL_INTERVAL,
            PLUGIN_BOOT_TIMEOUT,
        )?;

        Ok(())
    }

    /// Create a mock vGPU instance for bootload testing.
    ///
    /// Uses `Gfid(1)` (VF0) and the first registered vGPU type.
    pub(crate) fn mock_create_instance(
        &mut self,
        mm: &GpuMm,
        cmdq: &Cmdq,
        bar: &Bar0,
        h_client: u32,
        h_subdevice: u32,
        pdev: &pci::Device<device::Bound>,
    ) -> Result<usize> {
        // Use VF dbdf: PF bus/dev with function=4 (first VF).
        let dbdf = (pdev.dev_id() as u32) | 4;

        let instance = VgpuInstance {
            id: 0,
            gfid: Gfid(1),
            dbdf: Dbdf(dbdf),
            vgpu_type_idx: 0,
            vm_pid: 1,
            chid_offset: 0,
            num_chid: 0,
            num_plugin_channels: 0,
            fbmem_heap: None,
            mgmt_heap: None,
            active: false,
        };
        self.create_instance(mm, cmdq, bar, h_client, h_subdevice, instance)
    }

    /// Upload a hardcoded L40-1Q vGPU type to GSP and record it locally.
    ///
    /// Builds `NV2080_CTRL_VGPU_MGR_INTERNAL_PGPU_ADD_VGPU_TYPE_PARAMS` with a
    /// single `NVA081_CTRL_VGPU_INFO` entry from the embedded L40-1Q binary dump,
    /// sends it to GSP, then records the corresponding [`VgpuType`].
    pub(crate) fn upload_vgpu_type(
        &mut self,
        cmdq: &Cmdq,
        bar: &Bar0,
        h_client: u32,
        h_subdevice: u32,
    ) -> Result {
        let params_size = 8 + NVA081_MAX_VGPU_TYPES_PER_PGPU * NVA081_CTRL_VGPU_INFO_SIZE;
        let mut params = KVVec::from_elem(0u8, params_size, GFP_KERNEL)?;
        let p = params.as_mut_slice();

        p[0] = 1; // discardVgpuTypes
        p[4..8].copy_from_slice(&1u32.to_ne_bytes()); // vgpuInfoCount = 1
        p[8..8 + NVA081_CTRL_VGPU_INFO_SIZE].copy_from_slice(L40_1Q_VGPU_INFO);

        check_rmcontrol_status(
            cmdq, bar, CMD_PGPU_ADD_VGPU_TYPE, p, h_client, h_subdevice,
        )?;

        let mut name = [0u8; VGPU_TYPE_NAME_MAX];
        let src = b"NVIDIA L40-1Q";
        name[..src.len()].copy_from_slice(src);

        self.vgpu_types.push(
            VgpuType {
                vgpu_type_id: 871,
                name,
                vdev_id: 0x26b5_176f,
                pdev_id: 0x26b5,
                fb_length: 0x4000_0000,
                gsp_heap_size: 0x200_0000,
                bar1_length: 0x100,
                max_instance: 32,
                ecc_supported: 1,
                fb_reservation: 0,
            },
            GFP_KERNEL,
        )?;

        Ok(())
    }

    /// Create a vGPU instance: allocate resources, bootload the plugin, register.
    pub(crate) fn create_instance(
        &mut self,
        mm: &GpuMm,
        cmdq: &Cmdq,
        bar: &Bar0,
        h_client: u32,
        h_subdevice: u32,
        mut instance: VgpuInstance,
    ) -> Result<usize> {
        let vgpu_type_idx = instance.vgpu_type_idx;
        self.setup_chids(&mut instance)?;

        let vgpu_type = self.vgpu_types.get(vgpu_type_idx).ok_or(EINVAL)?;

        let fbmem = match self.alloc_guest_fb(mm, vgpu_type) {
            Ok(b) => b,
            Err(e) => {
                self.release_chids(&mut instance);
                return Err(e);
            }
        };
        let mgmt = match Self::alloc_plugin_heap(mm, vgpu_type) {
            Ok(b) => b,
            Err(e) => {
                self.release_chids(&mut instance);
                return Err(e);
            }
        };

        instance.fbmem_heap = Some(fbmem);
        instance.mgmt_heap = Some(mgmt);

        if let Err(e) = self.bootload_plugin(cmdq, bar, h_client, h_subdevice, &instance) {
            self.release_chids(&mut instance);
            return Err(e);
        }

        instance.active = true;
        self.register_instance(instance)
    }

    /// Destroy a vGPU instance: shutdown plugin, free resources, unregister.
    pub(crate) fn destroy_instance(
        &mut self,
        cmdq: &Cmdq,
        bar: &Bar0,
        h_client: u32,
        h_subdevice: u32,
        vf_id: i32,
    ) -> Result {
        let pos = self
            .instances
            .iter()
            .position(|inst| inst.id == vf_id)
            .ok_or(ENODEV)?;

        let instance = &self.instances[pos];
        if instance.active {
            let gfid = instance.gfid.0;
            if let Err(e) = self.shutdown_plugin(cmdq, bar, h_client, h_subdevice, gfid) {
                kernel::pr_warn!("vgpu: shutdown failed for gfid {}: {:?}\n", gfid, e);
            }
            if let Err(e) = self.cleanup_plugin(cmdq, bar, h_client, h_subdevice, gfid) {
                kernel::pr_warn!("vgpu: cleanup failed for gfid {}: {:?}\n", gfid, e);
            }
        }

        let instance = self.instances.remove(pos)?;
        self.release_chids_by_value(instance.chid_offset, instance.num_chid);
        Ok(())
    }

    /// Bootload the vGPU plugin task on GSP.
    ///
    /// Builds an RM-ABI-compatible [`BootloadParams`] and sends it to GSP.
    /// The params struct is heap-allocated and cast from a zeroed byte buffer
    /// to avoid a 6616-byte stack allocation.
    pub(crate) fn bootload_plugin(
        &self,
        cmdq: &Cmdq,
        bar: &Bar0,
        h_client: u32,
        h_subdevice: u32,
        instance: &VgpuInstance,
    ) -> Result {
        let vgpu_type = self.vgpu_types.get(instance.vgpu_type_idx).ok_or(EINVAL)?;
        let fbmem = instance.fbmem_heap.as_ref().ok_or(EINVAL)?;
        let mgmt = instance.mgmt_heap.as_ref().ok_or(EINVAL)?;

        let mut buf = KVVec::from_elem(0u8, size_of::<BootloadParams>(), GFP_KERNEL)?;
        // SAFETY: `buf` is exactly `size_of::<BootloadParams>()` bytes, heap-allocated
        // (at least 8-byte aligned), and zero-initialized.
        let params: &mut BootloadParams = unsafe {
            &mut *(buf.as_mut_ptr() as *mut BootloadParams)
        };

        params.dbdf = instance.dbdf.0;
        params.gfid = instance.gfid.0;
        params.vgpu_type = vgpu_type.vgpu_type_id;
        params.vm_pid = instance.vm_pid;
        params.num_channels = instance.num_chid;
        params.num_plugin_channels = instance.num_plugin_channels;

        for i in 0..NV2080_GPU_MAX_ENGINES {
            if self.engine_bitmap[i / 64] & (1u64 << (i % 64)) != 0 {
                params.chid_offset[i] = instance.chid_offset;
            }
        }

        params.num_guest_fb_segments = 1;
        params.guest_fb_phys_addr_list[0] = fbmem.addr();
        params.guest_fb_length_list[0] = fbmem.size();

        params.plugin_heap_memory_phys_addr = mgmt.addr();
        params.plugin_heap_memory_length = mgmt.size();

        let vgpu_log_off =
            self.comm_layout.init_task_log_offset + self.comm_layout.init_task_log_size;
        let kernel_log_off = vgpu_log_off + self.comm_layout.vgpu_task_log_size;

        params.init_task_log_buff_offset = mgmt.addr() + self.comm_layout.init_task_log_offset;
        params.init_task_log_buff_size = self.comm_layout.init_task_log_size;
        params.vgpu_task_log_buff_offset = mgmt.addr() + vgpu_log_off;
        params.vgpu_task_log_buff_size = self.comm_layout.vgpu_task_log_size;
        params.kernel_log_buff_offset = mgmt.addr() + kernel_log_off;
        params.kernel_log_buff_size = self.comm_layout.kernel_log_size;

        check_rmcontrol_status(
            cmdq, bar, CMD_VGPU_BOOTLOAD, buf.as_mut_slice(), h_client, h_subdevice,
        )
    }

    /// Send a gfid-only RM control command to GSP.
    fn send_gfid_command(
        &self,
        cmdq: &Cmdq,
        bar: &Bar0,
        cmd: u32,
        h_client: u32,
        h_subdevice: u32,
        gfid: u32,
    ) -> Result {
        let mut params = gfid.to_ne_bytes();
        check_rmcontrol_status(cmdq, bar, cmd, &mut params, h_client, h_subdevice)
    }

    /// Shutdown the vGPU plugin task on GSP.
    pub(crate) fn shutdown_plugin(
        &self,
        cmdq: &Cmdq,
        bar: &Bar0,
        h_client: u32,
        h_subdevice: u32,
        gfid: u32,
    ) -> Result {
        self.send_gfid_command(cmdq, bar, CMD_VGPU_SHUTDOWN, h_client, h_subdevice, gfid)
    }

    /// Cleanup the vGPU plugin task resources on GSP.
    pub(crate) fn cleanup_plugin(
        &self,
        cmdq: &Cmdq,
        bar: &Bar0,
        h_client: u32,
        h_subdevice: u32,
        gfid: u32,
    ) -> Result {
        self.send_gfid_command(cmdq, bar, CMD_VGPU_CLEANUP, h_client, h_subdevice, gfid)
    }

    /// Build the engine bitmap from GSP static config's `engineCaps`.
    ///
    /// `engineCaps` is a 96-bit bitmap where bit N indicates RM engine type N is present.
    pub(crate) fn build_engine_bitmap_from_caps(&mut self, engine_caps: &[u32; 3]) {
        self.engine_bitmap[0] = (engine_caps[1] as u64) << 32 | engine_caps[0] as u64;
        if ENGINE_BITMAP_WORDS > 1 {
            self.engine_bitmap[1] = engine_caps[2] as u64;
        }
    }

    /// Set up the CPU -> GSP plugin RPC channel and send initial configuration.
    ///
    /// Negotiates the stable RPC version, then sends config params via NVKV encoding.
    /// Must be called after bootload_plugin and wait_plugin_ready.
    pub(crate) fn setup_plugin_rpc(
        &self,
        instance: &VgpuInstance,
        bar_user: &mut BarUser,
        mm_gpu: &GpuMm,
        bar0: &Bar0,
        bar1: &Bar1,
        h_client: u32,
    ) -> Result {
        let mut rpc = PluginRpc::new(
            instance,
            bar_user,
            mm_gpu,
            bar1,
            self.comm_layout.total_size,
        )?;

        rpc.call(bar0, instance.gfid, RpcMsg::VersionNegotiation, &mut [])?;

        let vgpu_type = self.vgpu_types.get(instance.vgpu_type_idx).ok_or(EINVAL)?;
        let nvkv_data = encode_config_params_nvkv(instance, vgpu_type, h_client)?;
        let kv_pairs = nvkv_data.as_bytes();
        let kv_count = nvkv_data.kv_count();

        // msg_buff_v2 layout: [kv_count: u64][kv_pairs: u64[]]
        let mut buf = KVVec::from_elem(0u8, VGPU_GSP_MESSAGE_REGION_SIZE as usize, GFP_KERNEL)?;
        buf.as_mut_slice()[..8].copy_from_slice(&kv_count.to_ne_bytes());
        buf.as_mut_slice()[8..8 + kv_pairs.len()].copy_from_slice(kv_pairs);

        rpc.send_request(bar0, instance.gfid, RpcMsg::SetupConfigParamsAndInit, buf.as_slice())?;
        let result = rpc.recv_response(buf.as_mut_slice())?;

        // Firmware writes debug dump to msg_buff: [magic=0xDEBD0001][stable_rpc_mode][config_params...]
        dump_config_params_response(buf.as_slice());

        if result != 0 {
            kernel::pr_err!("vgpu: RPC SetupConfigParamsAndInit failed, result_code={:#x}\n", result);
            return Err(EIO);
        }
        Ok(())
    }
}

// --- CPU -> GSP Plugin RPC -------------------------------------------------

/// Byte offsets within VGPU_CPU_GSP_CTRL_BUFF_REGION.
mod ctrl_buf_off {
    pub(super) const VERSION: usize = 0;
    pub(super) const MESSAGE_TYPE: usize = 4;
    pub(super) const MESSAGE_SEQ_NUM: usize = 8;
    pub(super) const RESPONSE_BUFF_OFFSET: usize = 16;
    pub(super) const MESSAGE_BUFF_OFFSET: usize = 24;
    pub(super) const MIGRATION_BUFF_OFFSET: usize = 32;
    pub(super) const ERROR_BUFF_OFFSET: usize = 40;
}

/// Byte offsets within VGPU_CPU_GSP_RESPONSE_BUFF_REGION.
mod resp_buf_off {
    pub(super) const MESSAGE_SEQ_NUM_PROCESSED: usize = 4;
    pub(super) const RESULT_CODE: usize = 8;
}

/// RPC message types (mirrors NV_VGPU_CPU_RPC_MSG_*).
#[repr(u32)]
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
enum RpcMsg {
    VersionNegotiation = 1,
    SetupConfigParamsAndInit = 2,
    Reset = 3,
}

/// Doorbell register offset within BAR0 for the vGPU plugin.
const DOORBELL_BASE: usize = 0xb8_0000 + 0x2200;

/// ctrl_buff->version: version 2 enables stable RPC mode.
/// chips_a GSP plugin requires version >= 2.
const VGPU_CPU_GSP_CTRL_BUFF_VERSION: u32 = 2;

/// RPC response timeout and poll interval.
const RPC_TIMEOUT: kernel::time::Delta = kernel::time::Delta::from_secs(120);
const RPC_POLL_INTERVAL: kernel::time::Delta = kernel::time::Delta::from_millis(1);

/// CPU -> GSP plugin RPC channel state.
struct PluginRpc<'a> {
    access: mm::bar_user::BarAccess<'a>,
    ctrl_off: usize,
    resp_off: usize,
    msg_off: usize,
    msg_seq_num: u32,
}

impl<'a> PluginRpc<'a> {
    fn new(
        instance: &VgpuInstance,
        bar_user: &'a mut BarUser,
        mm_gpu: &'a GpuMm,
        bar1: &'a Bar1,
        comm_size: u64,
    ) -> Result<Self> {
        let mgmt = instance.mgmt_heap.as_ref().ok_or(EINVAL)?;
        let base = mgmt.addr();
        let page_size: u64 = PAGE_SIZE.into_safe_cast();
        let num_pages = ((comm_size + page_size - 1) / page_size) as usize;

        let mut pfns = KVec::new();
        for i in 0..num_pages {
            let i_u64: u64 = i.into_safe_cast();
            pfns.push(
                Pfn::from(VramAddress::new(base + i_u64 * page_size)),
                GFP_KERNEL,
            )?;
        }

        let access = bar_user.map(mm_gpu, bar1, &pfns, true)?;

        let ctrl_off: usize = 0;
        let resp_off = VGPU_GSP_CTRL_REGION_SIZE as usize;
        let msg_off = resp_off + VGPU_GSP_RESPONSE_REGION_SIZE as usize;
        let migration_off = msg_off + VGPU_GSP_MESSAGE_REGION_SIZE as usize;
        let error_off = migration_off + VGPU_GSP_MIGRATION_REGION_SIZE as usize;
        access.try_write32(VGPU_CPU_GSP_CTRL_BUFF_VERSION, ctrl_off + ctrl_buf_off::VERSION)?;
        access.try_write64(resp_off as u64, ctrl_off + ctrl_buf_off::RESPONSE_BUFF_OFFSET)?;
        access.try_write64(msg_off as u64, ctrl_off + ctrl_buf_off::MESSAGE_BUFF_OFFSET)?;
        access.try_write64(migration_off as u64, ctrl_off + ctrl_buf_off::MIGRATION_BUFF_OFFSET)?;
        access.try_write64(error_off as u64, ctrl_off + ctrl_buf_off::ERROR_BUFF_OFFSET)?;

        Ok(Self {
            access,
            ctrl_off,
            resp_off,
            msg_off,
            msg_seq_num: 0,
        })
    }

    fn trigger_doorbell(&self, bar0: &Bar0, gfid: Gfid) {
        let v: u32 = gfid.0 * 32 + 17;
        bar0.write32(v, DOORBELL_BASE);
        let _ = bar0.read32(DOORBELL_BASE);
    }

    fn send_request(
        &mut self,
        bar0: &Bar0,
        gfid: Gfid,
        msg_type: RpcMsg,
        data: &[u8],
    ) -> Result {
        // Zero the message buffer
        let mut z = 0usize;
        while z < VGPU_GSP_MESSAGE_REGION_SIZE as usize {
            self.access.try_write32(0, self.msg_off + z)?;
            z += 4;
        }

        let mut off = 0usize;
        while off + 4 <= data.len() {
            let val = u32::from_ne_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
            self.access.try_write32(val, self.msg_off + off)?;
            off += 4;
        }
        if off < data.len() {
            let mut tail = [0u8; 4];
            tail[..data.len() - off].copy_from_slice(&data[off..]);
            let val = u32::from_ne_bytes(tail);
            self.access.try_write32(val, self.msg_off + off)?;
        }

        self.access.try_write32(msg_type as u32, self.ctrl_off + ctrl_buf_off::MESSAGE_TYPE)?;
        self.msg_seq_num += 1;
        self.access.try_write32(self.msg_seq_num, self.ctrl_off + ctrl_buf_off::MESSAGE_SEQ_NUM)?;

        self.trigger_doorbell(bar0, gfid);
        Ok(())
    }

    fn recv_response(&self, data: &mut [u8]) -> Result<u32> {
        let expected = self.msg_seq_num;

        kernel::io::poll::read_poll_timeout(
            || self.access.try_read32(self.resp_off + resp_buf_off::MESSAGE_SEQ_NUM_PROCESSED),
            |val| *val == expected,
            RPC_POLL_INTERVAL,
            RPC_TIMEOUT,
        )?;

        let result = self.access.try_read32(self.resp_off + resp_buf_off::RESULT_CODE)?;

        let mut off = 0usize;
        while off + 4 <= data.len() {
            let val = self.access.try_read32(self.msg_off + off)?;
            data[off..off + 4].copy_from_slice(&val.to_ne_bytes());
            off += 4;
        }

        Ok(result)
    }

    fn call(
        &mut self,
        bar0: &Bar0,
        gfid: Gfid,
        msg_type: RpcMsg,
        data: &mut [u8],
    ) -> Result {
        self.send_request(bar0, gfid, msg_type, data)?;
        let result = self.recv_response(data)?;
        if result != 0 {
            kernel::pr_err!("vgpu: RPC {:?} failed, result_code={:#x}\n", msg_type, result);
            return Err(EIO);
        }
        Ok(())
    }

}

// --- NVKV Encoder ----------------------------------------------------------

const NVKV_OPCODE_IMM32: u64 = 0x0;
const NVKV_OPCODE_SEQ64: u64 = 0x2;
const NVKV_OPCODE_ARRAY8: u64 = 0x3;

const fn nvkv_imm32(key: u64, val: u32) -> u64 {
    (NVKV_OPCODE_IMM32 << 28) | (key & 0xFFFF) | ((val as u64) << 32)
}
const fn nvkv_seq64(key: u64, count: u32) -> u64 {
    (NVKV_OPCODE_SEQ64 << 28) | (key & 0xFFFF) | ((count as u64) << 32)
}
const fn nvkv_array8(key: u64, count: u32) -> u64 {
    (NVKV_OPCODE_ARRAY8 << 28) | (key & 0xFFFF) | ((count as u64) << 32)
}

struct NvkvWriter {
    buf: KVec<u64>,
}

impl NvkvWriter {
    fn new() -> Result<Self> {
        Ok(Self { buf: KVec::new() })
    }
    fn push(&mut self, val: u64) -> Result {
        self.buf.push(val, GFP_KERNEL)?;
        Ok(())
    }
    fn put_u32(&mut self, key: u64, val: u32) -> Result {
        self.push(nvkv_imm32(key, val))
    }
    fn put_u64(&mut self, key: u64, val: u64) -> Result {
        self.push(nvkv_seq64(key, 1))?;
        self.push(val)
    }
    fn put_bytes(&mut self, key: u64, data: &[u8]) -> Result {
        self.push(nvkv_array8(key, data.len() as u32))?;
        let mut off = 0usize;
        while off < data.len() {
            let mut qword = [0u8; 8];
            let end = (off + 8).min(data.len());
            qword[..end - off].copy_from_slice(&data[off..end]);
            self.push(u64::from_ne_bytes(qword))?;
            off += 8;
        }
        Ok(())
    }
    fn kv_count(&self) -> u64 {
        self.buf.len() as u64
    }
    fn as_bytes(&self) -> &[u8] {
        let ptr = self.buf.as_slice().as_ptr() as *const u8;
        let len = self.buf.len() * 8;
        // SAFETY: u64 slice is contiguous and properly aligned.
        unsafe { core::slice::from_raw_parts(ptr, len) }
    }
}

fn encode_config_params_nvkv(
    instance: &VgpuInstance,
    vgpu_type: &VgpuType,
    h_client: u32,
) -> Result<NvkvWriter> {
    let mut w = NvkvWriter::new()?;
    w.put_bytes(0x001, &[0u8; 16])?;          // VGPU_UUID
    w.put_u32(0x002, instance.dbdf.0)?;        // DBDF
    w.put_u32(0x003, 0)?;                      // DRIVER_VM_VF_DBDF
    w.put_u32(0x004, 0)?;                        // VGPU_DEVICE_INSTANCE_ID
    w.put_u32(0x005, vgpu_type.vgpu_type_id)?; // VGPU_TYPE
    w.put_u32(0x006, instance.vm_pid)?;        // VM_PID
    w.put_u32(0x010, 0)?;                        // SWIZZ_ID
    w.put_u32(0x011, instance.num_chid)?;      // NUM_CHANNELS
    w.put_u32(0x012, 3)?;                      // NUM_PLUGIN_CHANNELS
    w.put_u32(0x020, 0)?;                      // VMM_CAP
    w.put_u32(0x021, 0x4000)?;                 // MIGRATION_FEATURE
    w.put_u32(0x022, 4)?;                      // HYPERVISOR_TYPE (KVM)
    w.put_u32(0x023, 2)?;                      // HOST_CPU_ARCH (X86_64)
    w.put_u64(0x024, 4096)?;                   // HOST_PAGE_SIZE
    // FEATURE_FLAGS: U64 bitfield packing 15 booleans (bits 0-14)
    //   bit 3: ENABLE_UVM = 1
    //   bit 5: VMM_MIGRATION_SUPPORTED = 1
    let feature_flags: u64 = (1 << 3) | (1 << 5);
    w.put_u64(0x030, feature_flags)?;          // FEATURE_FLAGS
    w.put_u32(0x040, 0)?;                      // PLACEMENT_ID
    w.put_u32(0x042, 0)?;                      // CHANNEL_USAGE_THRESHOLD_PCT
    Ok(w)
}

// --- Debug dump from firmware -----------------------------------------------
// Firmware writes to msg_buff: [magic=0xDEBD0001][stable_rpc_mode][config_params struct]
// config_params struct offsets match NV_VGPU_CPU_RPC_DATA_COPY_CONFIG_PARAMS.

fn dump_config_params_response(buf: &[u8]) {
    fn r32(b: &[u8], off: usize) -> u32 {
        if off + 4 <= b.len() {
            u32::from_ne_bytes([b[off], b[off+1], b[off+2], b[off+3]])
        } else { 0 }
    }
    fn r64(b: &[u8], off: usize) -> u64 {
        if off + 8 <= b.len() {
            u64::from_ne_bytes([b[off], b[off+1], b[off+2], b[off+3],
                                b[off+4], b[off+5], b[off+6], b[off+7]])
        } else { 0 }
    }

    let magic = r32(buf, 0);
    if magic != 0xDEBD_0002 {
        kernel::pr_err!("vgpu: no debug dump in msg_buff (magic={:#x})\n", magic);
        return;
    }
    let mode = r32(buf, 4);
    let stage = r32(buf, 8);
    let err = r32(buf, 12);
    // Config params start at offset 16
    let p = &buf[16..];

    kernel::pr_info!("vgpu: GSP stage={} err={} stable_rpc={}: dbdf={:#x} instId={:#x} type={} pid={} swizz={:#x} ch={} pch={} hyp={} page={} uvm={}\n",
        stage, err, mode,
        r32(p, 16),  // dbdf
        r32(p, 24),  // vgpu_device_instance_id
        r32(p, 28),  // vgpu_type
        r32(p, 32),  // vm_pid
        r32(p, 36),  // swizz_id
        r32(p, 40),  // num_channels
        r32(p, 44),  // num_plugin_channels
        r32(p, 56),  // hypervisor_type
        r64(p, 64),  // host_page_size
        p.get(75).copied().unwrap_or(0),  // enable_uvm
    );
}
