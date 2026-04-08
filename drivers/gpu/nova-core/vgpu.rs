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
    mm::{self, GpuMm, VramBlock},
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

/// Prebuilt NVA081_CTRL_VGPU_INFO for L40-1Q (type 871).
const L40_1Q_VGPU_INFO: &[u8] = include_bytes!("l40_1q.bin");
const NVA081_CTRL_VGPU_INFO_SIZE: usize = 5224;
const NVA081_MAX_VGPU_TYPES_PER_PGPU: usize = 64;

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
    if x == 0 {
        return 0;
    }
    1 << (31 - x.leading_zeros())
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
            chid_alloc: ChidAllocator::new(),
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
        let aligned_total = self.gsp_config.total_fbmem_size.div_ceil(seg) * seg;
        let per_instance = aligned_total / u64::from(vgpu_type.max_instance)
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
            self.chid_alloc
                .free(instance.chid_offset, instance.num_chid);
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
        cmdq: &Cmdq,
        bar: &Bar0,
        total_vram: u64,
    ) -> Result {
        self.gsp_config.total_fbmem_size = total_vram;
        // TODO: query actual available CHIDs from GSP instead of hardcoding.
        self.gsp_config.total_avail_chids = 2048;
        self.build_engine_bitmap(cmdq, bar)?;
        self.init_comm_layout();
        self.init_chid_allocator();
        Ok(())
    }

    /// Create a vGPU instance: allocate resources, bootload the plugin, register.
    pub(crate) fn create_instance(
        &mut self,
        mm: &GpuMm,
        cmdq: &Cmdq,
        bar: &Bar0,
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

        if let Err(e) = self.bootload_plugin(cmdq, bar, &instance) {
            self.release_chids(&mut instance);
            return Err(e);
        }

        instance.active = true;
        self.register_instance(instance)
    }

    /// Destroy a vGPU instance: shutdown plugin, free resources, unregister.
    pub(crate) fn destroy_instance(&mut self, cmdq: &Cmdq, bar: &Bar0, vf_id: i32) -> Result {
        let pos = self
            .instances
            .iter()
            .position(|inst| inst.id == vf_id)
            .ok_or(ENODEV)?;

        let instance = &self.instances[pos];
        if instance.active {
            let gfid = instance.gfid.0;
            if let Err(e) = self.shutdown_plugin(cmdq, bar, gfid) {
                kernel::pr_warn!("vgpu: shutdown failed for gfid {}: {:?}\n", gfid, e);
            }
            if let Err(e) = self.cleanup_plugin(cmdq, bar, gfid) {
                kernel::pr_warn!("vgpu: cleanup failed for gfid {}: {:?}\n", gfid, e);
            }
        }

        let instance = self.instances.remove(pos)?;
        self.release_chids_by_value(instance.chid_offset, instance.num_chid);
        Ok(())
    }

    /// Upload a hardcoded L40-1Q vGPU type to GSP and record it locally.
    ///
    /// Builds `NV2080_CTRL_VGPU_MGR_INTERNAL_PGPU_ADD_VGPU_TYPE_PARAMS` with a
    /// single `NVA081_CTRL_VGPU_INFO` entry from the embedded L40-1Q binary dump,
    /// sends it to GSP, then records the corresponding [`VgpuType`].
    pub(crate) fn upload_vgpu_type(&mut self, cmdq: &Cmdq, bar: &Bar0) -> Result {
        let params_size = 8 + NVA081_MAX_VGPU_TYPES_PER_PGPU * NVA081_CTRL_VGPU_INFO_SIZE;
        let mut params = KVec::with_capacity(params_size, GFP_KERNEL)?;
        params.resize(params_size, 0u8, GFP_KERNEL)?;
        let p = params.as_mut_slice();

        p[0] = 1; // discardVgpuTypes
        p[4..8].copy_from_slice(&1u32.to_ne_bytes()); // vgpuInfoCount = 1
        p[8..8 + NVA081_CTRL_VGPU_INFO_SIZE].copy_from_slice(L40_1Q_VGPU_INFO);

        send_vgpu_command(cmdq, bar, CMD_PGPU_ADD_VGPU_TYPE, p)?;

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

    /// Bootload the vGPU plugin task on GSP.
    ///
    /// Builds an RM-ABI-compatible [`BootloadParams`] and sends it to GSP.
    /// The params struct is heap-allocated and cast from a zeroed byte buffer
    /// to avoid a 6616-byte stack allocation.
    pub(crate) fn bootload_plugin(
        &self,
        cmdq: &Cmdq,
        bar: &Bar0,
        instance: &VgpuInstance,
    ) -> Result {
        let vgpu_type = self.vgpu_types.get(instance.vgpu_type_idx).ok_or(EINVAL)?;
        let fbmem = instance.fbmem_heap.as_ref().ok_or(EINVAL)?;
        let mgmt = instance.mgmt_heap.as_ref().ok_or(EINVAL)?;

        let mut buf = KVec::with_capacity(size_of::<BootloadParams>(), GFP_KERNEL)?;
        buf.resize(size_of::<BootloadParams>(), 0u8, GFP_KERNEL)?;
        // SAFETY: `buf` is exactly `size_of::<BootloadParams>()` bytes, heap-allocated
        // (at least 8-byte aligned), and zero-initialized.
        let params: &mut BootloadParams =
            unsafe { &mut *buf.as_mut_ptr().cast::<BootloadParams>() };

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

        send_vgpu_command(cmdq, bar, CMD_VGPU_BOOTLOAD, buf.as_mut_slice())
    }

    /// Send a gfid-only command to GSP.
    fn send_gfid_command(&self, cmdq: &Cmdq, bar: &Bar0, cmd: u32, gfid: u32) -> Result {
        let mut params = gfid.to_ne_bytes();
        send_vgpu_command(cmdq, bar, cmd, &mut params)
    }

    /// Shutdown the vGPU plugin task on GSP.
    pub(crate) fn shutdown_plugin(&self, cmdq: &Cmdq, bar: &Bar0, gfid: u32) -> Result {
        self.send_gfid_command(cmdq, bar, CMD_VGPU_SHUTDOWN, gfid)
    }

    /// Cleanup the vGPU plugin task resources on GSP.
    pub(crate) fn cleanup_plugin(&self, cmdq: &Cmdq, bar: &Bar0, gfid: u32) -> Result {
        self.send_gfid_command(cmdq, bar, CMD_VGPU_CLEANUP, gfid)
    }

    /// Build the engine bitmap by querying GSP via the device info table.
    pub(crate) fn build_engine_bitmap(&mut self, cmdq: &Cmdq, bar: &Bar0) -> Result {
        let mut base_index: u32 = 0;

        loop {
            let mut buf = [0u8; size_of::<DeviceInfoTableParams>()];
            buf[0..4].copy_from_slice(&base_index.to_ne_bytes());

            send_vgpu_command(cmdq, bar, CMD_GET_DEVICE_INFO_TABLE, &mut buf)?;

            // SAFETY: `buf` is exactly `size_of::<DeviceInfoTableParams>()` bytes.
            let params: &DeviceInfoTableParams =
                unsafe { &*buf.as_ptr().cast::<DeviceInfoTableParams>() };

            let n = (params.num_entries as usize).min(DEVICE_INFO_TABLE_MAX_ENTRIES);
            for i in 0..n {
                let rm_engine_type = params.entries[i].engine_data[ENGINE_INFO_TYPE_RM_ENGINE_TYPE];
                let eid = rm_engine_type as usize;
                if eid > 0 && eid < NV2080_GPU_MAX_ENGINES {
                    self.engine_bitmap[eid / 64] |= 1u64 << (eid % 64);
                }
            }

            if params.b_more == 0 {
                break;
            }
            base_index += params.num_entries;
        }

        Ok(())
    }
}
