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
    sync::Arc,
    time::{
        delay::fsleep,
        Delta,
        Instant,
        Monotonic, //
    },
};

use crate::{
    driver::{
        Bar0,
        Bar1, //
    },
    gpu::Architecture,
    gsp::{
        cmdq::Cmdq,
        nvkv, //
    },
    mm::GpuMm, //
};

/// VRAM allocation with RAII lifetime. Drop frees blocks back to buddy allocator.
pub(crate) struct VramBlock {
    _blocks: Pin<KBox<AllocatedBlocks>>,
    pub addr: u64,
    pub size: u64,
}

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
pub(crate) struct Bar1Map {
    bar1: Arc<Devres<Bar1>>,
    pub offset: u64,
    #[expect(dead_code)]
    pub size: u64,
}

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

    #[expect(dead_code)]
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
    pub(crate) instances: KVec<VgpuInstance>,
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
            instances: KVec::new(),
        })
    }

    pub(crate) fn set_vgpu_enabled(&mut self, enabled: bool) {
        self.vgpu_enabled = enabled;
    }

    fn next_id(&mut self) -> i32 {
        self.instances.len() as i32
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

// --- Gfid / Dbdf ---

/// Guest Function ID. GFID 0 is reserved for PF, VFs start at 1.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Gfid(pub u32);

/// PCI address encoding: domain[31:16] bus[15:8] devfn[7:0].
#[repr(transparent)]
#[derive(Clone, Copy)]
pub(crate) struct Dbdf(pub u32);

// --- VgpuType ---

/// vGPU type descriptor, populated from QUERY_VGPU_PROPERTIES NVKV response.
pub(crate) struct VgpuType {
    pub name: [u8; 64],
    pub class: [u8; 64],
    pub vgpu_type_id: u32,
    pub bar1_length: u64,
    pub max_instance: u32,
    pub ecc_supported: u32,
    pub profile_size: u64,
    pub max_fps: u32,
    pub num_heads: u32,
    pub max_res_x: u32,
    pub max_res_y: u32,
    pub fb_length: u64,
    pub gsp_heap_size: u64,
    pub fb_reservation: u64,
}

impl Default for VgpuType {
    fn default() -> Self {
        Self {
            name: [0u8; 64],
            class: [0u8; 64],
            vgpu_type_id: 0,
            bar1_length: 0,
            max_instance: 0,
            ecc_supported: 0,
            profile_size: 0,
            max_fps: 0,
            num_heads: 0,
            max_res_x: 0,
            max_res_y: 0,
            fb_length: 0,
            gsp_heap_size: 0,
            fb_reservation: 0,
        }
    }
}

/// NVKV key constants for QUERY_VGPU_PROPERTIES response decoding.
mod vgpu_prop_keys {
    pub(super) const TYPE_NAME: u16 = 0x3100;
    pub(super) const CLASS: u16 = 0x3101;
    pub(super) const TYPE_ID: u16 = 0x3102;
    pub(super) const BAR1_LENGTH: u16 = 0x3103;
    pub(super) const MAX_INSTANCE: u16 = 0x3104;
    pub(super) const ECC: u16 = 0x3105;
    pub(super) const PROFILE_SIZE: u16 = 0x3106;
    pub(super) const MAX_FPS: u16 = 0x3107;
    pub(super) const NUM_HEADS: u16 = 0x3108;
    pub(super) const MAX_RES_X: u16 = 0x3109;
    pub(super) const MAX_RES_Y: u16 = 0x310A;
    pub(super) const DEV_ID: u16 = 0x310B;
    pub(super) const SUBSYSTEM_ID: u16 = 0x310C;
    pub(super) const FB_LENGTH: u16 = 0x310D;
    pub(super) const GSP_HEAP_SIZE: u16 = 0x310E;
    pub(super) const FB_RESERVATION: u16 = 0x310F;
}

pub(crate) mod gmcapi {
    pub(crate) const VGPU_MGMT_QUERY_PROPERTIES: u32 = 0x1000_0006;
    pub(crate) const VGPU_MGMT_QUERY_ASSIGNED_VF: u32 = 0x1000_0007;
    pub(crate) const VGPU_BOOTLOAD: u32 = 0x1000_0020;
    pub(crate) const VGPU_SHUTDOWN: u32 = 0x1000_0021;
    pub(crate) const VGPU_SHUTDOWN_COMPLETE: u32 = 0x1000_0022;
    pub(crate) const VGPU_CLEANUP: u32 = 0x1000_0023;
}

/// Plugin RPC channel for vGPU plugin communication.
pub(crate) struct PluginRpc {
    bar1_map: Bar1Map,
}

const GSP_PLUGIN_BOOTLOADED: u32 = 0x4E65_4A6F;
const CTRL_BUF_MSG_SEQ_NUM_OFFSET: u64 = 8;
const PLUGIN_BOOT_TIMEOUT_MS: u64 = 10_000;

impl PluginRpc {
    pub(crate) fn new(bar1_map: Bar1Map) -> Self {
        Self { bar1_map }
    }

    /// Poll ctrl_buf for plugin boot completion magic.
    pub(crate) fn wait_plugin_ready(&self, dev: &device::Device<device::Bound>) -> Result {
        let start = Instant::<Monotonic>::now();
        let timeout = Delta::from_millis(PLUGIN_BOOT_TIMEOUT_MS as i64);
        loop {
            let val = self.bar1_map.read32(dev, CTRL_BUF_MSG_SEQ_NUM_OFFSET)?;
            if val == GSP_PLUGIN_BOOTLOADED {
                return Ok(());
            }
            if start.elapsed() > timeout {
                return Err(ETIMEDOUT);
            }
            fsleep(Delta::from_millis(1));
        }
    }
}

// --- VgpuInstance ---

/// A live vGPU instance with allocated resources.
#[expect(dead_code)]
pub(crate) struct VgpuInstance {
    pub id: i32,
    pub gfid: Gfid,
    pub dbdf: Dbdf,
    pub vgpu_type: VgpuType,
    pub vm_pid: u32,
    pub chid_offset: u32,
    pub num_chid: u32,
    pub num_plugin_channels: u32,
    pub fbmem_heap: Option<VramBlock>,
    pub mgmt_heap: Option<VramBlock>,
    pub plugin_rpc: Option<PluginRpc>,
    pub active: bool,
}

// --- GMCAPI query helpers ---

/// Query the vGPU type assigned to a VF by its DBDF.
#[expect(dead_code)]
pub(crate) fn query_assigned_vf_type(cmdq: &Cmdq, bar: &Bar0, dbdf: Dbdf) -> Result<u32> {
    let in_params = u64::from(dbdf.0).to_le_bytes();
    let resp = cmdq.send_gmc_and_receive(
        bar,
        gmcapi::VGPU_MGMT_QUERY_ASSIGNED_VF,
        &in_params,
        64,
    )?;
    if resp.status != 0 {
        return Err(EIO);
    }
    if resp.payload.len() < 4 {
        return Err(ENODEV);
    }
    Ok(u32::from_le_bytes(
        resp.payload[..4].try_into().map_err(|_| EINVAL)?,
    ))
}

/// Query vGPU type properties and decode NVKV response.
#[expect(dead_code)]
pub(crate) fn query_vgpu_type(cmdq: &Cmdq, bar: &Bar0, type_id: u32) -> Result<VgpuType> {
    let in_params = type_id.to_le_bytes();
    let resp = cmdq.send_gmc_and_receive(
        bar,
        gmcapi::VGPU_MGMT_QUERY_PROPERTIES,
        &in_params,
        4096,
    )?;
    if resp.status != 0 {
        return Err(EIO);
    }

    let mut vt = VgpuType::default();
    nvkv::nvkv_decode(&resp.payload, |key, _index, value| match key {
        vgpu_prop_keys::TYPE_NAME => nvkv::nvkv_read_string8(&value, &mut vt.name),
        vgpu_prop_keys::CLASS => nvkv::nvkv_read_string8(&value, &mut vt.class),
        vgpu_prop_keys::TYPE_ID => vt.vgpu_type_id = nvkv::nvkv_read_u32(&value),
        vgpu_prop_keys::BAR1_LENGTH => vt.bar1_length = nvkv::nvkv_read_u64(&value),
        vgpu_prop_keys::MAX_INSTANCE => vt.max_instance = nvkv::nvkv_read_u32(&value),
        vgpu_prop_keys::ECC => vt.ecc_supported = nvkv::nvkv_read_u32(&value),
        vgpu_prop_keys::PROFILE_SIZE => vt.profile_size = nvkv::nvkv_read_u64(&value),
        vgpu_prop_keys::MAX_FPS => vt.max_fps = nvkv::nvkv_read_u32(&value),
        vgpu_prop_keys::NUM_HEADS => vt.num_heads = nvkv::nvkv_read_u32(&value),
        vgpu_prop_keys::MAX_RES_X => vt.max_res_x = nvkv::nvkv_read_u32(&value),
        vgpu_prop_keys::MAX_RES_Y => vt.max_res_y = nvkv::nvkv_read_u32(&value),
        vgpu_prop_keys::FB_LENGTH => vt.fb_length = nvkv::nvkv_read_u64(&value),
        vgpu_prop_keys::GSP_HEAP_SIZE => vt.gsp_heap_size = nvkv::nvkv_read_u64(&value),
        vgpu_prop_keys::FB_RESERVATION => vt.fb_reservation = nvkv::nvkv_read_u64(&value),
        _ => {}
    })?;

    Ok(vt)
}

// --- VgpuManager: create / destroy ---

impl VgpuManager {
    /// Create a new vGPU instance with allocated resources and bootload the plugin.
    #[expect(dead_code)]
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn create_instance(
        &mut self,
        dev: &device::Device<device::Bound>,
        cmdq: &Cmdq,
        bar: &Bar0,
        mm: &GpuMm,
        bar1: &Arc<Devres<Bar1>>,
        chid_alloc: &mut ChidAllocator,
        gfid: Gfid,
        dbdf: Dbdf,
        vgpu_type: VgpuType,
        vm_pid: u32,
    ) -> Result<VgpuInstance> {
        let num_chid =
            self.gsp_config.total_avail_chids / vgpu_type.max_instance.max(1);
        let chid_offset = chid_alloc.alloc(num_chid)?;

        let fb_size = vgpu_type.fb_length;
        let fb_align = self.gsp_config.vmmu_segment_size;
        let fbmem = alloc_vram(mm, fb_size, fb_align)
            .inspect_err(|_| chid_alloc.free(chid_offset, num_chid))?;

        let mgmt = alloc_vram(mm, vgpu_type.gsp_heap_size.max(self.comm_layout.total_size), 4096)
            .inspect_err(|_| chid_alloc.free(chid_offset, num_chid))?;

        let bar1_map = Bar1Map::new(bar1, mgmt.addr, self.comm_layout.total_size)?;
        let plugin_rpc = PluginRpc::new(bar1_map);

        let mut instance = VgpuInstance {
            id: self.next_id(),
            gfid,
            dbdf,
            vgpu_type,
            vm_pid,
            chid_offset,
            num_chid,
            num_plugin_channels: 3,
            fbmem_heap: Some(fbmem),
            mgmt_heap: Some(mgmt),
            plugin_rpc: Some(plugin_rpc),
            active: false,
        };

        bootload(dev, cmdq, bar, &instance, &self.engine_masks, &self.comm_layout)?;
        instance.active = true;

        Ok(instance)
    }

    /// Destroy a vGPU instance by GFID: send GSP shutdown sequence, then free resources.
    #[expect(dead_code)]
    pub(crate) fn destroy_instance(
        &mut self,
        cmdq: &Cmdq,
        bar: &Bar0,
        chid_alloc: &mut ChidAllocator,
        gfid: Gfid,
    ) -> Result {
        let idx = self
            .instances
            .iter()
            .position(|i| i.gfid == gfid)
            .ok_or(ENOENT)?;

        let gfid_val = self.instances[idx].gfid.0;
        let mut payload = KVec::<u8>::new();
        payload.extend_from_slice(&gfid_val.to_le_bytes(), GFP_KERNEL)?;

        cmdq.send_gmc_fire_and_forget(bar, gmcapi::VGPU_SHUTDOWN, payload.as_slice())?;

        cmdq.wait_gmc_event(bar, Delta::from_secs(10), |cmd_id, _data| {
            cmd_id == gmcapi::VGPU_SHUTDOWN_COMPLETE
        })?;

        cmdq.send_gmc_no_response(bar, gmcapi::VGPU_CLEANUP, payload.as_slice())?;

        let instance = self.instances.remove(idx).map_err(|_| EINVAL)?;
        chid_alloc.free(instance.chid_offset, instance.num_chid);

        Ok(())
    }
}

mod bootload_keys {
    pub(super) const DBDF: u16 = 0x0001;
    pub(super) const GFID: u16 = 0x0002;
    pub(super) const VGPU_TYPE: u16 = 0x0003;
    pub(super) const VM_PID: u16 = 0x0004;
    #[expect(dead_code)]
    pub(super) const SWIZZ_ID: u16 = 0x0005;
    pub(super) const NUM_CHANNELS: u16 = 0x0006;
    pub(super) const NUM_PLUGIN_CHANNELS: u16 = 0x0007;
    pub(super) const GUEST_FB_SEGMENT_COUNT: u16 = 0x0008;

    pub(super) const CHANNEL_MAPPING: u16 = 0x1001;
    pub(super) const GUEST_FB_SEGMENT_PHYS_ADDR: u16 = 0x1002;
    pub(super) const GUEST_FB_SEGMENT_LENGTH: u16 = 0x1003;
    pub(super) const PLUGIN_HEAP_PHYS_ADDR: u16 = 0x1004;
    pub(super) const PLUGIN_HEAP_LENGTH: u16 = 0x1005;
    pub(super) const CTRL_BUFF_OFFSET: u16 = 0x1006;
    pub(super) const INIT_TASK_LOG_OFFSET: u16 = 0x1007;
    pub(super) const INIT_TASK_LOG_SIZE: u16 = 0x1008;
    pub(super) const VGPU_TASK_LOG_OFFSET: u16 = 0x1009;
    pub(super) const VGPU_TASK_LOG_SIZE: u16 = 0x100A;
    pub(super) const KERNEL_LOG_OFFSET: u16 = 0x100B;
    pub(super) const KERNEL_LOG_SIZE: u16 = 0x100C;
}

/// Build a 32-bit GMC engine ID from type and instance index.
///
/// Format: `type[15:0] | index[31:16]` per `gmcapi_engine_types.h`.
fn gmc_engine_id(engine_type: usize, index: usize) -> u64 {
    (engine_type as u64) | ((index as u64) << 16)
}

/// Send GMCAPI VGPU_BOOTLOAD with NVKV-encoded parameters, then wait for
/// the plugin to signal readiness via the ctrl_buf magic value.
pub(crate) fn bootload(
    dev: &device::Device<device::Bound>,
    cmdq: &Cmdq,
    bar: &Bar0,
    instance: &VgpuInstance,
    engine_masks: &GmcEngineMasks,
    comm_layout: &CommBuffLayout,
) -> Result {
    let mut kvs: KVec<u64> = KVec::new();

    nvkv::nvkv_push_imm32(&mut kvs, bootload_keys::DBDF, instance.dbdf.0)?;
    nvkv::nvkv_push_imm32(&mut kvs, bootload_keys::GFID, instance.gfid.0)?;
    nvkv::nvkv_push_imm32(
        &mut kvs,
        bootload_keys::VGPU_TYPE,
        instance.vgpu_type.vgpu_type_id,
    )?;
    nvkv::nvkv_push_imm32(&mut kvs, bootload_keys::VM_PID, instance.vm_pid)?;
    nvkv::nvkv_push_imm32(&mut kvs, bootload_keys::NUM_CHANNELS, instance.num_chid)?;
    nvkv::nvkv_push_imm32(
        &mut kvs,
        bootload_keys::NUM_PLUGIN_CHANNELS,
        instance.num_plugin_channels,
    )?;

    let mut channel_map: KVec<u64> = KVec::new();
    for engine_type in 1..NVGMC_ENGINE_TYPE_COUNT {
        let mask = engine_masks.masks[engine_type];
        let mut bit = 0u32;
        let mut remaining = mask;
        while remaining != 0 {
            if remaining & 1 != 0 {
                let engine_id = gmc_engine_id(engine_type, bit as usize);
                let entry = engine_id | ((instance.chid_offset as u64) << 32);
                channel_map.push(entry, GFP_KERNEL)?;
            }
            remaining >>= 1;
            bit += 1;
        }
    }
    nvkv::nvkv_push_array64(&mut kvs, bootload_keys::CHANNEL_MAPPING, channel_map.as_slice())?;

    let fb = instance.fbmem_heap.as_ref().ok_or(EINVAL)?;
    nvkv::nvkv_push_imm32(&mut kvs, bootload_keys::GUEST_FB_SEGMENT_COUNT, 1)?;
    nvkv::nvkv_push_seq64(
        &mut kvs,
        bootload_keys::GUEST_FB_SEGMENT_PHYS_ADDR,
        &[fb.addr],
    )?;
    nvkv::nvkv_push_seq64(
        &mut kvs,
        bootload_keys::GUEST_FB_SEGMENT_LENGTH,
        &[fb.size],
    )?;

    let mgmt = instance.mgmt_heap.as_ref().ok_or(EINVAL)?;
    nvkv::nvkv_push_seq64(
        &mut kvs,
        bootload_keys::PLUGIN_HEAP_PHYS_ADDR,
        &[mgmt.addr],
    )?;
    nvkv::nvkv_push_seq64(
        &mut kvs,
        bootload_keys::PLUGIN_HEAP_LENGTH,
        &[mgmt.size],
    )?;

    nvkv::nvkv_push_seq64(&mut kvs, bootload_keys::CTRL_BUFF_OFFSET, &[0u64])?;
    nvkv::nvkv_push_seq64(
        &mut kvs,
        bootload_keys::INIT_TASK_LOG_OFFSET,
        &[mgmt.addr + comm_layout.init_task_log_offset],
    )?;
    nvkv::nvkv_push_seq64(
        &mut kvs,
        bootload_keys::INIT_TASK_LOG_SIZE,
        &[comm_layout.init_task_log_size],
    )?;
    let vgpu_log_offset =
        mgmt.addr + comm_layout.init_task_log_offset + comm_layout.init_task_log_size;
    nvkv::nvkv_push_seq64(
        &mut kvs,
        bootload_keys::VGPU_TASK_LOG_OFFSET,
        &[vgpu_log_offset],
    )?;
    nvkv::nvkv_push_seq64(
        &mut kvs,
        bootload_keys::VGPU_TASK_LOG_SIZE,
        &[comm_layout.vgpu_task_log_size],
    )?;
    let kernel_log_offset = vgpu_log_offset + comm_layout.vgpu_task_log_size;
    nvkv::nvkv_push_seq64(
        &mut kvs,
        bootload_keys::KERNEL_LOG_OFFSET,
        &[kernel_log_offset],
    )?;
    nvkv::nvkv_push_seq64(
        &mut kvs,
        bootload_keys::KERNEL_LOG_SIZE,
        &[comm_layout.kernel_log_size],
    )?;

    // SAFETY: `kvs` is a valid `KVec<u64>` and we reinterpret it as a byte
    // slice of the same total size. The pointer is valid for `len * 8` bytes.
    let payload: &[u8] = unsafe {
        core::slice::from_raw_parts(kvs.as_ptr().cast::<u8>(), kvs.len() * 8)
    };
    cmdq.send_gmc_fire_and_forget(bar, gmcapi::VGPU_BOOTLOAD, payload)?;

    let rpc = instance.plugin_rpc.as_ref().ok_or(EINVAL)?;
    rpc.wait_plugin_ready(dev)
}
