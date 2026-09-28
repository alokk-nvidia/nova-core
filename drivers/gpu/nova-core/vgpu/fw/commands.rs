// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

//! Wire types and codecs for vGPU commands.
//!
//! Defines request encoders and response schemas independently of command
//! submission and instance lifecycle operations.

use kernel::{
    alloc::ArrayVec,
    bitfield,
    num::casts::usize_as_u64,
    prelude::*, //
};

use crate::{
    gsp::nvkv::{
        nvkv_decode,
        nvkv_encode,
        Array,
        Encodable,
        Encoder,
        Index,
        Key,
        KeyId,
        Required, //
    },
    mm::vram::VramRegion, //
};

use super::bindings;

/// Message types supported by the nova-core plugin RPC channel.
#[derive(Clone, Copy)]
#[repr(u32)]
pub(crate) enum RpcMessage {
    VersionNegotiation = bindings::MESSAGE_NV_VGPU_CPU_RPC_MSG_VERSION_NEGOTIATION,
    SetupConfigParamsAndInit = bindings::MESSAGE_NV_VGPU_CPU_RPC_MSG_SETUP_CONFIG_PARAMS_AND_INIT,
    Reset = bindings::MESSAGE_NV_VGPU_CPU_RPC_MSG_RESET,
    UpdateBmeState = bindings::MESSAGE_NV_VGPU_CPU_RPC_MSG_UPDATE_BME_STATE,
}

bitfield! {
    pub(crate) struct Dbdf(u32) {
        2:0 function;
        7:3 device;
        15:8 bus;
        31:16 domain;
    }
}

#[derive(Clone, Copy)]
struct SwizzId(u32);

impl SwizzId {
    const WHOLE_GPU: Self = Self(0xFFFF_FFFF);
}

impl From<SwizzId> for u32 {
    fn from(value: SwizzId) -> Self {
        value.0
    }
}

bitfield! {
    pub(crate) struct ChannelMapEntry(u64) {
        15:0 engine_type;
        31:16 index;
        63:32 chid_offset;
    }
}

impl ChannelMapEntry {
    const KEY: KeyId = 0x1001;

    pub(crate) fn new(engine_type: u16, index: u16, chid_offset: u32) -> Self {
        Self::zeroed()
            .with_engine_type(engine_type)
            .with_index(index)
            .with_chid_offset(chid_offset)
    }
}

impl Encodable for KVVec<ChannelMapEntry> {
    fn encode(&self, encoder: &mut Encoder) -> Result {
        // SAFETY: `ChannelMapEntry` is a `bitfield!` over `u64`, i.e.
        // `#[repr(transparent)]` around a `u64`, so the entries are
        // layout-compatible with `u64` and can be viewed as a `u64` slice.
        let slice = unsafe { core::slice::from_raw_parts(self.as_ptr().cast::<u64>(), self.len()) };
        encoder.encode_array64(ChannelMapEntry::KEY, Index::new::<0>(), slice)
    }
}

bitfield! {
    struct VgpuBootloadOptions(u64) {
    }
}

nvkv_encode! {
    struct VgpuBootloadRequest {
        dbdf: Key<Dbdf, { Self::DBDF_KEY }, u32>,
        gfid: Key<u32, { Self::GFID_KEY }>,
        vgpu_type: Key<u32, { Self::VGPU_TYPE_KEY }>,
        vm_pid: Key<u32, { Self::VM_PID_KEY }>,
        swizz_id: Key<SwizzId, { Self::SWIZZ_ID_KEY }, u32>,
        num_channels: Key<u32, { Self::NUM_CHANNELS_KEY }>,
        num_plugin_channels: Key<u32, { Self::NUM_PLUGIN_CHANNELS_KEY }>,
        guest_fb_segment_count: Key<u32, { Self::GUEST_FB_SEGMENT_COUNT_KEY }>,
        options: Key<VgpuBootloadOptions, { Self::OPTIONS_KEY }, u64>,
        channel_mapping: KVVec<ChannelMapEntry>,
        guest_fb_segment_phys_addr: Array<u64, 8, { Self::GUEST_FB_SEGMENT_PHYS_ADDR_KEY }>,
        guest_fb_segment_length: Array<u64, 8, { Self::GUEST_FB_SEGMENT_LENGTH_KEY }>,
        plugin_heap_phys_addr: Key<u64, { Self::PLUGIN_HEAP_PHYS_ADDR_KEY }>,
        plugin_heap_length: Key<u64, { Self::PLUGIN_HEAP_LENGTH_KEY }>,
        ctrl_buff_offset: Key<u64, { Self::CTRL_BUFF_OFFSET_KEY }>,
        init_task_log_offset: Key<u64, { Self::INIT_TASK_LOG_OFFSET_KEY }>,
        init_task_log_size: Key<u64, { Self::INIT_TASK_LOG_SIZE_KEY }>,
        vgpu_task_log_offset: Key<u64, { Self::VGPU_TASK_LOG_OFFSET_KEY }>,
        vgpu_task_log_size: Key<u64, { Self::VGPU_TASK_LOG_SIZE_KEY }>,
        kernel_log_offset: Key<u64, { Self::KERNEL_LOG_OFFSET_KEY }>,
        kernel_log_size: Key<u64, { Self::KERNEL_LOG_SIZE_KEY }>,
        mig_rm_heap_phys_addr: Key<u64, { Self::MIG_RM_HEAP_PHYS_ADDR_KEY }>,
        mig_rm_heap_length: Key<u64, { Self::MIG_RM_HEAP_LENGTH_KEY }>,
    }
}

impl VgpuBootloadRequest {
    const DBDF_KEY: KeyId = 0x0001;
    const GFID_KEY: KeyId = 0x0002;
    const VGPU_TYPE_KEY: KeyId = 0x0003;
    const VM_PID_KEY: KeyId = 0x0004;
    const SWIZZ_ID_KEY: KeyId = 0x0005;
    const NUM_CHANNELS_KEY: KeyId = 0x0006;
    const NUM_PLUGIN_CHANNELS_KEY: KeyId = 0x0007;
    const GUEST_FB_SEGMENT_COUNT_KEY: KeyId = 0x0008;
    const OPTIONS_KEY: KeyId = 0x1000;
    const GUEST_FB_SEGMENT_PHYS_ADDR_KEY: KeyId = 0x1002;
    const GUEST_FB_SEGMENT_LENGTH_KEY: KeyId = 0x1003;
    const PLUGIN_HEAP_PHYS_ADDR_KEY: KeyId = 0x1004;
    const PLUGIN_HEAP_LENGTH_KEY: KeyId = 0x1005;
    const CTRL_BUFF_OFFSET_KEY: KeyId = 0x1006;
    const INIT_TASK_LOG_OFFSET_KEY: KeyId = 0x1007;
    const INIT_TASK_LOG_SIZE_KEY: KeyId = 0x1008;
    const VGPU_TASK_LOG_OFFSET_KEY: KeyId = 0x1009;
    const VGPU_TASK_LOG_SIZE_KEY: KeyId = 0x100A;
    const KERNEL_LOG_OFFSET_KEY: KeyId = 0x100B;
    const KERNEL_LOG_SIZE_KEY: KeyId = 0x100C;
    const MIG_RM_HEAP_PHYS_ADDR_KEY: KeyId = 0x100D;
    const MIG_RM_HEAP_LENGTH_KEY: KeyId = 0x100E;
}

/// Identity, channel mapping and VRAM regions passed to a GSP plugin at boot.
///
/// Region addresses are physical VRAM addresses, including the log fields whose wire
/// names contain `offset`. The control-buffer offset is relative to the plugin heap.
pub(crate) struct BootloadInfo<'a> {
    pub(crate) dbdf: Dbdf,
    pub(crate) gfid: u32,
    pub(crate) vgpu_type: u32,
    pub(crate) vm_pid: u32,
    pub(crate) num_channels: u32,
    pub(crate) num_plugin_channels: u32,
    pub(crate) channel_mapping: KVVec<ChannelMapEntry>,
    pub(crate) guest_fb: &'a VramRegion,
    pub(crate) plugin_heap: &'a VramRegion,
    pub(crate) ctrl_buffer_offset: u64,
    pub(crate) init_log: &'a VramRegion,
    pub(crate) vgpu_log: &'a VramRegion,
    pub(crate) kernel_log: &'a VramRegion,
}

/// Encodes a `VGPU_BOOTLOAD` request using the typed NVKV schema.
pub(crate) fn encode_vgpu_bootload(info: BootloadInfo<'_>) -> Result<KVVec<u64>> {
    let request = VgpuBootloadRequest {
        dbdf: info.dbdf.into(),
        gfid: info.gfid.into(),
        vgpu_type: info.vgpu_type.into(),
        vm_pid: info.vm_pid.into(),
        swizz_id: SwizzId::WHOLE_GPU.into(),
        num_channels: info.num_channels.into(),
        num_plugin_channels: info.num_plugin_channels.into(),
        guest_fb_segment_count: 1.into(),
        options: VgpuBootloadOptions::zeroed().into(),
        channel_mapping: info.channel_mapping,
        guest_fb_segment_phys_addr: Array::new(&[info.guest_fb.address()])?,
        guest_fb_segment_length: Array::new(&[info.guest_fb.size()])?,
        plugin_heap_phys_addr: info.plugin_heap.address().into(),
        plugin_heap_length: info.plugin_heap.size().into(),
        ctrl_buff_offset: info.ctrl_buffer_offset.into(),
        init_task_log_offset: info.init_log.address().into(),
        init_task_log_size: info.init_log.size().into(),
        vgpu_task_log_offset: info.vgpu_log.address().into(),
        vgpu_task_log_size: info.vgpu_log.size().into(),
        kernel_log_offset: info.kernel_log.address().into(),
        kernel_log_size: info.kernel_log.size().into(),
        // The current non-SMIG firmware path does not map a separate MIG-RM heap.
        mig_rm_heap_phys_addr: 0.into(),
        mig_rm_heap_length: 0.into(),
    };

    let mut encoder = Encoder::new();
    request.encode(&mut encoder)?;
    Ok(encoder.finish())
}

nvkv_decode! {
    pub(crate) struct VgpuPropertiesSchema => VgpuProperties {
        // TODO: `name`/`class` required?
        name: Array<u8, { VgpuProperties::STRING_LEN }, { Self::TYPE_NAME_KEY }>,
        class: Array<u8, { VgpuProperties::STRING_LEN }, { Self::CLASS_KEY }>,
        type_id: Required<Key<u32, { Self::TYPE_ID_KEY }>>,
        bar1_length: Required<Key<u64, { Self::BAR1_LENGTH_KEY }>>,
        max_instance: Required<Key<u32, { Self::MAX_INSTANCE_KEY }>>,
        ecc: Key<u32, { Self::ECC_KEY }>,
        profile_size: Required<Key<u64, { Self::PROFILE_SIZE_KEY }>>,
        max_fps: Key<u32, { Self::MAX_FPS_KEY }>,
        num_heads: Key<u32, { Self::NUM_HEADS_KEY }>,
        max_res_x: Key<u32, { Self::MAX_RES_X_KEY }>,
        max_res_y: Key<u32, { Self::MAX_RES_Y_KEY }>,
        dev_id: Required<Key<u32, { Self::DEV_ID_KEY }>>,
        subsystem_id: Required<Key<u32, { Self::SUBSYSTEM_ID_KEY }>>,
        fb_length: Required<Key<u64, { Self::FB_LENGTH_KEY }>>,
        gsp_heap_size: Required<Key<u64, { Self::GSP_HEAP_SIZE_KEY }>>,
        fb_reservation: Required<Key<u64, { Self::FB_RESERVATION_KEY }>>,
    }
}

impl VgpuPropertiesSchema {
    const TYPE_NAME_KEY: KeyId = 0x3100;
    const CLASS_KEY: KeyId = 0x3101;
    const TYPE_ID_KEY: KeyId = 0x3102;
    const BAR1_LENGTH_KEY: KeyId = 0x3103;
    const MAX_INSTANCE_KEY: KeyId = 0x3104;
    const ECC_KEY: KeyId = 0x3105;
    const PROFILE_SIZE_KEY: KeyId = 0x3106;
    const MAX_FPS_KEY: KeyId = 0x3107;
    const NUM_HEADS_KEY: KeyId = 0x3108;
    const MAX_RES_X_KEY: KeyId = 0x3109;
    const MAX_RES_Y_KEY: KeyId = 0x310A;
    const DEV_ID_KEY: KeyId = 0x310B;
    const SUBSYSTEM_ID_KEY: KeyId = 0x310C;
    const FB_LENGTH_KEY: KeyId = 0x310D;
    const GSP_HEAP_SIZE_KEY: KeyId = 0x310E;
    const FB_RESERVATION_KEY: KeyId = 0x310F;
}

pub(crate) struct VgpuProperties {
    name: ArrayVec<u8, { Self::STRING_LEN }>,
    class: ArrayVec<u8, { Self::STRING_LEN }>,
    pub(crate) type_id: u32,
    pub(crate) bar1_length: u64,
    pub(crate) max_instance: u32,
    ecc: u32,
    profile_size: u64,
    max_fps: u32,
    num_heads: u32,
    max_res_x: u32,
    max_res_y: u32,
    pub(crate) dev_id: u32,
    pub(crate) subsystem_id: u32,
    pub(crate) fb_length: u64,
    pub(crate) gsp_heap_size: u64,
    fb_reservation: u64,
}

impl VgpuProperties {
    const STRING_LEN: usize = 64;
}

#[derive(Clone, Copy)]
#[repr(u32)]
enum HypervisorType {
    Unknown = 4,
}

impl From<HypervisorType> for u32 {
    fn from(value: HypervisorType) -> Self {
        // CAST: `HypervisorType` uses the wire field's `u32` representation.
        value as u32
    }
}

#[derive(Clone, Copy)]
#[repr(u32)]
enum CpuArch {
    Aarch64 = 1,
    X86_64 = 2,
}

impl CpuArch {
    fn host() -> Result<Self> {
        if cfg!(target_arch = "x86_64") {
            Ok(Self::X86_64)
        } else if cfg!(target_arch = "aarch64") {
            Ok(Self::Aarch64)
        } else {
            Err(EOPNOTSUPP)
        }
    }
}

impl From<CpuArch> for u32 {
    fn from(value: CpuArch) -> Self {
        // CAST: `CpuArch` uses the wire field's `u32` representation.
        value as u32
    }
}

#[derive(Clone, Copy)]
struct MigrationFeature(u32);

impl MigrationFeature {
    const PRESERVE_CTX_BUF: Self = Self(0x4000);
}

impl From<MigrationFeature> for u32 {
    fn from(value: MigrationFeature) -> Self {
        value.0
    }
}

bitfield! {
    struct FeatureFlags(u64) {
        3:3 enable_uvm => bool;
        5:5 vmm_migration => bool;
    }
}

nvkv_encode! {
    struct PluginConfigParamsRequest {
        uuid: Key<[u8; 16], { Self::UUID_KEY }>,
        dbdf: Key<Dbdf, { Self::DBDF_KEY }, u32>,
        device_instance_id: Key<u32, { Self::DEVICE_INSTANCE_ID_KEY }>,
        vgpu_type: Key<u32, { Self::VGPU_TYPE_KEY }>,
        vm_pid: Key<u32, { Self::VM_PID_KEY }>,
        swizz_id: Key<SwizzId, { Self::SWIZZ_ID_KEY }, u32>,
        num_channels: Key<u32, { Self::NUM_CHANNELS_KEY }>,
        num_plugin_channels: Key<u32, { Self::NUM_PLUGIN_CHANNELS_KEY }>,
        vmm_cap: Key<u32, { Self::VMM_CAP_KEY }>,
        migration_feature: Key<MigrationFeature, { Self::MIGRATION_FEATURE_KEY }, u32>,
        hypervisor_type: Key<HypervisorType, { Self::HYPERVISOR_TYPE_KEY }, u32>,
        cpu_arch: Key<CpuArch, { Self::CPU_ARCH_KEY }, u32>,
        page_size: Key<u64, { Self::PAGE_SIZE_KEY }>,
        feature_flags: Key<FeatureFlags, { Self::FEATURE_FLAGS_KEY }, u64>,
    }
}

impl PluginConfigParamsRequest {
    const UUID_KEY: KeyId = 0x0001;
    const DBDF_KEY: KeyId = 0x0002;
    const DEVICE_INSTANCE_ID_KEY: KeyId = 0x0004;
    const VGPU_TYPE_KEY: KeyId = 0x0005;
    const VM_PID_KEY: KeyId = 0x0006;
    const SWIZZ_ID_KEY: KeyId = 0x0010;
    const NUM_CHANNELS_KEY: KeyId = 0x0011;
    const NUM_PLUGIN_CHANNELS_KEY: KeyId = 0x0012;
    const VMM_CAP_KEY: KeyId = 0x0020;
    const MIGRATION_FEATURE_KEY: KeyId = 0x0021;
    const HYPERVISOR_TYPE_KEY: KeyId = 0x0022;
    const CPU_ARCH_KEY: KeyId = 0x0023;
    const PAGE_SIZE_KEY: KeyId = 0x0024;
    const FEATURE_FLAGS_KEY: KeyId = 0x0030;
}

/// Encodes plugin configuration parameters using the typed NVKV schema.
pub(crate) fn encode_plugin_config_params(
    uuid: [u8; 16],
    dbdf: Dbdf,
    vgpu_type: u32,
    vm_pid: u32,
    num_channels: u32,
    num_plugin_channels: u32,
) -> Result<KVVec<u64>> {
    let request = PluginConfigParamsRequest {
        uuid: uuid.into(),
        dbdf: dbdf.into(),
        // The full host PCI address distinguishes VFs within a VM, including VFs
        // from different PFs. This opaque identity lasts for the host device lifetime.
        device_instance_id: dbdf.into_raw().into(),
        vgpu_type: vgpu_type.into(),
        vm_pid: vm_pid.into(),
        swizz_id: SwizzId::WHOLE_GPU.into(),
        num_channels: num_channels.into(),
        num_plugin_channels: num_plugin_channels.into(),
        // Advertise no optional VMM capabilities (vmioplugin.h VMM_CAP_*),
        // independently of the plugin feature_flags and migration_feature below.
        vmm_cap: 0.into(),
        migration_feature: MigrationFeature::PRESERVE_CTX_BUF.into(),
        hypervisor_type: HypervisorType::Unknown.into(),
        cpu_arch: CpuArch::host()?.into(),
        page_size: usize_as_u64(kernel::page::PAGE_SIZE).into(),
        feature_flags: FeatureFlags::zeroed()
            .with_enable_uvm(false)
            .with_vmm_migration(true)
            .into(),
    };

    let mut encoder = Encoder::new();
    request.encode(&mut encoder)?;
    Ok(encoder.finish())
}

#[repr(C)]
#[derive(IntoBytes, zerocopy_derive::Immutable)]
pub(crate) struct AllocCeutilsRequest {
    pub(crate) gfid: u32,
    pub(crate) fixed_chid: u32,
    pub(crate) force_ceid: u32,
    pub(crate) swizz_id: u32,
}

static_assert!(size_of::<AllocCeutilsRequest>() == 16);

#[repr(C)]
#[derive(FromBytes)]
pub(crate) struct AllocCeutilsResponse {
    pub(crate) semaphore_address: u64,
    pub(crate) semaphore_aperture: u32,
    _reserved: u32,
}

static_assert!(size_of::<AllocCeutilsResponse>() == 16);

#[repr(C)]
#[derive(IntoBytes, zerocopy_derive::Immutable)]
pub(crate) struct FreeCeutilsRequest {
    pub(crate) gfid: u32,
}

static_assert!(size_of::<FreeCeutilsRequest>() == 4);

#[repr(C)]
#[derive(IntoBytes, zerocopy_derive::Immutable)]
pub(crate) struct ScrubGuestFbRequest {
    pub(crate) gfid: u32,
    pub(crate) reserved: u32,
    pub(crate) fb_offset: u64,
    pub(crate) fb_size: u64,
}

static_assert!(size_of::<ScrubGuestFbRequest>() == 24);

#[repr(C)]
#[derive(FromBytes)]
pub(crate) struct ScrubGuestFbResponse {
    pub(crate) work_id: u64,
}

static_assert!(size_of::<ScrubGuestFbResponse>() == 8);

nvkv_encode! {
    struct PluginSetBmeRequest {
        bme_enable: Key<bool, { Self::BME_ENABLE_KEY }, u32>,
    }
}

impl PluginSetBmeRequest {
    const BME_ENABLE_KEY: KeyId = 0x0100;
}

/// Encodes a plugin BME state update using the typed NVKV schema.
pub(crate) fn encode_plugin_set_bme(enable: bool) -> Result<KVVec<u64>> {
    let request = PluginSetBmeRequest {
        bme_enable: enable.into(),
    };

    let mut encoder = Encoder::new();
    request.encode(&mut encoder)?;
    Ok(encoder.finish())
}
