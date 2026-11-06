// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

//! Wire types and codecs for vGPU commands.
//!
//! Defines request encoders and response schemas independently of command
//! submission and instance lifecycle operations.

use kernel::{
    alloc::ArrayVec,
    bitfield,
    prelude::*, //
};

use crate::{
    gsp::nvkv::{
        nvkv_decode,
        nvkv_encode,
        Array,
        Encodable,
        Encoder,
        Index, //
        Key,
        KeyId,
        Required,
    },
    mm::vram::VramRegion,
    vgpu::Dbdf, //
};

use super::bindings;

/// Message types supported by the nova-core plugin RPC channel.
#[derive(Clone, Copy)]
#[repr(u32)]
pub(crate) enum RpcMessage {
    VersionNegotiation = bindings::MESSAGE_NV_VGPU_CPU_RPC_MSG_VERSION_NEGOTIATION,
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
        dbdf: Key<Dbdf, 0x0001, u32>,
        gfid: Key<u32, 0x0002>,
        vgpu_type: Key<u32, 0x0003>,
        vm_pid: Key<u32, 0x0004>,
        swizz_id: Key<SwizzId, 0x0005, u32>,
        num_channels: Key<u32, 0x0006>,
        num_plugin_channels: Key<u32, 0x0007>,
        guest_fb_segment_count: Key<u32, 0x0008>,
        options: Key<VgpuBootloadOptions, 0x1000, u64>,
        channel_mapping: KVVec<ChannelMapEntry>,
        guest_fb_segment_phys_addr: Array<u64, 8, 0x1002>,
        guest_fb_segment_length: Array<u64, 8, 0x1003>,
        plugin_heap_phys_addr: Key<u64, 0x1004>,
        plugin_heap_length: Key<u64, 0x1005>,
        ctrl_buff_offset: Key<u64, 0x1006>,
        init_task_log_offset: Key<u64, 0x1007>,
        init_task_log_size: Key<u64, 0x1008>,
        vgpu_task_log_offset: Key<u64, 0x1009>,
        vgpu_task_log_size: Key<u64, 0x100A>,
        kernel_log_offset: Key<u64, 0x100B>,
        kernel_log_size: Key<u64, 0x100C>,
        mig_rm_heap_phys_addr: Key<u64, 0x100D>,
        mig_rm_heap_length: Key<u64, 0x100E>,
    }
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
        guest_fb_segment_phys_addr: Array::new(&[info.guest_fb.range().start])?,
        guest_fb_segment_length: Array::new(&[info.guest_fb.len()])?,
        plugin_heap_phys_addr: info.plugin_heap.range().start.into(),
        plugin_heap_length: info.plugin_heap.len().into(),
        ctrl_buff_offset: info.ctrl_buffer_offset.into(),
        init_task_log_offset: info.init_log.range().start.into(),
        init_task_log_size: info.init_log.len().into(),
        vgpu_task_log_offset: info.vgpu_log.range().start.into(),
        vgpu_task_log_size: info.vgpu_log.len().into(),
        kernel_log_offset: info.kernel_log.range().start.into(),
        kernel_log_size: info.kernel_log.len().into(),
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
        name: Array<u8, { VgpuProperties::STRING_LEN }, 0x3100>,
        class: Array<u8, { VgpuProperties::STRING_LEN }, 0x3101>,
        type_id: Required<Key<u32, 0x3102>>,
        bar1_length: Required<Key<u64, 0x3103>>,
        max_instance: Required<Key<u32, 0x3104>>,
        ecc: Key<u32, 0x3105>,
        profile_size: Required<Key<u64, 0x3106>>,
        max_fps: Key<u32, 0x3107>,
        num_heads: Key<u32, 0x3108>,
        max_res_x: Key<u32, 0x3109>,
        max_res_y: Key<u32, 0x310A>,
        dev_id: Required<Key<u32, 0x310B>>,
        subsystem_id: Required<Key<u32, 0x310C>>,
        fb_length: Required<Key<u64, 0x310D>>,
        gsp_heap_size: Required<Key<u64, 0x310E>>,
        fb_reservation: Required<Key<u64, 0x310F>>,
    }
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

// TODO: extract from bindings
#[repr(C)]
#[derive(FromBytes, IntoBytes)]
pub(crate) struct AssignedVfVgpuType {
    pub(crate) vgpu_type_id: u32,
    _swizz_id: u32,
    _placement_id: u32,
}
