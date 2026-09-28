// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

//! GSP plugin communication buffer mappings and access.

use kernel::{
    num::casts::u32_as_usize,
    prelude::*,
    sync::Mutex, //
};

use crate::mm::{
    bar_user::{
        BarMapping,
        BarUser, //
    },
    vram::VramRegion,
    GpuMm, //
};

use super::fw::{
    self,
    RawControlRegion,
    RawResponseRegion,
    RpcMessage,
    RpcResponse, //
};

static_assert!(
    fw::VGPU_CPU_GSP_CTRL_BUFF_REGION_SIZE
        + fw::VGPU_CPU_GSP_RESPONSE_BUFF_REGION_SIZE
        + fw::VGPU_CPU_GSP_MESSAGE_BUFF_REGION_SIZE
        + fw::VGPU_CPU_GSP_MIGRATION_BUFF_REGION_SIZE
        + fw::VGPU_CPU_GSP_ERROR_BUFF_REGION_SIZE
        + fw::VGPU_CPU_GSP_INIT_TASK_LOG_BUFF_REGION_SIZE
        + fw::VGPU_CPU_GSP_VGPU_TASK_LOG_BUFF_REGION_SIZE
        + fw::VGPU_CPU_GSP_KERNEL_TASK_LOG_BUFF_REGION_SIZE
        + fw::VGPU_CPU_GSP_GUEST_RPC_TRACE_BUFF_REGION_SIZE
        == fw::VGPU_CPU_GSP_COMMUNICATION_BUFF_TOTAL_SIZE
);
static_assert!(
    size_of::<RawControlRegion>() == u32_as_usize(fw::VGPU_CPU_GSP_CTRL_BUFF_REGION_SIZE)
);
static_assert!(
    size_of::<RawResponseRegion>() == u32_as_usize(fw::VGPU_CPU_GSP_RESPONSE_BUFF_REGION_SIZE)
);

/// Physical VRAM regions containing the vGPU plugin logs.
pub(super) struct PluginLogRegions {
    pub(super) init: VramRegion,
    pub(super) vgpu: VramRegion,
    pub(super) kernel: VramRegion,
}

fn take_region(region: &VramRegion, cursor: &mut u64, size: u32) -> Result<VramRegion> {
    let end = cursor.checked_add(u64::from(size)).ok_or(EOVERFLOW)?;
    let subregion = region.subregion(*cursor..end)?;
    *cursor = end;
    Ok(subregion)
}

/// BAR1 mapping of the plugin communication region in its management heap.
///
/// r000 layout, with byte offsets from the management heap (not to scale):
///
/// ```text
/// 0x000000 +----------------------------------------+
///          | Control (boot-ready marker)      4 KiB |
/// 0x001000 +----------------------------------------+
///          | Response                         4 KiB |
/// 0x002000 +----------------------------------------+
///          | Message                          4 KiB |
/// 0x003000 +----------------------------------------+
///          | Migration                        2 MiB |
/// 0x203000 +----------------------------------------+
///          | Error                            4 KiB |
/// 0x204000 +----------------------------------------+
///          | Init task log                  128 KiB |
/// 0x224000 +----------------------------------------+
///          | vGPU task log                  256 KiB |
/// 0x264000 +----------------------------------------+
///          | Kernel task log                 64 KiB |
/// 0x274000 +----------------------------------------+
///          | Guest RPC trace                 64 KiB |
/// 0x284000 +----------------------------------------+
/// ```
pub(super) struct CommBufferRegion<'map, 'gpu> {
    map: BarMapping<'map, 'gpu>,
    control: VramRegion,
    response: VramRegion,
    message: VramRegion,
    migration: VramRegion,
    error: VramRegion,
    init_log: VramRegion,
    vgpu_log: VramRegion,
    kernel_log: VramRegion,
    guest_trace: VramRegion,
}

impl<'map, 'gpu> CommBufferRegion<'map, 'gpu> {
    /// Map the communication portion of a plugin management heap.
    pub(super) fn new(
        bar_user: &'map BarUser<'gpu>,
        mm: &'map Mutex<GpuMm<'gpu>>,
        management_heap: &VramRegion,
    ) -> Result<Self> {
        let total_size = u64::from(fw::VGPU_CPU_GSP_COMMUNICATION_BUFF_TOTAL_SIZE);
        let region = management_heap.subregion(0..total_size)?;
        let mut cursor = 0;

        let control = take_region(&region, &mut cursor, fw::VGPU_CPU_GSP_CTRL_BUFF_REGION_SIZE)?;
        let response = take_region(
            &region,
            &mut cursor,
            fw::VGPU_CPU_GSP_RESPONSE_BUFF_REGION_SIZE,
        )?;
        let message = take_region(
            &region,
            &mut cursor,
            fw::VGPU_CPU_GSP_MESSAGE_BUFF_REGION_SIZE,
        )?;
        let migration = take_region(
            &region,
            &mut cursor,
            fw::VGPU_CPU_GSP_MIGRATION_BUFF_REGION_SIZE,
        )?;
        let error = take_region(
            &region,
            &mut cursor,
            fw::VGPU_CPU_GSP_ERROR_BUFF_REGION_SIZE,
        )?;
        let init_log = take_region(
            &region,
            &mut cursor,
            fw::VGPU_CPU_GSP_INIT_TASK_LOG_BUFF_REGION_SIZE,
        )?;
        let vgpu_log = take_region(
            &region,
            &mut cursor,
            fw::VGPU_CPU_GSP_VGPU_TASK_LOG_BUFF_REGION_SIZE,
        )?;
        let kernel_log = take_region(
            &region,
            &mut cursor,
            fw::VGPU_CPU_GSP_KERNEL_TASK_LOG_BUFF_REGION_SIZE,
        )?;
        let guest_trace = take_region(
            &region,
            &mut cursor,
            fw::VGPU_CPU_GSP_GUEST_RPC_TRACE_BUFF_REGION_SIZE,
        )?;

        let map = BarMapping::new(bar_user, mm, region, true)?;

        Ok(Self {
            map,
            control,
            response,
            message,
            migration,
            error,
            init_log,
            vgpu_log,
            kernel_log,
            guest_trace,
        })
    }

    fn region_offset(&self, region: &VramRegion) -> Result<usize> {
        let offset = region
            .address()
            .checked_sub(self.map.region().address())
            .ok_or(EINVAL)?;
        if offset.checked_add(region.size()).ok_or(EOVERFLOW)? > self.map.region().size() {
            return Err(EINVAL);
        }

        usize::try_from(offset).map_err(|_| EOVERFLOW)
    }

    fn io_offset(&self, region: &VramRegion, field: usize, width: usize) -> Result<usize> {
        let field_end = field.checked_add(width).ok_or(EOVERFLOW)?;
        if u64::try_from(field_end).map_err(|_| EOVERFLOW)? > region.size() {
            return Err(EINVAL);
        }

        self.region_offset(region)?
            .checked_add(field)
            .ok_or(EOVERFLOW)
    }

    fn read_u32(&self, region: &VramRegion, field: usize) -> Result<u32> {
        self.map
            .try_read32(self.io_offset(region, field, size_of::<u32>())?)
    }

    fn write_u8(&self, region: &VramRegion, field: usize, value: u8) -> Result {
        self.map
            .try_write8(value, self.io_offset(region, field, size_of::<u8>())?)
    }

    fn write_u32(&self, region: &VramRegion, field: usize, value: u32) -> Result {
        self.map
            .try_write32(value, self.io_offset(region, field, size_of::<u32>())?)
    }

    fn write_u64(&self, region: &VramRegion, field: usize, value: u64) -> Result {
        self.map
            .try_write64(value, self.io_offset(region, field, size_of::<u64>())?)
    }

    /// Return the physical regions occupied by the three plugin logs.
    pub(super) fn plugin_logs(&self) -> PluginLogRegions {
        PluginLogRegions {
            init: self.init_log.clone(),
            vgpu: self.vgpu_log.clone(),
            kernel: self.kernel_log.clone(),
        }
    }

    /// Clear a previous boot marker before starting the plugin.
    pub(super) fn clear_plugin_ready(&self) -> Result {
        let offset = self.io_offset(
            &self.control,
            core::mem::offset_of!(RawControlRegion, __bindgen_anon_1.message_seq_num),
            size_of::<u32>(),
        )?;
        self.map.try_write32(0, offset)?;
        // Complete the posted clear before firmware can publish its new marker.
        self.map.try_read32(offset)?;
        Ok(())
    }

    /// Return whether firmware has published the plugin boot marker.
    pub(super) fn is_plugin_ready(&self) -> Result<bool> {
        let value = self.read_u32(
            &self.control,
            core::mem::offset_of!(RawControlRegion, __bindgen_anon_1.message_seq_num),
        )?;

        Ok(value == fw::GSP_PLUGIN_BOOTLOADED)
    }

    /// Initialize the shared control and response buffers for plugin RPC.
    pub(super) fn initialize(&self) -> Result {
        self.write_u64(
            &self.control,
            core::mem::offset_of!(RawControlRegion, __bindgen_anon_1.response_buff_offset),
            u64::try_from(self.region_offset(&self.response)?)?,
        )?;
        self.write_u64(
            &self.control,
            core::mem::offset_of!(RawControlRegion, __bindgen_anon_1.message_buff_offset),
            u64::try_from(self.region_offset(&self.message)?)?,
        )?;
        self.write_u64(
            &self.control,
            core::mem::offset_of!(RawControlRegion, __bindgen_anon_1.migration_buff_offset),
            u64::try_from(self.region_offset(&self.migration)?)?,
        )?;
        self.write_u64(
            &self.control,
            core::mem::offset_of!(RawControlRegion, __bindgen_anon_1.error_buff_offset),
            u64::try_from(self.region_offset(&self.error)?)?,
        )?;
        self.write_u64(
            &self.control,
            core::mem::offset_of!(
                RawControlRegion,
                __bindgen_anon_1.guest_rpc_trace_buff_offset
            ),
            u64::try_from(self.region_offset(&self.guest_trace)?)?,
        )?;
        self.write_u32(
            &self.control,
            core::mem::offset_of!(
                RawControlRegion,
                __bindgen_anon_1.migration_buf_cpu_access_offset
            ),
            0,
        )?;
        self.write_u8(
            &self.control,
            core::mem::offset_of!(RawControlRegion, __bindgen_anon_1.is_migration_in_progress),
            0,
        )?;
        self.write_u32(
            &self.control,
            core::mem::offset_of!(RawControlRegion, __bindgen_anon_1.error_buff_cpu_get_idx),
            0,
        )?;
        self.write_u32(
            &self.control,
            core::mem::offset_of!(
                RawControlRegion,
                __bindgen_anon_1.guest_rpc_trace_buff_cpu_get_idx
            ),
            0,
        )?;
        self.write_u32(
            &self.control,
            core::mem::offset_of!(RawControlRegion, __bindgen_anon_1.attached_vgpu_count),
            1,
        )?;
        self.write_u8(
            &self.control,
            core::mem::offset_of!(RawControlRegion, __bindgen_anon_1.is_gr_init_done),
            0,
        )?;

        self.write_u32(
            &self.control,
            core::mem::offset_of!(RawControlRegion, __bindgen_anon_1.message_type),
            0,
        )?;
        // Replace the boot-ready marker with the initial RPC sequence.
        self.write_u32(
            &self.control,
            core::mem::offset_of!(RawControlRegion, __bindgen_anon_1.message_seq_num),
            0,
        )?;
        self.write_u32(
            &self.response,
            core::mem::offset_of!(
                RawResponseRegion,
                __bindgen_anon_1.message_seq_num_processed
            ),
            0,
        )?;
        self.write_u32(
            &self.response,
            core::mem::offset_of!(RawResponseRegion, __bindgen_anon_1.result_code),
            0,
        )?;
        self.write_u32(
            &self.control,
            core::mem::offset_of!(RawControlRegion, __bindgen_anon_1.version),
            fw::VGPU_CPU_GSP_CTRL_BUFF_VERSION,
        )
    }

    /// Copy and publish one RPC request to firmware.
    pub(super) fn submit(&self, message: RpcMessage, sequence: u32, data: &[u8]) -> Result {
        if u64::try_from(data.len()).map_err(|_| EOVERFLOW)? > self.message.size() {
            return Err(E2BIG);
        }

        for (index, chunk) in data.chunks(size_of::<u32>()).enumerate() {
            let mut bytes = [0u8; size_of::<u32>()];
            bytes[..chunk.len()].copy_from_slice(chunk);
            let field = index.checked_mul(size_of::<u32>()).ok_or(EOVERFLOW)?;
            self.write_u32(&self.message, field, u32::from_le_bytes(bytes))?;
        }

        self.write_u32(
            &self.control,
            core::mem::offset_of!(RawControlRegion, __bindgen_anon_1.message_type),
            // CAST: `RpcMessage` has a `u32` representation.
            message as u32,
        )?;
        self.write_u32(
            &self.control,
            core::mem::offset_of!(RawControlRegion, __bindgen_anon_1.message_seq_num),
            sequence,
        )
    }

    /// Read firmware's response for an expected RPC sequence.
    pub(super) fn response(&self, expected_sequence: u32) -> Result<RpcResponse> {
        let sequence = self.read_u32(
            &self.response,
            core::mem::offset_of!(
                RawResponseRegion,
                __bindgen_anon_1.message_seq_num_processed
            ),
        )?;
        if sequence != expected_sequence {
            return Ok(RpcResponse::Pending { sequence });
        }

        let status = self.read_u32(
            &self.response,
            core::mem::offset_of!(RawResponseRegion, __bindgen_anon_1.result_code),
        )?;
        Ok(RpcResponse::Complete { status })
    }

    /// Invalidate the PTEs and release the communication mapping.
    pub(super) fn unmap(&mut self) -> Result {
        self.map.unmap()
    }
}
