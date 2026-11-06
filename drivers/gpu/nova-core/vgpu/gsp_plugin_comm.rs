// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

//! GSP plugin communication buffer mappings and access.

use core::mem::offset_of;

use kernel::{
    io::{
        io_project,
        io_read,
        io_write,
        Io,
        Mmio, //
    },
    num::casts::FromSafeCast,
    prelude::*, //
};

use crate::mm::{
    bar_user::{
        BarMapping,
        BarUser, //
    },
    vram::VramRegion,
    SharedGpuMm, //
};

use super::fw::{
    self,
    RawControlRegion,
    RawResponseRegion, //
};

/// Layout of the plugin communication region, located at the start of the management heap.
///
/// This type is never instantiated; it only describes the layout of the region for I/O
/// projections.
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
#[repr(C)]
struct CommBuffer {
    control: RawControlRegion,
    response: RawResponseRegion,
    message: [u8; cv!(fw::VGPU_CPU_GSP_MESSAGE_BUFF_REGION_SIZE)],
    migration: [u8; cv!(fw::VGPU_CPU_GSP_MIGRATION_BUFF_REGION_SIZE)],
    error: [u8; cv!(fw::VGPU_CPU_GSP_ERROR_BUFF_REGION_SIZE)],
    init_log: [u8; cv!(fw::VGPU_CPU_GSP_INIT_TASK_LOG_BUFF_REGION_SIZE)],
    vgpu_log: [u8; cv!(fw::VGPU_CPU_GSP_VGPU_TASK_LOG_BUFF_REGION_SIZE)],
    kernel_log: [u8; cv!(fw::VGPU_CPU_GSP_KERNEL_TASK_LOG_BUFF_REGION_SIZE)],
    guest_trace: [u8; cv!(fw::VGPU_CPU_GSP_GUEST_RPC_TRACE_BUFF_REGION_SIZE)],
}

static_assert!(size_of::<CommBuffer>() == cv!(fw::VGPU_CPU_GSP_COMMUNICATION_BUFF_TOTAL_SIZE));
static_assert!(size_of::<RawControlRegion>() == cv!(fw::VGPU_CPU_GSP_CTRL_BUFF_REGION_SIZE));
static_assert!(size_of::<RawResponseRegion>() == cv!(fw::VGPU_CPU_GSP_RESPONSE_BUFF_REGION_SIZE));

/// Physical VRAM regions containing the vGPU plugin logs.
#[expect(dead_code)]
pub(super) struct PluginLogRegions {
    pub(super) init: VramRegion,
    pub(super) vgpu: VramRegion,
    pub(super) kernel: VramRegion,
}

/// BAR1 mapping of the plugin communication region in its management heap.
///
/// See [`CommBuffer`] for the layout of the region.
pub(super) struct CommBufferRegion<'map, 'gpu> {
    map: BarMapping<'map, 'gpu>,
}

#[expect(dead_code)]
impl<'map, 'gpu> CommBufferRegion<'map, 'gpu> {
    /// Map the communication portion of a plugin management heap.
    pub(super) fn new(
        bar_user: &'map BarUser<'gpu>,
        mm: &'map SharedGpuMm<'gpu>,
        management_heap: &VramRegion,
    ) -> Result<Self> {
        let region = management_heap.subregion(0..u64::from_safe_cast(size_of::<CommBuffer>()))?;

        Ok(Self {
            map: BarMapping::new(bar_user, mm, region, true)?,
        })
    }

    /// Returns the MMIO view of the communication region.
    ///
    /// Fails if the region has been unmapped.
    fn view(&self) -> Result<Mmio<'_, CommBuffer>> {
        self.map.view()
    }

    /// Return the physical regions occupied by the three plugin logs.
    pub(super) fn plugin_logs(&self) -> Result<PluginLogRegions> {
        let log_region = |offset, size| {
            let start = u64::from_safe_cast(offset);

            self.map.region().subregion(start..start + u64::from(size))
        };

        Ok(PluginLogRegions {
            init: log_region(
                offset_of!(CommBuffer, init_log),
                fw::VGPU_CPU_GSP_INIT_TASK_LOG_BUFF_REGION_SIZE,
            )?,
            vgpu: log_region(
                offset_of!(CommBuffer, vgpu_log),
                fw::VGPU_CPU_GSP_VGPU_TASK_LOG_BUFF_REGION_SIZE,
            )?,
            kernel: log_region(
                offset_of!(CommBuffer, kernel_log),
                fw::VGPU_CPU_GSP_KERNEL_TASK_LOG_BUFF_REGION_SIZE,
            )?,
        })
    }

    /// Clear a previous boot marker before starting the plugin.
    pub(super) fn clear_plugin_ready(&self) -> Result {
        let seq_num = io_project!(self.view()?, .control.__bindgen_anon_1.message_seq_num);

        seq_num.write_val(0);
        // Complete the posted clear before firmware can publish its new marker.
        seq_num.read_val();

        Ok(())
    }

    /// Return whether firmware has published the plugin boot marker.
    pub(super) fn is_plugin_ready(&self) -> Result<bool> {
        let value = io_read!(self.view()?, .control.__bindgen_anon_1.message_seq_num);

        Ok(value == fw::GSP_PLUGIN_BOOTLOADED)
    }

    /// Initialize the shared control and response buffers for plugin RPC.
    #[expect(dead_code)]
    pub(super) fn initialize(&self) -> Result {
        let comm = self.view()?;
        let control = io_project!(comm, .control.__bindgen_anon_1);
        let response = io_project!(comm, .response.__bindgen_anon_1);

        io_write!(
            control,
            .response_buff_offset,
            u64::from_safe_cast(offset_of!(CommBuffer, response))
        );
        io_write!(
            control,
            .message_buff_offset,
            u64::from_safe_cast(offset_of!(CommBuffer, message))
        );
        io_write!(
            control,
            .migration_buff_offset,
            u64::from_safe_cast(offset_of!(CommBuffer, migration))
        );
        io_write!(
            control,
            .error_buff_offset,
            u64::from_safe_cast(offset_of!(CommBuffer, error))
        );
        io_write!(
            control,
            .guest_rpc_trace_buff_offset,
            u64::from_safe_cast(offset_of!(CommBuffer, guest_trace))
        );
        io_write!(control, .migration_buf_cpu_access_offset, 0);
        io_write!(control, .is_migration_in_progress, 0);
        io_write!(control, .error_buff_cpu_get_idx, 0);
        io_write!(control, .guest_rpc_trace_buff_cpu_get_idx, 0);
        io_write!(control, .attached_vgpu_count, 1);
        io_write!(control, .is_gr_init_done, 0);

        io_write!(control, .message_type, 0);
        // Replace the boot-ready marker with the initial RPC sequence.
        io_write!(control, .message_seq_num, 0);
        io_write!(response, .message_seq_num_processed, 0);
        io_write!(response, .result_code, 0);
        io_write!(control, .version, fw::VGPU_CPU_GSP_CTRL_BUFF_VERSION);

        Ok(())
    }

    /// Invalidate the PTEs and release the communication mapping.
    pub(super) fn unmap(&mut self) -> Result {
        self.map.unmap()
    }
}
