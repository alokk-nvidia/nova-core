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
    RawControlRegion, //
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
    init_log: VramRegion,
    vgpu_log: VramRegion,
    kernel_log: VramRegion,
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
        take_region(
            &region,
            &mut cursor,
            fw::VGPU_CPU_GSP_RESPONSE_BUFF_REGION_SIZE,
        )?;
        take_region(
            &region,
            &mut cursor,
            fw::VGPU_CPU_GSP_MESSAGE_BUFF_REGION_SIZE,
        )?;
        take_region(
            &region,
            &mut cursor,
            fw::VGPU_CPU_GSP_MIGRATION_BUFF_REGION_SIZE,
        )?;
        take_region(
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
        take_region(
            &region,
            &mut cursor,
            fw::VGPU_CPU_GSP_GUEST_RPC_TRACE_BUFF_REGION_SIZE,
        )?;

        let map = BarMapping::new(bar_user, mm, region, true)?;

        Ok(Self {
            map,
            control,
            init_log,
            vgpu_log,
            kernel_log,
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

    /// Invalidate the PTEs and release the communication mapping.
    #[expect(dead_code)]
    pub(super) fn unmap(&mut self) -> Result {
        self.map.unmap()
    }
}
