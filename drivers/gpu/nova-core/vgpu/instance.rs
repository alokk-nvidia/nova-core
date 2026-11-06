// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

use core::num::{
    NonZero,
    NonZeroUsize, //
};

use kernel::{
    prelude::*,
    ptr::Alignment,
    sizes::SizeConstants, //
};

use crate::{
    gpu::ChannelIdReservation,
    mm::GpuMm, //
    num::FromSafeCast,
};

use super::{
    fw::commands::VgpuProperties,
    vram::{
        VgpuVramLayout,
        VgpuVramSlot,
        VgpuVramSlotPool, //
    },
    Dbdf,
    VgpuManagerInner, //
};

/// Guest Function ID validated against one device's total number of VFs.
///
/// GFID 0 is reserved for the PF; VFs start at 1. Keeping the nonzero `u16`
/// preserves the PCI SR-IOV range when forming a plugin doorbell handle.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Gfid(NonZero<u16>);

impl Gfid {
    /// Validates an external GFID for a device supporting `total_vfs` VFs.
    #[expect(dead_code)]
    pub(super) fn new(gfid: u32, total_vfs: NonZero<u16>) -> Result<Self> {
        let gfid = u16::try_from(gfid).map_err(|_| EINVAL)?;
        let gfid = NonZero::new(gfid).ok_or(EINVAL)?;

        if gfid > total_vfs {
            Err(EINVAL)
        } else {
            Ok(Self(gfid))
        }
    }

    #[expect(dead_code)]
    pub(super) const fn get(self) -> u16 {
        self.0.get()
    }
}

/// Resource requirements for one vGPU type.
struct VgpuType {
    vgpu_type_id: u32,
    max_instance: u32,
    fb_length: u64,
    gsp_heap_size: u64,
}

impl From<&VgpuProperties> for VgpuType {
    fn from(properties: &VgpuProperties) -> Self {
        Self {
            vgpu_type_id: properties.type_id,
            max_instance: properties.max_instance,
            fb_length: properties.fb_length,
            gsp_heap_size: properties.gsp_heap_size,
        }
    }
}

/// A vGPU instance and the resources reserved for it.
#[expect(dead_code)]
struct VgpuInstance<'gpu> {
    gfid: Gfid,
    dbdf: Dbdf,
    vgpu_type: VgpuType,
    vm_pid: u32,
    chids: ChannelIdReservation<'gpu>,
    vram_slot: VgpuVramSlot,
}

/// Identity and firmware profile used to allocate an instance.
pub(super) struct InstanceInfo {
    gfid: Gfid,
    dbdf: Dbdf,
    vgpu_type: VgpuType,
    vm_pid: u32,
}

#[expect(dead_code)]
impl InstanceInfo {
    pub(super) fn new(gfid: Gfid, dbdf: Dbdf, properties: &VgpuProperties, vm_pid: u32) -> Self {
        Self {
            gfid,
            dbdf,
            vgpu_type: VgpuType::from(properties),
            vm_pid,
        }
    }
}

/// Registry of live vGPU instances.
pub(super) struct VgpuInstances<'gpu> {
    instances: KVec<VgpuInstance<'gpu>>,
    vram_slots: Option<VgpuVramSlotPool>,
}

#[expect(dead_code)]
impl<'gpu> VgpuInstances<'gpu> {
    pub(super) const fn new() -> Self {
        Self {
            instances: KVec::new(),
            vram_slots: None,
        }
    }

    fn alloc_vram_slot(&mut self, mm: &GpuMm<'_>, layout: VgpuVramLayout) -> Result<VgpuVramSlot> {
        let allocator = match self.vram_slots.as_mut() {
            // Don't replace the allocator if the current one already matches the requested layout.
            Some(allocator) if allocator.matches_layout(layout) => allocator,
            // The allocator cannot be replaced if there already are instances.
            Some(_) if !self.instances.is_empty() => return Err(EBUSY),
            _ => {
                // Release the previous pool's VRAM so the new one can reuse it.
                self.vram_slots = None;
                self.vram_slots.insert(VgpuVramSlotPool::new(mm, layout)?)
            }
        };

        // Find the lowest slot index not used by any instance.
        let index = (0..layout.max_slots())
            .find(|i| !self.instances.iter().any(|inst| inst.vram_slot.index == *i))
            .ok_or(ENOSPC)?;

        allocator.slot(index)
    }

    /// Creates and returns a [`VgpuInstance`].
    fn allocate_instance(
        &mut self,
        vgpu: &VgpuManagerInner<'gpu>,
        info: InstanceInfo,
    ) -> Result<VgpuInstance<'gpu>> {
        let InstanceInfo {
            gfid,
            dbdf,
            vgpu_type,
            vm_pid,
        } = info;

        let instance_exists = self
            .instances
            .iter()
            .any(|instance| instance.gfid == gfid || instance.dbdf == dbdf);
        if instance_exists {
            return Err(EEXIST);
        }

        let type_id = vgpu_type.vgpu_type_id;
        let num_type_instances = self
            .instances
            .iter()
            .filter(|instance| instance.vgpu_type.vgpu_type_id == type_id)
            .count();
        if num_type_instances >= usize::from_safe_cast(vgpu_type.max_instance) {
            return Err(ENOSPC);
        }

        let channels_per_instance = vgpu
            .total_channels
            .checked_div(vgpu_type.max_instance)
            .map(usize::from_safe_cast)
            .and_then(NonZeroUsize::new)
            .ok_or(EINVAL)?;
        let chids = vgpu
            .chid_pool
            .reserve_ids(channels_per_instance, Alignment::SZ_1)?;

        let vram_layout = VgpuVramLayout::new(
            type_id,
            vgpu_type.max_instance,
            vgpu_type.fb_length,
            vgpu_type.gsp_heap_size,
            vgpu.vmmu_segment_size,
        )?;

        let vram_slot = vgpu
            .mm
            .with_locked(|mm| self.alloc_vram_slot(mm, vram_layout))?;

        Ok(VgpuInstance {
            gfid,
            dbdf,
            vgpu_type,
            vm_pid,
            chids,
            vram_slot,
        })
    }

    /// Remove an instance and release its channel and VRAM reservations.
    fn destroy_instance(&mut self, gfid: Gfid) -> Result {
        let instance_index = self
            .instances
            .iter()
            .position(|instance| instance.gfid == gfid)
            .ok_or(ENOENT)?;
        let instance = self.instances.remove(instance_index).map_err(|_| EIO)?;
        drop(instance);
        Ok(())
    }
}

impl VgpuManagerInner<'_> {
    /// Allocate, register, and activate a vGPU instance.
    ///
    /// Keep the registry locked through creation or rollback so duplicate checks and
    /// type limits remain stable. MM and BAR-user locks are acquired only within
    /// individual memory operations, in that order.
    #[expect(dead_code)]
    fn create_instance(&self, info: InstanceInfo) -> Result {
        let mut instances = self.instances.lock();

        // Reserve capacity before acquiring resources so registration cannot allocate.
        instances.instances.reserve(1, GFP_KERNEL)?;

        let instance = instances.allocate_instance(self, info)?;

        // Register live instances, and failed ones whose resources firmware may still own.
        instances
            .instances
            .push_within_capacity(instance)
            .map_err(|_| EIO)?;

        Ok(())
    }
}
