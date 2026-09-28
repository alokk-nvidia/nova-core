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
};

use super::{
    commands::Dbdf,
    vram::{
        VgpuVramLayout,
        VgpuVramSlot,
        VgpuVramSlotAllocator, //
    },
    VgpuManager, //
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

/// Resource requirements and device identity for one vGPU type.
#[expect(dead_code)]
pub(super) struct VgpuType {
    vgpu_type_id: u32,
    bar1_length: u64,
    max_instance: u32,
    pci_dev_id: u32,
    pci_subsys_id: u32,
    fb_length: u64,
    gsp_heap_size: u64,
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
    pub(super) const fn new(gfid: Gfid, dbdf: Dbdf, vgpu_type: VgpuType, vm_pid: u32) -> Self {
        Self {
            gfid,
            dbdf,
            vgpu_type,
            vm_pid,
        }
    }
}

/// Registry of live vGPU instances.
pub(super) struct VgpuInstances<'gpu> {
    instances: KVec<VgpuInstance<'gpu>>,
    vram_slots: Option<VgpuVramSlotAllocator>,
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
        let replace_empty_pool = match self.vram_slots.as_ref() {
            Some(allocator) if allocator.is_empty() => !allocator.matches_layout(layout)?,
            _ => false,
        };
        if replace_empty_pool {
            self.vram_slots = None;
        }

        if let Some(allocator) = self.vram_slots.as_mut() {
            return allocator.alloc(layout);
        }

        let mut allocator = VgpuVramSlotAllocator::new(mm, layout)?;
        let slot = allocator.alloc(layout)?;
        self.vram_slots = Some(allocator);
        Ok(slot)
    }

    /// Allocate resources and register a new inactive vGPU instance.
    fn allocate_instance(
        &mut self,
        mm: &GpuMm<'_>,
        vgpu: &VgpuManager<'gpu>,
        info: InstanceInfo,
    ) -> Result<Gfid> {
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
        let max_instances = usize::try_from(vgpu_type.max_instance).map_err(|_| EOVERFLOW)?;
        if max_instances == 0 || num_type_instances >= max_instances {
            return Err(ENOSPC);
        }

        // Reserve registry capacity before acquiring resources so publishing
        // the completed instance cannot fail due to memory pressure.
        self.instances.reserve(1, GFP_KERNEL)?;

        let channels_per_instance = vgpu
            .total_channels
            .checked_div(vgpu_type.max_instance)
            .ok_or(EINVAL)?;
        let channels_per_instance =
            usize::try_from(channels_per_instance).map_err(|_| EOVERFLOW)?;
        let channels_per_instance = NonZeroUsize::new(channels_per_instance).ok_or(EINVAL)?;
        let chids = vgpu
            .chid_pool
            .reserve_ids(channels_per_instance, Alignment::SZ_1)?;

        let vram_layout = VgpuVramLayout {
            type_id,
            max_slots: vgpu_type.max_instance,
            fb_size: vgpu_type.fb_length,
            heap_size: vgpu_type.gsp_heap_size,
            fb_align: vgpu.vmmu_segment_size,
        };
        let vram_slot = self.alloc_vram_slot(mm, vram_layout)?;

        let instance = VgpuInstance {
            gfid,
            dbdf,
            vgpu_type,
            vm_pid,
            chids,
            vram_slot,
        };
        self.instances
            .push_within_capacity(instance)
            .map_err(|_| EIO)?;

        Ok(gfid)
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
