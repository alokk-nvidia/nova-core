// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

//! VRAM slot allocation for vGPU instances.

use kernel::{
    prelude::*,
    sizes::SizeConstants,
    sync::Arc, //
};

use crate::mm::{
    vram::{
        VramBlock,
        VramRegion, //
    },
    GpuMm, //
};

const VRAM_SLOT_MIN_ALIGN: u64 = u64::SZ_4K;
// `VRAM_SLOT_MIN_ALIGN` is used for alignment.
static_assert!(VRAM_SLOT_MIN_ALIGN.is_power_of_two());

/// Describes the VRAM layout used by vGPU instances.
#[derive(Clone, Copy, PartialEq)]
pub(super) struct VgpuVramLayout {
    // Firmware ID of the vGPU type this layout was built for.
    type_id: u32,
    // Maximum number of slots with this layout.
    max_slots: u32,
    // Size of the framebuffer for a vGPU instance.
    fb_size: u64,
    // Size of heap for a vGPU instance.
    heap_size: u64,
    // Alignment of the framebuffer for vGPU instances.
    fb_align: u64,
}

#[expect(dead_code)]
impl VgpuVramLayout {
    pub(super) fn new(
        type_id: u32,
        max_slots: u32,
        fb_size: u64,
        heap_size: u64,
        fb_align: u64,
    ) -> Result<Self> {
        if max_slots == 0 || fb_size == 0 || heap_size == 0 {
            return Err(EINVAL);
        }

        let fb_align = core::cmp::max(fb_align, VRAM_SLOT_MIN_ALIGN);

        if !fb_align.is_power_of_two()
            || !fb_size.is_multiple_of(fb_align)
            || !heap_size.is_multiple_of(VRAM_SLOT_MIN_ALIGN)
        {
            return Err(EINVAL);
        }

        Ok(Self {
            type_id,
            max_slots,
            fb_size,
            heap_size,
            fb_align,
        })
    }

    pub(super) fn max_slots(&self) -> u32 {
        self.max_slots
    }
}

/// A description of a VRAM slot.
///
/// All device accesses and mappings of its regions must end before dropping the slot.
#[must_use]
pub(super) struct VgpuVramSlot {
    /// Framebuffer VRAM region.
    pub(super) fbmem: VramRegion,
    /// Management heap VRAM region.
    pub(super) mgmt_heap: VramRegion,
    /// Index of this slot.
    pub(super) index: u32,
}

/// A pool of VRAM slots, split according to a [`VgpuVramLayout`].
pub(super) struct VgpuVramSlotPool {
    backing: Arc<VramBlock>,
    layout: VgpuVramLayout,
    fb_region_size: u64,
}

impl VgpuVramSlotPool {
    /// Reserve the VRAM for all possible slots of `layout`, and returns a [`VgpuVramSlotPool`] that
    /// can provide these slots.
    pub(super) fn new(mm: &GpuMm<'_>, layout: VgpuVramLayout) -> Result<Self> {
        let max_slots = u64::from(layout.max_slots);
        // Keep guest VRAM slots contiguous so each starts at `fb_align`;
        // interleaving page-aligned heaps could misalign later slots.
        let fb_region_size = layout.fb_size.checked_mul(max_slots).ok_or(EINVAL)?;
        let heap_region_size = layout.heap_size.checked_mul(max_slots).ok_or(EINVAL)?;
        let pool_size = fb_region_size.checked_add(heap_region_size).ok_or(EINVAL)?;

        let backing = mm.alloc_vram_range(0..pool_size, cv!(VRAM_SLOT_MIN_ALIGN))?;
        if !backing.range().start.is_multiple_of(layout.fb_align) {
            return Err(EINVAL);
        }

        Ok(Self {
            backing,
            layout,
            fb_region_size,
        })
    }

    /// Returns `true` if `layout` is identical to the layout used by this pool.
    pub(super) fn matches_layout(&self, layout: VgpuVramLayout) -> bool {
        self.layout == layout
    }

    /// Returns the VRAM slot corresponding to `index`, or `EINVAL` if `index` is not a valid slot
    /// index for the pool's layout.
    pub(super) fn slot(&self, index: u32) -> Result<VgpuVramSlot> {
        if index >= self.layout.max_slots {
            return Err(EINVAL);
        }

        let fb_size = self.layout.fb_size;
        let fb_offset = fb_size.checked_mul(index.into()).ok_or(EINVAL)?;
        let fb_end = fb_offset.checked_add(fb_size).ok_or(EINVAL)?;

        let heap_size = self.layout.heap_size;
        let heap_slot_offset = heap_size.checked_mul(index.into()).ok_or(EINVAL)?;
        let heap_offset = self
            .fb_region_size
            .checked_add(heap_slot_offset)
            .ok_or(EINVAL)?;
        let heap_end = heap_offset.checked_add(heap_size).ok_or(EINVAL)?;

        let fbmem = self.backing.region(fb_offset..fb_end)?;
        let mgmt_heap = self.backing.region(heap_offset..heap_end)?;

        Ok(VgpuVramSlot {
            fbmem,
            mgmt_heap,
            index,
        })
    }
}
