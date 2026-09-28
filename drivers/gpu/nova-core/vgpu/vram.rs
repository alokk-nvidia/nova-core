// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

//! VRAM slot allocation for vGPU instances.

use kernel::{
    bitmap::BitmapVec,
    prelude::*,
    sync::{
        new_mutex,
        Arc,
        Mutex, //
    }, //
};

use crate::mm::{
    vram::{
        VramBlock,
        VramRegion, //
    },
    GpuMm, //
};

const VRAM_SLOT_MIN_ALIGN: u64 = 4096;

#[derive(Clone, Copy, PartialEq)]
pub(super) struct VgpuVramLayout {
    pub(super) type_id: u32,
    pub(super) max_slots: u32,
    pub(super) fb_size: u64,
    pub(super) heap_size: u64,
    pub(super) fb_align: u64,
}

impl VgpuVramLayout {
    fn validated(mut self) -> Result<Self> {
        if self.max_slots == 0 || self.fb_size == 0 || self.heap_size == 0 {
            return Err(EINVAL);
        }
        self.fb_align = core::cmp::max(self.fb_align, VRAM_SLOT_MIN_ALIGN);
        if !self.fb_align.is_power_of_two() {
            return Err(EINVAL);
        }
        if self.fb_size & (self.fb_align - 1) != 0
            || self.heap_size & (VRAM_SLOT_MIN_ALIGN - 1) != 0
        {
            return Err(EINVAL);
        }
        Ok(self)
    }
}

/// Slot bitmap shared by the allocator and its outstanding slots.
#[pin_data]
struct SlotBitmap {
    #[pin]
    used: Mutex<BitmapVec>,
}

impl SlotBitmap {
    fn new(count: usize) -> Result<Arc<Self>> {
        let used = BitmapVec::new(count, GFP_KERNEL)?;
        Arc::pin_init(
            pin_init!(Self {
                used <- new_mutex!(used),
            }),
            GFP_KERNEL,
        )
    }

    fn alloc(self: &Arc<Self>) -> Result<Slot> {
        let mut used = self.used.lock();
        let index = used.next_zero_bit(0).ok_or(ENOSPC)?;
        used.set_bit(index);
        Ok(Slot {
            bitmap: self.clone(),
            index,
        })
    }

    fn is_empty(&self) -> bool {
        self.used.lock().last_bit().is_none()
    }
}

/// A slot that clears its bitmap entry on drop.
struct Slot {
    bitmap: Arc<SlotBitmap>,
    index: usize,
}

impl Drop for Slot {
    fn drop(&mut self) {
        let mut used = self.bitmap.used.lock();
        debug_assert_eq!(used.next_bit(self.index), Some(self.index));
        used.clear_bit(self.index);
    }
}

/// A VRAM slot that clears its bitmap entry when dropped.
///
/// All device accesses and mappings of its regions must end before dropping the slot.
/// Clearing the entry locks a sleeping mutex, so dropping the slot may sleep.
#[must_use]
pub(super) struct VgpuVramSlot {
    pub(super) fbmem: VramRegion,
    pub(super) mgmt_heap: VramRegion,
    _slot: Slot,
}

pub(super) struct VgpuVramSlotAllocator {
    backing: Arc<VramBlock>,
    layout: VgpuVramLayout,
    fb_region_size: u64,
    slot_bitmap: Arc<SlotBitmap>,
}

impl VgpuVramSlotAllocator {
    pub(super) fn new(mm: &GpuMm<'_>, layout: VgpuVramLayout) -> Result<Self> {
        let layout = layout.validated()?;
        let max_slots = u64::from(layout.max_slots);
        // Keep guest VRAM slots contiguous so each starts at `fb_align`;
        // interleaving page-aligned heaps could misalign later slots.
        let fb_region_size = layout.fb_size.checked_mul(max_slots).ok_or(EINVAL)?;
        let heap_region_size = layout.heap_size.checked_mul(max_slots).ok_or(EINVAL)?;
        let pool_size = fb_region_size.checked_add(heap_region_size).ok_or(EINVAL)?;

        let slot_bitmap = SlotBitmap::new(usize::try_from(layout.max_slots).map_err(|_| EINVAL)?)?;
        let backing = mm.alloc_vram_range(0..pool_size, VRAM_SLOT_MIN_ALIGN)?;
        if !backing.address().is_multiple_of(layout.fb_align) {
            return Err(EINVAL);
        }

        Ok(Self {
            backing,
            layout,
            fb_region_size,
            slot_bitmap,
        })
    }

    pub(super) fn matches_layout(&self, layout: VgpuVramLayout) -> Result<bool> {
        Ok(self.layout == layout.validated()?)
    }

    pub(super) fn alloc(&mut self, layout: VgpuVramLayout) -> Result<VgpuVramSlot> {
        if !self.matches_layout(layout)? {
            return Err(EBUSY);
        }

        let slot = self.slot_bitmap.alloc()?;
        let index = u64::try_from(slot.index).map_err(|_| EINVAL)?;

        let fb_size = self.layout.fb_size;
        let fb_offset = fb_size.checked_mul(index).ok_or(EINVAL)?;
        let fb_end = fb_offset.checked_add(fb_size).ok_or(EINVAL)?;

        let heap_size = self.layout.heap_size;
        let heap_slot_offset = heap_size.checked_mul(index).ok_or(EINVAL)?;
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
            _slot: slot,
        })
    }

    pub(super) fn is_empty(&self) -> bool {
        self.slot_bitmap.is_empty()
    }
}
