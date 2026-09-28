// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

//! VRAM allocation.

use core::ops::Range;

use kernel::{
    gpu::buddy::{
        AllocatedBlocks,
        GpuBuddyAllocFlags,
        GpuBuddyAllocMode, //
    },
    prelude::*,
    ptr::Alignment,
    sync::Arc, //
};

use crate::num::IntoSafeCast;

use super::{
    GpuMm,
    PAGE_SIZE, //
};

/// A physically contiguous VRAM allocation.
pub(crate) struct VramBlock {
    _blocks: Pin<KBox<AllocatedBlocks>>,
    address: u64,
    size: u64,
}

impl VramBlock {
    pub(crate) const fn address(&self) -> u64 {
        self.address
    }

    /// Creates a byte range relative to this allocation, retaining its backing storage.
    pub(crate) fn region(self: &Arc<Self>, range: Range<u64>) -> Result<VramRegion> {
        VramRegion::new(self.clone(), range)
    }
}

/// A byte range that keeps its VRAM allocation alive.
#[derive(Clone)]
pub(crate) struct VramRegion {
    backing: Arc<VramBlock>,
    address: u64,
    size: u64,
}

impl VramRegion {
    fn new(backing: Arc<VramBlock>, range: Range<u64>) -> Result<Self> {
        if range.start >= range.end || range.end > backing.size {
            return Err(EINVAL);
        }

        let size = range.end - range.start;
        let address = backing.address.checked_add(range.start).ok_or(EOVERFLOW)?;
        backing.address.checked_add(range.end).ok_or(EOVERFLOW)?;

        Ok(Self {
            backing,
            address,
            size,
        })
    }

    pub(crate) const fn address(&self) -> u64 {
        self.address
    }

    pub(crate) const fn size(&self) -> u64 {
        self.size
    }

    /// Creates a nonempty subregion relative to this view's start.
    pub(crate) fn subregion(&self, range: Range<u64>) -> Result<Self> {
        if range.start >= range.end || range.end > self.size {
            return Err(EINVAL);
        }

        let size = range.end - range.start;
        let address = self.address.checked_add(range.start).ok_or(EOVERFLOW)?;
        address.checked_add(size).ok_or(EOVERFLOW)?;

        Ok(Self {
            backing: self.backing.clone(),
            address,
            size,
        })
    }
}

impl GpuMm<'_> {
    /// Allocates an exact byte range relative to the buddy allocator's base.
    pub(crate) fn alloc_vram_range(&self, range: Range<u64>, align: u64) -> Result<Arc<VramBlock>> {
        let size = range.end.checked_sub(range.start).ok_or(EINVAL)?;

        let align = align.max(PAGE_SIZE.into_safe_cast());
        let min_block_size = usize::try_from(align).map_err(|_| EOVERFLOW)?;
        let min_block_size = Alignment::new_checked(min_block_size).ok_or(EINVAL)?;

        let buddy = self.buddy();
        let base = buddy.base_offset();
        let address = base.checked_add(range.start).ok_or(EOVERFLOW)?;
        base.checked_add(range.end).ok_or(EOVERFLOW)?;
        if !address.is_multiple_of(align) {
            return Err(EINVAL);
        }

        let blocks = KBox::pin_init(
            buddy.alloc_blocks(
                GpuBuddyAllocMode::Range(range),
                size,
                min_block_size,
                GpuBuddyAllocFlags::default(),
            ),
            GFP_KERNEL,
        )?;

        Ok(Arc::new(
            VramBlock {
                _blocks: blocks,
                address,
                size,
            },
            GFP_KERNEL,
        )?)
    }
}
