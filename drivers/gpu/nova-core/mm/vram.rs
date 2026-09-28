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
