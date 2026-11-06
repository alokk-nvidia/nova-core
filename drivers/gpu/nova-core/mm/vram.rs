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
    num::casts::FromSafeCast,
    prelude::*,
    ptr::Alignment,
    sync::Arc, //
};

use super::{
    GpuMm,
    PAGE_SIZE, //
};

/// A physically contiguous VRAM allocation.
///
/// # Invariants
///
/// `range.start <= range.end`.
pub(crate) struct VramBlock {
    _blocks: Pin<KBox<AllocatedBlocks>>,
    range: Range<u64>,
}

impl VramBlock {
    /// Returns the range covered by this VRAM block.
    pub(crate) const fn range(&self) -> &Range<u64> {
        &self.range
    }
}

impl GpuMm<'_> {
    /// Allocates an exact byte range relative to the buddy allocator's base.
    pub(crate) fn alloc_vram_range(
        &self,
        range: Range<u64>,
        align: Alignment,
    ) -> Result<Arc<VramBlock>> {
        const PAGE_SIZE_ALIGN: Alignment = Alignment::new::<PAGE_SIZE>();

        let align = if align > PAGE_SIZE_ALIGN {
            align
        } else {
            PAGE_SIZE_ALIGN
        };
        let size = range.end.checked_sub(range.start).ok_or(EINVAL)?;
        let buddy = self.buddy();
        let base = buddy.base_offset();
        let start = base.checked_add(range.start).ok_or(EOVERFLOW)?;
        let end = base.checked_add(range.end).ok_or(EOVERFLOW)?;
        if !start.is_multiple_of(u64::from_safe_cast(align.as_usize())) {
            return Err(EINVAL);
        }

        let blocks = KBox::pin_init(
            buddy.alloc_blocks(
                GpuBuddyAllocMode::Range(range),
                size,
                align,
                GpuBuddyAllocFlags::default(),
            ),
            GFP_KERNEL,
        )?;

        Ok(Arc::new(
            VramBlock {
                _blocks: blocks,
                // INVARIANT: `range.start <= range.end` was checked above, hence
                // `address <= end`.
                range: start..end,
            },
            GFP_KERNEL,
        )?)
    }
}
