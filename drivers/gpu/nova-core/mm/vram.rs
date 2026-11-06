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

/// Rebases the nonempty range `rel`, relative to the start of `base`, onto `base`.
///
/// Returns `EINVAL` if `rel` is empty or does not fit within `base`.
fn rebase(base: &Range<u64>, rel: Range<u64>) -> Result<Range<u64>> {
    let start = base.start.checked_add(rel.start).ok_or(EINVAL)?;
    let end = base.start.checked_add(rel.end).ok_or(EINVAL)?;

    // Check that the rebased range on not empty and fits within `base`.
    if start >= end || end > base.end {
        return Err(EINVAL);
    }

    Ok(start..end)
}

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

    /// Creates a byte range relative to this allocation, retaining its backing storage.
    pub(crate) fn region(self: &Arc<Self>, range: Range<u64>) -> Result<VramRegion> {
        Ok(VramRegion {
            backing: self.clone(),
            // INVARIANT: `rebase` returns a nonempty range.
            range: rebase(&self.range, range)?,
        })
    }
}

/// A byte range that keeps its VRAM allocation alive.
///
/// # Invariants
///
/// `range.start < range.end`, and `range` lies within the range of `backing`.
#[derive(Clone)]
pub(crate) struct VramRegion {
    backing: Arc<VramBlock>,
    range: Range<u64>,
}

impl VramRegion {
    /// Returns the range covered by this region.
    pub(crate) const fn range(&self) -> &Range<u64> {
        &self.range
    }

    /// Returns the length of this region in bytes.
    pub(crate) const fn len(&self) -> u64 {
        self.range.end.saturating_sub(self.range.start)
    }

    /// Creates a nonempty subregion relative to this view's start.
    pub(crate) fn subregion(&self, range: Range<u64>) -> Result<Self> {
        Ok(Self {
            backing: self.backing.clone(),
            // INVARIANT: `rebase` returns a nonempty range.
            range: rebase(&self.range, range)?,
        })
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
