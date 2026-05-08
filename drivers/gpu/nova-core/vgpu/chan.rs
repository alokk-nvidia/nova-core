// SPDX-License-Identifier: GPL-2.0

use kernel::prelude::*;

/// Channel ID allocator using a bitmap over 2048 channels.
#[expect(dead_code)]
pub(crate) struct ChidAllocator {
    bitmap: [u64; 32],
    total: u32,
}

impl ChidAllocator {
    pub(crate) fn new(total: u32) -> Self {
        Self {
            bitmap: [0u64; 32],
            total,
        }
    }

    /// Allocate `count` contiguous channels, aligned to `count` boundary.
    pub(crate) fn alloc(&mut self, count: u32) -> Result<u32> {
        if count == 0 {
            return Err(EINVAL);
        }
        let stride = count as usize;
        let total_bits = self.bitmap.len() * 64;
        let mut offset = 0usize;

        while offset + stride <= total_bits {
            if self.is_range_free(offset, stride) {
                self.set_range(offset, stride);
                return Ok(offset as u32);
            }
            offset += stride;
        }
        Err(ENOSPC)
    }

    /// Free `count` channels starting at `offset`.
    pub(crate) fn free(&mut self, offset: u32, count: u32) {
        let start = offset as usize;
        for i in start..start + count as usize {
            let word = i / 64;
            let bit = i % 64;
            self.bitmap[word] &= !(1u64 << bit);
        }
    }

    fn is_range_free(&self, start: usize, count: usize) -> bool {
        for i in start..start + count {
            let word = i / 64;
            let bit = i % 64;
            if self.bitmap[word] & (1u64 << bit) != 0 {
                return false;
            }
        }
        true
    }

    fn set_range(&mut self, start: usize, count: usize) {
        for i in start..start + count {
            let word = i / 64;
            let bit = i % 64;
            self.bitmap[word] |= 1u64 << bit;
        }
    }
}
