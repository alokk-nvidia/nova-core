// SPDX-License-Identifier: GPL-2.0

//! Virtual Function I/O (VFIO) abstractions.

use crate::{
    bindings,
    prelude::*,
    uaccess::{
        UserPtr,
        UserSlice, //
    },
};

#[cfg(CONFIG_VFIO_PCI_CORE)]
pub mod pci;

/// Reset the VFIO device.
pub const DEVICE_RESET: u32 = bindings::VFIO_DEVICE_RESET;

/// Reset the PCI slot or bus containing a VFIO device.
pub const DEVICE_PCI_HOT_RESET: u32 = bindings::VFIO_DEVICE_PCI_HOT_RESET;

/// Capability buffer supplied by VFIO for a region-info callback.
#[cfg(CONFIG_VFIO_PCI_CORE)]
pub struct InfoCap<'a> {
    raw: &'a mut bindings::vfio_info_cap,
}

/// Opaque wrapper around a `char __user *` buffer from a VFIO read or write callback.
///
/// Provides bounds-checked copying of kernel data into the user-space buffer.
///
/// This type cannot be constructed by driver code — it is only created by the
/// callback trampoline.
pub struct UserBuf {
    ptr: UserPtr,
    count: usize,
}

impl UserBuf {
    /// Returns the buffer length in bytes.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` when the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Limit this callback's transfer to at most `count` bytes.
    pub fn truncate(&mut self, count: usize) {
        self.count = self.count.min(count);
    }

    /// Overwrite the part of `data` that overlaps a completed region read.
    ///
    /// `read_offset` is the starting region offset and `read_len` is the number of
    /// bytes returned by the read. `data_offset` locates `data` within the same region.
    pub fn write_overlapping(
        &self,
        read_offset: u64,
        read_len: usize,
        data_offset: u64,
        data: &[u8],
    ) -> Result {
        if read_len > self.len() {
            return Err(EINVAL);
        }
        let read_end = read_offset.checked_add(read_len as u64).ok_or(EOVERFLOW)?;
        let data_end = data_offset
            .checked_add(data.len() as u64)
            .ok_or(EOVERFLOW)?;
        let start = read_offset.max(data_offset);
        let end = read_end.min(data_end);
        if start >= end {
            return Ok(());
        }

        // These differences are bounded by `read_len` and `data.len()`.
        let buf_offset = (start - read_offset) as usize;
        let data_start = (start - data_offset) as usize;
        let data_end = (end - data_offset) as usize;
        self.write_at(buf_offset, &data[data_start..data_end])
    }

    /// Copy `data` to the user buffer at byte offset `offset`.
    ///
    /// Returns [`EFAULT`] if the copy fails, [`EINVAL`] if the write would
    /// exceed the buffer bounds.
    fn write_at(&self, offset: usize, data: &[u8]) -> Result {
        let end = offset.checked_add(data.len()).ok_or(EOVERFLOW)?;
        if end > self.count {
            return Err(EINVAL);
        }
        let dest = self.ptr.wrapping_byte_add(offset);
        let mut writer = UserSlice::new(dest, data.len()).writer();
        writer.write_slice(data)
    }
}
