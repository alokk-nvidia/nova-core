// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

//! GSP plugin logs exposed through debugfs.
//!
//! With debugfs mounted at `/sys/kernel/debug`, the directory uses the VF's
//! PCI domain:bus:device.function address:
//!
//! ```text
//! /sys/kernel/debug/nova-core/<VF-DBDF>-vgpu/
//! |-- init_log
//! |-- vgpu_log
//! `-- kernel_log
//! ```

use kernel::{
    debugfs,
    fs::file,
    prelude::*,
    uaccess::UserSliceWriter, //
};

use crate::{
    firmware::gsp::BuildId,
    gpu::Spec,
    gsp::{
        build_log_buffer_header,
        LOG_BUFFER_HEADER_SIZE, //
    },
    vgpu::gsp_plugin_comm::{
        MappedPluginLogBuffer,
        MappedPluginLogBuffers, //
    },
};

const LOG_READ_CHUNK_SIZE: usize = 4096;

/// A vGPU plugin log buffer backed by VRAM, read via BAR1 MMIO.
///
/// An optional header lets `nvlog_decoder` identify the GPU architecture and firmware build.
struct VgpuLogBuffer<'gpu> {
    buffer: MappedPluginLogBuffer<'gpu>,
    header: [u8; LOG_BUFFER_HEADER_SIZE],
    header_len: usize,
}

impl<'gpu> VgpuLogBuffer<'gpu> {
    fn new(
        buffer: MappedPluginLogBuffer<'gpu>,
        spec: Spec,
        build_id: Option<&BuildId>,
        task_prefix: &str,
    ) -> Self {
        let (header, header_len) = match build_id {
            Some(bid) => (
                build_log_buffer_header(spec, bid, task_prefix),
                LOG_BUFFER_HEADER_SIZE,
            ),
            None => ([0u8; LOG_BUFFER_HEADER_SIZE], 0),
        };

        Self {
            buffer,
            header,
            header_len,
        }
    }
}

impl debugfs::BinaryWriter for VgpuLogBuffer<'_> {
    fn write_to_slice(
        &self,
        writer: &mut UserSliceWriter,
        offset: &mut file::Offset,
    ) -> Result<usize> {
        if offset.is_negative() {
            return Err(EINVAL);
        }

        let offset_val: usize = (*offset).try_into().map_err(|_| EINVAL)?;
        let total_len = self
            .header_len
            .checked_add(self.buffer.size())
            .ok_or(EOVERFLOW)?;

        if offset_val >= total_len {
            return Ok(0);
        }

        let count = (total_len - offset_val).min(writer.len());
        if count == 0 {
            return Ok(0);
        }

        // Keep the staging buffer on the heap to avoid a page-sized kernel stack object.
        let staging_size = count.min(LOG_READ_CHUNK_SIZE);
        let mut staging = KVec::new();
        staging.resize(staging_size, 0, GFP_KERNEL)?;

        let mut written = 0usize;
        let result: Result = (|| {
            while written < count {
                let chunk_len = (count - written).min(staging.len());
                let chunk = &mut staging[..chunk_len];
                let chunk_offset = offset_val.checked_add(written).ok_or(EOVERFLOW)?;
                let mut filled = 0usize;

                if chunk_offset < self.header_len {
                    let header_len = (self.header_len - chunk_offset).min(chunk_len);
                    chunk[..header_len]
                        .copy_from_slice(&self.header[chunk_offset..chunk_offset + header_len]);
                    filled = header_len;
                }

                if filled < chunk_len {
                    let log_offset = chunk_offset
                        .checked_add(filled)
                        .ok_or(EOVERFLOW)?
                        .checked_sub(self.header_len)
                        .ok_or(EINVAL)?;

                    self.buffer.read(log_offset, &mut chunk[filled..])?;
                }

                writer.write_slice(chunk)?;
                written = written.checked_add(chunk_len).ok_or(EOVERFLOW)?;
            }
            Ok(())
        })();
        if written == 0 {
            result?;
        }

        *offset = (*offset)
            .checked_add(i64::try_from(written).map_err(|_| EOVERFLOW)?)
            .ok_or(EOVERFLOW)?;
        Ok(written)
    }
}

/// The three plugin log streams for one vGPU instance.
pub(super) struct VgpuLogBuffers<'gpu> {
    init_log: VgpuLogBuffer<'gpu>,
    vgpu_log: VgpuLogBuffer<'gpu>,
    kernel_log: VgpuLogBuffer<'gpu>,
}

impl<'gpu> VgpuLogBuffers<'gpu> {
    pub(super) fn new(
        buffers: MappedPluginLogBuffers<'gpu>,
        spec: Spec,
        build_id: Option<&BuildId>,
    ) -> Self {
        let (init, vgpu, kernel) = buffers.into_parts();

        Self {
            init_log: VgpuLogBuffer::new(init, spec, build_id, "INIT"),
            vgpu_log: VgpuLogBuffer::new(vgpu, spec, build_id, "VGPU"),
            kernel_log: VgpuLogBuffer::new(kernel, spec, build_id, "KRNL"),
        }
    }

    pub(super) fn register_debugfs<'data, 'dir>(
        logs: &'data Self,
        dir: &'dir debugfs::ScopedDir<'data, 'dir>,
    ) {
        dir.read_binary_file(c"init_log", &logs.init_log);
        dir.read_binary_file(c"vgpu_log", &logs.vgpu_log);
        dir.read_binary_file(c"kernel_log", &logs.kernel_log);
    }
}
