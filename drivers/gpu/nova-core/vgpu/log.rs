// SPDX-License-Identifier: GPL-2.0

use kernel::{
    debugfs,
    devres::Devres,
    fs::file,
    io::Io,
    prelude::*,
    sync::Arc,
    uaccess::UserSliceWriter, //
};

use crate::{
    driver::Bar1,
    firmware::BuildId,
    gpu::Chipset,
    gsp::{build_log_buffer_header, LOG_BUFFER_HEADER_SIZE},
    vgpu::{
        consts::plugin_rpc,
        Bar1Map, //
    },
};

/// A single vGPU plugin log buffer backed by VRAM, read via BAR1 MMIO.
///
/// The GSP plugin writes encoded log entries into the management heap in
/// VRAM at a known offset.  This struct holds a clone of the BAR1
/// [`Devres`] and the GPU virtual address from the [`Bar1Map`], allowing
/// debugfs readers to access the log data without a device reference.
///
/// A [`LOG_BUFFER_HEADER_SIZE`]-byte header is prepended so that
/// `nvlog_decoder` can identify the GPU architecture and firmware build.
pub(crate) struct VgpuLogBuffer {
    bar1: Arc<Devres<Bar1>>,
    gpu_va_addr: u64,
    offset: u64,
    size: usize,
    header: [u8; LOG_BUFFER_HEADER_SIZE],
    header_len: usize,
}

impl VgpuLogBuffer {
    fn new(
        bar1_map: &Bar1Map,
        offset: u64,
        size: usize,
        chipset: Chipset,
        build_id: Option<&BuildId>,
        task_prefix: &str,
    ) -> Self {
        let (header, header_len) = match build_id {
            Some(bid) => (
                build_log_buffer_header(chipset, bid, task_prefix),
                LOG_BUFFER_HEADER_SIZE,
            ),
            None => ([0u8; LOG_BUFFER_HEADER_SIZE], 0),
        };

        Self {
            bar1: bar1_map.bar1_arc().clone(),
            gpu_va_addr: bar1_map.gpu_va_addr(),
            offset,
            size,
            header,
            header_len,
        }
    }
}

impl debugfs::BinaryWriter for VgpuLogBuffer {
    fn write_to_slice(
        &self,
        writer: &mut UserSliceWriter,
        offset: &mut file::Offset,
    ) -> Result<usize> {
        if offset.is_negative() {
            return Err(EINVAL);
        }

        let offset_val: usize = (*offset).try_into().map_err(|_| EINVAL)?;
        let total_len = self.header_len + self.size;

        if offset_val >= total_len {
            return Ok(0);
        }

        let count = (total_len - offset_val).min(writer.len());
        if count == 0 {
            return Ok(0);
        }

        let mut written = 0usize;

        if offset_val < self.header_len {
            let hdr_start = offset_val;
            let hdr_count = (self.header_len - hdr_start).min(count);
            writer.write_slice(&self.header[hdr_start..hdr_start + hdr_count])?;
            written += hdr_count;
        }

        if written < count {
            let bar = self.bar1.try_access().ok_or(ENXIO)?;
            let buf_start = offset_val.saturating_sub(self.header_len);
            let remaining = count - written;
            let mut pos = buf_start;
            let mut buf_written = 0usize;

            while buf_written < remaining {
                let vram_off = self.offset + pos as u64;
                let aligned_off = vram_off & !3;
                let within = (vram_off & 3) as usize;

                let val = bar.try_read32((self.gpu_va_addr + aligned_off) as usize)?;
                let bytes = val.to_le_bytes();

                let avail = 4 - within;
                let chunk = avail.min(remaining - buf_written);

                writer.write_slice(&bytes[within..within + chunk])?;
                buf_written += chunk;
                pos += chunk;
            }

            written += buf_written;
        }

        *offset += written as i64;
        Ok(written)
    }
}

/// Aggregated log buffers for a single vGPU instance.
///
/// Each vGPU plugin produces three log streams within the management heap:
/// - `init_log`: init task log (128 KB)
/// - `vgpu_log`: vGPU task log (256 KB)
/// - `kernel_log`: kernel task log (64 KB)
pub(crate) struct VgpuLogBuffers {
    init_log: VgpuLogBuffer,
    vgpu_log: VgpuLogBuffer,
    kernel_log: VgpuLogBuffer,
}

impl VgpuLogBuffers {
    pub(crate) fn new(
        bar1_map: &Bar1Map,
        chipset: Chipset,
        build_id: Option<&BuildId>,
    ) -> Self {
        Self {
            init_log: VgpuLogBuffer::new(
                bar1_map,
                plugin_rpc::INIT_TASK_LOG_OFFSET,
                plugin_rpc::INIT_LOG_SIZE as usize,
                chipset,
                build_id,
                "INIT",
            ),
            vgpu_log: VgpuLogBuffer::new(
                bar1_map,
                plugin_rpc::VGPU_TASK_LOG_OFFSET,
                plugin_rpc::VGPU_LOG_SIZE as usize,
                chipset,
                build_id,
                "VGPU",
            ),
            kernel_log: VgpuLogBuffer::new(
                bar1_map,
                plugin_rpc::KERNEL_LOG_OFFSET,
                plugin_rpc::KERNEL_LOG_SIZE as usize,
                chipset,
                build_id,
                "KERN",
            ),
        }
    }

    /// Register debugfs binary files for these log buffers within a scoped directory.
    pub(crate) fn register_debugfs<'data, 'dir>(
        logs: &'data VgpuLogBuffers,
        dir: &'dir debugfs::ScopedDir<'data, 'dir>,
    ) {
        dir.read_binary_file(c"init_log", &logs.init_log);
        dir.read_binary_file(c"vgpu_log", &logs.vgpu_log);
        dir.read_binary_file(c"kernel_log", &logs.kernel_log);
    }
}
