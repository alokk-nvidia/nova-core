// SPDX-License-Identifier: GPL-2.0

mod boot;
mod nvkv;

use kernel::{
    debugfs,
    device,
    dma::{
        Coherent,
        CoherentBox,
        DmaAddress, //
    },
    fs::file,
    io::{
        self,
        io_project,
        io_write, //
    },
    pci,
    prelude::*,
    ptr::KnownSize,
    transmute::{
        AsBytes,
        FromBytes, //
    },
    uaccess::UserSliceWriter, //
};

pub(crate) mod cmdq;
pub(crate) mod commands;
mod fw;

pub(crate) use fw::{
    GspFwWprMeta,
    LibosParams, //
};

use crate::{
    driver::Bar0,
    falcon::{
        gsp::Gsp as GspFalcon,
        sec2::Sec2 as Sec2Falcon,
        Falcon, //
    },
    firmware::BuildId,
    gpu::Chipset,
    gsp::cmdq::Cmdq,
    gsp::fw::{
        GspArgumentsPadded,
        LibosMemoryRegionInitArgument, //
    },
    num,
};

pub(crate) const GSP_PAGE_SHIFT: usize = 12;
pub(crate) const GSP_PAGE_SIZE: usize = 1 << GSP_PAGE_SHIFT;

/// Common context for the GSP boot process.
pub(crate) struct GspBootContext<'a> {
    pub(crate) pdev: &'a pci::Device<device::Bound>,
    pub(crate) bar: &'a Bar0,
    pub(crate) chipset: Chipset,
    pub(crate) gsp_falcon: &'a Falcon<GspFalcon>,
    pub(crate) sec2_falcon: &'a Falcon<Sec2Falcon>,
    pub(crate) fsp_falcon: Option<Falcon<crate::falcon::fsp::Fsp>>,
    pub(crate) vgpu_requested: bool,
}

impl GspBootContext<'_> {
    pub(crate) fn dev(&self) -> &device::Device<device::Bound> {
        self.pdev.as_ref()
    }
}

/// Number of GSP pages to use in a RM log buffer.
const RM_LOG_BUFFER_NUM_PAGES: usize = 0x10;
const LOG_BUFFER_SIZE: usize = RM_LOG_BUFFER_NUM_PAGES * GSP_PAGE_SIZE;

/// Array of page table entries, as understood by the GSP bootloader.
#[repr(C)]
struct PteArray<const NUM_ENTRIES: usize>([u64; NUM_ENTRIES]);

/// SAFETY: arrays of `u64` implement `FromBytes` and we are but a wrapper around one.
unsafe impl<const NUM_ENTRIES: usize> FromBytes for PteArray<NUM_ENTRIES> {}

/// SAFETY: arrays of `u64` implement `AsBytes` and we are but a wrapper around one.
unsafe impl<const NUM_ENTRIES: usize> AsBytes for PteArray<NUM_ENTRIES> {}

impl<const NUM_PAGES: usize> PteArray<NUM_PAGES> {
    /// Initialize a new page table array mapping `NUM_PAGES` GSP pages starting at address `start`.
    fn init<T: KnownSize + ?Sized>(
        view: io::View<'_, Coherent<T>, Self>,
        start: DmaAddress,
    ) -> Result<()> {
        for i in 0..NUM_PAGES {
            io_write!(
                view,
                .0[build: i],
                start
                    .checked_add(num::usize_as_u64(i) << GSP_PAGE_SHIFT)
                    .ok_or(EOVERFLOW)?
            );
        }

        Ok(())
    }
}

/// Size of the header prepended to debugfs log buffer dumps.
///
/// This header makes each dump self-describing so that decoding tools can
/// identify the firmware build, GPU architecture, and metadata format without
/// out-of-band information.
const LOG_BUFFER_HEADER_SIZE: usize = 0x48;

/// Build a log buffer header from GPU and firmware metadata.
///
/// Layout (all little-endian):
///   0x00  gpuArch (u32)
///   0x04  gpuImpl (u32)
///   0x08  version (u32) = 2
///   0x0C  buildIdLength (u32)
///   0x10  taskPrefix[8]
///   0x18  localToGlobalTimerDelta (u64) = 0
///   0x20  buildId[32]
///   0x40  flags (u32) = 1 (packed metadata)
///   0x44  reserved (u32) = 0
fn build_log_buffer_header(
    chipset: Chipset,
    build_id: &BuildId,
    task_prefix: &str,
) -> [u8; LOG_BUFFER_HEADER_SIZE] {
    let mut h = [0u8; LOG_BUFFER_HEADER_SIZE];
    let chipset_val = chipset as u32;

    h[0x00..0x04].copy_from_slice(&(chipset_val >> 4).to_le_bytes());
    h[0x04..0x08].copy_from_slice(&(chipset_val & 0xF).to_le_bytes());
    h[0x08..0x0C].copy_from_slice(&2u32.to_le_bytes());

    let bid = build_id.as_bytes();
    h[0x0C..0x10].copy_from_slice(&(bid.len() as u32).to_le_bytes());

    let prefix = task_prefix.as_bytes();
    let prefix_len = prefix.len().min(8);
    h[0x10..0x10 + prefix_len].copy_from_slice(&prefix[..prefix_len]);

    h[0x20..0x20 + bid.len()].copy_from_slice(bid);
    h[0x40..0x44].copy_from_slice(&1u32.to_le_bytes());

    h
}

/// The logging buffers are byte queues that contain encoded printf-like
/// messages from GSP-RM.  They need to be decoded by a special application
/// that can parse the buffers.
///
/// The 'loginit' buffer contains logs from early GSP-RM init and
/// exception dumps.  The 'logrm' buffer contains the subsequent logs. Both are
/// written to directly by GSP-RM and can be any multiple of GSP_PAGE_SIZE.
///
/// The physical address map for the log buffer is stored in the buffer
/// itself, starting with offset 1. Offset 0 contains the "put" pointer (pp).
/// Initially, pp is equal to 0. If the buffer has valid logging data in it,
/// then pp points to index into the buffer where the next logging entry will
/// be written. Therefore, the logging data is valid if:
///   1 <= pp < sizeof(buffer)/sizeof(u64)
///
/// When a build ID is available, the debugfs file for this buffer prepends
/// a header so the dump is self-describing.
struct LogBuffer {
    header: [u8; LOG_BUFFER_HEADER_SIZE],
    header_len: usize,
    buffer: Coherent<[u8; LOG_BUFFER_SIZE]>,
}

impl LogBuffer {
    /// Creates a new `LogBuffer` mapped on `dev`.
    fn new(
        dev: &device::Device<device::Bound>,
        chipset: Chipset,
        build_id: Option<&BuildId>,
        task_prefix: &str,
    ) -> Result<Self> {
        let buffer: Coherent<[u8; LOG_BUFFER_SIZE]> = Coherent::zeroed(dev, GFP_KERNEL)?;

        let start_addr = buffer.dma_handle();
        let pte_view = io_project!(
            buffer,
            [build: size_of::<u64>()..][build: ..RM_LOG_BUFFER_NUM_PAGES * size_of::<u64>()]
        )
        .try_cast::<PteArray<RM_LOG_BUFFER_NUM_PAGES>>()?;
        PteArray::init(pte_view, start_addr)?;

        let (header, header_len) = match build_id {
            Some(bid) => (
                build_log_buffer_header(chipset, bid, task_prefix),
                LOG_BUFFER_HEADER_SIZE,
            ),
            None => ([0u8; LOG_BUFFER_HEADER_SIZE], 0),
        };

        Ok(Self {
            header,
            header_len,
            buffer,
        })
    }
}

impl debugfs::BinaryWriter for LogBuffer {
    fn write_to_slice(
        &self,
        writer: &mut UserSliceWriter,
        offset: &mut file::Offset,
    ) -> Result<usize> {
        if offset.is_negative() {
            return Err(EINVAL);
        }

        let offset_val: usize = (*offset).try_into().map_err(|_| EINVAL)?;
        let total_len = self.header_len + self.buffer.size();

        if offset_val >= total_len {
            return Ok(0);
        }

        let count = (total_len - offset_val).min(writer.len());
        if count == 0 {
            return Ok(0);
        }

        let mut written = 0;

        if offset_val < self.header_len {
            let hdr_start = offset_val;
            let hdr_count = (self.header_len - hdr_start).min(count);
            writer.write_slice(&self.header[hdr_start..hdr_start + hdr_count])?;
            written += hdr_count;
        }

        if written < count {
            let buf_start = offset_val.saturating_sub(self.header_len);
            let buf_count = count - written;
            writer.write_dma(&self.buffer, buf_start, buf_count)?;
            written += buf_count;
        }

        *offset += written as i64;
        Ok(written)
    }
}

/// Log buffers used by GSP-RM for debug logging.
///
/// r000+ firmware expects log buffers for all LIBOS3 tasks. Each buffer is
/// registered as a libos memory region entry, identified by its id8 name.
struct LogBuffers {
    /// Init task log buffer (LOGINIT, 64KB).
    loginit: LogBuffer,
    /// Interrupt task log buffer (LOGINTR, 64KB).
    logintr: LogBuffer,
    /// RM task log buffer (LOGRM, 64KB).
    logrm: LogBuffer,
    /// MNOC task log buffer (LOGMNOC, 64KB).
    logmnoc: LogBuffer,
    /// Root task log buffer (LOGROOT, 4KB).
    logroot: LogBuffer,
    /// RM state monitor task log buffer (LOGRMON, 4KB).
    logrmon: LogBuffer,
}

/// GSP runtime data.
#[pin_data]
pub(crate) struct Gsp {
    /// Libos arguments.
    pub(crate) libos: Coherent<[LibosMemoryRegionInitArgument]>,
    /// Log buffers for all LIBOS3 tasks, exposed via debugfs.
    #[pin]
    logs: debugfs::Scope<LogBuffers>,
    /// Command queue.
    #[pin]
    pub(crate) cmdq: Cmdq,
    /// RM arguments.
    rmargs: Coherent<GspArgumentsPadded>,
    /// RM state monitor buffer (required by r000+ GSP-RM for diagnostics).
    rm_state_monitor: Coherent<[u8; GSP_PAGE_SIZE]>,
}

impl Gsp {
    // Creates an in-place initializer for a `Gsp` manager for `pdev`.
    pub(crate) fn new<'a>(
        pdev: &'a pci::Device<device::Bound>,
        chipset: Chipset,
        build_id: Option<&'a BuildId>,
    ) -> impl PinInit<Self, Error> + 'a {
        pin_init::pin_init_scope(move || {
            let dev = pdev.as_ref();

            let loginit = LogBuffer::new(dev, chipset, build_id, "INIT")?;
            let logintr = LogBuffer::new(dev, chipset, build_id, "INTR")?;
            let logrm = LogBuffer::new(dev, chipset, build_id, "RM")?;
            let logmnoc = LogBuffer::new(dev, chipset, build_id, "MNOC")?;
            let logroot = LogBuffer::new(dev, chipset, build_id, "ROOT")?;
            let logrmon = LogBuffer::new(dev, chipset, build_id, "RMON")?;

            Ok(try_pin_init!(Self {
                cmdq <- Cmdq::new(dev),
                rm_state_monitor: Coherent::zeroed(dev, GFP_KERNEL)?,
                rmargs: Coherent::init(
                    dev,
                    GFP_KERNEL,
                    GspArgumentsPadded::new(&cmdq, None, rm_state_monitor),
                )?,
                libos: {
                    let mut libos = CoherentBox::zeroed_slice(
                        dev,
                        GSP_PAGE_SIZE / size_of::<LibosMemoryRegionInitArgument>(),
                        GFP_KERNEL,
                    )?;

                    libos.init_at(0, LibosMemoryRegionInitArgument::new("LOGINIT", &loginit.buffer))?;
                    libos.init_at(1, LibosMemoryRegionInitArgument::new("LOGINTR", &logintr.buffer))?;
                    libos.init_at(2, LibosMemoryRegionInitArgument::new("LOGRM", &logrm.buffer))?;
                    libos.init_at(3, LibosMemoryRegionInitArgument::new("LOGMNOC", &logmnoc.buffer))?;
                    libos.init_at(4, LibosMemoryRegionInitArgument::new("LOGROOT", &logroot.buffer))?;
                    libos.init_at(5, LibosMemoryRegionInitArgument::new("LOGRMON", &logrmon.buffer))?;
                    libos.init_at(6, LibosMemoryRegionInitArgument::new("RMARGS", rmargs))?;

                    libos.into()
                },
                logs <- {
                    let log_buffers = LogBuffers {
                        loginit,
                        logintr,
                        logrm,
                        logmnoc,
                        logroot,
                        logrmon,
                    };

                    #[allow(static_mut_refs)]
                    // SAFETY: `DEBUGFS_ROOT` is created before driver registration and cleared
                    // after driver unregistration, so no probe() can race with its modification.
                    //
                    // PANIC: `DEBUGFS_ROOT` cannot be `None` here.  It is set before driver
                    // registration and cleared after driver unregistration, so it is always
                    // `Some` for the entire lifetime that probe() can be called.
                    let log_parent: &debugfs::Dir = unsafe { crate::DEBUGFS_ROOT.as_ref() }
                        .expect("DEBUGFS_ROOT not initialized");

                    log_parent.scope(log_buffers, dev.name(), |logs, dir| {
                        dir.read_binary_file(c"loginit", &logs.loginit);
                        dir.read_binary_file(c"logintr", &logs.logintr);
                        dir.read_binary_file(c"logrm", &logs.logrm);
                        dir.read_binary_file(c"logmnoc", &logs.logmnoc);
                        dir.read_binary_file(c"logroot", &logs.logroot);
                        dir.read_binary_file(c"logrmon", &logs.logrmon);
                    })
                },
            }))
        })
    }
}
