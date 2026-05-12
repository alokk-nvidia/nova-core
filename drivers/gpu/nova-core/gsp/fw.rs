// SPDX-License-Identifier: GPL-2.0

pub(crate) mod commands;
mod r000_00;
mod r570_144;

use r000_00 as r000;
use r570_144 as r570;

use core::ops::Range;

use kernel::{
    bitfield,
    dma::Coherent,
    io::{
        self,
        io_read,
        io_write, //
    },
    prelude::*,
    ptr::{
        Alignable,
        Alignment,
        KnownSize, //
    },
    sizes::{
        SZ_128K,
        SZ_1M, //
    },
    transmute::{
        AsBytes,
        FromBytes, //
    },
};

use crate::{
    fb::{
        FbLayout,
        PMU_RESERVED_SIZE, //
    },
    firmware::gsp::GspFirmware,
    gpu::Chipset,
    gsp::{
        cmdq::Cmdq,
        GSP_PAGE_SHIFT,
        GSP_PAGE_SIZE, //
    },
    num::{
        self,
        FromSafeCast, //
    },
};

/// Maximum size of a single GSP message queue element in bytes.
pub(crate) const GSP_MSG_QUEUE_ELEMENT_SIZE_MAX: usize =
    num::u32_as_usize(r570::GSP_MSG_QUEUE_ELEMENT_SIZE_MAX);

/// Empty type to group methods related to heap parameters for running the GSP firmware.
enum GspFwHeapParams {}

/// Minimum required alignment for the GSP heap.
const GSP_HEAP_ALIGNMENT: Alignment = Alignment::new::<{ 1 << 20 }>();

// These constants override the generated bindings for architecture-specific heap sizing.
// See Open RM: kgspCalculateGspFwHeapSize and related functions.
//
// 14MB for Hopper/Blackwell+.
const GSP_FW_HEAP_PARAM_BASE_RM_SIZE_GH100: u64 = 14 * num::usize_as_u64(SZ_1M);
// 142MB client alloc for ~188MB total.
const GSP_FW_HEAP_PARAM_CLIENT_ALLOC_SIZE_GH100: u64 = 142 * num::usize_as_u64(SZ_1M);
// Hopper/Blackwell+ minimum heap size: 170MB (88 + 12 + 70).
// See Open RM: GSP_FW_HEAP_SIZE_OVERRIDE_LIBOS3_BAREMETAL_MIN_MB for the base 88MB,
// plus Hopper+ additions in kgspCalculateGspFwHeapSize_GH100.
const GSP_FW_HEAP_SIZE_OVERRIDE_LIBOS3_BAREMETAL_MIN_MB_HOPPER: u64 = 170;

impl GspFwHeapParams {
    /// Returns the amount of GSP-RM heap memory used during GSP-RM boot and initialization (up to
    /// and including the first client subdevice allocation).
    fn base_rm_size(chipset: Chipset) -> u64 {
        use crate::gpu::Architecture;
        match chipset.arch() {
            Architecture::Hopper | Architecture::BlackwellGB10x | Architecture::BlackwellGB20x => {
                GSP_FW_HEAP_PARAM_BASE_RM_SIZE_GH100
            }
            _ => u64::from(r570::GSP_FW_HEAP_PARAM_BASE_RM_SIZE_TU10X),
        }
    }

    /// Returns the amount of heap memory required to support a single channel allocation.
    fn client_alloc_size(chipset: Chipset) -> Result<u64> {
        use crate::gpu::Architecture;
        let size = match chipset.arch() {
            Architecture::Hopper | Architecture::BlackwellGB10x | Architecture::BlackwellGB20x => {
                GSP_FW_HEAP_PARAM_CLIENT_ALLOC_SIZE_GH100
            }
            _ => u64::from(r570::GSP_FW_HEAP_PARAM_CLIENT_ALLOC_SIZE),
        };
        size.align_up(GSP_HEAP_ALIGNMENT).ok_or(EINVAL)
    }

    /// Returns the amount of memory to reserve for management purposes for a framebuffer of size
    /// `fb_size`.
    fn management_overhead(fb_size: u64) -> Result<u64> {
        let fb_size_gb = fb_size.div_ceil(u64::from_safe_cast(kernel::sizes::SZ_1G));

        u64::from(r570::GSP_FW_HEAP_PARAM_SIZE_PER_GB_FB)
            .saturating_mul(fb_size_gb)
            .align_up(GSP_HEAP_ALIGNMENT)
            .ok_or(EINVAL)
    }
}

/// Heap memory requirements and constraints for a given version of the GSP LIBOS.
pub(crate) struct LibosParams {
    /// The base amount of heap required by the GSP operating system, in bytes.
    carveout_size: u64,
    /// The minimum and maximum sizes allowed for the GSP FW heap, in bytes.
    allowed_heap_size: Range<u64>,
}

impl LibosParams {
    /// Version 2 of the GSP LIBOS (Turing and GA100)
    const LIBOS2: LibosParams = LibosParams {
        carveout_size: num::u32_as_u64(r570::GSP_FW_HEAP_PARAM_OS_SIZE_LIBOS2),
        allowed_heap_size: num::u32_as_u64(r570::GSP_FW_HEAP_SIZE_OVERRIDE_LIBOS2_MIN_MB)
            * num::usize_as_u64(SZ_1M)
            ..num::u32_as_u64(r570::GSP_FW_HEAP_SIZE_OVERRIDE_LIBOS2_MAX_MB)
                * num::usize_as_u64(SZ_1M),
    };

    /// Version 3 of the GSP LIBOS (GA102+)
    const LIBOS3: LibosParams = LibosParams {
        carveout_size: num::u32_as_u64(r570::GSP_FW_HEAP_PARAM_OS_SIZE_LIBOS3_BAREMETAL),
        allowed_heap_size: num::u32_as_u64(r570::GSP_FW_HEAP_SIZE_OVERRIDE_LIBOS3_BAREMETAL_MIN_MB)
            * num::usize_as_u64(SZ_1M)
            ..num::u32_as_u64(r570::GSP_FW_HEAP_SIZE_OVERRIDE_LIBOS3_BAREMETAL_MAX_MB)
                * num::usize_as_u64(SZ_1M),
    };

    /// Hopper/Blackwell+ GPUs need a larger minimum heap size than the bindings specify.
    /// The r570 bindings set LIBOS3_BAREMETAL_MIN_MB to 88MB, but Hopper/Blackwell+ actually
    /// requires 170MB (88 + 12 + 70).
    const LIBOS_HOPPER: LibosParams = LibosParams {
        carveout_size: num::u32_as_u64(r570::GSP_FW_HEAP_PARAM_OS_SIZE_LIBOS3_BAREMETAL),
        allowed_heap_size: GSP_FW_HEAP_SIZE_OVERRIDE_LIBOS3_BAREMETAL_MIN_MB_HOPPER
            * num::usize_as_u64(SZ_1M)
            ..num::u32_as_u64(r570::GSP_FW_HEAP_SIZE_OVERRIDE_LIBOS3_BAREMETAL_MAX_MB)
                * num::usize_as_u64(SZ_1M),
    };

    /// Returns the libos parameters corresponding to `chipset`.
    pub(crate) fn from_chipset(chipset: Chipset) -> &'static LibosParams {
        use crate::gpu::Architecture;
        match chipset.arch() {
            Architecture::Turing => &Self::LIBOS2,
            Architecture::Ampere if chipset == Chipset::GA100 => &Self::LIBOS2,
            Architecture::Ampere | Architecture::Ada => &Self::LIBOS3,
            Architecture::Hopper | Architecture::BlackwellGB10x | Architecture::BlackwellGB20x => {
                &Self::LIBOS_HOPPER
            }
        }
    }

    /// Returns the amount of memory (in bytes) to allocate for the WPR heap for a framebuffer size
    /// of `fb_size` (in bytes) for `chipset`.
    pub(crate) fn wpr_heap_size(&self, chipset: Chipset, fb_size: u64) -> Result<u64> {
        // The WPR heap will contain the following:
        // LIBOS carveout,
        Ok(self
            .carveout_size
            // RM boot working memory,
            .saturating_add(GspFwHeapParams::base_rm_size(chipset))
            // One RM client,
            .saturating_add(GspFwHeapParams::client_alloc_size(chipset)?)
            // Overhead for memory management.
            .saturating_add(GspFwHeapParams::management_overhead(fb_size)?)
            // Clamp to the supported heap sizes.
            .clamp(self.allowed_heap_size.start, self.allowed_heap_size.end - 1))
    }
}

/// Structure passed to the GSP bootloader, containing the framebuffer layout as well as the DMA
/// addresses of the GSP bootloader and firmware.
#[repr(transparent)]
pub(crate) struct GspFwWprMeta {
    inner: r570::GspFwWprMeta,
}

// SAFETY: Padding is explicit and does not contain uninitialized data.
unsafe impl AsBytes for GspFwWprMeta {}

// SAFETY: This struct only contains integer types for which all bit patterns
// are valid.
unsafe impl FromBytes for GspFwWprMeta {}

type GspFwWprMetaBootResumeInfo = r570::GspFwWprMeta__bindgen_ty_1;
type GspFwWprMetaBootInfo = r570::GspFwWprMeta__bindgen_ty_1__bindgen_ty_1;

impl GspFwWprMeta {
    /// Returns an initializer for a `GspFwWprMeta` suitable for booting `gsp_firmware` using the
    /// `fb_layout` layout.
    pub(crate) fn new<'a>(
        gsp_firmware: &'a GspFirmware,
        fb_layout: &'a FbLayout,
    ) -> impl Init<Self> + 'a {
        #[allow(non_snake_case)]
        let init_inner = init!(r570::GspFwWprMeta {
            // CAST: we want to store the bits of `GSP_FW_WPR_META_MAGIC` unmodified.
            magic: r570::GSP_FW_WPR_META_MAGIC as u64,
            revision: u64::from(r570::GSP_FW_WPR_META_REVISION),
            sysmemAddrOfRadix3Elf: gsp_firmware.radix3_dma_handle(),
            sizeOfRadix3Elf: u64::from_safe_cast(gsp_firmware.size()),
            sysmemAddrOfBootloader: gsp_firmware.bootloader.ucode.dma_handle(),
            sizeOfBootloader: u64::from_safe_cast(gsp_firmware.bootloader.ucode.size()),
            bootloaderCodeOffset: u64::from(gsp_firmware.bootloader.code_offset),
            bootloaderDataOffset: u64::from(gsp_firmware.bootloader.data_offset),
            bootloaderManifestOffset: u64::from(gsp_firmware.bootloader.manifest_offset),
            __bindgen_anon_1: GspFwWprMetaBootResumeInfo {
                __bindgen_anon_1: GspFwWprMetaBootInfo {
                    sysmemAddrOfSignature: gsp_firmware.signatures.dma_handle(),
                    sizeOfSignature: u64::from_safe_cast(gsp_firmware.signatures.size()),
                },
            },
            gspFwRsvdStart: fb_layout.heap.start,
            nonWprHeapOffset: fb_layout.heap.start,
            nonWprHeapSize: fb_layout.heap.end - fb_layout.heap.start,
            gspFwWprStart: fb_layout.wpr2.start,
            gspFwHeapOffset: fb_layout.wpr2_heap.start,
            gspFwHeapSize: fb_layout.wpr2_heap.end - fb_layout.wpr2_heap.start,
            gspFwOffset: fb_layout.fw_image.start,
            bootBinOffset: fb_layout.boot.start,
            frtsOffset: fb_layout.frts.start,
            frtsSize: fb_layout.frts.end - fb_layout.frts.start,
            gspFwWprEnd: fb_layout
                .vga_workspace
                .start
                .align_down(Alignment::new::<SZ_128K>()),
            gspFwHeapVfPartitionCount: fb_layout.vf_partition_count,
            fbSize: fb_layout.fb.end - fb_layout.fb.start,
            vgaWorkspaceOffset: fb_layout.vga_workspace.start,
            vgaWorkspaceSize: fb_layout.vga_workspace.end - fb_layout.vga_workspace.start,
            pmuReservedSize: PMU_RESERVED_SIZE,
            ..Zeroable::init_zeroed()
        });

        init!(GspFwWprMeta {
            inner <- init_inner,
        })
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(u32)]
pub(crate) enum MsgFunction {
    // Common function codes
    AllocChannelDma = r570::NV_VGPU_MSG_FUNCTION_ALLOC_CHANNEL_DMA,
    AllocCtxDma = r570::NV_VGPU_MSG_FUNCTION_ALLOC_CTX_DMA,
    AllocDevice = r570::NV_VGPU_MSG_FUNCTION_ALLOC_DEVICE,
    AllocMemory = r570::NV_VGPU_MSG_FUNCTION_ALLOC_MEMORY,
    AllocObject = r570::NV_VGPU_MSG_FUNCTION_ALLOC_OBJECT,
    AllocRoot = r570::NV_VGPU_MSG_FUNCTION_ALLOC_ROOT,
    BindCtxDma = r570::NV_VGPU_MSG_FUNCTION_BIND_CTX_DMA,
    ContinuationRecord = r570::NV_VGPU_MSG_FUNCTION_CONTINUATION_RECORD,
    Free = r570::NV_VGPU_MSG_FUNCTION_FREE,
    GetGspStaticInfo = r570::NV_VGPU_MSG_FUNCTION_GET_GSP_STATIC_INFO,
    GetStaticInfo = r570::NV_VGPU_MSG_FUNCTION_GET_STATIC_INFO,
    GspInitPostObjGpu = r570::NV_VGPU_MSG_FUNCTION_GSP_INIT_POST_OBJGPU,
    GspRmControl = r570::NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL,
    Log = r570::NV_VGPU_MSG_FUNCTION_LOG,
    MapMemory = r570::NV_VGPU_MSG_FUNCTION_MAP_MEMORY,
    Nop = r570::NV_VGPU_MSG_FUNCTION_NOP,
    SetGuestSystemInfo = r570::NV_VGPU_MSG_FUNCTION_SET_GUEST_SYSTEM_INFO,

    // Event codes
    GspInitDone = r570::NV_VGPU_MSG_EVENT_GSP_INIT_DONE,
    GspLockdownNotice = r570::NV_VGPU_MSG_EVENT_GSP_LOCKDOWN_NOTICE,
    GspPostNoCat = r570::NV_VGPU_MSG_EVENT_GSP_POST_NOCAT_RECORD,
    MmuFaultQueued = r570::NV_VGPU_MSG_EVENT_MMU_FAULT_QUEUED,
    OsErrorLog = r570::NV_VGPU_MSG_EVENT_OS_ERROR_LOG,
    PostEvent = r570::NV_VGPU_MSG_EVENT_POST_EVENT,
    RcTriggered = r570::NV_VGPU_MSG_EVENT_RC_TRIGGERED,
    UcodeLibOsPrint = r570::NV_VGPU_MSG_EVENT_UCODE_LIBOS_PRINT,
}

impl TryFrom<u32> for MsgFunction {
    type Error = kernel::error::Error;

    fn try_from(value: u32) -> Result<MsgFunction> {
        match value {
            // Common function codes
            r570::NV_VGPU_MSG_FUNCTION_ALLOC_CHANNEL_DMA => Ok(MsgFunction::AllocChannelDma),
            r570::NV_VGPU_MSG_FUNCTION_ALLOC_CTX_DMA => Ok(MsgFunction::AllocCtxDma),
            r570::NV_VGPU_MSG_FUNCTION_ALLOC_DEVICE => Ok(MsgFunction::AllocDevice),
            r570::NV_VGPU_MSG_FUNCTION_ALLOC_MEMORY => Ok(MsgFunction::AllocMemory),
            r570::NV_VGPU_MSG_FUNCTION_ALLOC_OBJECT => Ok(MsgFunction::AllocObject),
            r570::NV_VGPU_MSG_FUNCTION_ALLOC_ROOT => Ok(MsgFunction::AllocRoot),
            r570::NV_VGPU_MSG_FUNCTION_BIND_CTX_DMA => Ok(MsgFunction::BindCtxDma),
            r570::NV_VGPU_MSG_FUNCTION_CONTINUATION_RECORD => Ok(MsgFunction::ContinuationRecord),
            r570::NV_VGPU_MSG_FUNCTION_FREE => Ok(MsgFunction::Free),
            r570::NV_VGPU_MSG_FUNCTION_GET_GSP_STATIC_INFO => Ok(MsgFunction::GetGspStaticInfo),
            r570::NV_VGPU_MSG_FUNCTION_GET_STATIC_INFO => Ok(MsgFunction::GetStaticInfo),
            r570::NV_VGPU_MSG_FUNCTION_GSP_INIT_POST_OBJGPU => Ok(MsgFunction::GspInitPostObjGpu),
            r570::NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL => Ok(MsgFunction::GspRmControl),
            r570::NV_VGPU_MSG_FUNCTION_LOG => Ok(MsgFunction::Log),
            r570::NV_VGPU_MSG_FUNCTION_MAP_MEMORY => Ok(MsgFunction::MapMemory),
            r570::NV_VGPU_MSG_FUNCTION_NOP => Ok(MsgFunction::Nop),
            r570::NV_VGPU_MSG_FUNCTION_SET_GUEST_SYSTEM_INFO => Ok(MsgFunction::SetGuestSystemInfo),

            // Event codes
            r570::NV_VGPU_MSG_EVENT_GSP_INIT_DONE => Ok(MsgFunction::GspInitDone),
            r570::NV_VGPU_MSG_EVENT_GSP_LOCKDOWN_NOTICE => Ok(MsgFunction::GspLockdownNotice),
            r570::NV_VGPU_MSG_EVENT_GSP_POST_NOCAT_RECORD => Ok(MsgFunction::GspPostNoCat),
            r570::NV_VGPU_MSG_EVENT_MMU_FAULT_QUEUED => Ok(MsgFunction::MmuFaultQueued),
            r570::NV_VGPU_MSG_EVENT_OS_ERROR_LOG => Ok(MsgFunction::OsErrorLog),
            r570::NV_VGPU_MSG_EVENT_POST_EVENT => Ok(MsgFunction::PostEvent),
            r570::NV_VGPU_MSG_EVENT_RC_TRIGGERED => Ok(MsgFunction::RcTriggered),
            r570::NV_VGPU_MSG_EVENT_UCODE_LIBOS_PRINT => Ok(MsgFunction::UcodeLibOsPrint),
            _ => Err(EINVAL),
        }
    }
}

impl MsgFunction {
    /// Returns true if this is a GSP-initiated async event (NV_VGPU_MSG_EVENT_*), as opposed to
    /// a command response (NV_VGPU_MSG_FUNCTION_*).
    pub(crate) fn is_event(&self) -> bool {
        matches!(
            self,
            Self::GspInitDone
                | Self::GspLockdownNotice
                | Self::PostEvent
                | Self::RcTriggered
                | Self::MmuFaultQueued
                | Self::OsErrorLog
                | Self::GspPostNoCat
                | Self::UcodeLibOsPrint //
        )
    }
}

impl From<MsgFunction> for u32 {
    fn from(value: MsgFunction) -> Self {
        // CAST: `MsgFunction` is `repr(u32)` and can thus be cast losslessly.
        value as u32
    }
}

/// Struct containing the arguments required to pass a memory buffer to the GSP
/// for use during initialisation.
///
/// The GSP only understands 4K pages (GSP_PAGE_SIZE), so even if the kernel is
/// configured for a larger page size (e.g. 64K pages), we need to give
/// the GSP an array of 4K pages. Since we only create physically contiguous
/// buffers the math to calculate the addresses is simple.
///
/// The buffers must be a multiple of GSP_PAGE_SIZE.  GSP-RM also currently
/// ignores the @kind field for LOGINIT, LOGINTR, and LOGRM, but expects the
/// buffers to be physically contiguous anyway.
///
/// The memory allocated for the arguments must remain until the GSP sends the
/// init_done RPC.
#[repr(transparent)]
pub(crate) struct LibosMemoryRegionInitArgument {
    inner: r570::LibosMemoryRegionInitArgument,
}

// SAFETY: Padding is explicit and does not contain uninitialized data.
unsafe impl AsBytes for LibosMemoryRegionInitArgument {}

// SAFETY: This struct only contains integer types for which all bit patterns
// are valid.
unsafe impl FromBytes for LibosMemoryRegionInitArgument {}

impl LibosMemoryRegionInitArgument {
    pub(crate) fn new<'a, A: AsBytes + FromBytes + KnownSize + ?Sized>(
        name: &'static str,
        obj: &'a Coherent<A>,
    ) -> impl Init<Self> + 'a {
        /// Generates the `ID8` identifier required for some GSP objects.
        fn id8(name: &str) -> u64 {
            let mut bytes = [0u8; core::mem::size_of::<u64>()];

            for (c, b) in name.bytes().rev().zip(&mut bytes) {
                *b = c;
            }

            u64::from_ne_bytes(bytes)
        }

        #[allow(non_snake_case)]
        let init_inner = init!(r570::LibosMemoryRegionInitArgument {
            id8: id8(name),
            pa: obj.dma_handle(),
            size: num::usize_as_u64(obj.size()),
            kind: num::u32_into_u8::<{ r570::LibosMemoryRegionKind_LIBOS_MEMORY_REGION_CONTIGUOUS }>(
            ),
            loc: num::u32_into_u8::<{ r570::LibosMemoryRegionLoc_LIBOS_MEMORY_REGION_LOC_SYSMEM }>(
            ),
            ..Zeroable::init_zeroed()
        });

        init!(LibosMemoryRegionInitArgument {
            inner <- init_inner,
        })
    }
}

/// TX header for setting up a message queue with the GSP.
#[repr(transparent)]
pub(crate) struct MsgqTxHeader(r570::msgqTxHeader);

impl MsgqTxHeader {
    /// Create a new v0 TX queue header.
    ///
    /// # Arguments
    ///
    /// * `msgq_size` - Total size of the message queue structure, in bytes.
    /// * `rx_hdr_offset` - Offset, in bytes, of the start of the RX header in the message queue
    ///   structure.
    /// * `msg_count` - Number of messages that can be sent, i.e. the number of memory pages
    ///   allocated for the message queue in the message queue structure.
    #[expect(dead_code)]
    pub(crate) fn new(msgq_size: u32, rx_hdr_offset: u32, msg_count: u32) -> Self {
        Self(r570::msgqTxHeader {
            version: 0,
            size: msgq_size,
            msgSize: num::usize_into_u32::<GSP_PAGE_SIZE>(),
            msgCount: msg_count,
            writePtr: 0,
            flags: 1,
            rxHdrOff: rx_hdr_offset,
            entryOff: num::usize_into_u32::<GSP_PAGE_SIZE>(),
        })
    }

    /// Returns the value of the write pointer for this queue.
    pub(crate) fn write_ptr<T: KnownSize + ?Sized>(this: io::View<'_, Coherent<T>, Self>) -> u32 {
        io_read!(this, .0.writePtr)
    }

    /// Sets the value of the write pointer for this queue.
    pub(crate) fn set_write_ptr<T: KnownSize + ?Sized>(
        this: io::View<'_, Coherent<T>, Self>,
        val: u32,
    ) {
        io_write!(this, .0.writePtr, val)
    }

    /// Create a new v2 TX queue header.
    ///
    /// The msgq v2 layout drops the in-memory `writePtr`/`rxHdrOff`/`flags`
    /// fields (the ring pointers live in BAR0 registers) and splits the
    /// `version` word into `versionMajor`/`versionMinor`. The two layouts
    /// occupy the same 32 bytes, so the wrapper byte-aliases either one.
    ///
    /// # Arguments
    ///
    /// * `msgq_size` - Total size of the message queue structure, in bytes.
    /// * `msg_size` - Size of each message slot, in bytes.
    /// * `msg_count` - Number of message slots in the ring.
    /// * `entry_off` - Byte offset from the start of the queue at which the
    ///   message data array begins.
    pub(crate) fn new_v2(msgq_size: u32, msg_size: u32, msg_count: u32, entry_off: u32) -> Self {
        let v2 = r000::msgqTxHeader {
            versionMajor: 2,
            versionMinor: 0,
            size: msgq_size,
            msgSize: msg_size,
            msgCount: msg_count,
            entryOff: entry_off,
            reserved: [0; 3],
        };
        // SAFETY: `r000::msgqTxHeader` and `r570::msgqTxHeader` are both
        // `repr(C)` POD with size 32 bytes. The wrapper holds a 32-byte
        // chunk that the GSP reads as v2 once handed off via `dma_write!`.
        Self(unsafe { core::mem::transmute::<r000::msgqTxHeader, r570::msgqTxHeader>(v2) })
    }
}

// SAFETY: Padding is explicit and does not contain uninitialized data.
unsafe impl AsBytes for MsgqTxHeader {}

/// RX header for setting up a message queue with the GSP.
#[repr(transparent)]
pub(crate) struct MsgqRxHeader(r570::msgqRxHeader);

/// Header for the message RX queue.
#[expect(dead_code)]
impl MsgqRxHeader {
    /// Creates a new RX queue header.
    pub(crate) fn new() -> Self {
        Self(Default::default())
    }

    /// Returns the value of the read pointer for this queue.
    pub(crate) fn read_ptr<T: KnownSize + ?Sized>(this: io::View<'_, Coherent<T>, Self>) -> u32 {
        io_read!(this, .0.readPtr)
    }

    /// Sets the value of the read pointer for this queue.
    pub(crate) fn set_read_ptr<T: KnownSize + ?Sized>(
        this: io::View<'_, Coherent<T>, Self>,
        val: u32,
    ) {
        io_write!(this, .0.readPtr, val)
    }
}

// SAFETY: Padding is explicit and does not contain uninitialized data.
unsafe impl AsBytes for MsgqRxHeader {}

bitfield! {
    struct MsgHeaderVersion(u32) {
        31:24 major;
        23:16 minor;
    }
}

impl MsgHeaderVersion {
    const MAJOR_TOT: u8 = 3;
    const MINOR_TOT: u8 = 0;

    fn new() -> Self {
        Self::from_raw(0)
            .with_major(Self::MAJOR_TOT)
            .with_minor(Self::MINOR_TOT)
    }
}

impl r000::rpc_message_header_v {
    fn init(cmd_size: usize, function: MsgFunction, sequence: u32) -> impl Init<Self, Error> {
        type RpcMessageHeader = r000::rpc_message_header_v;

        try_init!(RpcMessageHeader {
            header_version: MsgHeaderVersion::new().into(),
            signature: r570::NV_VGPU_MSG_SIGNATURE_VALID,
            function: function.into(),
            length: size_of::<Self>()
                .checked_add(cmd_size)
                .ok_or(EOVERFLOW)
                .and_then(|v| v.try_into().map_err(|_| EINVAL))?,
            rpc_result: 0xffffffff,
            rpc_result_private: 0xffffffff,
            sequence,
            ..Zeroable::init_zeroed()
        })
    }
}

/// MCTP magic value identifying valid GSP message queue elements.
pub(super) const MCTP_MAGIC: u32 = 0x4D43_5450;

/// GSP Message Element (r000 MCTP/NVDM format).
///
/// This is the transport-layer header for messages exchanged with GSP-RM.
/// r000 firmware uses MCTP/NVDM framing instead of the r570 `GSP_MSG_QUEUE_ELEMENT`.
#[repr(C)]
pub(crate) struct GspMsgElement {
    mctp_magic: u32,
    mctp_payload_size: u32,
    mctp_header: u32,
    nvdm_header: u32,
    nvdm_payload_size: u32,
    reserved: u32,
    rpc: r000::rpc_message_header_v,
}

impl GspMsgElement {
    /// Creates a new message element.
    ///
    /// # Arguments
    ///
    /// * `rpc_seq` - RPC-level sequence number for `rpc_message_header_v.sequence`.
    ///   Set to 0 for async (fire-and-forget) commands, or to the sequence counter
    ///   for command/response pairs.
    /// * `cmd_size` - Size of the command (not including the message element), in bytes.
    /// * `function` - Function of the message.
    pub(crate) fn init(
        rpc_seq: u32,
        cmd_size: usize,
        function: MsgFunction,
    ) -> impl Init<Self, Error> {
        use crate::mctp;
        type RpcMessageHeader = r000::rpc_message_header_v;

        try_init!(GspMsgElement {
            mctp_magic: MCTP_MAGIC,
            mctp_payload_size: (size_of::<Self>() - 8 + cmd_size) as u32,
            mctp_header: mctp::TransportHeader::new(true, true).into(),
            nvdm_header: mctp::NvdmHeader::new(mctp::nvdm_type::RM_RPC).into(),
            nvdm_payload_size: (size_of::<RpcMessageHeader>() + cmd_size) as u32,
            reserved: 0u32,
            rpc <- RpcMessageHeader::init(cmd_size, function, rpc_seq),
        })
    }

    /// Returns the length of the message's payload (command data after the RPC header).
    pub(crate) fn payload_length(&self) -> usize {
        num::u32_as_usize(self.nvdm_payload_size)
            .saturating_sub(size_of::<r000::rpc_message_header_v>())
    }

    /// Returns the total length of the message, transport and RPC headers included.
    pub(crate) fn length(&self) -> usize {
        8 + num::u32_as_usize(self.mctp_payload_size)
    }

    /// Returns `true` if the MCTP magic field contains the expected value.
    pub(crate) fn has_valid_magic(&self) -> bool {
        self.mctp_magic == MCTP_MAGIC
    }

    // Returns the sequence number of the message.
    pub(crate) fn sequence(&self) -> u32 {
        self.rpc.sequence
    }

    // Returns the function of the message, if it is valid, or the invalid function number as an
    // error.
    pub(crate) fn function(&self) -> Result<MsgFunction, u32> {
        self.rpc.function.try_into().map_err(|_| self.rpc.function)
    }

    // Returns the number of elements (i.e. memory pages) used by this message.
    pub(crate) fn element_count(&self) -> u32 {
        self.length().div_ceil(GSP_PAGE_SIZE) as u32
    }
}

// SAFETY: All fields are integer types or contain only integer types, with no
// uninitialized padding bytes.
unsafe impl AsBytes for GspMsgElement {}

// SAFETY: All fields are integer types for which all bit patterns are valid.
unsafe impl FromBytes for GspMsgElement {}

/// GMC API message header.
///
/// Matches the `GMCAPI_HEADER` struct from Open RM (`gmcapi_base.h`). The two
/// middle `u32` fields form a union in C: for requests they carry `size` and
/// `max_response_size`, for responses they carry `status` and `size`.
///
/// Open RM reference: `interface/gmcapi/gmcapi_base.h`
#[repr(C)]
#[derive(Zeroable)]
pub(crate) struct GmcApiHeader {
    /// GMC command identifier (from `GMCAPI_COMMANDS`).
    pub(crate) command_id: u32,
    reserved1: u32,
    /// Sequence number for matching requests to responses.
    pub(crate) sequence: u64,
    /// Request: payload size in bytes. Response: `NV_STATUS` code.
    pub(crate) size_or_status: u32,
    /// Request: maximum response size. Response: actual response size.
    pub(crate) max_resp_or_resp_size: u32,
    reserved2: u64,
}

static_assert!(size_of::<GmcApiHeader>() == 32);
static_assert!(core::mem::offset_of!(GmcApiHeader, command_id) == 0);
static_assert!(core::mem::offset_of!(GmcApiHeader, sequence) == 8);
static_assert!(core::mem::offset_of!(GmcApiHeader, size_or_status) == 16);
static_assert!(core::mem::offset_of!(GmcApiHeader, max_resp_or_resp_size) == 20);
static_assert!(core::mem::offset_of!(GmcApiHeader, reserved2) == 24);

// SAFETY: All fields are integer types with no uninitialized padding bytes.
unsafe impl AsBytes for GmcApiHeader {}

// SAFETY: All fields are integer types for which all bit patterns are valid.
unsafe impl FromBytes for GmcApiHeader {}

/// GSP GMC Message Element (r000 MCTP/NVDM format).
///
/// Same transport prefix as [`GspMsgElement`] but with a [`GmcApiHeader`]
/// instead of `rpc_message_header_v`. The `nvdm_header` DWORD carries
/// `nvdm_type::GMCAPI` (`0x26`) to route the message through the GSP GMC
/// dispatch path.
#[repr(C)]
pub(crate) struct GspGmcMsgElement {
    mctp_magic: u32,
    mctp_payload_size: u32,
    mctp_header: u32,
    nvdm_header: u32,
    nvdm_payload_size: u32,
    reserved: u32,
    pub(crate) gmc: GmcApiHeader,
}

static_assert!(
    core::mem::offset_of!(GspGmcMsgElement, mctp_magic)
        == core::mem::offset_of!(GspMsgElement, mctp_magic)
);
static_assert!(
    core::mem::offset_of!(GspGmcMsgElement, mctp_payload_size)
        == core::mem::offset_of!(GspMsgElement, mctp_payload_size)
);
static_assert!(
    core::mem::offset_of!(GspGmcMsgElement, mctp_header)
        == core::mem::offset_of!(GspMsgElement, mctp_header)
);
static_assert!(
    core::mem::offset_of!(GspGmcMsgElement, nvdm_header)
        == core::mem::offset_of!(GspMsgElement, nvdm_header)
);
static_assert!(
    core::mem::offset_of!(GspGmcMsgElement, nvdm_payload_size)
        == core::mem::offset_of!(GspMsgElement, nvdm_payload_size)
);
static_assert!(
    core::mem::offset_of!(GspGmcMsgElement, reserved)
        == core::mem::offset_of!(GspMsgElement, reserved)
);

impl GspGmcMsgElement {
    /// Creates a new GMC message element.
    ///
    /// `payload_size` is the size of the command payload following this header.
    /// `command_id` is the GMC command identifier.
    /// `sequence` is the sequence number for request/response matching.
    /// `max_response_size` is the maximum expected response payload size.
    pub(crate) fn init(
        command_id: u32,
        sequence: u64,
        payload_size: usize,
        max_response_size: u32,
    ) -> impl Init<Self, Error> {
        use crate::mctp;

        try_init!(GspGmcMsgElement {
            mctp_magic: MCTP_MAGIC,
            mctp_payload_size: (size_of::<Self>() - 8 + payload_size) as u32,
            mctp_header: mctp::TransportHeader::new(true, true).into(),
            nvdm_header: mctp::NvdmHeader::new(mctp::nvdm_type::GMCAPI).into(),
            nvdm_payload_size: (size_of::<GmcApiHeader>() + payload_size) as u32,
            reserved: 0u32,
            gmc: GmcApiHeader {
                command_id,
                reserved1: 0,
                sequence,
                size_or_status: payload_size as u32,
                max_resp_or_resp_size: max_response_size,
                reserved2: 0,
            },
        })
    }

    /// Returns the length of the response payload (data after the [`GmcApiHeader`]).
    pub(crate) fn payload_length(&self) -> usize {
        num::u32_as_usize(self.nvdm_payload_size).saturating_sub(size_of::<GmcApiHeader>())
    }

    /// Returns the total length of the message, transport and GMC headers included.
    pub(crate) fn length(&self) -> usize {
        8 + num::u32_as_usize(self.mctp_payload_size)
    }

    /// Returns `true` if the MCTP magic field contains the expected value.
    pub(crate) fn has_valid_magic(&self) -> bool {
        self.mctp_magic == MCTP_MAGIC
    }

    /// Returns the number of elements (i.e. memory pages) used by this message.
    pub(crate) fn element_count(&self) -> u32 {
        self.length().div_ceil(GSP_PAGE_SIZE) as u32
    }
}

// SAFETY: All fields are integer types with no uninitialized padding bytes.
unsafe impl AsBytes for GspGmcMsgElement {}

// SAFETY: All fields are integer types for which all bit patterns are valid.
unsafe impl FromBytes for GspGmcMsgElement {}

/// Optional bindata (ucodes) firmware info for GSP startup arguments.
pub(crate) struct BindataArgs {
    /// DMA address of the radix3 level 0 page table for the bindata firmware.
    pub(crate) radix3: u64,
    /// Size in bytes of the bindata firmware.
    pub(crate) size: u64,
}

/// Magic value for the `GSP_ARGUMENTS_CACHED` header.
const GSP_ARGUMENTS_MAGIC_VALUE: u32 = 0x2050_5347;

/// Flag indicating the GSP stack should be placed in DMEM.
const GSP_ARGUMENTS_FLAG_STACK_IN_DMEM: u64 = 0x02;

/// Arguments for GSP startup.
#[repr(transparent)]
#[derive(Zeroable)]
pub(crate) struct GspArgumentsCached {
    inner: r000::GSP_ARGUMENTS_CACHED,
}

impl GspArgumentsCached {
    /// Creates the arguments for starting the GSP up using `cmdq` as its command queue.
    ///
    /// If `bindata` is provided, the GSP will be told where to find the ucodes firmware.
    ///
    /// `state_monitor` is the RM state monitor buffer (4KB). GSP-RM maps this buffer
    /// during init for diagnostics.
    pub(crate) fn new(
        cmdq: &Cmdq,
        bindata: Option<&BindataArgs>,
        state_monitor: &Coherent<[u8; GSP_PAGE_SIZE]>,
    ) -> Self {
        let mut args = r000::GSP_ARGUMENTS_CACHED {
            magic: GSP_ARGUMENTS_MAGIC_VALUE,
            size: size_of::<r000::GSP_ARGUMENTS_CACHED>() as u16,
            flags: GSP_ARGUMENTS_FLAG_STACK_IN_DMEM,
            messageQueueInitArguments: MessageQueueInitArguments::new(cmdq),
            ..Default::default()
        };

        if let Some(bindata) = bindata {
            args.bindataArgs.radix3 = bindata.radix3;
            args.bindataArgs.size = bindata.size;
        }

        args.rmStateMonitorBufferArgs.pa = state_monitor.dma_handle();
        args.rmStateMonitorBufferArgs.size = state_monitor.size() as u64;

        Self { inner: args }
    }
}

// SAFETY: Padding is explicit and will not contain uninitialized data.
unsafe impl AsBytes for GspArgumentsCached {}

/// On Turing and GA100, the entries in the `LibosMemoryRegionInitArgument`
/// must all be a multiple of GSP_PAGE_SIZE in size, so add padding to force it
/// to that size.
#[repr(C)]
#[derive(Zeroable)]
pub(crate) struct GspArgumentsPadded {
    pub(crate) inner: GspArgumentsCached,
    _padding: [u8; GSP_PAGE_SIZE - core::mem::size_of::<r000::GSP_ARGUMENTS_CACHED>()],
}

impl GspArgumentsPadded {
    pub(crate) fn new<'a>(
        cmdq: &'a Cmdq,
        bindata: Option<&'a BindataArgs>,
        state_monitor: &'a Coherent<[u8; GSP_PAGE_SIZE]>,
    ) -> impl Init<Self> + 'a {
        init!(GspArgumentsPadded {
            inner: GspArgumentsCached::new(cmdq, bindata, state_monitor),
            ..Zeroable::init_zeroed()
        })
    }
}

// SAFETY: Padding is explicit and will not contain uninitialized data.
unsafe impl AsBytes for GspArgumentsPadded {}

// SAFETY: This struct only contains integer types for which all bit patterns
// are valid.
unsafe impl FromBytes for GspArgumentsPadded {}

/// Init arguments for the message queue.
type MessageQueueInitArguments = r000::MESSAGE_QUEUE_INIT_ARGUMENTS;

impl MessageQueueInitArguments {
    /// Creates a new init arguments structure for `cmdq`.
    #[allow(non_snake_case)]
    fn new(cmdq: &Cmdq) -> Self {
        MessageQueueInitArguments {
            sharedMemPhysAddr: cmdq.dma_handle,
            pageTableEntryCount: num::usize_into_u32::<{ Cmdq::NUM_PTES }>(),
            cmdQueueOffset: num::usize_as_u64(Cmdq::CMDQ_OFFSET),
            statQueueOffset: num::usize_as_u64(Cmdq::STATQ_OFFSET),

            queueElementHdrSize: (size_of::<GspMsgElement>()
                - size_of::<r000::rpc_message_header_v>()) as u32,
            queueElementSizeMin: GSP_PAGE_SIZE as u32,
            queueElementSizeMax: (GSP_PAGE_SIZE * 16) as u32,
            queueHeaderAlign: 4,
            queueElementAlign: GSP_PAGE_SHIFT as u32,

            ..Default::default()
        }
    }
}
