// SPDX-License-Identifier: GPL-2.0

//! MCTP/NVDM protocol support for inter-processor communication.
//!
//! MCTP (Management Component Transport Protocol, DMTF DSP0236) provides a
//! transport-agnostic message exchange format. NVDM (NVIDIA Data Model) is a
//! vendor-defined message layer on top of MCTP.
//!
//! Multiple GPU subsystems (FSP, GSP, SEC2) use MCTP/NVDM framing. This module
//! provides the shared header types and helpers for all of them.
//!
//! Open RM reference:
//!   `arch/nvalloc/common/inc/mctp_format.h`
//!   `arch/nvalloc/common/inc/nvdm_format.h`

use kernel::bitfield;

// MCTP transport header bit layout (DMTF DSP0236):
//
//  31  30  29:28  27  26:24  23:16  15:8  7:4   3:0
// [SOM|EOM| SEQ  | TO| TAG | SEID | DEID| RSVD| VER]
//
// TAG, TO, and RSVD are always zero in current usage.
bitfield! {
    pub(crate) struct TransportHeader(u32) {
        31:31 som;
        30:30 eom;
        29:28 seq;
        23:16 seid;
        15:8  deid;
        3:0   version;
    }
}

impl TransportHeader {
    /// Create a new MCTP transport header.
    ///
    /// TAG and TO fields are always zero (not needed when messages are
    /// single-packet and do not use request/response tag correlation).
    pub(crate) fn new(som: bool, eom: bool) -> Self {
        Self::from_raw(0)
            .with_const_version::<1>()
            .with_som(som)
            .with_eom(eom)
    }
}

// NVDM over MCTP message header bit layout:
//
//  31:24      23:8        7     6:0
// [NVDM_TYPE | VENDOR_ID | IC | MSG_TYPE]
//
// IC (Instance ID) is always zero in current usage.
bitfield! {
    pub(crate) struct NvdmHeader(u32) {
        31:24 nvdm_type;
        23:8  vendor_id;
        6:0   msg_type;
    }
}

/// MCTP message type for vendor-defined PCI messages (DMTF DSP0236).
pub(crate) const MSG_TYPE_VENDOR_PCI: u8 = 0x7e;

/// NVIDIA PCI Vendor ID.
pub(crate) const VENDOR_ID_NV: u16 = 0x10de;

impl NvdmHeader {
    /// Create a new NVDM header for NVIDIA vendor-defined PCI messages.
    ///
    /// Sets MCTP message type to Vendor Defined PCI (0x7E) and vendor ID to
    /// NVIDIA (0x10DE). The IC (Instance ID) field is always zero.
    pub(crate) fn new(nvdm_type: u8) -> Self {
        Self::from_raw(0)
            .with_const_msg_type::<{ MSG_TYPE_VENDOR_PCI as u32 }>()
            .with_vendor_id(VENDOR_ID_NV)
            .with_nvdm_type(nvdm_type)
    }
}

/// NVDM message type constants.
///
/// These values come from a shared namespace across all GPU subsystems.
/// Open RM reference: `arch/nvalloc/common/inc/nvdm_format.h`
pub(crate) mod nvdm_type {
    /// PRC (Product Reconfiguration Control) message.
    pub(crate) const PRC: u8 = 0x13;
    /// Chain of Trust (FSP boot).
    pub(crate) const COT: u8 = 0x14;
    /// FSP/SEC2 command response.
    pub(crate) const FSP_RESPONSE: u8 = 0x15;
    /// RM RPC message (GSP command queue).
    pub(crate) const RM_RPC: u8 = 0x25;
    /// GMC API message (GSP command queue).
    pub(crate) const GMCAPI: u8 = 0x26;
}
