// SPDX-License-Identifier: GPL-2.0

//! MMU v3 page table types for Hopper and later GPUs.
//!
//! This module defines MMU version 3 specific types (Hopper and later GPUs).
//!
//! Key differences from MMU v2:
//! - Unified 40-bit address field for all apertures (v2 had separate sys/vid fields).
//! - PCF (Page Classification Field) replaces separate privilege/RO/atomic/cache bits.
//! - KIND field is 4 bits (not 8).
//! - IS_PTE bit in PDE to support large pages directly.
//! - No COMPTAGLINE field (compression handled differently in v3).
//! - No separate ENCRYPTED bit.
//!
//! Bit field layouts derived from the NVIDIA OpenRM documentation:
//! `open-gpu-kernel-modules/src/common/inc/swref/published/hopper/gh100/dev_mmu.h`

#![expect(dead_code)]

use kernel::bitfield;
use kernel::num::Bounded;
use kernel::prelude::*;
use pin_init::Zeroable;

use super::{
    AperturePde,
    AperturePte,
    DualPdeOps,
    PageTableLevel,
    PdeOps,
    PteOps,
    VaLevelIndex, //
};
use crate::mm::{
    Pfn,
    VirtualAddress,
    VramAddress, //
};

// Bounded to version 3 Pfn conversion.
impl_pfn_bounded!(40);

bitfield! {
    /// MMU v3 57-bit virtual address layout.
    pub(super) struct VirtualAddressV3(u64) {
        /// Page offset [11:0].
        11:0    offset;
        /// PT index [20:12].
        20:12   pt_idx;
        /// PDE0 index [28:21].
        28:21   pde0_idx;
        /// PDE1 index [37:29].
        37:29   pde1_idx;
        /// PDE2 index [46:38].
        46:38   pde2_idx;
        /// PDE3 index [55:47].
        55:47   pde3_idx;
        /// PDE4 index [56].
        56:56   pde4_idx;
    }
}

impl VirtualAddressV3 {
    /// Create a [`VirtualAddressV3`] from a [`VirtualAddress`].
    pub(super) fn new(va: VirtualAddress) -> Self {
        Self::from_raw(va.raw_u64())
    }
}

impl VaLevelIndex for VirtualAddressV3 {
    fn level_index(&self, level: u64) -> u64 {
        match level {
            0 => self.pde4_idx().get(),
            1 => self.pde3_idx().get(),
            2 => self.pde2_idx().get(),
            3 => self.pde1_idx().get(),
            4 => self.pde0_idx().get(),
            5 => self.pt_idx().get(),
            _ => 0,
        }
    }
}

/// PDE levels for MMU v3 (6-level hierarchy).
pub(super) const PDE_LEVELS: &[PageTableLevel] = &[
    PageTableLevel::Pdb,
    PageTableLevel::L1,
    PageTableLevel::L2,
    PageTableLevel::L3,
    PageTableLevel::L4,
];

/// PTE level for MMU v3.
pub(super) const PTE_LEVEL: PageTableLevel = PageTableLevel::L5;

/// Dual PDE level for MMU v3 (128-bit entries).
pub(super) const DUAL_PDE_LEVEL: PageTableLevel = PageTableLevel::L4;

bitfield! {
    /// Page Classification Field for PTEs (5 bits) in MMU v3.
    pub(in crate::mm) struct PtePcf(u8) {
        /// Bypass L2 cache (0=cached, 1=bypass).
        0:0     uncached;
        /// Access counting disabled (0=enabled, 1=disabled).
        1:1     acd;
        /// Read-only access (0=read-write, 1=read-only).
        2:2     read_only;
        /// Atomics disabled (0=enabled, 1=disabled).
        3:3     no_atomic;
        /// Privileged access only (0=regular, 1=privileged).
        4:4     privileged;
    }
}

impl PtePcf {
    /// Create PCF for read-write mapping (cached, no atomics, regular mode).
    fn rw() -> Self {
        Self::zeroed().with_no_atomic(true)
    }

    /// Create PCF for read-only mapping (cached, no atomics, regular mode).
    fn ro() -> Self {
        Self::zeroed().with_read_only(true).with_no_atomic(true)
    }

    /// Get the raw `u8` value.
    fn raw_u8(&self) -> u8 {
        self.into_raw()
    }
}

impl From<Bounded<u64, 5>> for PtePcf {
    fn from(val: Bounded<u64, 5>) -> Self {
        Self::from_raw(u8::from(val))
    }
}

impl From<PtePcf> for Bounded<u64, 5> {
    fn from(pcf: PtePcf) -> Self {
        Bounded::from_expr(u64::from(pcf.into_raw()) & 0x1F)
    }
}

bitfield! {
    /// Page Classification Field for PDEs (3 bits) in MMU v3.
    ///
    /// Controls Address Translation Services (ATS) and caching.
    pub(in crate::mm) struct PdePcf(u8) {
        /// Bypass L2 cache (0=cached, 1=bypass).
        0:0     uncached;
        /// ATS disabled (0=enabled, 1=disabled).
        1:1     no_ats;
    }
}

impl PdePcf {
    /// Create PCF for cached mapping with ATS enabled (default).
    fn cached() -> Self {
        Self::zeroed()
    }

    /// Get the raw `u8` value.
    fn raw_u8(&self) -> u8 {
        self.into_raw()
    }
}

impl From<Bounded<u64, 3>> for PdePcf {
    fn from(val: Bounded<u64, 3>) -> Self {
        Self::from_raw(u8::from(val))
    }
}

impl From<PdePcf> for Bounded<u64, 3> {
    fn from(pcf: PdePcf) -> Self {
        Bounded::from_expr(u64::from(pcf.into_raw()) & 0x7)
    }
}

bitfield! {
    /// Page Table Entry for MMU v3.
    pub(in crate::mm) struct Pte(u64) {
        /// Entry is valid.
        0:0     valid;
        /// Memory aperture type.
        2:1     aperture => AperturePte;
        /// Page Classification Field.
        7:3     pcf => PtePcf;
        /// Surface kind (4 bits, 0x0=pitch, 0xF=invalid).
        11:8    kind;
        /// Physical frame number (for all apertures).
        51:12   frame_number => Pfn;
        /// Peer GPU ID for peer memory (0-7).
        63:61   peer_id;
    }
}

impl Pte {
    /// Create a PTE from a `u64` value.
    pub(super) fn new_raw(val: u64) -> Self {
        Self::from_raw(val)
    }

    /// Create a valid PTE for video memory.
    fn new_vram_inner(frame: Pfn, writable: bool) -> Self {
        let pcf = if writable { PtePcf::rw() } else { PtePcf::ro() };
        Self::zeroed()
            .with_valid(true)
            .with_aperture(AperturePte::VideoMemory)
            .with_pcf(pcf)
            .with_frame_number(frame)
    }
}

impl PteOps for Pte {
    fn new(val: u64) -> Self {
        Self::from_raw(val)
    }

    fn invalid() -> Self {
        Self::zeroed()
    }

    fn new_vram(pfn: Pfn, writable: bool) -> Self {
        Self::new_vram_inner(pfn, writable)
    }

    fn is_valid(&self) -> bool {
        self.valid().into_bool()
    }

    fn frame_number(&self) -> Pfn {
        Pte::frame_number(*self)
    }

    fn raw_u64(&self) -> u64 {
        self.into_raw()
    }
}

bitfield! {
    /// Page Directory Entry for MMU v3 (Hopper+).
    ///
    /// Note: v3 uses a unified 40-bit address field (v2 had separate sys/vid address fields).
    pub(in crate::mm) struct Pde(u64) {
        /// Entry is a PTE (0=PDE, 1=large page PTE).
        0:0     is_pte;
        /// Memory aperture type.
        2:1     aperture => AperturePde;
        /// Page Classification Field (3 bits for PDE).
        5:3     pcf => PdePcf;
        /// Table frame number (40-bit unified address).
        51:12   table_frame => Pfn;
    }
}

impl Pde {
    /// Create a PDE from a `u64` value.
    pub(super) fn new_raw(val: u64) -> Self {
        Self::from_raw(val)
    }

    /// Create a valid PDE pointing to a page table in video memory.
    fn new_vram_inner(table_pfn: Pfn) -> Self {
        Self::zeroed()
            .with_is_pte(false)
            .with_aperture(AperturePde::VideoMemory)
            .with_table_frame(table_pfn)
    }
}

impl PdeOps for Pde {
    fn new(val: u64) -> Self {
        Self::from_raw(val)
    }

    fn new_vram(table_pfn: Pfn) -> Self {
        Self::new_vram_inner(table_pfn)
    }

    fn invalid() -> Self {
        Self::zeroed().with_aperture(AperturePde::Invalid)
    }

    fn is_valid(&self) -> bool {
        Pde::aperture(*self) != AperturePde::Invalid
    }

    fn aperture(&self) -> AperturePde {
        Pde::aperture(*self)
    }

    fn table_vram_address(&self) -> VramAddress {
        debug_assert!(
            Pde::aperture(*self) == AperturePde::VideoMemory,
            "table_vram_address called on non-VRAM PDE (aperture: {:?})",
            Pde::aperture(*self)
        );
        VramAddress::from(self.table_frame())
    }

    fn raw_u64(&self) -> u64 {
        self.into_raw()
    }
}

bitfield! {
    /// Big Page Table pointer in Dual PDE (MMU v3).
    ///
    /// 64-bit lower word of the 128-bit Dual PDE.
    pub(super) struct DualPdeBig(u64) {
        /// Entry is a PTE (for large pages).
        0:0     is_pte;
        /// Memory aperture type.
        2:1     aperture => AperturePde;
        /// Page Classification Field.
        5:3     pcf => PdePcf;
        /// Table frame (table address 256-byte aligned).
        51:8    table_frame;
    }
}

impl DualPdeBig {
    /// Create a big page table pointer from a `u64` value.
    fn new(val: u64) -> Self {
        Self::from_raw(val)
    }

    /// Create an invalid big page table pointer.
    fn invalid() -> Self {
        Self::zeroed().with_aperture(AperturePde::Invalid)
    }

    /// Create a valid big PDE pointing to a page table in video memory.
    fn new_vram(table_addr: VramAddress) -> Result<Self> {
        // Big page table addresses must be 256-byte aligned (shift 8).
        if table_addr.raw_u64() & 0xFF != 0 {
            return Err(EINVAL);
        }

        let table_frame = Bounded::from_expr(table_addr.raw_u64() >> 8);
        Ok(Self::zeroed()
            .with_is_pte(false)
            .with_aperture(AperturePde::VideoMemory)
            .with_table_frame(table_frame))
    }

    /// Check if this big PDE is valid.
    fn is_valid(&self) -> bool {
        self.aperture() != AperturePde::Invalid
    }

    /// Get the VRAM address of the big page table.
    fn table_vram_address(&self) -> VramAddress {
        debug_assert!(
            self.aperture() == AperturePde::VideoMemory,
            "table_vram_address called on non-VRAM DualPdeBig (aperture: {:?})",
            self.aperture()
        );
        VramAddress::new(self.table_frame().get() << 8)
    }

    /// Get the raw `u64` value.
    pub(super) fn raw_u64(&self) -> u64 {
        self.into_raw()
    }
}

/// Dual PDE at Level 4 for MMU v3 - 128-bit entry.
///
/// Contains both big (64KB) and small (4KB) page table pointers:
/// - Lower 64 bits: Big Page Table pointer.
/// - Upper 64 bits: Small Page Table pointer.
///
/// ## Note
///
/// The big and small page table pointers have different address layouts:
/// - Big address = field value << 8 (256-byte alignment).
/// - Small address = field value << 12 (4KB alignment).
///
/// This is why `DualPdeBig` is a separate type from `Pde`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(in crate::mm) struct DualPde {
    /// Big Page Table pointer.
    pub(super) big: DualPdeBig,
    /// Small Page Table pointer.
    pub(super) small: Pde,
}

// SAFETY: Both `DualPdeBig` and `Pde` fields are `Zeroable` (bitfield types are Zeroable).
unsafe impl Zeroable for DualPde {}

impl DualPde {
    /// Check if the big page table pointer is valid.
    fn has_big(&self) -> bool {
        self.big.is_valid()
    }
}

impl DualPdeOps for DualPde {
    fn new(big: u64, small: u64) -> Self {
        Self {
            big: DualPdeBig::new(big),
            small: PdeOps::new(small),
        }
    }

    fn new_small(table_pfn: Pfn) -> Self {
        Self {
            big: DualPdeBig::invalid(),
            small: PdeOps::new_vram(table_pfn),
        }
    }

    fn has_small(&self) -> bool {
        PdeOps::is_valid(&self.small)
    }

    fn small_vram_address(&self) -> VramAddress {
        PdeOps::table_vram_address(&self.small)
    }

    fn big_raw_u64(&self) -> u64 {
        self.big.raw_u64()
    }

    fn small_raw_u64(&self) -> u64 {
        PdeOps::raw_u64(&self.small)
    }
}
