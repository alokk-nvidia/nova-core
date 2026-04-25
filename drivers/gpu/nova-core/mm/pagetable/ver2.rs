// SPDX-License-Identifier: GPL-2.0

//! MMU v2 page table types for Turing, Ampere and Ada GPUs.
//!
//! This module defines MMU version 2 specific types (Turing, Ampere and Ada GPUs).
//!
//! Bit field layouts derived from the NVIDIA OpenRM documentation:
//! `open-gpu-kernel-modules/src/common/inc/swref/published/turing/tu102/dev_mmu.h`

#![expect(dead_code)]

use kernel::bitfield;
use kernel::num::Bounded;
use pin_init::Zeroable;

use super::{
    AperturePde,
    AperturePte,
    PageTableLevel,
    VaLevelIndex, //
};
use crate::mm::{
    Pfn,
    VirtualAddress,
    VramAddress, //
};

// Bounded to version 2 Pfn bitfield conversions:
// 25 bits for video memory frame numbers (bits 32:8).
impl_pfn_bounded!(25);
// 46 bits for system memory frame numbers (bits 53:8).
impl_pfn_bounded!(46);

bitfield! {
    /// MMU v2 49-bit virtual address layout.
    pub(super) struct VirtualAddressV2(u64) {
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
        /// PDE3 index [48:47].
        48:47   pde3_idx;
    }
}

impl VirtualAddressV2 {
    /// Create a [`VirtualAddressV2`] from a [`VirtualAddress`].
    pub(super) fn new(va: VirtualAddress) -> Self {
        Self::from_raw(va.raw_u64())
    }
}

impl VaLevelIndex for VirtualAddressV2 {
    fn level_index(&self, level: u64) -> u64 {
        match level {
            0 => self.pde3_idx().get(),
            1 => self.pde2_idx().get(),
            2 => self.pde1_idx().get(),
            3 => self.pde0_idx().get(),
            4 => self.pt_idx().get(),
            _ => 0,
        }
    }
}

/// `PDE` levels for MMU v2 (5-level hierarchy: `PDB` -> `L1` -> `L2` -> `L3` -> `L4`).
pub(super) const PDE_LEVELS: &[PageTableLevel] = &[
    PageTableLevel::Pdb,
    PageTableLevel::L1,
    PageTableLevel::L2,
    PageTableLevel::L3,
];

/// `PTE` level for MMU v2.
pub(super) const PTE_LEVEL: PageTableLevel = PageTableLevel::L4;

/// Dual `PDE` level for MMU v2 (128-bit entries).
pub(super) const DUAL_PDE_LEVEL: PageTableLevel = PageTableLevel::L3;

// Page Table Entry (PTE) for MMU v2 - 64-bit entry at level 4.
bitfield! {
    /// Page Table Entry for MMU v2.
    pub(in crate::mm) struct Pte(u64) {
        /// Entry is valid.
        0:0     valid;
        /// Memory aperture type.
        2:1     aperture => AperturePte;
        /// Volatile (bypass L2 cache).
        3:3     volatile;
        /// Encryption enabled (Confidential Computing).
        4:4     encrypted;
        /// Privileged access only.
        5:5     privilege;
        /// Write protection.
        6:6     read_only;
        /// Atomic operations disabled.
        7:7     atomic_disable;
        /// Frame number for system memory.
        53:8    frame_number_sys => Pfn;
        /// Frame number for video memory.
        32:8    frame_number_vid => Pfn;
        /// Peer GPU ID for peer memory (0-7).
        35:33   peer_id;
        /// Compression tag line bits.
        53:36   comptagline;
        /// Surface kind/format.
        63:56   kind;
    }
}

impl Pte {
    /// Create a `PTE` from a `u64` value.
    pub(super) fn new(val: u64) -> Self {
        Self::from_raw(val)
    }

    /// Create a valid `PTE` for video memory.
    pub(super) fn new_vram(pfn: Pfn, writable: bool) -> Self {
        Self::zeroed()
            .with_valid(true)
            .with_aperture(AperturePte::VideoMemory)
            .with_frame_number_vid(pfn)
            .with_read_only(!writable)
    }

    /// Create an invalid `PTE`.
    pub(super) fn invalid() -> Self {
        Self::zeroed()
    }

    /// Get the frame number based on aperture type.
    pub(super) fn frame_number(&self) -> Pfn {
        match self.aperture() {
            AperturePte::VideoMemory => self.frame_number_vid(),
            _ => self.frame_number_sys(),
        }
    }

    /// Get the raw `u64` value.
    pub(super) fn raw_u64(&self) -> u64 {
        self.into_raw()
    }
}

// Page Directory Entry (PDE) for MMU v2 - 64-bit entry at levels 0-2.
bitfield! {
    /// Page Directory Entry for MMU v2.
    pub(in crate::mm) struct Pde(u64) {
        /// Valid bit (inverted logic).
        0:0     valid_inverted;
        /// Memory aperture type.
        2:1     aperture => AperturePde;
        /// Volatile (bypass L2 cache).
        3:3     volatile;
        /// Disable Address Translation Services.
        5:5     no_ats;
        /// Table frame number for system memory.
        53:8    table_frame_sys => Pfn;
        /// Table frame number for video memory.
        32:8    table_frame_vid => Pfn;
        /// Peer GPU ID (0-7).
        35:33   peer_id;
    }
}

impl Pde {
    /// Create a `PDE` from a `u64` value.
    pub(super) fn new(val: u64) -> Self {
        Self::from_raw(val)
    }

    /// Create a valid `PDE` pointing to a page table in video memory.
    pub(super) fn new_vram(table_pfn: Pfn) -> Self {
        Self::zeroed()
            .with_valid_inverted(false) // 0 = valid
            .with_aperture(AperturePde::VideoMemory)
            .with_table_frame_vid(table_pfn)
    }

    /// Create an invalid `PDE`.
    pub(super) fn invalid() -> Self {
        Self::zeroed()
            .with_valid_inverted(true)
            .with_aperture(AperturePde::Invalid)
    }

    /// Check if this `PDE` is valid.
    pub(super) fn is_valid(&self) -> bool {
        !self.valid_inverted().into_bool() && self.aperture() != AperturePde::Invalid
    }

    /// Get the table frame number based on aperture type.
    fn table_frame(&self) -> Pfn {
        match self.aperture() {
            AperturePde::VideoMemory => self.table_frame_vid(),
            _ => self.table_frame_sys(),
        }
    }

    /// Get the `VRAM` address of the page table.
    pub(super) fn table_vram_address(&self) -> VramAddress {
        debug_assert!(
            self.aperture() == AperturePde::VideoMemory,
            "table_vram_address called on non-VRAM PDE (aperture: {:?})",
            self.aperture()
        );
        VramAddress::from(self.table_frame_vid())
    }

    /// Get the raw `u64` value of the `PDE`.
    pub(super) fn raw_u64(&self) -> u64 {
        self.into_raw()
    }
}

/// Dual `PDE` at Level 3 - 128-bit entry of Large/Small Page Table pointers.
///
/// The dual `PDE` supports both large (64KB) and small (4KB) page tables.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(in crate::mm) struct DualPde {
    /// Large/Big Page Table pointer (lower 64 bits).
    pub(super) big: Pde,
    /// Small Page Table pointer (upper 64 bits).
    pub(super) small: Pde,
}

impl DualPde {
    /// Create a dual `PDE` from raw 128-bit value (two `u64`s).
    pub(super) fn new(big: u64, small: u64) -> Self {
        Self {
            big: Pde::new(big),
            small: Pde::new(small),
        }
    }

    /// Create a dual `PDE` with only the small page table pointer set.
    ///
    /// Note: The big (LPT) portion is set to 0, not `Pde::invalid()`.
    /// According to hardware documentation, clearing bit 0 of the 128-bit
    /// entry makes the PDE behave as a "normal" PDE. Using `Pde::invalid()`
    /// would set bit 0 (valid_inverted), which breaks page table walking.
    pub(super) fn new_small(table_pfn: Pfn) -> Self {
        Self {
            big: Pde::new(0),
            small: Pde::new_vram(table_pfn),
        }
    }

    /// Check if the small page table pointer is valid.
    pub(super) fn has_small(&self) -> bool {
        self.small.is_valid()
    }

    /// Check if the big page table pointer is valid.
    fn has_big(&self) -> bool {
        self.big.is_valid()
    }

    /// Get the small page table `Pfn`.
    fn small_pfn(&self) -> Pfn {
        self.small.table_frame()
    }
}
