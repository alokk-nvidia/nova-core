// SPDX-License-Identifier: GPL-2.0

//! Memory management subsystems for nova-core.

#![expect(dead_code)]

/// Implements `From` conversions between a frame-number type ([`Pfn`] or [`Vfn`])
/// and `Bounded<u64, N>` for bitfield interop.
///
/// Each MMU version module should invoke this for the specific bit widths used by that version's
/// PTE/PDE bitfield definitions.
macro_rules! impl_frame_number_bounded {
    ($type:ty, $bits:literal) => {
        impl From<Bounded<u64, $bits>> for $type {
            fn from(val: Bounded<u64, $bits>) -> Self {
                Self::new(val.get())
            }
        }

        impl From<$type> for Bounded<u64, $bits> {
            fn from(v: $type) -> Self {
                Bounded::from_expr(v.raw() & ::kernel::bits::genmask_u64(0..=($bits - 1)))
            }
        }
    };
}

/// Implements `From` conversions between [`Pfn`] and `Bounded<u64, N>` for bitfield interop.
macro_rules! impl_pfn_bounded {
    ($bits:literal) => {
        impl_frame_number_bounded!(Pfn, $bits);
    };
}

pub(crate) mod pramin;
pub(super) mod tlb;

use kernel::{
    bitfield,
    device,
    devres::Devres,
    gpu::buddy::{
        GpuBuddy,
        GpuBuddyParams, //
    },
    num::Bounded,
    prelude::*,
    sizes::SZ_4K,
    sync::Arc, //
};

use crate::{
    driver::Bar0,
    num::u64_as_usize, //
};

pub(crate) use tlb::Tlb;

/// GPU Memory Manager - owns all core MM components.
///
/// Provides centralized ownership of memory management resources:
/// - [`GpuBuddy`] allocator for VRAM page table allocation.
/// - [`pramin::Pramin`] for direct VRAM access.
/// - [`Tlb`] manager for translation buffer flush operations.
#[pin_data]
pub(crate) struct GpuMm {
    buddy: GpuBuddy,
    #[pin]
    pramin: pramin::Pramin,
    #[pin]
    tlb: Tlb,
}

impl GpuMm {
    /// Create a pin-initializer for `GpuMm`.
    ///
    /// `pramin_vram_region` is the full physical VRAM range (including GSP-reserved
    /// areas). PRAMIN window accesses are validated against this range.
    pub(crate) fn new(
        bar: Arc<Devres<Bar0>>,
        dev: &device::Device<device::Bound>,
        buddy_params: GpuBuddyParams,
        pramin_vram_region: core::ops::Range<u64>,
    ) -> Result<impl PinInit<Self>> {
        let buddy = GpuBuddy::new(buddy_params)?;
        let tlb_init = Tlb::new(bar.clone());
        let pramin_init = pramin::Pramin::new(bar, dev, pramin_vram_region)?;

        Ok(pin_init!(Self {
            buddy,
            pramin <- pramin_init,
            tlb <- tlb_init,
        }))
    }

    /// Access the [`GpuBuddy`] allocator.
    pub(crate) fn buddy(&self) -> &GpuBuddy {
        &self.buddy
    }

    /// Access the [`pramin::Pramin`].
    pub(crate) fn pramin(&self) -> &pramin::Pramin {
        &self.pramin
    }

    /// Access the [`Tlb`] manager.
    pub(crate) fn tlb(&self) -> &Tlb {
        &self.tlb
    }
}

/// Page size in bytes (4 KiB).
pub(crate) const PAGE_SIZE: usize = SZ_4K;

bitfield! {
    /// Physical VRAM address in GPU video memory.
    pub(crate) struct VramAddress(u64) {
        /// Offset within 4KB page.
        11:0    offset;
        /// Physical frame number.
        63:12   frame_number => Pfn;
    }
}

impl VramAddress {
    /// Create a new VRAM address from a raw value.
    pub(crate) const fn new(addr: u64) -> Self {
        Self::from_raw(addr)
    }

    /// Get the raw address value as `usize` (useful for MMIO offsets).
    pub(crate) const fn raw(&self) -> usize {
        u64_as_usize(self.into_raw())
    }

    /// Get the raw address value as `u64`.
    pub(crate) const fn raw_u64(&self) -> u64 {
        self.into_raw()
    }
}

impl PartialOrd for VramAddress {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VramAddress {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.into_raw().cmp(&other.into_raw())
    }
}

impl From<Pfn> for VramAddress {
    fn from(pfn: Pfn) -> Self {
        Self::zeroed().with_frame_number(pfn)
    }
}

bitfield! {
    /// Virtual address in GPU address space.
    pub(crate) struct VirtualAddress(u64) {
        /// Offset within 4KB page.
        11:0    offset;
        /// Virtual frame number.
        63:12   frame_number => Vfn;
    }
}

impl VirtualAddress {
    /// Create a new virtual address from a raw value.
    pub(crate) const fn new(addr: u64) -> Self {
        Self::from_raw(addr)
    }

    /// Get the raw address value as `u64`.
    pub(crate) const fn raw_u64(&self) -> u64 {
        self.into_raw()
    }
}

impl From<Vfn> for VirtualAddress {
    fn from(vfn: Vfn) -> Self {
        Self::zeroed().with_frame_number(vfn)
    }
}

/// Physical Frame Number.
///
/// Represents a physical page in VRAM.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Pfn(u64);

impl Pfn {
    /// Create a new PFN from a frame number.
    pub(crate) const fn new(frame_number: u64) -> Self {
        Self(frame_number)
    }

    /// Get the raw frame number.
    pub(crate) const fn raw(self) -> u64 {
        self.0
    }
}

impl From<VramAddress> for Pfn {
    fn from(addr: VramAddress) -> Self {
        addr.frame_number()
    }
}

impl From<u64> for Pfn {
    fn from(val: u64) -> Self {
        Self(val)
    }
}

impl From<Pfn> for u64 {
    fn from(pfn: Pfn) -> Self {
        pfn.0
    }
}

impl_pfn_bounded!(52);

/// Virtual Frame Number.
///
/// Represents a virtual page in GPU address space.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Vfn(u64);

impl Vfn {
    /// Create a new VFN from a frame number.
    pub(crate) const fn new(frame_number: u64) -> Self {
        Self(frame_number)
    }

    /// Get the raw frame number.
    pub(crate) const fn raw(self) -> u64 {
        self.0
    }
}

impl From<VirtualAddress> for Vfn {
    fn from(addr: VirtualAddress) -> Self {
        addr.frame_number()
    }
}

impl From<u64> for Vfn {
    fn from(val: u64) -> Self {
        Self(val)
    }
}

impl From<Vfn> for u64 {
    fn from(vfn: Vfn) -> Self {
        vfn.0
    }
}

impl_frame_number_bounded!(Vfn, 52);
