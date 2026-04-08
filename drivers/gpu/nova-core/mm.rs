// SPDX-License-Identifier: GPL-2.0

//! Memory management subsystems for nova-core.

#![expect(dead_code)]

pub(crate) mod bar_user;
pub(crate) mod pagetable;
pub(crate) mod pramin;
pub(crate) mod tlb;
pub(crate) mod vmm;

use kernel::{
    devres::Devres,
    gpu::buddy::{
        AllocatedBlocks,
        GpuBuddy,
        GpuBuddyAllocFlags,
        GpuBuddyAllocMode,
        GpuBuddyParams, //
    },
    prelude::*,
    ptr::Alignment,
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
    buddy: Option<GpuBuddy>,
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
        buddy_params: GpuBuddyParams,
        pramin_vram_region: core::ops::Range<u64>,
    ) -> Result<impl PinInit<Self>> {
        // TODO: Once NVKV key extraction is implemented in get_gsp_info(),
        // usable_fb_region will carry real values and buddy init will succeed.
        let buddy = if buddy_params.size > 0 {
            Some(GpuBuddy::new(buddy_params)?)
        } else {
            None
        };
        let tlb_init = Tlb::new(bar.clone());
        let pramin_init = pramin::Pramin::new(bar, pramin_vram_region)?;

        Ok(pin_init!(Self {
            buddy,
            pramin <- pramin_init,
            tlb <- tlb_init,
        }))
    }

    /// Access the [`GpuBuddy`] allocator.
    pub(crate) fn buddy(&self) -> Result<&GpuBuddy> {
        self.buddy.as_ref().ok_or(ENOMEM)
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
    pub(crate) struct VramAddress(u64), "Physical VRAM address in GPU video memory" {
        11:0    offset          as u64, "Offset within 4KB page";
        63:12   frame_number    as u64 => Pfn, "Physical frame number";
    }
}

impl VramAddress {
    /// Create a new VRAM address from a raw value.
    pub(crate) const fn new(addr: u64) -> Self {
        Self(addr)
    }

    /// Get the raw address value as `usize` (useful for MMIO offsets).
    pub(crate) const fn raw(&self) -> usize {
        u64_as_usize(self.0)
    }

    /// Get the raw address value as `u64`.
    pub(crate) const fn raw_u64(&self) -> u64 {
        self.0
    }
}

impl PartialEq for VramAddress {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for VramAddress {}

impl PartialOrd for VramAddress {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VramAddress {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl From<Pfn> for VramAddress {
    fn from(pfn: Pfn) -> Self {
        Self::default().set_frame_number(pfn)
    }
}

/// A block of allocated VRAM. Freed when dropped.
pub(crate) struct VramBlock {
    blocks: Pin<KBox<AllocatedBlocks>>,
    addr: u64,
    size: u64,
}

impl VramBlock {
    /// The physical VRAM address of this block.
    pub(crate) fn addr(&self) -> u64 {
        self.addr
    }

    /// The size of this block in bytes.
    pub(crate) fn size(&self) -> u64 {
        self.size
    }

    /// The physical frame number for this block.
    pub(crate) fn pfn(&self) -> Pfn {
        Pfn::from(VramAddress::new(self.addr))
    }
}

/// Allocate a contiguous block of VRAM.
pub(crate) fn alloc_vram(mm: &GpuMm, size: u64, align: u64) -> Result<VramBlock> {
    let align_val = align.max(PAGE_SIZE as u64) as usize;
    let alignment = Alignment::new_checked(align_val).ok_or(EINVAL)?;
    let blocks = KBox::pin_init(
        mm.buddy()?.alloc_blocks(
            GpuBuddyAllocMode::Simple,
            size,
            alignment,
            GpuBuddyAllocFlags::default(),
        ),
        GFP_KERNEL,
    )?;
    let first = blocks.iter().next().ok_or(ENOMEM)?;
    Ok(VramBlock {
        addr: first.offset(),
        size: first.size(),
        blocks,
    })
}

// GPU virtual address.
bitfield! {
    pub(crate) struct VirtualAddress(u64), "Virtual address in GPU address space" {
        11:0    offset          as u64, "Offset within 4KB page";
        63:12   frame_number    as u64 => Vfn, "Virtual frame number";
    }
}

impl VirtualAddress {
    /// Create a new virtual address from a raw value.
    #[expect(dead_code)]
    pub(crate) const fn new(addr: u64) -> Self {
        Self(addr)
    }

    /// Get the raw address value as `u64`.
    pub(crate) const fn raw_u64(&self) -> u64 {
        self.0
    }
}

impl From<Vfn> for VirtualAddress {
    fn from(vfn: Vfn) -> Self {
        Self::default().set_frame_number(vfn)
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
