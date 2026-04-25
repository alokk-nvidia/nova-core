// SPDX-License-Identifier: GPL-2.0

//! Direct VRAM access through the PRAMIN aperture.
//!
//! PRAMIN provides a 1MB sliding window into VRAM through BAR0, allowing the CPU to access
//! video memory directly. Access is managed through a two-level API:
//!
//! - [`Pramin`]: The parent object that owns the BAR0 reference and synchronization lock.
//! - [`PraminWindow`]: A guard object that holds exclusive PRAMIN access for its lifetime.
//!
//! The PRAMIN aperture is a 1MB region at a fixed offset from BAR0. The window base is
//! controlled by an architecture-specific register and is 64KB aligned.
//!
//! # Examples
//!
//! ## Basic read/write
//!
//! ```no_run
//! use crate::driver::Bar0;
//! use crate::gpu::Chipset;
//! use crate::mm::pramin;
//! use kernel::device;
//! use kernel::devres::Devres;
//! use kernel::prelude::*;
//! use kernel::sync::Arc;
//!
//! fn example(
//!     devres_bar: Arc<Devres<Bar0>>,
//!     dev: &device::Device<device::Bound>,
//!     chipset: Chipset,
//!     vram_region: core::ops::Range<u64>,
//! ) -> Result<()> {
//!     let pramin = Arc::pin_init(
//!         pramin::Pramin::new(devres_bar, dev, chipset, vram_region)?,
//!         GFP_KERNEL,
//!     )?;
//!     let mut window = pramin.get_window(dev)?;
//!
//!     // Write and read back.
//!     window.try_write32(0x100, 0xDEADBEEF)?;
//!     let val = window.try_read32(0x100)?;
//!     assert_eq!(val, 0xDEADBEEF);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Auto-repositioning across VRAM regions
//!
//! ```no_run
//! use crate::driver::Bar0;
//! use crate::gpu::Chipset;
//! use crate::mm::pramin;
//! use kernel::device;
//! use kernel::devres::Devres;
//! use kernel::prelude::*;
//! use kernel::sync::Arc;
//!
//! fn example(
//!     devres_bar: Arc<Devres<Bar0>>,
//!     dev: &device::Device<device::Bound>,
//!     chipset: Chipset,
//!     vram_region: core::ops::Range<u64>,
//! ) -> Result<()> {
//!     let pramin = Arc::pin_init(
//!         pramin::Pramin::new(devres_bar, dev, chipset, vram_region)?,
//!         GFP_KERNEL,
//!     )?;
//!     let mut window = pramin.get_window(dev)?;
//!
//!     // Access first 1MB region.
//!     window.try_write32(0x100, 0x11111111)?;
//!
//!     // Access at 2MB - window auto-repositions.
//!     window.try_write32(0x200000, 0x22222222)?;
//!
//!     // Back to first region - window repositions again.
//!     let val = window.try_read32(0x100)?;
//!     assert_eq!(val, 0x11111111);
//!
//!     Ok(())
//! }
//! ```

#![expect(unused)]

use core::ops::Range;

use crate::{
    bounded_enum,
    driver::Bar0,
    num::IntoSafeCast,
    regs, //
};

use kernel::{
    devres::Devres,
    io::Io,
    new_mutex,
    num::Bounded,
    prelude::*,
    revocable::RevocableGuard,
    sizes::{
        SZ_1M,
        SZ_64K, //
    },
    sync::{
        lock::mutex::MutexGuard,
        Arc,
        Mutex, //
    },
};

bounded_enum! {
    /// Target memory type for the BAR0 window register.
    ///
    /// Only VRAM is supported; Hopper+ GPUs do not support other targets.
    #[derive(Debug)]
    pub(crate) enum Bar0WindowTarget with TryFrom<Bounded<u32, 2>> {
        /// Video RAM (GPU framebuffer memory).
        Vram = 0,
    }
}

/// PRAMIN aperture base offset in BAR0.
const PRAMIN_BASE: usize = 0x700000;

/// PRAMIN aperture size (1MB).
const PRAMIN_SIZE: usize = SZ_1M;

/// Generate a PRAMIN read accessor.
macro_rules! define_pramin_read {
    ($name:ident, $ty:ty) => {
        #[doc = concat!("Read a `", stringify!($ty), "` from VRAM at the given offset.")]
        pub(crate) fn $name(&mut self, vram_offset: usize) -> Result<$ty> {
            let (bar_offset, new_base) =
                self.compute_window(vram_offset, ::core::mem::size_of::<$ty>())?;

            if let Some(base) = new_base {
                Self::write_window_base(&self.bar, base)?;
                *self.state = base;
            }
            self.bar.$name(bar_offset)
        }
    };
}

/// Generate a PRAMIN write accessor.
macro_rules! define_pramin_write {
    ($name:ident, $ty:ty) => {
        #[doc = concat!("Write a `", stringify!($ty), "` to VRAM at the given offset.")]
        pub(crate) fn $name(&mut self, vram_offset: usize, value: $ty) -> Result {
            let (bar_offset, new_base) =
                self.compute_window(vram_offset, ::core::mem::size_of::<$ty>())?;

            if let Some(base) = new_base {
                Self::write_window_base(&self.bar, base)?;
                *self.state = base;
            }
            self.bar.$name(value, bar_offset)
        }
    };
}

/// PRAMIN aperture manager.
///
/// Call [`Pramin::get_window()`] to acquire exclusive PRAMIN access.
#[pin_data]
pub(crate) struct Pramin {
    bar: Arc<Devres<Bar0>>,
    /// Valid VRAM region. Accesses outside this range are rejected.
    vram_region: Range<u64>,
    /// PRAMIN aperture state, protected by a mutex.
    ///
    /// # Invariants
    ///
    /// This lock is acquired during the DMA fence signaling critical path.
    /// It must NEVER be held across any reclaimable CPU memory / allocations
    /// (`GFP_KERNEL`), because the memory reclaim path can call
    /// `dma_fence_wait()`, which would deadlock with this lock held.
    #[pin]
    state: Mutex<u64>,
}

impl Pramin {
    /// Create a pin-initializer for PRAMIN.
    ///
    /// `vram_region` specifies the valid VRAM address range.
    pub(crate) fn new(
        bar: Arc<Devres<Bar0>>,
        vram_region: Range<u64>,
    ) -> Result<impl PinInit<Self>> {
        let bar_access = bar.try_access().ok_or(ENODEV)?;
        let current_base = Self::read_window_base(&bar_access);

        Ok(pin_init!(Self {
            bar,
            vram_region,
            state <- new_mutex!(current_base, "pramin_state"),
        }))
    }

    /// Acquire exclusive PRAMIN access.
    ///
    /// Returns a [`PraminWindow`] guard that provides VRAM read/write accessors.
    /// The [`PraminWindow`] is exclusive and only one can exist at a time.
    pub(crate) fn get_window(&self) -> Result<PraminWindow<'_>> {
        let bar = self.bar.try_access().ok_or(ENODEV)?;
        let state = self.state.lock();
        Ok(PraminWindow {
            bar,
            vram_region: self.vram_region.clone(),
            state,
        })
    }

    /// Read the current window base from the BAR0_WINDOW register.
    fn read_window_base(bar: &Bar0) -> u64 {
        let reg = bar.read(regs::NV_PBUS_BAR0_WINDOW);

        // TODO: Convert to Bounded<u64, 40> when available.
        u64::from(reg.window_base()) << 16
    }
}

/// PRAMIN window guard for direct VRAM access.
///
/// This guard holds exclusive access to the PRAMIN aperture. The window auto-repositions
/// when accessing VRAM offsets outside the current 1MB range.
///
/// Only one [`PraminWindow`] can exist at a time per [`Pramin`] instance (enforced by the
/// internal `MutexGuard`).
pub(crate) struct PraminWindow<'a> {
    bar: RevocableGuard<'a, Bar0>,
    vram_region: Range<u64>,
    state: MutexGuard<'a, u64>,
}

impl PraminWindow<'_> {
    /// Write a new window base to the BAR0_WINDOW register.
    fn write_window_base(bar: &Bar0, base: u64) -> Result {
        // CAST: After >> 16, a VRAM address fits in u32.
        let window_base = (base >> 16) as u32;
        bar.write_reg(
            regs::NV_PBUS_BAR0_WINDOW::zeroed()
                .with_target(Bar0WindowTarget::Vram)
                .try_with_window_base(window_base)?,
        );
        Ok(())
    }

    /// Compute window parameters for a VRAM access.
    ///
    /// Returns (`bar_offset`, `new_base`) where:
    /// - `bar_offset`: The BAR0 offset to use for the access.
    /// - `new_base`: `Some(base)` if window needs repositioning, `None` otherwise.
    fn compute_window(
        &self,
        vram_offset: usize,
        access_size: usize,
    ) -> Result<(usize, Option<u64>)> {
        // Validate VRAM offset is within the valid VRAM region.
        let vram_addr = vram_offset as u64;
        let end_addr = vram_addr.checked_add(access_size as u64).ok_or(EINVAL)?;
        if vram_addr < self.vram_region.start || end_addr > self.vram_region.end {
            return Err(EINVAL);
        }

        // Check if access fits within the current 1MB window.
        let current_base = *self.state;
        if vram_addr >= current_base {
            let offset_in_window: usize = (vram_addr - current_base).into_safe_cast();
            if offset_in_window + access_size <= PRAMIN_SIZE {
                return Ok((PRAMIN_BASE + offset_in_window, None));
            }
        }

        // Access doesn't fit in current window - reposition.
        // Hardware requires 64KB alignment for the window base register.
        let needed_base = vram_addr & !(SZ_64K as u64 - 1);
        let offset_in_window: usize = (vram_addr - needed_base).into_safe_cast();

        // Verify access fits in the 1MB window from the new base.
        if offset_in_window + access_size > PRAMIN_SIZE {
            return Err(EINVAL);
        }

        Ok((PRAMIN_BASE + offset_in_window, Some(needed_base)))
    }

    define_pramin_read!(try_read8, u8);
    define_pramin_read!(try_read16, u16);
    define_pramin_read!(try_read32, u32);
    define_pramin_read!(try_read64, u64);

    define_pramin_write!(try_write8, u8);
    define_pramin_write!(try_write16, u16);
    define_pramin_write!(try_write32, u32);
    define_pramin_write!(try_write64, u64);
}
