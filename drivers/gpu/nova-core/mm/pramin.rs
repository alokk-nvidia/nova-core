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
    device,
    devres::Devres,
    io::Io,
    new_mutex,
    num::Bounded,
    prelude::*,
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
                Self::write_window_base(self.bar, base)?;
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
                Self::write_window_base(self.bar, base)?;
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
        dev: &device::Device<device::Bound>,
        vram_region: Range<u64>,
    ) -> Result<impl PinInit<Self>> {
        let bar_access = bar.access(dev)?;
        let current_base = Self::read_window_base(bar_access);

        Ok(pin_init!(Self {
            bar,
            vram_region,
            state <- new_mutex!(current_base, "pramin_state"),
        }))
    }

    /// Returns the valid VRAM region for this PRAMIN instance.
    fn vram_region(&self) -> &Range<u64> {
        &self.vram_region
    }

    /// Acquire exclusive PRAMIN access.
    ///
    /// Returns a [`PraminWindow`] guard that provides VRAM read/write accessors.
    /// The [`PraminWindow`] is exclusive and only one can exist at a time.
    pub(crate) fn get_window<'a>(
        &'a self,
        dev: &'a device::Device<device::Bound>,
    ) -> Result<PraminWindow<'a>> {
        let bar = self.bar.access(dev)?;
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
    bar: &'a Bar0,
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

/// Offset within the VRAM region to use as the self-test area.
#[cfg(CONFIG_NOVA_MM_SELFTESTS)]
const SELFTEST_REGION_OFFSET: usize = 0x1000;

/// Test read/write at byte-aligned locations.
#[cfg(CONFIG_NOVA_MM_SELFTESTS)]
fn test_byte_readwrite(
    dev: &kernel::device::Device,
    win: &mut PraminWindow<'_>,
    base: usize,
) -> Result {
    for i in 0u8..4 {
        let offset = base + 1 + usize::from(i);
        let val = 0xA0 + i;
        win.try_write8(offset, val)?;
        let read_val = win.try_read8(offset)?;
        if read_val != val {
            dev_err!(
                dev,
                "PRAMIN: FAIL - offset {:#x}: wrote {:#x}, read {:#x}\n",
                offset,
                val,
                read_val
            );
            return Err(EIO);
        }
    }
    Ok(())
}

/// Test writing a `u32` and reading back as individual `u8`s.
#[cfg(CONFIG_NOVA_MM_SELFTESTS)]
fn test_u32_as_bytes(
    dev: &kernel::device::Device,
    win: &mut PraminWindow<'_>,
    base: usize,
) -> Result {
    let offset = base + 0x10;
    let val: u32 = 0xDEADBEEF;
    win.try_write32(offset, val)?;

    // Read back as individual bytes (little-endian: EF BE AD DE).
    let expected_bytes: [u8; 4] = [0xEF, 0xBE, 0xAD, 0xDE];
    for (i, &expected) in expected_bytes.iter().enumerate() {
        let read_val = win.try_read8(offset + i)?;
        if read_val != expected {
            dev_err!(
                dev,
                "PRAMIN: FAIL - offset {:#x}: expected {:#x}, read {:#x}\n",
                offset + i,
                expected,
                read_val
            );
            return Err(EIO);
        }
    }
    Ok(())
}

/// Test window repositioning across 1MB boundaries.
#[cfg(CONFIG_NOVA_MM_SELFTESTS)]
fn test_window_reposition(
    dev: &kernel::device::Device,
    win: &mut PraminWindow<'_>,
    base: usize,
) -> Result {
    let offset_a: usize = base;
    let offset_b: usize = base + 0x200000; // base + 2MB (different 1MB region).
    let val_a: u32 = 0x11111111;
    let val_b: u32 = 0x22222222;

    win.try_write32(offset_a, val_a)?;
    win.try_write32(offset_b, val_b)?;

    let read_b = win.try_read32(offset_b)?;
    if read_b != val_b {
        dev_err!(
            dev,
            "PRAMIN: FAIL - offset {:#x}: expected {:#x}, read {:#x}\n",
            offset_b,
            val_b,
            read_b
        );
        return Err(EIO);
    }

    let read_a = win.try_read32(offset_a)?;
    if read_a != val_a {
        dev_err!(
            dev,
            "PRAMIN: FAIL - offset {:#x}: expected {:#x}, read {:#x}\n",
            offset_a,
            val_a,
            read_a
        );
        return Err(EIO);
    }
    Ok(())
}

/// Test that offsets outside the VRAM region are rejected.
#[cfg(CONFIG_NOVA_MM_SELFTESTS)]
fn test_invalid_offset(
    dev: &kernel::device::Device,
    win: &mut PraminWindow<'_>,
    vram_end: u64,
) -> Result {
    let invalid_offset: usize = vram_end.into_safe_cast();
    let result = win.try_read32(invalid_offset);
    if result.is_ok() {
        dev_err!(
            dev,
            "PRAMIN: FAIL - read at invalid offset {:#x} should have failed\n",
            invalid_offset
        );
        return Err(EIO);
    }
    Ok(())
}

/// Test that misaligned multi-byte accesses are rejected.
#[cfg(CONFIG_NOVA_MM_SELFTESTS)]
fn test_misaligned_access(
    dev: &kernel::device::Device,
    win: &mut PraminWindow<'_>,
    base: usize,
) -> Result {
    // `u16` at odd offset (not 2-byte aligned).
    let offset_u16 = base + 0x21;
    if win.try_write16(offset_u16, 0xABCD).is_ok() {
        dev_err!(
            dev,
            "PRAMIN: FAIL - misaligned u16 write at {:#x} should have failed\n",
            offset_u16
        );
        return Err(EIO);
    }

    // `u32` at 2-byte-aligned (not 4-byte-aligned) offset.
    let offset_u32 = base + 0x32;
    if win.try_write32(offset_u32, 0x12345678).is_ok() {
        dev_err!(
            dev,
            "PRAMIN: FAIL - misaligned u32 write at {:#x} should have failed\n",
            offset_u32
        );
        return Err(EIO);
    }

    // `u64` read at 4-byte-aligned (not 8-byte-aligned) offset.
    let offset_u64 = base + 0x44;
    if win.try_read64(offset_u64).is_ok() {
        dev_err!(
            dev,
            "PRAMIN: FAIL - misaligned u64 read at {:#x} should have failed\n",
            offset_u64
        );
        return Err(EIO);
    }
    Ok(())
}

/// Run PRAMIN self-tests during boot if self-tests are enabled.
#[cfg(CONFIG_NOVA_MM_SELFTESTS)]
pub(crate) fn run_self_test(
    pdev: &device::Device<device::Bound>,
    pramin: &Pramin,
    chipset: crate::gpu::Chipset,
) -> Result {
    use crate::gpu::Architecture;

    let dev = pdev;

    // PRAMIN uses NV_PBUS_BAR0_WINDOW which is only available on pre-Hopper GPUs.
    // Hopper+ uses NV_XAL_EP_BAR0_WINDOW instead, requiring a separate HAL that
    // has not been implemented yet.
    if !matches!(
        chipset.arch(),
        Architecture::Turing | Architecture::Ampere | Architecture::Ada
    ) {
        dev_info!(
            dev,
            "PRAMIN: Skipping self-tests for {:?} (only pre-Hopper supported)\n",
            chipset
        );
        return Ok(());
    }

    dev_info!(dev, "PRAMIN: Starting self-test...\n");

    let vram_region = pramin.vram_region();
    let base: usize = vram_region.start.into_safe_cast();
    let base = base + SELFTEST_REGION_OFFSET;
    let vram_end = vram_region.end;
    let mut win = pramin.get_window(pdev)?;

    test_byte_readwrite(dev, &mut win, base)?;
    test_u32_as_bytes(dev, &mut win, base)?;
    test_window_reposition(dev, &mut win, base)?;
    test_invalid_offset(dev, &mut win, vram_end)?;
    test_misaligned_access(dev, &mut win, base)?;

    dev_info!(dev, "PRAMIN: All self-tests PASSED\n");
    Ok(())
}
