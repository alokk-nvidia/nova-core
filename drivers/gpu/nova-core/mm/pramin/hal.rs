// SPDX-License-Identifier: GPL-2.0

//! BAR0 window register abstraction for PRAMIN access.
//!
//! Pre-Hopper GPUs use `NV_PBUS_BAR0_WINDOW` (0x1700), while Hopper and later
//! architectures use `NV_XAL_EP_BAR0_WINDOW` (0x10FD40).

use kernel::prelude::*;

use crate::{
    driver::Bar0,
    gpu::{
        Architecture,
        Chipset, //
    },
};

mod gh100;
mod tu102;

/// Hardware abstraction for the BAR0 window register used by PRAMIN.
pub(crate) trait Bar0WindowHal: Send + Sync {
    /// Read the current window base address from the BAR0 window register.
    ///
    /// Returns the VRAM byte address corresponding to the start of the
    /// current 1 MB PRAMIN window.
    fn read_window_base(&self, bar: &Bar0) -> u64;

    /// Write a new window base address to the BAR0 window register.
    ///
    /// `base` is a 64 KB-aligned VRAM byte address.
    fn write_window_base(&self, bar: &Bar0, base: u64) -> Result;
}

/// Returns the BAR0 window HAL for `chipset`.
pub(crate) fn pramin_hal(chipset: Chipset) -> &'static dyn Bar0WindowHal {
    match chipset.arch() {
        Architecture::Turing | Architecture::Ampere | Architecture::Ada => tu102::TU102_HAL,
        Architecture::Hopper | Architecture::BlackwellGB10x | Architecture::BlackwellGB20x => {
            gh100::GH100_HAL
        }
    }
}
