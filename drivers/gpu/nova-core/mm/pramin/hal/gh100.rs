// SPDX-License-Identifier: GPL-2.0

//! Hopper+ BAR0 window via NV_XAL_EP_BAR0_WINDOW (0x10FD40).

use kernel::{
    io::Io,
    prelude::*, //
};

use crate::{
    driver::Bar0,
    regs, //
};

use super::Bar0WindowHal;

struct Gh100;

impl Bar0WindowHal for Gh100 {
    fn read_window_base(&self, bar: &Bar0) -> u64 {
        let reg = bar.read(regs::NV_XAL_EP_BAR0_WINDOW);
        u64::from(reg.base()) << 16
    }

    fn write_window_base(&self, bar: &Bar0, base: u64) -> Result {
        let window_base = (base >> 16) as u32;
        bar.write_reg(regs::NV_XAL_EP_BAR0_WINDOW::zeroed().try_with_base(window_base)?);
        Ok(())
    }
}

const GH100: Gh100 = Gh100;
pub(super) const GH100_HAL: &dyn Bar0WindowHal = &GH100;
