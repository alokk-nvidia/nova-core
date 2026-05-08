// SPDX-License-Identifier: GPL-2.0

//! Pre-Hopper BAR0 window via NV_PBUS_BAR0_WINDOW (0x1700).

use kernel::{
    io::Io,
    prelude::*, //
};

use crate::{
    driver::Bar0,
    mm::pramin::Bar0WindowTarget,
    regs, //
};

use super::Bar0WindowHal;

struct Tu102;

impl Bar0WindowHal for Tu102 {
    fn read_window_base(&self, bar: &Bar0) -> u64 {
        let reg = bar.read(regs::NV_PBUS_BAR0_WINDOW);
        u64::from(reg.window_base()) << 16
    }

    fn write_window_base(&self, bar: &Bar0, base: u64) -> Result {
        let window_base = (base >> 16) as u32;
        bar.write_reg(
            regs::NV_PBUS_BAR0_WINDOW::zeroed()
                .with_target(Bar0WindowTarget::Vram)
                .try_with_window_base(window_base)?,
        );
        Ok(())
    }
}

const TU102: Tu102 = Tu102;
pub(super) const TU102_HAL: &dyn Bar0WindowHal = &TU102;
