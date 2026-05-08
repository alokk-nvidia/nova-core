// SPDX-License-Identifier: GPL-2.0

use kernel::{
    device,
    prelude::*,
    time::{
        delay::fsleep,
        Delta,
        Instant,
        Monotonic, //
    },
};

use crate::vgpu::{
    consts::plugin_rpc as consts,
    Bar1Map, //
};

/// Plugin RPC channel for vGPU plugin communication.
pub(crate) struct PluginRpc {
    bar1_map: Bar1Map,
}

impl PluginRpc {
    pub(crate) fn new(bar1_map: Bar1Map) -> Self {
        Self { bar1_map }
    }

    /// Poll ctrl_buf for plugin boot completion magic.
    pub(crate) fn wait_plugin_ready(&self, dev: &device::Device<device::Bound>) -> Result {
        let start = Instant::<Monotonic>::now();
        let timeout = Delta::from_millis(consts::PLUGIN_BOOT_TIMEOUT_MS as i64);
        loop {
            let val = self
                .bar1_map
                .read32(dev, consts::CTRL_BUF_MSG_SEQ_NUM_OFFSET)?;
            if val == consts::GSP_PLUGIN_BOOTLOADED {
                return Ok(());
            }
            if start.elapsed() > timeout {
                return Err(ETIMEDOUT);
            }
            fsleep(Delta::from_millis(1));
        }
    }
}
