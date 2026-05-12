// SPDX-License-Identifier: GPL-2.0

//! fwctl driver for nova-core GMC API pass-through.

use core::ptr::NonNull;

use kernel::{
    devres::Devres,
    fwctl,
    fwctl::{
        DeviceType,
        FwRpcResponse,
        RpcScope, //
    },
    prelude::*,
    sync::Arc, //
};

use crate::{
    driver::Bar0,
    gsp::cmdq::Cmdq, //
};

/// Device data embedded in the fwctl device allocation.
///
/// Holds the resources needed to send GMC commands to the GSP.
pub(crate) struct NovaCoreFwCtlData {
    bar: Arc<Devres<Bar0>>,
    cmdq: NonNull<Cmdq>,
}

// SAFETY: `cmdq` points to a pinned `Cmdq` inside `Gsp` whose lifetime is
// guaranteed by the device model: the fwctl `Registration` lives in a `Devres`
// tied to the same parent, and `Gsp` (owner of `Cmdq`) is dropped after the
// `Devres` group runs. `Cmdq` methods take `&self` and use internal locking.
unsafe impl Send for NovaCoreFwCtlData {}
// SAFETY: See Send impl above. All access goes through `Cmdq`'s internal mutex.
unsafe impl Sync for NovaCoreFwCtlData {}

impl NovaCoreFwCtlData {
    /// Creates a new `NovaCoreFwCtlData`.
    ///
    /// # Safety
    ///
    /// `cmdq` must point to a `Cmdq` that outlives the fwctl registration.
    pub(crate) unsafe fn new(bar: Arc<Devres<Bar0>>, cmdq: *const Cmdq) -> Self {
        Self {
            bar,
            // SAFETY: Caller guarantees `cmdq` is non-null and valid.
            cmdq: unsafe { NonNull::new_unchecked(cmdq.cast_mut()) },
        }
    }

    fn cmdq(&self) -> &Cmdq {
        // SAFETY: The pointer is valid for the lifetime of this struct (see
        // Send/Sync safety comments).
        unsafe { self.cmdq.as_ref() }
    }
}

/// Per-FD user context for nova-core fwctl.
pub(crate) struct NovaCoreFwCtl;

impl fwctl::Operations for NovaCoreFwCtl {
    type DeviceData = NovaCoreFwCtlData;
    const DEVICE_TYPE: DeviceType = DeviceType::NovaCore;

    fn open(_device: &fwctl::Device<Self>) -> Result<impl PinInit<Self, Error>, Error> {
        // SAFETY: `NovaCoreFwCtl` is a unit struct with no fields to initialize.
        Ok(unsafe { pin_init::init_from_closure(move |_slot: *mut Self| Ok(())) })
    }

    fn fw_rpc(
        _this: &Self,
        device: &fwctl::Device<Self>,
        scope: RpcScope,
        rpc_in: &mut [u8],
    ) -> Result<FwRpcResponse, Error> {
        let data = device.data();

        if rpc_in.len() < RPC_HEADER_SIZE {
            return Err(EINVAL);
        }

        let command_id = u32::from_le_bytes(rpc_in[0..4].try_into().map_err(|_| EINVAL)?);

        if !is_command_permitted(scope, command_id) {
            return Err(EPERM);
        }

        let payload = &rpc_in[RPC_HEADER_SIZE..];

        let bar = data.bar.access(device.parent())?;
        let response = data.cmdq().send_gmc_and_receive(
            bar,
            command_id,
            payload,
            GSP_GMC_MAX_RESPONSE_SIZE,
        )?;

        if response.status != 0 {
            return Err(EIO);
        }

        let mut out = KVec::with_capacity(RPC_HEADER_SIZE + response.payload.len(), GFP_KERNEL)?;
        out.extend_from_slice(&command_id.to_le_bytes(), GFP_KERNEL)?;
        out.extend_from_slice(&0u32.to_le_bytes(), GFP_KERNEL)?;
        out.extend_from_slice(&response.payload, GFP_KERNEL)?;

        Ok(FwRpcResponse::NewBuffer(out))
    }
}

/// Size of `fwctl_rpc_nova_core` (command_id + reserved).
const RPC_HEADER_SIZE: usize = 8;

const GSP_GMC_MAX_RESPONSE_SIZE: u32 = 16384;

/// Check whether `command_id` is permitted for the given `scope`.
fn is_command_permitted(_scope: RpcScope, _command_id: u32) -> bool {
    // TODO: Populate with the per-scope permitted command table once vGPU
    // commands land in gmcapi_table.h.
    true
}
