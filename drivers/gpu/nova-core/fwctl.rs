// SPDX-License-Identifier: GPL-2.0

//! fwctl driver for nova-core GMC API pass-through.

use core::convert::Infallible;

use kernel::{
    fwctl::{
        self,
        DeviceType,
        FwRpcResponse,
        RpcScope, //
    },
    prelude::*, //
};

use crate::gsp::{
    cmdq::{
        gmc::{
            CommandPayload,
            CommandToGsp,
            MessageFromGsp,
            MessagePayload, //
        },
        Cmdq,
    },
    GmcCommandId,
};

/// Resources used by fwctl callbacks while the GPU is bound.
#[pin_data]
pub(crate) struct NovaCoreFwCtlData<'a> {
    cmdq: &'a Cmdq<'a>,
}

impl<'a> NovaCoreFwCtlData<'a> {
    pub(crate) fn new(cmdq: &'a Cmdq<'a>) -> impl PinInit<Self, Error> {
        try_pin_init!(Self { cmdq })
    }
}

/// Per-file fwctl context for nova-core.
#[pin_data]
pub(crate) struct NovaCoreFwCtl {}

impl fwctl::Operations for NovaCoreFwCtl {
    type RegistrationData<'a> = NovaCoreFwCtlData<'a>;

    const DEVICE_TYPE: DeviceType = DeviceType::NovaCore;

    fn open<'a>(
        _device: &fwctl::Device<Self>,
        _reg_data: &Self::RegistrationData<'a>,
    ) -> impl PinInit<Self, Error> {
        try_pin_init!(Self {})
    }

    fn fw_rpc<'a>(
        _this: Pin<&Self>,
        _device: &fwctl::Device<Self>,
        reg_data: &Self::RegistrationData<'a>,
        scope: RpcScope,
        rpc_buf: &mut [u8],
        _max_output_len: usize,
    ) -> Result<FwRpcResponse, Error> {
        if rpc_buf.len() < RPC_HEADER_SIZE {
            return Err(EINVAL);
        }

        let (header, payload) = rpc_buf.split_at(RPC_HEADER_SIZE);
        let command_id = u32::from_le_bytes(header[..4].try_into().map_err(|_| EINVAL)?);
        let reserved = u32::from_le_bytes(header[4..].try_into().map_err(|_| EINVAL)?);

        if reserved != 0 {
            return Err(EINVAL);
        }
        if !matches!(scope, RpcScope::Configuration) {
            return Err(EPERM);
        }

        let response = send_command(reg_data.cmdq, command_id, payload)?;
        let response_len = RPC_HEADER_SIZE
            .checked_add(response.len())
            .ok_or(EOVERFLOW)?;
        let mut out = KVVec::with_capacity(response_len, GFP_KERNEL)?;
        out.extend_from_slice(&command_id.to_le_bytes(), GFP_KERNEL)?;
        out.extend_from_slice(&0u32.to_le_bytes(), GFP_KERNEL)?;
        out.extend_from_slice(&response, GFP_KERNEL)?;

        Ok(FwRpcResponse::NewBuffer(out))
    }
}

/// Size of `fwctl_rpc_nova_core` (`command_id` plus `reserved`).
const RPC_HEADER_SIZE: usize = 8;

const GSP_GMC_MAX_RESPONSE_SIZE: u32 = 16_384;

/// Firmware response bytes copied out before the message queue slot is consumed.
struct Response(KVec<u8>);

impl MessageFromGsp for Response {
    type InitError = Error;

    fn read(payload: &mut MessagePayload<'_>) -> Result<Self> {
        payload.flush_into_kvec(GFP_KERNEL).map(Self)
    }
}

// Define typed requests and the exact command allowlist together.
macro_rules! fwctl_commands {
    ($($command:ident),+ $(,)?) => {
        $(
            struct $command<'a>(&'a [u8]);

            impl CommandToGsp for $command<'_> {
                const COMMAND: GmcCommandId = GmcCommandId::$command;
                type Command = ();
                type Reply = Response;
                const MAX_REPLY_SIZE: u32 = GSP_GMC_MAX_RESPONSE_SIZE;
                type InitError = Infallible;

                fn init(&self) -> impl Init<Self::Command, Self::InitError> {
                    pin_init::init_zeroed()
                }

                fn variable_payload_len(&self) -> usize {
                    self.0.len()
                }

                fn init_variable_payload(&self, dst: &mut CommandPayload<'_>) -> Result {
                    dst.write_all(self.0)
                }
            }
        )+

        fn send_command(cmdq: &Cmdq<'_>, command_id: u32, payload: &[u8]) -> Result<KVec<u8>> {
            // CAST: `GmcCommandId` is `repr(u32)`, so these casts are lossless.
            match command_id {
                $(
                    id if id == GmcCommandId::$command as u32 => cmdq
                        .send_gmc_command($command(payload), |_| Ok(()))?
                        .map(|response| response.0)
                        .map_err(|_| EIO),
                )+
                _ => Err(EPERM),
            }
        }
    };
}

fwctl_commands! {
    AddVgpuType,
    QuerySupportedVgpuTypes,
    QueryCreatableVgpuTypes,
    AssignVgpuType,
    DeassignVgpuType,
    QueryVgpuProperties,
    QueryAssignedVfVgpuType,
}
