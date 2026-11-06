// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

//! vGPU firmware command operations.
//!
//! Sends requests, checks responses and coordinates the firmware command
//! sequences used by instance lifecycle operations.

use core::convert::Infallible;

use kernel::{
    device,
    prelude::*,
    time::Delta,
    transmute::AsBytes, //
};

use crate::{
    gsp::{
        cmdq::{
            gmc::{
                CommandPayload,
                CommandToGsp,
                MessageFromGsp,
                MessagePayload, //
            },
            Cmdq,
            NoReply, //
        },
        nvkv::{
            nvkv_words,
            Decoder,
            Schema,
            UnknownKeyPolicy, //
        },
        GmcCommandId,
        GSP_PAGE_SIZE, //
    },
    sbuffer::SBufferIter,
    vgpu::{
        fw::{
            commands::{
                encode_plugin_set_bme,
                AssignedVfVgpuType,
                VgpuProperties,
                VgpuPropertiesSchema, //
            },
            RpcMessage,
        },
        gsp_plugin_rpc::PluginRpc,
        instance::Gfid,
        Dbdf,
    },
};

struct QueryAssignedVfVgpuType(Dbdf);

impl CommandToGsp for QueryAssignedVfVgpuType {
    const COMMAND: GmcCommandId = GmcCommandId::QueryAssignedVfVgpuType;
    type Command = u64;
    type Reply = AssignedVfVgpuType;
    const MAX_REPLY_SIZE: u32 = cv!(size_of::<AssignedVfVgpuType>());
    type InitError = Infallible;

    fn init(&self) -> impl Init<Self::Command, Self::InitError> {
        u64::from(self.0.into_raw()).to_le()
    }
}

impl MessageFromGsp for AssignedVfVgpuType {
    type InitError = Error;

    fn read(payload: &mut MessagePayload<'_>) -> Result<Self, Self::InitError> {
        payload.read()
    }
}

/// Query the vGPU type assigned to a VF by its DBDF.
#[expect(dead_code)]
pub(super) fn query_assigned_vf_type(
    _dev: &device::Device,
    cmdq: &Cmdq<'_>,
    dbdf: Dbdf,
) -> Result<u32> {
    let reply = cmdq
        .send_gmc_command(QueryAssignedVfVgpuType(dbdf), |_| Ok(()))?
        .map_err(|_| EIO)?;

    Ok(reply.vgpu_type_id)
}

struct QueryVgpuProperties(u32);

impl CommandToGsp for QueryVgpuProperties {
    const COMMAND: GmcCommandId = GmcCommandId::QueryVgpuProperties;
    type Command = u32;
    type Reply = KBox<VgpuProperties>;
    /// The reply is NVKV-encoded and varies in length.
    const MAX_REPLY_SIZE: u32 = cv!(GSP_PAGE_SIZE);
    type InitError = Infallible;

    fn init(&self) -> impl Init<Self::Command, Self::InitError> {
        self.0.to_le()
    }
}

impl MessageFromGsp for KBox<VgpuProperties> {
    type InitError = Error;

    fn read(payload: &mut MessagePayload<'_>) -> Result<Self, Self::InitError> {
        let bytes = payload.flush_into_kvec(GFP_KERNEL)?;
        let words = nvkv_words(&bytes, &[])?;
        let decoder = Decoder::new(&words, UnknownKeyPolicy::Ignore);
        let mut schema = KBox::init(VgpuPropertiesSchema::init(), GFP_KERNEL)?;
        let properties = KBox::try_init(decoder.decode(&mut *schema)?, GFP_KERNEL)?;

        Ok(properties)
    }
}

/// Query and decode the firmware properties of one vGPU type.
#[expect(dead_code)]
pub(super) fn query_vgpu_properties(
    _dev: &device::Device,
    cmdq: &Cmdq<'_>,
    type_id: u32,
) -> Result<KBox<VgpuProperties>> {
    let properties = cmdq
        .send_gmc_command(QueryVgpuProperties(type_id), |_| Ok(()))?
        .map_err(|_| EIO)?;

    if properties.type_id != type_id || properties.max_instance == 0 {
        Err(EINVAL)
    } else {
        Ok(properties)
    }
}

/// The `BOOTLOAD_GSP_VGPU_PLUGIN_TASK` command, which carries NVKV-encoded parameters.
struct BootloadPluginTask<'a>(&'a [u64]);

impl CommandToGsp for BootloadPluginTask<'_> {
    const COMMAND: GmcCommandId = GmcCommandId::BootloadGspVgpuPluginTask;
    type Command = ();
    type Reply = ();
    const MAX_REPLY_SIZE: u32 = 0;
    const COMPLETION_TIMEOUT: Delta = Delta::from_secs(10);
    type InitError = Infallible;

    fn init(&self) -> impl Init<Self::Command, Self::InitError> {
        pin_init::init_zeroed()
    }

    fn variable_payload_len(&self) -> usize {
        size_of_val(self.0)
    }

    fn init_variable_payload(&self, dst: &mut CommandPayload<'_>) -> Result {
        dst.write_all(AsBytes::as_bytes(self.0))
    }
}

/// Send BOOTLOAD and check its firmware status.
pub(super) fn send_bootload(_dev: &device::Device, cmdq: &Cmdq<'_>, payload: &[u64]) -> Result {
    cmdq.send_gmc_command(BootloadPluginTask(payload), |_| Ok(()))?
        .map_err(|_| EIO)
}

/// The `SHUTDOWN_GSP_VGPU_PLUGIN_TASK` command.
///
/// The GSP does not answer it, and sends a `SHUTDOWN_GSP_VGPU_PLUGIN_TASK_COMPLETE` event carrying
/// the same GFID once the task has stopped.
struct ShutdownPluginTask(Gfid);

impl CommandToGsp for ShutdownPluginTask {
    const COMMAND: GmcCommandId = GmcCommandId::ShutdownGspVgpuPluginTask;
    type Command = u32;
    type Reply = NoReply;
    const MAX_REPLY_SIZE: u32 = 0;
    const COMPLETION_TIMEOUT: Delta = Delta::from_secs(10);
    type InitError = Infallible;

    fn init(&self) -> impl Init<Self::Command, Self::InitError> {
        u32::from(self.0.get()).to_le()
    }
}

/// The `CLEANUP_GSP_VGPU_PLUGIN_RESOURCES` command.
struct CleanupPluginResources(Gfid);

impl CommandToGsp for CleanupPluginResources {
    const COMMAND: GmcCommandId = GmcCommandId::CleanupGspVgpuPluginResources;
    type Command = u32;
    type Reply = ();
    const MAX_REPLY_SIZE: u32 = 0;
    type InitError = Infallible;

    fn init(&self) -> impl Init<Self::Command, Self::InitError> {
        u32::from(self.0.get()).to_le()
    }
}

/// Shut down a vGPU plugin task and wait for its completion event.
pub(super) fn send_shutdown(dev: &device::Device, cmdq: &Cmdq<'_>, gfid: Gfid) -> Result {
    cmdq.send_gmc_command_await_event(ShutdownPluginTask(gfid), |event| {
        let command = event.header().command();
        if !matches!(
            command.id(),
            Ok(GmcCommandId::ShutdownGspVgpuPluginTaskComplete)
        ) {
            dev_dbg!(dev, "shutdown: ignoring GMC event {:?}\n", command);
            return Ok(None);
        }

        let (payload_0, payload_1) = event.payload();
        let mut event_gfid = [0u8; size_of::<u32>()];
        SBufferIter::new_reader([payload_0, payload_1]).read_exact(&mut event_gfid)?;

        Ok((u32::from_le_bytes(event_gfid) == u32::from(gfid.get())).then_some(()))
    })
}

/// Release firmware resources after a plugin task has stopped.
pub(super) fn send_cleanup(dev: &device::Device, cmdq: &Cmdq<'_>, gfid: Gfid) -> Result {
    cmdq.send_gmc_command(CleanupPluginResources(gfid), |_| Ok(()))?
        .map_err(|_| EIO)?;

    dev_dbg!(dev, "cleanup: gfid={} done\n", gfid.get());
    Ok(())
}

/// Negotiate the host protocol with a bootloaded GSP plugin.
pub(super) fn negotiate_plugin_version(
    dev: &device::Device,
    rpc: &mut PluginRpc<'_, '_>,
) -> Result {
    rpc.rpc_call(dev, RpcMessage::VersionNegotiation, &[])
}

/// Send an instance's encoded configuration to the GSP plugin.
pub(super) fn send_plugin_config(
    dev: &device::Device,
    rpc: &mut PluginRpc<'_, '_>,
    config: &[u64],
) -> Result {
    rpc.rpc_call_nvkv(dev, RpcMessage::SetupConfigParamsAndInit, config)
}

/// Update the bus-mastering state reported to the GSP plugin.
pub(super) fn set_plugin_bme(
    dev: &device::Device,
    rpc: &mut PluginRpc<'_, '_>,
    enable: bool,
) -> Result {
    let bme = encode_plugin_set_bme(enable)?;
    rpc.rpc_call_nvkv(dev, RpcMessage::UpdateBmeState, &bme)
}
