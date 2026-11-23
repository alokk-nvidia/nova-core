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
    vgpu::{
        fw::commands::{
            AssignedVfVgpuType,
            VgpuProperties,
            VgpuPropertiesSchema, //
        },
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
