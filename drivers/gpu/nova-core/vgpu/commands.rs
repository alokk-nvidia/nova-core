// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

//! vGPU firmware command operations.
//!
//! Sends requests, checks responses and coordinates the firmware command
//! sequences used by instance lifecycle operations.

use kernel::{
    device,
    prelude::*, //
};

use crate::gsp::{
    cmdq::Cmdq,
    nvkv::{
        nvkv_words,
        Decoder,
        Schema,
        UnknownKeyPolicy, //
    }, //
};

use super::{
    fw::{
        commands::{
            VgpuPropertiesSchema, //
        },
        GMCAPI_CMD_QUERY_ASSIGNED_VF_VGPU_TYPE,
        GMCAPI_CMD_QUERY_VGPU_PROPERTIES, //
    }, //
};

pub(super) use super::fw::commands::{
    Dbdf,
    VgpuProperties, //
};

/// Reports a matching command's raw firmware status before translating rejection to `EIO`.
fn check_status(dev: &device::Device<device::Bound>, command_id: u32, status: u32) -> Result {
    if status == 0 {
        Ok(())
    } else {
        dev_err!(
            dev,
            "GMC command {:#x} rejected: NV_STATUS={:#x}\n",
            command_id,
            status,
        );
        Err(EIO)
    }
}

/// Query the vGPU type assigned to a VF by its DBDF.
#[expect(dead_code)]
pub(super) fn query_assigned_vf_type(
    dev: &device::Device<device::Bound>,
    cmdq: &Cmdq<'_>,
    dbdf: Dbdf,
) -> Result<u32> {
    let command_id = GMCAPI_CMD_QUERY_ASSIGNED_VF_VGPU_TYPE;
    let payload = u64::from(dbdf.into_raw()).to_le_bytes();
    // Preserve the firmware receive budget; only the leading type ID is consumed.
    let response = cmdq.send_gmc_and_receive(command_id, &payload, 64)?;
    check_status(dev, command_id, response.status)?;

    let bytes = response.payload.first_chunk::<4>().ok_or(ENODEV)?;
    Ok(u32::from_le_bytes(*bytes))
}

/// Query and decode the firmware properties of one vGPU type.
#[expect(dead_code)]
pub(super) fn query_vgpu_properties(
    dev: &device::Device<device::Bound>,
    cmdq: &Cmdq<'_>,
    type_id: u32,
) -> Result<KBox<VgpuProperties>> {
    let command_id = GMCAPI_CMD_QUERY_VGPU_PROPERTIES;
    // NVKV replies vary in length. This is a receive capacity, not their encoded size.
    let response = cmdq.send_gmc_and_receive(command_id, &type_id.to_le_bytes(), 4096)?;
    check_status(dev, command_id, response.status)?;

    let words = nvkv_words(&response.payload, &[])?;
    let decoder = Decoder::new(&words, UnknownKeyPolicy::Ignore);
    let mut schema = KBox::init(VgpuPropertiesSchema::init(), GFP_KERNEL)?;
    let properties = KBox::try_init(decoder.decode(&mut *schema)?, GFP_KERNEL)?;
    if properties.type_id != type_id || properties.max_instance == 0 {
        Err(EINVAL)
    } else {
        Ok(properties)
    }
}
