// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

//! vGPU firmware command operations.
//!
//! Sends requests, checks responses and coordinates the firmware command
//! sequences used by instance lifecycle operations.

use kernel::{
    device,
    num::casts::{
        usize_as_u64,
        usize_into_u32, //
    },
    prelude::*,
    time::Delta,
    transmute::AsBytes, //
};

use crate::{
    gsp::{
        cmdq::Cmdq,
        nvkv::{
            nvkv_words,
            Decoder,
            Schema,
            UnknownKeyPolicy, //
        }, //
    },
    mm::PAGE_SIZE, //
};

use super::{
    fw::{
        commands::{
            encode_plugin_set_bme,
            AllocCeutilsRequest,
            AllocCeutilsResponse,
            FreeCeutilsRequest,
            ScrubGuestFbRequest,
            ScrubGuestFbResponse,
            VgpuPropertiesSchema, //
        },
        RpcMessage,
        GMCAPI_CMD_BOOTLOAD_GSP_VGPU_PLUGIN_TASK,
        GMCAPI_CMD_CLEANUP_GSP_VGPU_PLUGIN_RESOURCES,
        GMCAPI_CMD_QUERY_ASSIGNED_VF_VGPU_TYPE,
        GMCAPI_CMD_QUERY_VGPU_PROPERTIES,
        GMCAPI_CMD_SHUTDOWN_GSP_VGPU_PLUGIN_TASK,
        GMCAPI_CMD_SHUTDOWN_GSP_VGPU_PLUGIN_TASK_COMPLETE,
        GMCAPI_CMD_VGPU_MGR_ALLOC_GSP_CEUTILS,
        GMCAPI_CMD_VGPU_MGR_FREE_GSP_CEUTILS,
        GMCAPI_CMD_VGPU_MGR_SCRUB_GUEST_FB,
        NV_ADDR_FBMEM, //
    },
    gsp_plugin_rpc::PluginRpc,
    instance::Gfid, //
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

/// Send BOOTLOAD and check its firmware status.
pub(super) fn send_bootload(
    dev: &device::Device<device::Bound>,
    cmdq: &Cmdq<'_>,
    payload: &[u64],
) -> Result {
    let command_id = GMCAPI_CMD_BOOTLOAD_GSP_VGPU_PLUGIN_TASK;
    let response = cmdq.send_gmc_and_receive_timeout(
        command_id,
        AsBytes::as_bytes(payload),
        0,
        Delta::from_secs(10),
    )?;
    check_status(dev, command_id, response.status)
}

/// Shut down a vGPU plugin task and wait for its completion event.
pub(super) fn send_shutdown(
    dev: &device::Device<device::Bound>,
    cmdq: &Cmdq<'_>,
    gfid: Gfid,
) -> Result {
    let payload = u32::from(gfid.get()).to_le_bytes();
    cmdq.send_gmc_and_wait_event(
        GMCAPI_CMD_SHUTDOWN_GSP_VGPU_PLUGIN_TASK,
        &payload,
        Delta::from_secs(10),
        |command_id, _max_response_size, _sequence, payload_0, payload_1| {
            Ok(
                command_id == GMCAPI_CMD_SHUTDOWN_GSP_VGPU_PLUGIN_TASK_COMPLETE
                    && payload
                        .iter()
                        .copied()
                        .eq(Iterator::chain(payload_0.iter(), payload_1)
                            .take(payload.len())
                            .copied()),
            )
        },
        |command_id, _max_response_size, _sequence, _payload_0, _payload_1| {
            dev_dbg!(
                dev,
                "shutdown: ignoring unrelated event command={:#x}\n",
                command_id,
            );
            Ok(())
        },
    )
}

/// Release firmware resources after a plugin task has stopped.
pub(super) fn send_cleanup(
    dev: &device::Device<device::Bound>,
    cmdq: &Cmdq<'_>,
    gfid: Gfid,
) -> Result {
    let command_id = GMCAPI_CMD_CLEANUP_GSP_VGPU_PLUGIN_RESOURCES;
    let response =
        cmdq.send_gmc_and_receive(command_id, &u32::from(gfid.get()).to_le_bytes(), 0)?;
    check_status(dev, command_id, response.status)?;
    dev_dbg!(dev, "cleanup: gfid={} done\n", gfid.get());
    Ok(())
}

/// Negotiate the host protocol with a bootloaded GSP plugin.
pub(super) fn negotiate_plugin_version(
    dev: &device::Device<device::Bound>,
    rpc: &mut PluginRpc<'_, '_>,
) -> Result {
    rpc.rpc_call(dev, RpcMessage::VersionNegotiation, &[])
}

/// Send an instance's encoded configuration to the GSP plugin.
pub(super) fn send_plugin_config(
    dev: &device::Device<device::Bound>,
    rpc: &mut PluginRpc<'_, '_>,
    config: &[u64],
) -> Result {
    rpc.rpc_call_nvkv(dev, RpcMessage::SetupConfigParamsAndInit, config)
}

/// Update the bus-mastering state reported to the GSP plugin.
pub(super) fn set_plugin_bme(
    dev: &device::Device<device::Bound>,
    rpc: &mut PluginRpc<'_, '_>,
    enable: bool,
) -> Result {
    let bme = encode_plugin_set_bme(enable)?;
    rpc.rpc_call_nvkv(dev, RpcMessage::UpdateBmeState, &bme)
}

/// Whether a failed allocation may still have transferred CHID ownership to firmware.
#[expect(dead_code)]
pub(super) enum CeUtilsAllocError {
    /// A matching firmware response explicitly rejected the allocation.
    NotOwned(Error),
    /// The request may have completed despite a transport or response-validation error.
    MayOwn(Error),
}

/// Allocate a CeUtils channel and validate its semaphore description.
#[expect(dead_code)]
pub(super) fn alloc_ceutils(
    dev: &device::Device<device::Bound>,
    cmdq: &Cmdq<'_>,
    gfid: Gfid,
    chid: u32,
) -> core::result::Result<u64, CeUtilsAllocError> {
    let request = AllocCeutilsRequest {
        gfid: u32::from(gfid.get()).to_le(),
        fixed_chid: chid.to_le(),
        force_ceid: u32::MAX.to_le(),
        swizz_id: 0,
    };

    dev_dbg!(dev, "alloc CeUtils: gfid={} chid={}\n", gfid.get(), chid,);

    let command_id = GMCAPI_CMD_VGPU_MGR_ALLOC_GSP_CEUTILS;
    let response = cmdq
        .send_gmc_and_receive(
            command_id,
            IntoBytes::as_bytes(&request),
            usize_into_u32::<{ size_of::<AllocCeutilsResponse>() }>(),
        )
        .map_err(CeUtilsAllocError::MayOwn)?;
    check_status(dev, command_id, response.status).map_err(CeUtilsAllocError::NotOwned)?;
    let (response, _) = AllocCeutilsResponse::read_from_prefix(&response.payload)
        .map_err(|_| CeUtilsAllocError::MayOwn(EMSGSIZE))?;

    let semaphore_address = u64::from_le(response.semaphore_address);
    let semaphore_aperture = u32::from_le(response.semaphore_aperture);
    let page_size = usize_as_u64(PAGE_SIZE);

    if semaphore_address == 0
        || !semaphore_address.is_multiple_of(page_size)
        || semaphore_aperture != NV_ADDR_FBMEM
    {
        return Err(CeUtilsAllocError::MayOwn(EINVAL));
    }

    dev_dbg!(
        dev,
        "alloc CeUtils: gfid={} semaphore={:#x}\n",
        gfid.get(),
        semaphore_address,
    );
    Ok(semaphore_address)
}

/// Release a CeUtils allocation, including one whose allocation reply was lost.
#[expect(dead_code)]
pub(super) fn free_ceutils(
    dev: &device::Device<device::Bound>,
    cmdq: &Cmdq<'_>,
    gfid: Gfid,
) -> Result {
    let request = FreeCeutilsRequest {
        gfid: u32::from(gfid.get()).to_le(),
    };

    dev_dbg!(dev, "free CeUtils: gfid={}\n", gfid.get());
    let command_id = GMCAPI_CMD_VGPU_MGR_FREE_GSP_CEUTILS;
    let response = cmdq.send_gmc_and_receive(command_id, IntoBytes::as_bytes(&request), 0)?;
    check_status(dev, command_id, response.status)
}

/// Submit an asynchronous guest FB scrub and return its work identifier.
#[expect(dead_code)]
pub(super) fn submit_ceutils_scrub(
    dev: &device::Device<device::Bound>,
    cmdq: &Cmdq<'_>,
    gfid: Gfid,
    fb_offset: u64,
    fb_size: u64,
) -> Result<u32> {
    let request = ScrubGuestFbRequest {
        gfid: u32::from(gfid.get()).to_le(),
        reserved: 0,
        fb_offset: fb_offset.to_le(),
        fb_size: fb_size.to_le(),
    };

    dev_dbg!(
        dev,
        "submit scrub: gfid={} offset={:#x} size={:#x}\n",
        gfid.get(),
        fb_offset,
        fb_size,
    );

    let command_id = GMCAPI_CMD_VGPU_MGR_SCRUB_GUEST_FB;
    let response = cmdq.send_gmc_and_receive(
        command_id,
        IntoBytes::as_bytes(&request),
        usize_into_u32::<{ size_of::<ScrubGuestFbResponse>() }>(),
    )?;
    check_status(dev, command_id, response.status)?;
    let (response, _) =
        ScrubGuestFbResponse::read_from_prefix(&response.payload).map_err(|_| EMSGSIZE)?;
    u32::try_from(u64::from_le(response.work_id)).map_err(|_| EOVERFLOW)
}
