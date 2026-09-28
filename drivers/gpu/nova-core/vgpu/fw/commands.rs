// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

//! Wire types and codecs for vGPU commands.
//!
//! Defines request encoders and response schemas independently of command
//! submission and instance lifecycle operations.

use kernel::{
    alloc::ArrayVec,
    bitfield, //
};

use crate::{
    gsp::nvkv::{
        nvkv_decode,
        Array,
        Key,
        KeyId,
        Required, //
    }, //
};

bitfield! {
    pub(crate) struct Dbdf(u32) {
        2:0 function;
        7:3 device;
        15:8 bus;
        31:16 domain;
    }
}

nvkv_decode! {
    pub(crate) struct VgpuPropertiesSchema => VgpuProperties {
        // TODO: `name`/`class` required?
        name: Array<u8, { VgpuProperties::STRING_LEN }, { Self::TYPE_NAME_KEY }>,
        class: Array<u8, { VgpuProperties::STRING_LEN }, { Self::CLASS_KEY }>,
        type_id: Required<Key<u32, { Self::TYPE_ID_KEY }>>,
        bar1_length: Required<Key<u64, { Self::BAR1_LENGTH_KEY }>>,
        max_instance: Required<Key<u32, { Self::MAX_INSTANCE_KEY }>>,
        ecc: Key<u32, { Self::ECC_KEY }>,
        profile_size: Required<Key<u64, { Self::PROFILE_SIZE_KEY }>>,
        max_fps: Key<u32, { Self::MAX_FPS_KEY }>,
        num_heads: Key<u32, { Self::NUM_HEADS_KEY }>,
        max_res_x: Key<u32, { Self::MAX_RES_X_KEY }>,
        max_res_y: Key<u32, { Self::MAX_RES_Y_KEY }>,
        dev_id: Required<Key<u32, { Self::DEV_ID_KEY }>>,
        subsystem_id: Required<Key<u32, { Self::SUBSYSTEM_ID_KEY }>>,
        fb_length: Required<Key<u64, { Self::FB_LENGTH_KEY }>>,
        gsp_heap_size: Required<Key<u64, { Self::GSP_HEAP_SIZE_KEY }>>,
        fb_reservation: Required<Key<u64, { Self::FB_RESERVATION_KEY }>>,
    }
}

impl VgpuPropertiesSchema {
    const TYPE_NAME_KEY: KeyId = 0x3100;
    const CLASS_KEY: KeyId = 0x3101;
    const TYPE_ID_KEY: KeyId = 0x3102;
    const BAR1_LENGTH_KEY: KeyId = 0x3103;
    const MAX_INSTANCE_KEY: KeyId = 0x3104;
    const ECC_KEY: KeyId = 0x3105;
    const PROFILE_SIZE_KEY: KeyId = 0x3106;
    const MAX_FPS_KEY: KeyId = 0x3107;
    const NUM_HEADS_KEY: KeyId = 0x3108;
    const MAX_RES_X_KEY: KeyId = 0x3109;
    const MAX_RES_Y_KEY: KeyId = 0x310A;
    const DEV_ID_KEY: KeyId = 0x310B;
    const SUBSYSTEM_ID_KEY: KeyId = 0x310C;
    const FB_LENGTH_KEY: KeyId = 0x310D;
    const GSP_HEAP_SIZE_KEY: KeyId = 0x310E;
    const FB_RESERVATION_KEY: KeyId = 0x310F;
}

pub(crate) struct VgpuProperties {
    name: ArrayVec<u8, { Self::STRING_LEN }>,
    class: ArrayVec<u8, { Self::STRING_LEN }>,
    pub(crate) type_id: u32,
    pub(crate) bar1_length: u64,
    pub(crate) max_instance: u32,
    ecc: u32,
    profile_size: u64,
    max_fps: u32,
    num_heads: u32,
    max_res_x: u32,
    max_res_y: u32,
    pub(crate) dev_id: u32,
    pub(crate) subsystem_id: u32,
    pub(crate) fb_length: u64,
    pub(crate) gsp_heap_size: u64,
    fb_reservation: u64,
}

impl VgpuProperties {
    const STRING_LEN: usize = 64;
}
