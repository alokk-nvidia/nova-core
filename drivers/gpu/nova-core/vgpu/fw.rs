// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

//! vGPU firmware interface.
//!
//! Exposes command constants and communication buffer layout types from
//! raw firmware bindings.

pub(super) mod commands;

use crate::gsp::bindings;

pub(super) const GMCAPI_CMD_QUERY_ASSIGNED_VF_VGPU_TYPE: u32 =
    bindings::GMCAPI_COMMANDS_GMCAPI_CMD_QUERY_ASSIGNED_VF_VGPU_TYPE;

pub(super) const GMCAPI_CMD_QUERY_VGPU_PROPERTIES: u32 =
    bindings::GMCAPI_COMMANDS_GMCAPI_CMD_QUERY_VGPU_PROPERTIES;
