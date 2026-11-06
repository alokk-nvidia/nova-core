// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

//! vGPU firmware interface.
//!
//! Exposes command constants and communication buffer layout types from
//! raw firmware bindings.

pub(super) mod commands;
mod r000_00;

// Alias to avoid repeating the version number with every use.
use r000_00 as bindings;

pub(super) use bindings::{
    GSP_PLUGIN_BOOTLOADED,
    VGPU_CPU_GSP_COMMUNICATION_BUFF_TOTAL_SIZE,
    VGPU_CPU_GSP_CTRL_BUFF_REGION as RawControlRegion,
    VGPU_CPU_GSP_CTRL_BUFF_REGION_SIZE,
    VGPU_CPU_GSP_CTRL_BUFF_VERSION,
    VGPU_CPU_GSP_ERROR_BUFF_REGION_SIZE,
    VGPU_CPU_GSP_GUEST_RPC_TRACE_BUFF_REGION_SIZE,
    VGPU_CPU_GSP_INIT_TASK_LOG_BUFF_REGION_SIZE,
    VGPU_CPU_GSP_KERNEL_TASK_LOG_BUFF_REGION_SIZE,
    VGPU_CPU_GSP_MESSAGE_BUFF_REGION_SIZE,
    VGPU_CPU_GSP_MIGRATION_BUFF_REGION_SIZE,
    VGPU_CPU_GSP_RESPONSE_BUFF_REGION as RawResponseRegion,
    VGPU_CPU_GSP_RESPONSE_BUFF_REGION_SIZE,
    VGPU_CPU_GSP_VGPU_TASK_LOG_BUFF_REGION_SIZE, //
};
