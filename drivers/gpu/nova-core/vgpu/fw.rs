// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

//! vGPU firmware interface.
//!
//! Exposes command constants and communication buffer layout types from
//! raw firmware bindings.

pub(super) mod commands;
mod r000_00;
