/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
 *
 * NVIDIA nova-core fwctl interface. All fwctl traffic uses the GMC API
 * transport. User-space supplies a command_id and optional payload; the
 * kernel constructs the GMC transport headers, validates the command_id
 * against the permitted set for the requested scope, and returns only the
 * response payload.
 *
 * The device_type for this file is FWCTL_DEVICE_TYPE_NOVA_CORE.
 */
#ifndef _UAPI_FWCTL_NOVA_CORE_H
#define _UAPI_FWCTL_NOVA_CORE_H

#include <linux/types.h>

/**
 * struct fwctl_rpc_nova_core - ioctl(FWCTL_RPC) in/out buffer format
 * @command_id: GMC API command identifier (from GMCAPI_COMMANDS).
 * @reserved: Must be zero.
 *
 * The request buffer passed to FWCTL_RPC begins with this header,
 * followed by the command-specific payload bytes (if any).
 *
 * The response buffer returned by FWCTL_RPC begins with this same
 * header (echoing the command_id), followed by the response payload.
 */
struct fwctl_rpc_nova_core {
	__u32 command_id;
	__u32 reserved;
};

#endif
