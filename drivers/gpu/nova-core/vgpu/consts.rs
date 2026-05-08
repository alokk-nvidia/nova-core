// SPDX-License-Identifier: GPL-2.0

/// NVKV key constants for QUERY_VGPU_PROPERTIES response decoding.
pub(crate) mod vgpu_prop_keys {
    pub(crate) const TYPE_NAME: u16 = 0x3100;
    pub(crate) const CLASS: u16 = 0x3101;
    pub(crate) const TYPE_ID: u16 = 0x3102;
    pub(crate) const BAR1_LENGTH: u16 = 0x3103;
    pub(crate) const MAX_INSTANCE: u16 = 0x3104;
    pub(crate) const ECC: u16 = 0x3105;
    pub(crate) const PROFILE_SIZE: u16 = 0x3106;
    pub(crate) const MAX_FPS: u16 = 0x3107;
    pub(crate) const NUM_HEADS: u16 = 0x3108;
    pub(crate) const MAX_RES_X: u16 = 0x3109;
    pub(crate) const MAX_RES_Y: u16 = 0x310A;
    pub(crate) const DEV_ID: u16 = 0x310B;
    pub(crate) const SUBSYSTEM_ID: u16 = 0x310C;
    pub(crate) const FB_LENGTH: u16 = 0x310D;
    pub(crate) const GSP_HEAP_SIZE: u16 = 0x310E;
    pub(crate) const FB_RESERVATION: u16 = 0x310F;
}

/// GMCAPI command IDs for vGPU management.
pub(crate) mod gmcapi {
    pub(crate) const VGPU_MGMT_ADD_TYPE: u32 = 0x1000_0001;
    pub(crate) const VGPU_MGMT_QUERY_SUPPORTED: u32 = 0x1000_0002;
    pub(crate) const VGPU_MGMT_QUERY_CREATABLE: u32 = 0x1000_0003;
    pub(crate) const VGPU_MGMT_ASSIGN_TYPE: u32 = 0x1000_0004;
    pub(crate) const VGPU_MGMT_DEASSIGN_TYPE: u32 = 0x1000_0005;
    pub(crate) const VGPU_MGMT_QUERY_PROPERTIES: u32 = 0x1000_0006;
    pub(crate) const VGPU_MGMT_QUERY_ASSIGNED_VF: u32 = 0x1000_0007;
    pub(crate) const VGPU_BOOTLOAD: u32 = 0x1000_0020;
    pub(crate) const VGPU_SHUTDOWN: u32 = 0x1000_0021;
    pub(crate) const VGPU_SHUTDOWN_COMPLETE: u32 = 0x1000_0022;
    pub(crate) const VGPU_CLEANUP: u32 = 0x1000_0023;
}

/// NVKV key constants for VGPU_BOOTLOAD command encoding.
pub(crate) mod bootload_keys {
    pub(crate) const DBDF: u16 = 0x0001;
    pub(crate) const GFID: u16 = 0x0002;
    pub(crate) const VGPU_TYPE: u16 = 0x0003;
    pub(crate) const VM_PID: u16 = 0x0004;
    pub(crate) const SWIZZ_ID: u16 = 0x0005;
    pub(crate) const NUM_CHANNELS: u16 = 0x0006;
    pub(crate) const NUM_PLUGIN_CHANNELS: u16 = 0x0007;
    pub(crate) const GUEST_FB_SEGMENT_COUNT: u16 = 0x0008;

    pub(crate) const OPTIONS: u16 = 0x1000;
    pub(crate) const CHANNEL_MAPPING: u16 = 0x1001;
    pub(crate) const GUEST_FB_SEGMENT_PHYS_ADDR: u16 = 0x1002;
    pub(crate) const GUEST_FB_SEGMENT_LENGTH: u16 = 0x1003;
    pub(crate) const PLUGIN_HEAP_PHYS_ADDR: u16 = 0x1004;
    pub(crate) const PLUGIN_HEAP_LENGTH: u16 = 0x1005;
    pub(crate) const CTRL_BUFF_OFFSET: u16 = 0x1006;
    pub(crate) const INIT_TASK_LOG_OFFSET: u16 = 0x1007;
    pub(crate) const INIT_TASK_LOG_SIZE: u16 = 0x1008;
    pub(crate) const VGPU_TASK_LOG_OFFSET: u16 = 0x1009;
    pub(crate) const VGPU_TASK_LOG_SIZE: u16 = 0x100A;
    pub(crate) const KERNEL_LOG_OFFSET: u16 = 0x100B;
    pub(crate) const KERNEL_LOG_SIZE: u16 = 0x100C;
    pub(crate) const MIG_RM_HEAP_PHYS_ADDR: u16 = 0x100D;
    pub(crate) const MIG_RM_HEAP_LENGTH: u16 = 0x100E;
}

/// Constants for vGPU plugin RPC communication.
pub(crate) mod plugin_rpc {
    pub(crate) const GSP_PLUGIN_BOOTLOADED: u32 = 0x4E65_4A6F;
    pub(crate) const CTRL_BUF_MSG_SEQ_NUM_OFFSET: u64 = 8;
    pub(crate) const PLUGIN_BOOT_TIMEOUT_MS: u64 = 10_000;

    pub(crate) const CTRL_SIZE: u64 = 4 * 1024;
    pub(crate) const RESPONSE_SIZE: u64 = 4 * 1024;
    pub(crate) const MESSAGE_SIZE: u64 = 4 * 1024;
    pub(crate) const MIGRATION_SIZE: u64 = 2 * 1024 * 1024;
    pub(crate) const ERROR_SIZE: u64 = 4 * 1024;
    pub(crate) const GUEST_RPC_TRACE_SIZE: u64 = 64 * 1024;

    pub(crate) const MIGRATION_BUFF_OFFSET: u64 = CTRL_SIZE + RESPONSE_SIZE + MESSAGE_SIZE;
    pub(crate) const ERROR_BUFF_OFFSET: u64 = MIGRATION_BUFF_OFFSET + MIGRATION_SIZE;
    pub(crate) const GUEST_RPC_TRACE_BUFF_OFFSET: u64 = ERROR_BUFF_OFFSET + ERROR_SIZE;

    pub(crate) const INIT_LOG_SIZE: u64 = 128 * 1024;
    pub(crate) const VGPU_LOG_SIZE: u64 = 256 * 1024;
    pub(crate) const KERNEL_LOG_SIZE: u64 = 64 * 1024;

    pub(crate) const INIT_TASK_LOG_OFFSET: u64 =
        CTRL_SIZE + RESPONSE_SIZE + MESSAGE_SIZE + MIGRATION_SIZE + ERROR_SIZE;
    pub(crate) const VGPU_TASK_LOG_OFFSET: u64 = INIT_TASK_LOG_OFFSET + INIT_LOG_SIZE;
    pub(crate) const KERNEL_LOG_OFFSET: u64 = VGPU_TASK_LOG_OFFSET + VGPU_LOG_SIZE;

    pub(crate) const CTRL_BUFF_VERSION: u32 = 2;
    pub(crate) const NV_VIRTUAL_FUNCTION_PRIV_DOORBELL: usize = 0xb8_0000 + 0x2200;

    pub(crate) const FEATURE_FLAG_ENABLE_UVM: u64 = 1 << 3;
    pub(crate) const FEATURE_FLAG_VMM_MIGRATION: u64 = 1 << 5;

    #[repr(u32)]
    #[derive(Clone, Copy)]
    pub(crate) enum RpcMsg {
        VersionNegotiation = 1,
        SetupConfigParamsAndInit = 2,
        Reset = 3,
        UpdateBmeState = 13,
    }
}

/// NVKV key constants for plugin config params encoding.
pub(crate) mod config_keys {
    pub(crate) const UUID: u16 = 0x001;
    pub(crate) const DBDF: u16 = 0x002;
    pub(crate) const DEV_INST: u16 = 0x004;
    pub(crate) const VGPU_TYPE: u16 = 0x005;
    pub(crate) const VM_PID: u16 = 0x006;
    pub(crate) const SWIZZ_ID: u16 = 0x010;
    pub(crate) const NUM_CHANNELS: u16 = 0x011;
    pub(crate) const NUM_PLUGIN_CHANNELS: u16 = 0x012;
    pub(crate) const VMM_CAP: u16 = 0x020;
    pub(crate) const MIGRATION_FEATURE: u16 = 0x021;
    pub(crate) const HYPERVISOR_TYPE: u16 = 0x022;
    pub(crate) const CPU_ARCH: u16 = 0x023;
    pub(crate) const PAGE_SIZE: u16 = 0x024;
    pub(crate) const FEATURE_FLAGS: u16 = 0x030;

    pub(crate) const HYPERVISOR_UNKNOWN: u32 = 4;
    pub(crate) const MIGRATION_FEATURE_KVM: u32 = 0x4000;
    pub(crate) const CPU_ARCH_X86_64: u32 = 2;
    pub(crate) const PAGE_SIZE_4K: u64 = 4096;
    pub(crate) const SWIZZ_ID_NONE: u32 = 0xFFFF_FFFF;
}

/// NVKV key constants for plugin set bme encoding.
pub(crate) mod set_bme_keys {
    pub(crate) const BME_ENABLE: u16 = 0x100;
}
