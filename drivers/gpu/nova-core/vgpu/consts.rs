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
    pub(crate) const VGPU_MGMT_QUERY_PROPERTIES: u32 = 0x1000_0006;
    pub(crate) const VGPU_MGMT_QUERY_ASSIGNED_VF: u32 = 0x1000_0007;
}
