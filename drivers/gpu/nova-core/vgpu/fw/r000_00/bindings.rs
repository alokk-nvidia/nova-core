// SPDX-License-Identifier: GPL-2.0

pub type __u8 = ffi::c_uchar;
pub type __u16 = ffi::c_ushort;
pub type __u32 = ffi::c_uint;
pub type __u64 = ffi::c_ulonglong;
pub type u8_ = __u8;
pub type u16_ = __u16;
pub type u32_ = __u32;
pub type u64_ = __u64;
pub const GSP_PLUGIN_BOOTLOADED: u32 = 1315261039;
pub const VGPU_CPU_GSP_CTRL_BUFF_VERSION: u32 = 2;
pub const VGPU_CPU_GSP_CTRL_BUFF_REGION_SIZE: u32 = 4096;
pub const VGPU_CPU_GSP_RESPONSE_BUFF_REGION_SIZE: u32 = 4096;
pub const VGPU_CPU_GSP_MESSAGE_BUFF_REGION_SIZE: u32 = 4096;
pub const VGPU_CPU_GSP_MIGRATION_BUFF_REGION_SIZE: u32 = 2097152;
pub const VGPU_CPU_GSP_ERROR_BUFF_REGION_SIZE: u32 = 4096;
pub const VGPU_CPU_GSP_INIT_TASK_LOG_BUFF_REGION_SIZE: u32 = 131072;
pub const VGPU_CPU_GSP_VGPU_TASK_LOG_BUFF_REGION_SIZE: u32 = 262144;
pub const VGPU_CPU_GSP_KERNEL_TASK_LOG_BUFF_REGION_SIZE: u32 = 65536;
pub const VGPU_CPU_GSP_GUEST_RPC_TRACE_BUFF_REGION_SIZE: u32 = 65536;
pub const VGPU_CPU_GSP_COMMUNICATION_BUFF_TOTAL_SIZE: u32 = 2637824;
pub type VGPU_CPU_GSP_BOOL = u32_;
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, MaybeZeroable)]
pub struct VGPU_CPU_GSP_VGX_VERSION {
    pub major_number: u32_,
    pub minor_number: u32_,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, MaybeZeroable)]
pub struct VGPU_CPU_GSP_GUEST_INFO {
    pub vgx_version: VGPU_CPU_GSP_VGX_VERSION,
    pub guest_driver_version_buffer_length: u32_,
    pub guest_version_buffer_length: u32_,
    pub guest_title_buffer_length: u32_,
    pub guest_changelist_number: u32_,
    pub guest_driver_version_buffer: [ffi::c_char; 256usize],
    pub guest_version_buffer: [ffi::c_char; 256usize],
    pub guest_title_buffer: [ffi::c_char; 256usize],
    pub guest_branch_buffer: [ffi::c_char; 256usize],
}
impl Default for VGPU_CPU_GSP_GUEST_INFO {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[repr(C)]
#[derive(Copy, Clone, MaybeZeroable)]
pub union VGPU_CPU_GSP_CTRL_BUFF_REGION {
    pub buf: [u8_; 4096usize],
    pub __bindgen_anon_1: VGPU_CPU_GSP_CTRL_BUFF_REGION__bindgen_ty_1,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, MaybeZeroable)]
pub struct VGPU_CPU_GSP_CTRL_BUFF_REGION__bindgen_ty_1 {
    pub version: u32_,
    pub message_type: u32_,
    pub message_seq_num: u32_,
    pub __bindgen_padding_0: [u8; 4usize],
    pub response_buff_offset: u64_,
    pub message_buff_offset: u64_,
    pub migration_buff_offset: u64_,
    pub error_buff_offset: u64_,
    pub guest_rpc_trace_buff_offset: u64_,
    pub migration_buf_cpu_access_offset: u32_,
    pub is_migration_in_progress: u8_,
    pub __bindgen_padding_1: [u8; 3usize],
    pub error_buff_cpu_get_idx: u32_,
    pub guest_rpc_trace_buff_cpu_get_idx: u32_,
    pub attached_vgpu_count: u32_,
    pub is_gr_init_done: u8_,
    pub __bindgen_padding_2: [u8; 3usize],
    pub host_info: [VGPU_CPU_GSP_CTRL_BUFF_REGION__bindgen_ty_1__bindgen_ty_1; 16usize],
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, MaybeZeroable)]
pub struct VGPU_CPU_GSP_CTRL_BUFF_REGION__bindgen_ty_1__bindgen_ty_1 {
    pub vgpu_type_id: u32_,
    pub host_gpu_pci_id: u32_,
    pub pci_dev_id: u32_,
    pub vgpu_uuid: [u8_; 16usize],
}
impl Default for VGPU_CPU_GSP_CTRL_BUFF_REGION {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
pub const MESSAGE_NV_VGPU_CPU_RPC_MSG_VERSION_NEGOTIATION: MESSAGE = 1;
pub const MESSAGE_NV_VGPU_CPU_RPC_MSG_SETUP_CONFIG_PARAMS_AND_INIT: MESSAGE = 2;
pub const MESSAGE_NV_VGPU_CPU_RPC_MSG_RESET: MESSAGE = 3;
pub const MESSAGE_NV_VGPU_CPU_RPC_MSG_MIGRATION_STOP_WORK: MESSAGE = 4;
pub const MESSAGE_NV_VGPU_CPU_RPC_MSG_MIGRATION_CANCEL_STOP: MESSAGE = 5;
pub const MESSAGE_NV_VGPU_CPU_RPC_MSG_MIGRATION_SAVE_STATE: MESSAGE = 6;
pub const MESSAGE_NV_VGPU_CPU_RPC_MSG_MIGRATION_CANCEL_SAVE: MESSAGE = 7;
pub const MESSAGE_NV_VGPU_CPU_RPC_MSG_MIGRATION_RESTORE_STATE: MESSAGE = 8;
pub const MESSAGE_NV_VGPU_CPU_RPC_MSG_MIGRATION_RESTORE_DEFERRED_STATE: MESSAGE = 9;
pub const MESSAGE_NV_VGPU_CPU_RPC_MSG_MIGRATION_RESUME_WORK: MESSAGE = 10;
pub const MESSAGE_NV_VGPU_CPU_RPC_MSG_CONSOLE_VNC_STATE: MESSAGE = 11;
pub const MESSAGE_NV_VGPU_CPU_RPC_MSG_VF_BAR0_REG_ACCESS: MESSAGE = 12;
pub const MESSAGE_NV_VGPU_CPU_RPC_MSG_UPDATE_BME_STATE: MESSAGE = 13;
pub const MESSAGE_NV_VGPU_CPU_RPC_MSG_RESET_MIGRATION_BUFFER_PTR: MESSAGE = 14;
pub const MESSAGE_NV_VGPU_CPU_RPC_MSG_RELEASE_CLIENT_DATABASE: MESSAGE = 15;
pub const MESSAGE_NV_VGPU_CPU_RPC_MSG_CHECK_IS_ALIVE: MESSAGE = 16;
pub const MESSAGE_NV_VGPU_CPU_RPC_MSG_SEND_STATIC_INFO: MESSAGE = 17;
pub const MESSAGE_NV_VGPU_CPU_RPC_MSG_MAX: MESSAGE = 18;
pub type MESSAGE = ffi::c_uint;
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, MaybeZeroable)]
pub struct VGPU_CPU_GSP_DISPLAYLESS_SURFACE {
    pub sequence_update_start: u64_,
    pub sequence_update_end: u64_,
    pub effective_fb_page_size: u32_,
    pub rect_width: u32_,
    pub rect_height: u32_,
    pub surface_width: u32_,
    pub surface_height: u32_,
    pub surface_size: u32_,
    pub surface_offset: u32_,
    pub surface_format: u32_,
    pub surface_kind: u32_,
    pub surface_pitch: u32_,
    pub surface_type: u32_,
    pub surface_block_height: u8_,
    pub __bindgen_padding_0: [u8; 3usize],
    pub is_blanking_enabled: VGPU_CPU_GSP_BOOL,
    pub is_flip_pending: VGPU_CPU_GSP_BOOL,
    pub is_free_pending: VGPU_CPU_GSP_BOOL,
    pub is_memory_blocklinear: VGPU_CPU_GSP_BOOL,
}
#[repr(C)]
#[derive(Copy, Clone, MaybeZeroable)]
pub union VGPU_CPU_GSP_RESPONSE_BUFF_REGION {
    pub buf: [u8_; 4096usize],
    pub __bindgen_anon_1: VGPU_CPU_GSP_RESPONSE_BUFF_REGION__bindgen_ty_1,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, MaybeZeroable)]
pub struct VGPU_CPU_GSP_RESPONSE_BUFF_REGION__bindgen_ty_1 {
    pub message_seq_num_received: u32_,
    pub message_seq_num_processed: u32_,
    pub result_code: u32_,
    pub guest_rpc_version: u32_,
    pub migration_buf_gsp_access_offset: u32_,
    pub migration_state_save_complete: u32_,
    pub is_migration_allowed: VGPU_CPU_GSP_BOOL,
    pub __bindgen_padding_0: [u8; 4usize],
    pub surface: [VGPU_CPU_GSP_DISPLAYLESS_SURFACE; 4usize],
    pub error_buff_gsp_put_idx: u32_,
    pub grid_license_state: u32_,
    pub guest_os_type: u32_,
    pub frl_config: u32_,
    pub guest_info: VGPU_CPU_GSP_GUEST_INFO,
    pub is_guest_info_populated: VGPU_CPU_GSP_BOOL,
    pub guest_rpc_trace_buff_gsp_put_idx: u32_,
}
impl Default for VGPU_CPU_GSP_RESPONSE_BUFF_REGION__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for VGPU_CPU_GSP_RESPONSE_BUFF_REGION {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
