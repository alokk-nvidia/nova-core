// SPDX-License-Identifier: GPL-2.0

use core::{
    convert::Infallible,
    ffi::FromBytesUntilNulError,
    ops::Range,
    str::Utf8Error, //
};

use kernel::{
    device,
    pci,
    prelude::*,
    transmute::AsBytes, //
};

use crate::{
    driver::Bar0,
    gpu::Chipset,
    gsp::{
        cmdq::{
            Cmdq,
            CommandToGsp,
            NoReply, //
        },
        fw::{
            commands::*,
            GspVfInfo,
            MsgFunction, //
        },
        nvkv, //
    },
    sbuffer::SBufferIter,
};

/// The `GspSetSystemInfo` command.
pub(crate) struct SetSystemInfo<'a> {
    pdev: &'a pci::Device<device::Bound>,
    chipset: Chipset,
    vf_info: Option<GspVfInfo>,
}

impl<'a> SetSystemInfo<'a> {
    /// Creates a new `GspSetSystemInfo` command using the parameters of `pdev`.
    pub(crate) fn new(
        pdev: &'a pci::Device<device::Bound>,
        chipset: Chipset,
        vf_info: Option<GspVfInfo>,
    ) -> Self {
        Self {
            pdev,
            chipset,
            vf_info,
        }
    }
}

impl<'a> CommandToGsp for SetSystemInfo<'a> {
    const FUNCTION: MsgFunction = MsgFunction::GspSetSystemInfo;
    const IS_ASYNC: bool = true;
    type Command = GspSetSystemInfo;
    type Reply = NoReply;
    type InitError = Error;

    fn init(&self) -> impl Init<Self::Command, Self::InitError> {
        GspSetSystemInfo::init(self.pdev, self.chipset, self.vf_info)
    }
}

struct RegistryEntry {
    key: &'static str,
    value: u32,
}

/// The `SetRegistry` command.
///
/// Registry entries are built dynamically at runtime based on the current
/// configuration (e.g. whether vGPU is enabled).
pub(crate) struct SetRegistry {
    entries: KVec<RegistryEntry>,
}

impl SetRegistry {
    /// Creates a new `SetRegistry` command.
    ///
    /// The base set of registry entries is always included. Additional entries
    /// are appended dynamically based on runtime conditions (e.g. vGPU).
    pub(crate) fn new(vgpu_requested: bool) -> Result<Self> {
        let mut entries = KVec::new();

        // RMSecBusResetEnable - enables PCI secondary bus reset
        entries.push(
            RegistryEntry {
                key: "RMSecBusResetEnable",
                value: 1,
            },
            GFP_KERNEL,
        )?;

        // RMForcePcieConfigSave - forces GSP-RM to preserve PCI configuration registers on
        // any PCI reset.
        entries.push(
            RegistryEntry {
                key: "RMForcePcieConfigSave",
                value: 1,
            },
            GFP_KERNEL,
        )?;

        // RMDevidCheckIgnore - allows GSP-RM to boot even if the PCI dev ID is not found
        // in the internal product name database.
        entries.push(
            RegistryEntry {
                key: "RMDevidCheckIgnore",
                value: 1,
            },
            GFP_KERNEL,
        )?;

        // RMSetSriovMode - required when vGPU is enabled.
        if vgpu_requested {
            entries.push(
                RegistryEntry {
                    key: "RMSetSriovMode",
                    value: 1,
                },
                GFP_KERNEL,
            )?;
        }

        Ok(Self { entries })
    }
}

impl CommandToGsp for SetRegistry {
    const FUNCTION: MsgFunction = MsgFunction::SetRegistry;
    const IS_ASYNC: bool = true;
    type Command = PackedRegistryTable;
    type Reply = NoReply;
    type InitError = Infallible;

    fn init(&self) -> impl Init<Self::Command, Self::InitError> {
        PackedRegistryTable::init(
            self.entries.len() as u32,
            self.variable_payload_len() as u32,
        )
    }

    fn variable_payload_len(&self) -> usize {
        let mut key_size = 0;
        for entry in self.entries.iter() {
            key_size += entry.key.len() + 1; // +1 for NULL terminator
        }
        self.entries.len() * size_of::<PackedRegistryEntry>() + key_size
    }

    fn init_variable_payload(
        &self,
        dst: &mut SBufferIter<core::array::IntoIter<&mut [u8], 2>>,
    ) -> Result {
        let string_data_start_offset = size_of::<PackedRegistryTable>()
            + self.entries.len() * size_of::<PackedRegistryEntry>();

        // Array for string data.
        let mut string_data = KVec::new();

        for entry in self.entries.iter() {
            dst.write_all(
                PackedRegistryEntry::new(
                    (string_data_start_offset + string_data.len()) as u32,
                    entry.value,
                )
                .as_bytes(),
            )?;

            let key_bytes = entry.key.as_bytes();
            string_data.extend_from_slice(key_bytes, GFP_KERNEL)?;
            string_data.push(0, GFP_KERNEL)?;
        }

        dst.write_all(string_data.as_slice())
    }
}

/// GMC command ID for `GSP_GET_STATIC_INFO`.
const GMC_CMD_GSP_GET_STATIC_INFO: u32 = 0x0001_0001;

/// Maximum response size for `GSP_GET_STATIC_INFO`.
const GSP_GET_STATIC_INFO_MAX_RESPONSE: u32 = 8192;

/// The reply from the GSP to the `GSP_GET_STATIC_INFO` GMC command.
pub(crate) struct GetGspStaticInfoReply {
    gpu_name: [u8; 64],
    /// BAR1 Page Directory Entry base address.
    pub(crate) bar1_pde_base: u64,
    /// Usable FB (VRAM) region for driver memory allocation.
    pub(crate) usable_fb_region: Range<u64>,
    /// End of VRAM.
    pub(crate) total_fb_end: u64,
}

/// Error type for [`GetGspStaticInfoReply::gpu_name`].
#[derive(Debug)]
pub(crate) enum GpuNameError {
    /// The GPU name string does not contain a null terminator.
    NoNullTerminator(FromBytesUntilNulError),

    /// The GPU name string contains invalid UTF-8.
    #[expect(dead_code)]
    InvalidUtf8(Utf8Error),
}

impl GetGspStaticInfoReply {
    /// Returns the name of the GPU as a string.
    ///
    /// Returns an error if the string given by the GSP does not contain a null terminator or
    /// contains invalid UTF-8.
    pub(crate) fn gpu_name(&self) -> core::result::Result<&str, GpuNameError> {
        CStr::from_bytes_until_nul(&self.gpu_name)
            .map_err(GpuNameError::NoNullTerminator)?
            .to_str()
            .map_err(GpuNameError::InvalidUtf8)
    }
}

/// Sends `GSP_GET_STATIC_INFO` via GMC and parses the NVKV response.
pub(crate) fn get_gsp_info(cmdq: &Cmdq, bar: &Bar0) -> Result<GetGspStaticInfoReply> {
    let response = cmdq.send_gmc_and_receive(
        bar,
        GMC_CMD_GSP_GET_STATIC_INFO,
        &[],
        GSP_GET_STATIC_INFO_MAX_RESPONSE,
    )?;

    if response.status != 0 {
        return Err(EIO);
    }

    const MAX_FB_REGIONS: usize = 16;

    let mut gpu_name = [0u8; 64];
    let mut bar1_pde_base: u64 = 0;
    let mut fb_region_count: usize = 0;
    let mut fb_base = [0u64; MAX_FB_REGIONS];
    let mut fb_limit = [0u64; MAX_FB_REGIONS];
    let mut fb_reserved = [0u64; MAX_FB_REGIONS];
    let mut fb_flags = [0u32; MAX_FB_REGIONS];

    nvkv::for_each_entry(&response.payload, |entry| {
        use nvkv::gsp_config_key::*;
        let idx = entry.index as usize;

        match entry.key {
            GPU_NAME_STRING => {
                let len = (entry.count as usize).min(gpu_name.len()).min(entry.data.len());
                gpu_name[..len].copy_from_slice(&entry.data[..len]);
            }
            FB_REGION_COUNT => {
                if let Some(v) = entry.as_imm32() {
                    fb_region_count = (v as usize).min(MAX_FB_REGIONS);
                }
            }
            FB_REGION_BASE if idx < MAX_FB_REGIONS => {
                if let Some(v) = entry.as_u64() {
                    fb_base[idx] = v;
                }
            }
            FB_REGION_LIMIT if idx < MAX_FB_REGIONS => {
                if let Some(v) = entry.as_u64() {
                    fb_limit[idx] = v;
                }
            }
            FB_REGION_RESERVED if idx < MAX_FB_REGIONS => {
                if let Some(v) = entry.as_u64() {
                    fb_reserved[idx] = v;
                }
            }
            FB_REGION_FLAGS if idx < MAX_FB_REGIONS => {
                if let Some(v) = entry.as_u32() {
                    fb_flags[idx] = v;
                }
            }
            BAR1_PDE_BASE => {
                if let Some(v) = entry.as_u64() {
                    bar1_pde_base = v;
                }
            }
            _ => {}
        }
    })?;

    let mut usable_fb_region = 0..0u64;
    let mut total_fb_end = 0u64;
    let mut found_usable = false;

    for i in 0..fb_region_count {
        let base = fb_base[i];
        let limit = fb_limit[i];
        if limit < base {
            continue;
        }

        let end = limit.saturating_add(1);
        if end > total_fb_end {
            total_fb_end = end;
        }

        // Same criteria as GspStaticConfigInfo::first_usable_fb_region():
        // not reserved, not protected, supports compression and ISO.
        let flags = fb_flags[i];
        let support_compressed = flags & 1 != 0;
        let support_iso = flags & 2 != 0;
        let is_protected = flags & 4 != 0;

        if !found_usable
            && fb_reserved[i] == 0
            && !is_protected
            && support_compressed
            && support_iso
        {
            usable_fb_region = base..end;
            found_usable = true;
        }
    }

    Ok(GetGspStaticInfoReply {
        gpu_name,
        bar1_pde_base,
        usable_fb_region,
        total_fb_end,
    })
}
