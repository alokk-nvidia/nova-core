// SPDX-License-Identifier: GPL-2.0

use core::{
    ffi::FromBytesUntilNulError,
    ops::Range,
    str::Utf8Error, //
};

use kernel::{
    device,
    pci,
    prelude::*, //
};

use crate::{
    driver::Bar0,
    gpu::{
        Architecture,
        Chipset, //
    },
    gsp::{
        cmdq::Cmdq,
        nvkv, //
    },
};

/// Maximum response size for the `GSP_INIT` reply.
const GSP_INIT_MAX_RESPONSE_SIZE: u32 = 8192;

/// The reply from the GSP to the `GSP_INIT` GMC command.
pub(crate) struct GetGspStaticInfoReply {
    gpu_name: [u8; 64],
    /// Usable FB (VRAM) region for driver memory allocation.
    pub(crate) usable_fb_region: Range<u64>,
    /// End of physical VRAM (exclusive), covering all FB regions.
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

/// GMC command id for `GSP_INIT`.
///
/// Matches `GMCAPI_COMMANDS_GMCAPI_CMD_GSP_INIT` in the r000 bindings.
const CMD_GSP_INIT: u32 = 0x0001_0001;

/// Hardcoded registry entries the driver always sends to GSP-RM.
///
/// `RMSecBusResetEnable` enables PCI secondary bus reset. `RMForcePcieConfigSave`
/// forces GSP-RM to preserve PCI configuration registers across any PCI reset.
/// `RMDevidCheckIgnore` allows GSP-RM to boot even if the PCI device id is not
/// found in its internal product name database.
const REGISTRY_ENTRIES: &[(&str, u32)] = &[
    ("RMSecBusResetEnable", 1),
    ("RMForcePcieConfigSave", 1),
    ("RMDevidCheckIgnore", 1),
];

/// Builds an NVKV-encoded `GSP_INIT` request payload.
///
/// The blob carries the system-info keys with values the driver actually
/// has, plus the registry entries from [`REGISTRY_ENTRIES`] as
/// `REGKEY_NAME` plus `REGKEY_VALUE_U32` pairs.
pub(crate) fn build_gsp_init_payload(
    pdev: &pci::Device<device::Bound>,
    chipset: Chipset,
) -> Result<KVec<u8>> {
    let mut nvkv = nvkv::Builder::new();

    nvkv.push_imm32(
        nvkv::sys_info_key::PCI_DEVICE_ID,
        (u32::from(pdev.device_id()) << 16) | u32::from(pdev.vendor_id().as_raw()),
    )?;
    nvkv.push_imm32(
        nvkv::sys_info_key::PCI_SUB_DEVICE_ID,
        (u32::from(pdev.subsystem_device_id()) << 16) | u32::from(pdev.subsystem_vendor_id()),
    )?;
    nvkv.push_imm32(
        nvkv::sys_info_key::PCI_REVISION_ID,
        u32::from(pdev.revision_id()),
    )?;

    // Hopper, Blackwell, and later moved the PCI config mirror window to
    // 0x092000. Older architectures continue to use the legacy 0x088000.
    let mirror_base = match chipset.arch() {
        Architecture::Turing | Architecture::Ampere | Architecture::Ada => 0x088000,
        Architecture::Hopper | Architecture::BlackwellGB10x | Architecture::BlackwellGB20x => {
            0x092000
        }
    };
    nvkv.push_imm32(nvkv::sys_info_key::PCI_CONFIG_MIRROR_BASE, mirror_base)?;
    nvkv.push_imm32(nvkv::sys_info_key::PCI_CONFIG_MIRROR_SIZE, 0x001000)?;

    let oor_arch = if cfg!(target_arch = "x86_64") {
        nvkv::oor_arch::X86_64
    } else if cfg!(target_arch = "aarch64") {
        nvkv::oor_arch::AARCH64
    } else if cfg!(target_arch = "powerpc64") {
        nvkv::oor_arch::PPC64LE
    } else if cfg!(target_arch = "arm") {
        nvkv::oor_arch::ARM
    } else if cfg!(target_arch = "riscv64") {
        nvkv::oor_arch::RISCV64
    } else {
        nvkv::oor_arch::NONE
    };
    nvkv.push_imm32(nvkv::sys_info_key::OOR_ARCH, oor_arch)?;

    for (name, value) in REGISTRY_ENTRIES {
        let mut name_bytes = KVec::with_capacity(name.len() + 1, GFP_KERNEL)?;
        name_bytes.extend_from_slice(name.as_bytes(), GFP_KERNEL)?;
        name_bytes.push(0, GFP_KERNEL)?;
        nvkv.push_array8(nvkv::sys_info_key::REGKEY_NAME, &name_bytes)?;
        nvkv.push_imm32(nvkv::sys_info_key::REGKEY_VALUE_U32, *value)?;
    }

    Ok(nvkv.finish())
}

/// Sends `GSP_INIT` and drives the receive loop until the reply arrives.
///
/// `payload` is the NVKV-encoded blob from [`build_gsp_init_payload`].
/// GSP-RM may interleave boot events between the send and the reply, so the
/// caller supplies an `on_boot_event` closure that handles those events.
/// The loop terminates when a GMC message arrives whose command id matches
/// [`CMD_GSP_INIT`]; the reply payload is decoded and returned.
pub(crate) fn gsp_init(
    cmdq: &Cmdq,
    bar: &Bar0,
    payload: &[u8],
    mut on_boot_event: impl FnMut(u32, &[u8]) -> Result,
) -> Result<GetGspStaticInfoReply> {
    cmdq.send_gmc_no_wait(bar, CMD_GSP_INIT, payload, GSP_INIT_MAX_RESPONSE_SIZE)?;

    loop {
        let reply = cmdq.receive_gmc_and_dispatch(
            bar,
            Cmdq::RECEIVE_TIMEOUT,
            |id, status, p0, p1| -> Result<Option<GetGspStaticInfoReply>> {
                if id == CMD_GSP_INIT {
                    if status != 0 {
                        return Err(EIO);
                    }
                    let mut blob = KVec::with_capacity(p0.len() + p1.len(), GFP_KERNEL)?;
                    blob.extend_from_slice(p0, GFP_KERNEL)?;
                    blob.extend_from_slice(p1, GFP_KERNEL)?;
                    let mut gpu_name = [0u8; 64];
                    if let Some(name_bytes) =
                        nvkv::find_array8(&blob, nvkv::gsp_config_key::GPU_NAME_STRING)?
                    {
                        let len = name_bytes.len().min(gpu_name.len());
                        gpu_name[..len].copy_from_slice(&name_bytes[..len]);
                    }

                    let usable_fb_region = first_usable_fb_region(&blob)?
                        .ok_or(ENODEV)?;
                    let total_fb_end = total_fb_end(&blob)?
                        .ok_or(ENODEV)?;

                    Ok(Some(GetGspStaticInfoReply {
                        gpu_name,
                        usable_fb_region,
                        total_fb_end,
                    }))
                } else {
                    on_boot_event(id, p0)?;
                    Ok(None)
                }
            },
        )??;

        if let Some(reply) = reply {
            return Ok(reply);
        }
    }
}

/// Extracts the first usable FB region from an NVKV-encoded GSP_INIT response.
///
/// Usable regions satisfy: not protected, supports compression and ISO,
/// and tag == TAG_NONE (not reserved by firmware).
fn first_usable_fb_region(blob: &[u8]) -> Result<Option<Range<u64>>> {
    use nvkv::gsp_config_key::*;

    let num_regions = match nvkv::find_imm32(blob, FB_REGION_COUNT)? {
        Some(n) => n,
        None => return Ok(None),
    };

    for i in 0..num_regions {
        let idx = i as u16;

        let base = match nvkv::find_seq64_indexed(blob, FB_REGION_BASE, idx)? {
            Some(v) => v,
            None => continue,
        };
        let limit = match nvkv::find_seq64_indexed(blob, FB_REGION_LIMIT, idx)? {
            Some(v) => v,
            None => continue,
        };
        let flags = nvkv::find_seq32_indexed(blob, FB_REGION_FLAGS, idx)?
            .unwrap_or(0);
        let tag = nvkv::find_seq32_indexed(blob, FB_REGION_TAG, idx)?
            .unwrap_or(u32::MAX);

        let compression_supported = flags & 1 != 0;
        let iso_supported = flags & 2 != 0;
        let protected = flags & 4 != 0;

        if !protected && compression_supported && iso_supported && tag == FB_REGION_TAG_NONE {
            if let Some(end) = limit.checked_add(1) {
                return Ok(Some(base..end));
            }
        }
    }

    Ok(None)
}

/// Compute the end of physical VRAM from all FB regions in the NVKV payload.
///
/// Returns the exclusive end address of the highest valid region.
fn total_fb_end(blob: &[u8]) -> Result<Option<u64>> {
    use nvkv::gsp_config_key::*;

    let num_regions = match nvkv::find_imm32(blob, FB_REGION_COUNT)? {
        Some(n) => n,
        None => return Ok(None),
    };

    let mut max_end: Option<u64> = None;

    for i in 0..num_regions {
        let idx = i as u16;

        let limit = match nvkv::find_seq64_indexed(blob, FB_REGION_LIMIT, idx)? {
            Some(v) => v,
            None => continue,
        };

        if let Some(end) = limit.checked_add(1) {
            max_end = Some(max_end.map_or(end, |cur| cur.max(end)));
        }
    }

    Ok(max_end)
}
