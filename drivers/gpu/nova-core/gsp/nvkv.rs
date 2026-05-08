// SPDX-License-Identifier: GPL-2.0

//! NVKV key/value codec for GMC API payloads.
//!
//! NVKV is a pushbuffer-like encoding used by GSP-RM to transmit structured
//! data as an array of `u64` values. Each entry starts with a header `u64`
//! containing the opcode, key, index, and count (or immediate value),
//! followed by zero or more data `u64`s.
//!
//! Open RM reference: `inc/libraries/nvkv/nvkv.h`.

use kernel::prelude::*;

const OPCODE_IMM32: u32 = 0x0;
const OPCODE_SEQ32: u32 = 0x1;
const OPCODE_SEQ64: u32 = 0x2;
const OPCODE_ARRAY8: u32 = 0x3;
const OPCODE_ARRAY32: u32 = 0x4;
const OPCODE_ARRAY64: u32 = 0x5;

fn header_key(hdr: u64) -> u16 {
    (hdr & 0xFFFF) as u16
}

fn header_index(hdr: u64) -> u16 {
    ((hdr >> 16) & 0xFFF) as u16
}

fn header_opcode(hdr: u64) -> u32 {
    ((hdr >> 28) & 0xF) as u32
}

fn header_count(hdr: u64) -> u32 {
    (hdr >> 32) as u32
}

/// Builds the leading header `u64` for an NVKV entry.
///
/// `count_or_value` carries the byte count for `ARRAY*` opcodes and the
/// immediate value for `IMM32`.
fn make_header(opcode: u32, key: u16, count_or_value: u32) -> u64 {
    u64::from(key) | ((u64::from(opcode) & 0xF) << 28) | (u64::from(count_or_value) << 32)
}

/// Number of data `u64`s that follow a given header.
fn data_u64s(opcode: u32, count: u32) -> Result<usize> {
    match opcode {
        OPCODE_IMM32 => Ok(0),
        OPCODE_SEQ32 => Ok(count.div_ceil(2) as usize),
        OPCODE_SEQ64 => Ok(count as usize),
        OPCODE_ARRAY8 => Ok((count as usize).div_ceil(8)),
        OPCODE_ARRAY32 => Ok((count as usize * 4).div_ceil(8)),
        OPCODE_ARRAY64 => Ok(count as usize),
        _ => Err(EINVAL),
    }
}

/// GSP static config key constants.
///
/// Open RM reference: `interface/gmcapi/gmcapi_gsp_config.h`.
pub(crate) mod gsp_config_key {
    /// GPU name string (`ARRAY8`, null-terminated).
    pub(crate) const GPU_NAME_STRING: u16 = 0x2000;
    /// Number of FB regions (`IMM32`).
    pub(crate) const FB_REGION_COUNT: u16 = 0x0010;
    /// FB region base address (`SEQ64`, indexed).
    pub(crate) const FB_REGION_BASE: u16 = 0x1011;
    /// FB region limit address (`SEQ64`, indexed).
    pub(crate) const FB_REGION_LIMIT: u16 = 0x1012;
    /// FB region flags (`SEQ32`, indexed). Bit 0=compression, 1=ISO, 2=protected.
    pub(crate) const FB_REGION_FLAGS: u16 = 0x0012;
    /// FB region tag (`SEQ32`, indexed). TAG_NONE (0) indicates usable.
    pub(crate) const FB_REGION_TAG: u16 = 0x0013;
    /// Tag value indicating no reservation.
    pub(crate) const FB_REGION_TAG_NONE: u32 = 0;
    /// BAR1 page directory entry base address (`SEQ64`).
    pub(crate) const BAR1_PDE_BASE: u16 = 0x1020;
    /// VMMU segment size (`SEQ64`, single value at index 0).
    pub(crate) const VMMU_SEGMENT_SIZE: u16 = 0x1050;
    /// Per-engine-type bitmask of available instances (`SEQ64`, indexed by
    /// `NVGMC_ENGINE_TYPE`). Each u64 is a bitmask where bit N = instance N.
    pub(crate) const ENGINE_MASK: u16 = 0x1100;
}

/// System-info key constants for `GMCAPI_CMD_GSP_INIT`.
///
/// Values match the `NVGMC_SI_*` `#define`s in r000 bindings
/// (`interface/gmcapi/gmcapi_system_info.h` and
/// `gmcapi_system_info_unstable.h`). Kept narrow to the keys the kernel
/// actually emits.
#[expect(dead_code)]
pub(crate) mod sys_info_key {
    pub(crate) const PCI_DEVICE_ID: u16 = 1;
    pub(crate) const PCI_SUB_DEVICE_ID: u16 = 2;
    pub(crate) const PCI_REVISION_ID: u16 = 3;
    pub(crate) const PCI_CONFIG_MIRROR_BASE: u16 = 16;
    pub(crate) const PCI_CONFIG_MIRROR_SIZE: u16 = 17;
    pub(crate) const OOR_ARCH: u16 = 112;
    pub(crate) const GPU_PHYS_ADDR: u16 = 4112;
    pub(crate) const GPU_PHYS_FB_ADDR: u16 = 4113;
    pub(crate) const GPU_PHYS_INST_ADDR: u16 = 4114;
    pub(crate) const NV_DOMAIN_BUS_DEVICE_FUNC: u16 = 4128;
    pub(crate) const MAX_USER_VA: u16 = 4146;
    pub(crate) const REGKEY_NAME: u16 = 12400;
    pub(crate) const REGKEY_VALUE_U32: u16 = 12401;
    /// VF total count (`IMM32`).
    pub(crate) const VF_TOTAL_VFS: u16 = 0x0080;
    /// VF first offset (`IMM32`).
    pub(crate) const VF_FIRST_VF_OFFSET: u16 = 0x0081;
    /// VF flags (`SEQ64`): bit0=64bit_bar0, bit1=64bit_bar1, bit2=64bit_bar2.
    pub(crate) const VF_FLAGS: u16 = 0x1003;
    /// VF first BAR0 address (`SEQ64`).
    pub(crate) const VF_FIRST_BAR0_ADDRESS: u16 = 0x1050;
    /// VF first BAR1 address (`SEQ64`).
    pub(crate) const VF_FIRST_BAR1_ADDRESS: u16 = 0x1051;
    /// VF first BAR2 address (`SEQ64`).
    pub(crate) const VF_FIRST_BAR2_ADDRESS: u16 = 0x1052;
}

/// `NVGMC_SI_OOR_ARCH` wire values.
///
/// Source: `NVGMC_SI_OOR_ARCH_*` in r000 bindings.
pub(crate) mod oor_arch {
    pub(crate) const NONE: u32 = 0;
    pub(crate) const X86_64: u32 = 1;
    pub(crate) const PPC64LE: u32 = 2;
    pub(crate) const ARM: u32 = 3;
    pub(crate) const AARCH64: u32 = 4;
    pub(crate) const RISCV64: u32 = 5;
}

/// Incremental builder for an NVKV-encoded payload.
///
/// Each `push_*` call appends one entry. The output is a `KVec<u8>` whose
/// length is always a multiple of 8.
pub(crate) struct Builder {
    bytes: KVec<u8>,
}

impl Builder {
    /// Creates an empty NVKV builder.
    pub(crate) fn new() -> Self {
        Self { bytes: KVec::new() }
    }

    /// Consumes the builder and returns the encoded byte buffer.
    pub(crate) fn finish(self) -> KVec<u8> {
        self.bytes
    }

    fn push_u64(&mut self, value: u64) -> Result {
        self.bytes
            .extend_from_slice(&value.to_le_bytes(), GFP_KERNEL)?;
        Ok(())
    }

    /// Appends an `OPCODE_IMM32` entry carrying a 32-bit immediate value.
    pub(crate) fn push_imm32(&mut self, key: u16, value: u32) -> Result {
        self.push_u64(make_header(OPCODE_IMM32, key, value))
    }

    /// Appends an `OPCODE_SEQ64` entry carrying a single `u64` value.
    pub(crate) fn push_seq64(&mut self, key: u16, value: u64) -> Result {
        self.push_u64(make_header(OPCODE_SEQ64, key, 1))?;
        self.push_u64(value)
    }

    /// Appends an `OPCODE_ARRAY8` entry whose payload is `data`.
    ///
    /// The header records the byte count exactly. The payload bytes are
    /// followed by zero-padding so the next entry starts on an 8-byte
    /// boundary.
    pub(crate) fn push_array8(&mut self, key: u16, data: &[u8]) -> Result {
        let count = u32::try_from(data.len()).map_err(|_| EINVAL)?;
        self.push_u64(make_header(OPCODE_ARRAY8, key, count))?;
        self.bytes.extend_from_slice(data, GFP_KERNEL)?;
        let pad = (8 - data.len() % 8) % 8;
        for _ in 0..pad {
            self.bytes.push(0, GFP_KERNEL)?;
        }
        Ok(())
    }
}

/// Find a key in an NVKV-encoded byte buffer and return its raw data bytes.
///
/// The `payload` is the raw response bytes from a `GMCAPI_DYNAMIC` command.
/// Returns `None` if the key is not found.
///
/// # Errors
///
/// Returns `EINVAL` if the NVKV data is malformed (bad opcode or truncated).
pub(crate) fn find_array8<'a>(payload: &'a [u8], target_key: u16) -> Result<Option<&'a [u8]>> {
    if payload.len() % 8 != 0 {
        return Err(EINVAL);
    }

    let qwords = payload.len() / 8;
    let mut pos = 0usize;

    while pos < qwords {
        let hdr = u64::from_le_bytes(
            payload[pos * 8..(pos + 1) * 8]
                .try_into()
                .map_err(|_| EINVAL)?,
        );
        pos += 1;

        let opcode = header_opcode(hdr);
        let key = header_key(hdr);
        let count = header_count(hdr);
        let n_data = data_u64s(opcode, count)?;

        if pos + n_data > qwords {
            return Err(EINVAL);
        }

        if key == target_key && opcode == OPCODE_ARRAY8 {
            let byte_offset = pos * 8;
            let byte_count = count as usize;
            if byte_offset + byte_count > payload.len() {
                return Err(EINVAL);
            }
            return Ok(Some(&payload[byte_offset..byte_offset + byte_count]));
        }

        pos += n_data;
    }

    Ok(None)
}

/// Find an `IMM32` key and return its 32-bit value.
///
/// Returns `None` if the key is not present.
pub(crate) fn find_imm32(payload: &[u8], target_key: u16) -> Result<Option<u32>> {
    if payload.len() % 8 != 0 {
        return Err(EINVAL);
    }

    let qwords = payload.len() / 8;
    let mut pos = 0usize;

    while pos < qwords {
        let hdr = u64::from_le_bytes(
            payload[pos * 8..(pos + 1) * 8]
                .try_into()
                .map_err(|_| EINVAL)?,
        );
        pos += 1;

        let opcode = header_opcode(hdr);
        let key = header_key(hdr);
        let count = header_count(hdr);
        let n_data = data_u64s(opcode, count)?;

        if pos + n_data > qwords {
            return Err(EINVAL);
        }

        if key == target_key && opcode == OPCODE_IMM32 {
            return Ok(Some(count));
        }

        pos += n_data;
    }

    Ok(None)
}

/// Find a `SEQ64` entry matching `target_key` and `target_index`, returning
/// the first `u64` value in the sequence.
///
/// In NVKV, indexed keys use the header's 12-bit index field (bits 27:16).
/// `SEQ64` entries encode a run of sequential keys starting at `target_key`;
/// the nth u64 in the payload corresponds to `target_key + n`.
pub(crate) fn find_seq64_indexed(
    payload: &[u8],
    target_key: u16,
    target_index: u16,
) -> Result<Option<u64>> {
    if payload.len() % 8 != 0 {
        return Err(EINVAL);
    }

    let qwords = payload.len() / 8;
    let mut pos = 0usize;

    while pos < qwords {
        let hdr = u64::from_le_bytes(
            payload[pos * 8..(pos + 1) * 8]
                .try_into()
                .map_err(|_| EINVAL)?,
        );
        pos += 1;

        let opcode = header_opcode(hdr);
        let key = header_key(hdr);
        let index = header_index(hdr);
        let count = header_count(hdr);
        let n_data = data_u64s(opcode, count)?;

        if pos + n_data > qwords {
            return Err(EINVAL);
        }

        if opcode == OPCODE_SEQ64 && index == target_index {
            let offset = target_key.wrapping_sub(key) as usize;
            if offset < n_data {
                let val = u64::from_le_bytes(
                    payload[(pos + offset) * 8..(pos + offset + 1) * 8]
                        .try_into()
                        .map_err(|_| EINVAL)?,
                );
                return Ok(Some(val));
            }
        }

        if opcode == OPCODE_ARRAY64 && key == target_key && target_index >= index {
            let offset = (target_index - index) as usize;
            if offset < n_data {
                let val = u64::from_le_bytes(
                    payload[(pos + offset) * 8..(pos + offset + 1) * 8]
                        .try_into()
                        .map_err(|_| EINVAL)?,
                );
                return Ok(Some(val));
            }
        }

        pos += n_data;
    }

    Ok(None)
}

/// Find a `SEQ32` entry matching `target_key` and `target_index`, returning
/// the first `u32` value in the sequence.
pub(crate) fn find_seq32_indexed(
    payload: &[u8],
    target_key: u16,
    target_index: u16,
) -> Result<Option<u32>> {
    if payload.len() % 8 != 0 {
        return Err(EINVAL);
    }

    let qwords = payload.len() / 8;
    let mut pos = 0usize;

    while pos < qwords {
        let hdr = u64::from_le_bytes(
            payload[pos * 8..(pos + 1) * 8]
                .try_into()
                .map_err(|_| EINVAL)?,
        );
        pos += 1;

        let opcode = header_opcode(hdr);
        let key = header_key(hdr);
        let index = header_index(hdr);
        let count = header_count(hdr);
        let n_data = data_u64s(opcode, count)?;

        if pos + n_data > qwords {
            return Err(EINVAL);
        }

        if opcode == OPCODE_SEQ32 && index == target_index {
            let offset = target_key.wrapping_sub(key) as usize;
            if offset < count as usize {
                let qword_idx = offset / 2;
                let qword = u64::from_le_bytes(
                    payload[(pos + qword_idx) * 8..(pos + qword_idx + 1) * 8]
                        .try_into()
                        .map_err(|_| EINVAL)?,
                );
                let val = if offset % 2 == 0 {
                    qword as u32
                } else {
                    (qword >> 32) as u32
                };
                return Ok(Some(val));
            }
        }

        pos += n_data;
    }

    Ok(None)
}

// --- Callback-based NVKV decoder ---

/// Decoded value from a single NVKV entry.
pub(crate) enum NvkvValue<'a> {
    Imm32(u32),
    Seq32(&'a [u8]),
    Seq64(&'a [u8]),
    Array8(&'a [u8]),
}

/// Iterate over all key-value pairs in an NVKV payload, calling `f` for each.
pub(crate) fn nvkv_decode<F>(payload: &[u8], mut f: F) -> Result
where
    F: FnMut(u16, u16, NvkvValue<'_>),
{
    if payload.len() % 8 != 0 {
        return Err(EINVAL);
    }

    let qwords = payload.len() / 8;
    let mut pos = 0usize;

    while pos < qwords {
        let hdr = u64::from_le_bytes(
            payload[pos * 8..(pos + 1) * 8]
                .try_into()
                .map_err(|_| EINVAL)?,
        );
        pos += 1;

        let opcode = header_opcode(hdr);
        let key = header_key(hdr);
        let index = header_index(hdr);
        let count = header_count(hdr);
        let n_data = data_u64s(opcode, count)?;

        if pos + n_data > qwords {
            return Err(EINVAL);
        }

        let data_bytes = &payload[pos * 8..(pos + n_data) * 8];

        match opcode {
            OPCODE_IMM32 => f(key, index, NvkvValue::Imm32(count)),
            OPCODE_SEQ32 => f(key, index, NvkvValue::Seq32(data_bytes)),
            OPCODE_SEQ64 => f(key, index, NvkvValue::Seq64(data_bytes)),
            OPCODE_ARRAY8 => {
                let byte_count = count as usize;
                let byte_offset = pos * 8;
                if byte_offset + byte_count > payload.len() {
                    return Err(EINVAL);
                }
                f(
                    key,
                    index,
                    NvkvValue::Array8(&payload[byte_offset..byte_offset + byte_count]),
                );
            }
            OPCODE_ARRAY64 => f(key, index, NvkvValue::Seq64(data_bytes)),
            _ => {}
        }

        pos += n_data;
    }

    Ok(())
}

/// Read a u32 from an NvkvValue.
pub(crate) fn nvkv_read_u32(val: &NvkvValue<'_>) -> u32 {
    match val {
        NvkvValue::Imm32(v) => *v,
        NvkvValue::Seq32(data) if data.len() >= 4 => {
            u32::from_le_bytes(data[..4].try_into().unwrap_or([0; 4]))
        }
        _ => 0,
    }
}

/// Read a u64 from an NvkvValue.
pub(crate) fn nvkv_read_u64(val: &NvkvValue<'_>) -> u64 {
    match val {
        NvkvValue::Seq64(data) if data.len() >= 8 => {
            u64::from_le_bytes(data[..8].try_into().unwrap_or([0; 8]))
        }
        NvkvValue::Imm32(v) => u64::from(*v),
        _ => 0,
    }
}

/// Copy a string from an NvkvValue::Array8 into a fixed buffer.
pub(crate) fn nvkv_read_string8(val: &NvkvValue<'_>, dst: &mut [u8]) {
    if let NvkvValue::Array8(data) = val {
        let len = data.len().min(dst.len());
        dst[..len].copy_from_slice(&data[..len]);
    }
}

// --- NVKV Encoding ---

/// Encode an IMM32 key-value pair and push to the kvs vector.
pub(crate) fn nvkv_push_imm32(kvs: &mut KVec<u64>, key: u16, value: u32) -> Result {
    let header = make_header(OPCODE_IMM32, key, value);
    kvs.push(header, GFP_KERNEL)?;
    Ok(())
}

/// Encode a SEQ32 key-value pair: header + ceil(N/2) data u64s.
///
/// `first_key` is the starting key; the Nth value maps to `first_key + N`.
pub(crate) fn nvkv_push_seq32(kvs: &mut KVec<u64>, first_key: u16, values: &[u32]) -> Result {
    let header = make_header(OPCODE_SEQ32, first_key, values.len() as u32);
    kvs.push(header, GFP_KERNEL)?;
    for pair in values.chunks(2) {
        let lo = u64::from(pair[0]);
        let hi = if pair.len() > 1 {
            u64::from(pair[1])
        } else {
            0
        };
        kvs.push(lo | (hi << 32), GFP_KERNEL)?;
    }
    Ok(())
}

/// Encode a SEQ64 key-value pair: header + N data u64s.
pub(crate) fn nvkv_push_seq64(kvs: &mut KVec<u64>, key: u16, values: &[u64]) -> Result {
    let header = make_header(OPCODE_SEQ64, key, values.len() as u32);
    kvs.push(header, GFP_KERNEL)?;
    for &v in values {
        kvs.push(v, GFP_KERNEL)?;
    }
    Ok(())
}

/// Encode an ARRAY64 key-value pair: header + N data u64s.
pub(crate) fn nvkv_push_array64(kvs: &mut KVec<u64>, key: u16, values: &[u64]) -> Result {
    let header = make_header(OPCODE_ARRAY64, key, values.len() as u32);
    kvs.push(header, GFP_KERNEL)?;
    for &v in values {
        kvs.push(v, GFP_KERNEL)?;
    }
    Ok(())
}
