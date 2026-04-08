// SPDX-License-Identifier: GPL-2.0

//! NVKV key/value decoder for GMC API responses.
//!
//! NVKV is a pushbuffer-like encoding used by GSP-RM to transmit structured
//! data as an array of `u64` values. Each entry starts with a header `u64`
//! containing the opcode, key, index, and count (or immediate value),
//! followed by zero or more data `u64`s.
//!
//! Open RM reference: `inc/libraries/nvkv/nvkv.h`

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

fn header_opcode(hdr: u64) -> u32 {
    ((hdr >> 28) & 0xF) as u32
}

fn header_count(hdr: u64) -> u32 {
    (hdr >> 32) as u32
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

fn header_index(hdr: u64) -> u32 {
    ((hdr >> 16) & 0xFFF) as u32
}

/// GSP static config key constants.
///
/// Open RM reference: `interface/gmcapi/gmcapi_gsp_config.h`
pub(crate) mod gsp_config_key {
    pub(crate) const FB_REGION_COUNT: u16 = 0x0010;
    pub(crate) const FB_REGION_FLAGS: u16 = 0x0012;
    pub(crate) const FB_REGION_BASE: u16 = 0x1011;
    pub(crate) const FB_REGION_LIMIT: u16 = 0x1012;
    pub(crate) const FB_REGION_RESERVED: u16 = 0x1013;
    pub(crate) const BAR1_PDE_BASE: u16 = 0x1020;
    /// GPU name string (ARRAY8, null-terminated).
    pub(crate) const GPU_NAME_STRING: u16 = 0x2000;
}

/// A decoded NVKV entry.
pub(crate) struct NvkvEntry<'a> {
    pub(crate) key: u16,
    pub(crate) index: u32,
    pub(crate) opcode: u32,
    pub(crate) count: u32,
    pub(crate) data: &'a [u8],
}

impl NvkvEntry<'_> {
    /// Read the value as an immediate u32 (OPCODE_IMM32).
    pub(crate) fn as_imm32(&self) -> Option<u32> {
        (self.opcode == OPCODE_IMM32).then_some(self.count)
    }

    /// Read the first u64 from a SEQ64 entry.
    pub(crate) fn as_u64(&self) -> Option<u64> {
        if self.opcode != OPCODE_SEQ64 || self.data.len() < 8 {
            return None;
        }
        Some(u64::from_le_bytes(self.data[..8].try_into().ok()?))
    }

    /// Read the first u32 from a SEQ32 entry, or from IMM32.
    pub(crate) fn as_u32(&self) -> Option<u32> {
        match self.opcode {
            OPCODE_IMM32 => Some(self.count),
            OPCODE_SEQ32 if self.data.len() >= 4 => {
                Some(u32::from_le_bytes(self.data[..4].try_into().ok()?))
            }
            _ => None,
        }
    }
}

/// Iterate all entries in an NVKV-encoded byte buffer.
///
/// SEQ32/SEQ64 entries encode multiple sequential keys in a single header.
/// This iterator unpacks them, calling `f` once per logical key-value pair.
///
/// Returns `Err(EINVAL)` if the data is malformed.
pub(crate) fn for_each_entry<F>(payload: &[u8], mut f: F) -> Result
where
    F: FnMut(NvkvEntry<'_>),
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
        let base_key = header_key(hdr);
        let index = header_index(hdr);
        let count = header_count(hdr);
        let n_data = data_u64s(opcode, count)?;

        if pos + n_data > qwords {
            return Err(EINVAL);
        }

        let data_start = pos * 8;

        match opcode {
            OPCODE_IMM32 => {
                f(NvkvEntry {
                    key: base_key,
                    index,
                    opcode,
                    count,
                    data: &[],
                });
            }
            OPCODE_SEQ32 => {
                for i in 0..count as usize {
                    let off = data_start + i * 4;
                    if off + 4 <= payload.len() {
                        let val = u32::from_le_bytes(
                            payload[off..off + 4].try_into().map_err(|_| EINVAL)?,
                        );
                        f(NvkvEntry {
                            key: base_key + i as u16,
                            index,
                            opcode: OPCODE_IMM32,
                            count: val,
                            data: &payload[off..off + 4],
                        });
                    }
                }
            }
            OPCODE_SEQ64 => {
                for i in 0..count as usize {
                    let off = data_start + i * 8;
                    if off + 8 <= payload.len() {
                        f(NvkvEntry {
                            key: base_key + i as u16,
                            index,
                            opcode: OPCODE_SEQ64,
                            count: 1,
                            data: &payload[off..off + 8],
                        });
                    }
                }
            }
            _ => {
                let data_end = (pos + n_data) * 8;
                f(NvkvEntry {
                    key: base_key,
                    index,
                    opcode,
                    count,
                    data: &payload[data_start..data_end],
                });
            }
        }

        pos += n_data;
    }

    Ok(())
}
