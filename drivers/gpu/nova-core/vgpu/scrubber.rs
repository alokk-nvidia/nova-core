// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

//! Per-VM CeUtils guest VRAM scrubbing.

use kernel::{
    device,
    prelude::*,
    sync::Mutex,
    time::{
        delay::fsleep,
        Delta,
        Instant,
        Monotonic, //
    },
    types::ScopeGuard, //
};

use crate::{
    gsp::cmdq::Cmdq,
    mm::{
        bar_user::BarUser,
        vram::VramRegion,
        GpuMm,
        Pfn,
        VramAddress, //
    },
    vgpu::instance::Gfid, //
};

use super::commands::{
    self,
    CeUtilsAllocError, //
};

// Semaphore layout from OpenRM `channel_utils.h`.
const NV_CEUTILS_SEMA_PAGE_MAGIC: u32 = 0xce5e_5ea0;

#[repr(C)]
struct CeUtilsSemaphoreHeader {
    magic: u32,
    payload: u32,
}

static_assert!(size_of::<CeUtilsSemaphoreHeader>() == 8);

const SEMA_PAGE_MAGIC_OFFSET: usize = core::mem::offset_of!(CeUtilsSemaphoreHeader, magic);
const SEMA_PAGE_PAYLOAD_OFFSET: usize = core::mem::offset_of!(CeUtilsSemaphoreHeader, payload);

const SCRUB_REQUEST_SIZE: u64 = 4 * 1024 * 1024 * 1024;

// Host timeout policy; not a firmware ABI value.
const SCRUB_TIMEOUT: Delta = Delta::from_secs(5);

/// A firmware-owned per-VM CeUtils allocation.
///
/// Submitted scrubs must complete before releasing this allocation or its VRAM.
pub(super) struct CeUtils {
    gfid: Gfid,
    semaphore_address: u64,
}

impl CeUtils {
    pub(super) fn allocate(
        dev: &device::Device<device::Bound>,
        cmdq: &Cmdq<'_>,
        gfid: Gfid,
        chid: u32,
    ) -> core::result::Result<Self, CeUtilsAllocError> {
        let semaphore_address = commands::alloc_ceutils(dev, cmdq, gfid, chid)?;
        Ok(Self {
            gfid,
            semaphore_address,
        })
    }

    /// Scrub the complete guest VRAM and wait for semaphore completion.
    pub(super) fn scrub_guest_fb(
        &self,
        dev: &device::Device<device::Bound>,
        cmdq: &Cmdq<'_>,
        bar_user: &BarUser<'_>,
        mm: &Mutex<GpuMm<'_>>,
        fb: &VramRegion,
    ) -> Result {
        let mut offset = fb.address();
        let end = offset.checked_add(fb.size()).ok_or(EOVERFLOW)?;
        while offset < end {
            let size = core::cmp::min(SCRUB_REQUEST_SIZE, end - offset);
            let work_id = commands::submit_ceutils_scrub(dev, cmdq, self.gfid, offset, size)?;
            wait_scrub_complete(bar_user, mm, dev, self.semaphore_address, work_id)?;
            offset = offset.checked_add(size).ok_or(EOVERFLOW)?;
        }

        Ok(())
    }
}

/// Poll the GSP-owned CeUtils semaphore page through a temporary BAR1 map.
fn wait_scrub_complete(
    bar_user: &BarUser<'_>,
    mm: &Mutex<GpuMm<'_>>,
    dev: &device::Device<device::Bound>,
    semaphore_address: u64,
    work_id: u32,
) -> Result {
    let pfn = Pfn::from(VramAddress::from_raw(semaphore_address));
    let semaphore_map = bar_user.map(&mut mm.lock(), &[pfn], false)?;
    let semaphore_map = ScopeGuard::new_with_data(semaphore_map, |mapping| {
        if let Err(error) = mapping.release(&mut mm.lock()) {
            dev_err!(
                dev,
                "failed to release semaphore BAR1 mapping: {:?}\n",
                error
            );
        }
    });

    let result = (|| {
        let magic = semaphore_map.try_read32(SEMA_PAGE_MAGIC_OFFSET)?;
        if magic != NV_CEUTILS_SEMA_PAGE_MAGIC {
            dev_warn!(
                dev,
                "bad CeUtils semaphore magic {:#x}, expected {:#x}\n",
                magic,
                NV_CEUTILS_SEMA_PAGE_MAGIC,
            );
            return Err(EIO);
        }

        let start = Instant::<Monotonic>::now();
        loop {
            let value = semaphore_map.try_read32(SEMA_PAGE_PAYLOAD_OFFSET)?;
            if value.wrapping_sub(work_id) < 0x8000_0000 {
                dev_dbg!(
                    dev,
                    "scrub completed after {:?}: semaphore={:#x}, target={:#x}\n",
                    start.elapsed(),
                    value,
                    work_id,
                );
                return Ok(());
            }

            if start.elapsed() >= SCRUB_TIMEOUT {
                dev_warn!(
                    dev,
                    "scrub timed out: semaphore={:#x}, target={:#x}\n",
                    value,
                    work_id,
                );
                return Err(ETIMEDOUT);
            }
            fsleep(Delta::from_millis(1));
        }
    })();

    let cleanup = semaphore_map.dismiss().release(&mut mm.lock());
    match result {
        Ok(()) => cleanup,
        Err(error) => {
            if let Err(cleanup_error) = cleanup {
                dev_err!(
                    dev,
                    "failed to release semaphore BAR1 mapping after error {:?}: {:?}\n",
                    error,
                    cleanup_error,
                );
            }
            Err(error)
        }
    }
}
