// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

use core::num::{
    NonZero,
    NonZeroUsize, //
};

use kernel::{
    device,
    prelude::*,
    ptr::Alignment,
    sizes::SizeConstants,
    time::{
        delay::fsleep,
        Delta,
        Instant,
        Monotonic, //
    }, //
};

use crate::{
    driver::Bar0,
    gpu::ChannelIdReservation,
    gsp::{
        cmdq::Cmdq,
        commands::FifoEngineList, //
    },
    mm::GpuMm,
};

use super::{
    commands::{
        free_ceutils,
        negotiate_plugin_version,
        send_bootload,
        send_cleanup,
        send_plugin_config,
        send_shutdown,
        set_plugin_bme,
        CeUtilsAllocError,
        Dbdf, //
    },
    fw::commands::{
        encode_plugin_config_params,
        encode_vgpu_bootload,
        BootloadInfo,
        ChannelMapEntry, //
    },
    gsp_plugin_comm::CommBufferRegion,
    gsp_plugin_rpc::PluginRpc,
    scrubber::CeUtils,
    vram::{
        VgpuVramLayout,
        VgpuVramSlot,
        VgpuVramSlotAllocator, //
    },
    VgpuManager, //
};

/// Ready limit used by `vmiopd_negotiate_cpu_gsp_version()` for the same marker.
const PLUGIN_READY_TIMEOUT: Delta = Delta::from_secs(10);

/// Per-engine channel budget reserved by the full SR-IOV plugin.
///
/// The supported GB20x path keeps firmware's non-heavy default, which uses
/// `PLUGIN_ALLOCATED_CHANNELS_PER_ENGINE` rather than the heavy-mode budget.
const PLUGIN_CHANNELS_PER_ENGINE: u32 = 3;

/// Build the typed channel mapping from the GSP FIFO engine list.
fn channel_mapping(
    fifo_engine_list: &FifoEngineList,
    chid_offset: u32,
) -> Result<KVVec<ChannelMapEntry>> {
    let mut mapping = KVVec::new();
    for &gmc_id in fifo_engine_list.gmc_ids() {
        // CAST: The mask leaves only the low 16 bits of the GMC engine ID.
        let engine_type = (gmc_id & u32::from(u16::MAX)) as u16;
        // CAST: Shifting a `u32` by 16 leaves at most 16 bits.
        let index = (gmc_id >> u16::BITS) as u16;
        mapping.push(
            ChannelMapEntry::new(engine_type, index, chid_offset),
            GFP_KERNEL,
        )?;
    }
    Ok(mapping)
}

fn wait_plugin_ready(
    dev: &device::Device<device::Bound>,
    comm: &CommBufferRegion<'_, '_>,
) -> Result {
    let start = Instant::<Monotonic>::now();

    loop {
        if comm.is_plugin_ready()? {
            dev_dbg!(dev, "vGPU plugin ready after {:?}\n", start.elapsed());
            return Ok(());
        }
        if start.elapsed() >= PLUGIN_READY_TIMEOUT {
            return Err(ETIMEDOUT);
        }
        fsleep(Delta::from_millis(1));
    }
}

/// Guest Function ID validated against one device's total number of VFs.
///
/// GFID 0 is reserved for the PF; VFs start at 1. Keeping the nonzero `u16`
/// preserves the PCI SR-IOV range when forming a plugin doorbell handle.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Gfid(NonZero<u16>);

impl Gfid {
    /// Validates an external GFID for a device supporting `total_vfs` VFs.
    #[expect(dead_code)]
    pub(super) fn new(gfid: u32, total_vfs: NonZero<u16>) -> Result<Self> {
        let gfid = u16::try_from(gfid).map_err(|_| EINVAL)?;
        let gfid = NonZero::new(gfid).ok_or(EINVAL)?;

        if gfid > total_vfs {
            Err(EINVAL)
        } else {
            Ok(Self(gfid))
        }
    }

    pub(super) const fn get(self) -> u16 {
        self.0.get()
    }
}

/// Resource requirements and device identity for one vGPU type.
#[expect(dead_code)]
pub(super) struct VgpuType {
    vgpu_type_id: u32,
    bar1_length: u64,
    max_instance: u32,
    pci_dev_id: u32,
    pci_subsys_id: u32,
    fb_length: u64,
    gsp_heap_size: u64,
}

/// A vGPU instance and the resources reserved for it.
struct VgpuInstance<'gpu> {
    gfid: Gfid,
    dbdf: Dbdf,
    vgpu_type: VgpuType,
    vm_pid: u32,
    num_plugin_channels: u32,
    plugin_rpc: PluginRpc<'gpu, 'gpu>,
    // Unmap the communication region before returning its slot and channel IDs.
    vram_slot: VgpuVramSlot,
    chids: ChannelIdReservation<'gpu>,
    ceutils: Option<CeUtils>,
    needs_teardown: bool,
    /// An uncertain or failed operation retains resources until device removal.
    failure: Option<Error>,
}

impl<'gpu> VgpuInstance<'gpu> {
    fn initialize(&mut self, vgpu: &VgpuManager<'gpu>) -> Result {
        let ceutils_chid =
            u32::try_from(self.chids.end.checked_sub(1).ok_or(EINVAL)?).map_err(|_| EOVERFLOW)?;
        let ceutils = match CeUtils::allocate(vgpu.dev, vgpu.cmdq, self.gfid, ceutils_chid) {
            Ok(ceutils) => ceutils,
            Err(CeUtilsAllocError::NotOwned(error)) => return Err(error),
            Err(CeUtilsAllocError::MayOwn(error)) => {
                self.failure = Some(error);
                dev_err!(vgpu.dev, "CeUtils allocation failed: {:?}\n", error);
                return Err(error);
            }
        };

        let result = ceutils.scrub_guest_fb(
            vgpu.dev,
            vgpu.cmdq,
            vgpu.bar_user,
            vgpu.mm,
            &self.vram_slot.fbmem,
        );
        self.ceutils = Some(ceutils);
        if let Err(error) = result {
            // A failed wait does not establish that the submitted scrub has stopped.
            self.failure = Some(error);
        }
        result
    }

    fn activate(&mut self, vgpu: &VgpuManager<'gpu>) -> Result {
        let dev = vgpu.dev;
        self.bootload(dev, vgpu.cmdq, &vgpu.fifo_engine_list)?;
        self.plugin_rpc.init_rpc()?;
        negotiate_plugin_version(dev, &mut self.plugin_rpc)?;
        self.configure_plugin(dev)?;
        set_plugin_bme(dev, &mut self.plugin_rpc, true)?;

        Ok(())
    }

    /// Tear down once, retaining all remaining resources if an operation fails.
    fn teardown(&mut self, vgpu: &VgpuManager<'gpu>) -> Result {
        if let Some(error) = self.failure {
            return Err(error);
        }
        let result = (|| {
            self.shutdown(vgpu.dev, vgpu.cmdq)?;
            if let Some(ceutils) = self.ceutils.as_ref() {
                ceutils.scrub_guest_fb(
                    vgpu.dev,
                    vgpu.cmdq,
                    vgpu.bar_user,
                    vgpu.mm,
                    &self.vram_slot.fbmem,
                )?;
                free_ceutils(vgpu.dev, vgpu.cmdq, self.gfid)?;
                self.ceutils = None;
            }
            if self.needs_teardown {
                send_cleanup(vgpu.dev, vgpu.cmdq, self.gfid)?;
                self.needs_teardown = false;
            }
            self.plugin_rpc.unmap()
        })();
        if let Err(error) = result {
            self.failure = Some(error);
        }
        result
    }

    /// Bootload the GSP vGPU plugin and wait for its BAR1 ready indication.
    fn bootload(
        &mut self,
        dev: &device::Device<device::Bound>,
        cmdq: &Cmdq<'_>,
        fifo_engine_list: &FifoEngineList,
    ) -> Result {
        let fb = &self.vram_slot.fbmem;
        let mgmt = &self.vram_slot.mgmt_heap;
        let logs = self.plugin_rpc.comm().plugin_logs();

        let payload = encode_vgpu_bootload(BootloadInfo {
            dbdf: self.dbdf,
            gfid: u32::from(self.gfid.get()),
            vgpu_type: self.vgpu_type.vgpu_type_id,
            vm_pid: self.vm_pid,
            num_channels: u32::try_from(self.chids.len()).map_err(|_| EOVERFLOW)?,
            num_plugin_channels: self.num_plugin_channels,
            channel_mapping: channel_mapping(
                fifo_engine_list,
                u32::try_from(self.chids.start).map_err(|_| EOVERFLOW)?,
            )?,
            guest_fb: fb,
            plugin_heap: mgmt,
            ctrl_buffer_offset: 0,
            init_log: &logs.init,
            vgpu_log: &logs.vgpu,
            kernel_log: &logs.kernel,
        })?;

        dev_dbg!(
            dev,
            "bootload: gfid={} sending {} typed NVKV bytes\n",
            self.gfid.get(),
            payload.len() * size_of::<u64>(),
        );

        self.plugin_rpc.comm().clear_plugin_ready()?;
        self.needs_teardown = true;
        send_bootload(dev, cmdq, &payload)?;

        wait_plugin_ready(dev, self.plugin_rpc.comm())?;

        dev_dbg!(dev, "bootload: gfid={} plugin ready\n", self.gfid.get());
        Ok(())
    }

    fn configure_plugin(&mut self, dev: &device::Device<device::Bound>) -> Result {
        let config = encode_plugin_config_params(
            [0; 16],
            self.dbdf,
            self.vgpu_type.vgpu_type_id,
            self.vm_pid,
            u32::try_from(self.chids.len().checked_sub(1).ok_or(EINVAL)?).map_err(|_| EOVERFLOW)?,
            self.num_plugin_channels,
        )?;

        send_plugin_config(dev, &mut self.plugin_rpc, &config)
    }

    /// Stop the plugin when firmware may own instance resources.
    fn shutdown(&mut self, dev: &device::Device<device::Bound>, cmdq: &Cmdq<'_>) -> Result {
        if self.needs_teardown {
            send_shutdown(dev, cmdq, self.gfid)?;
            dev_dbg!(dev, "shutdown: gfid={} stopped\n", self.gfid.get());
        }
        Ok(())
    }
}

/// Identity and firmware profile used to allocate an instance.
pub(super) struct InstanceInfo {
    gfid: Gfid,
    dbdf: Dbdf,
    vgpu_type: VgpuType,
    vm_pid: u32,
}

#[expect(dead_code)]
impl InstanceInfo {
    pub(super) const fn new(gfid: Gfid, dbdf: Dbdf, vgpu_type: VgpuType, vm_pid: u32) -> Self {
        Self {
            gfid,
            dbdf,
            vgpu_type,
            vm_pid,
        }
    }
}

/// Registry of live vGPU instances.
pub(super) struct VgpuInstances<'gpu> {
    instances: KVec<VgpuInstance<'gpu>>,
    vram_slots: Option<VgpuVramSlotAllocator>,
}

#[expect(dead_code)]
impl<'gpu> VgpuInstances<'gpu> {
    pub(super) const fn new() -> Self {
        Self {
            instances: KVec::new(),
            vram_slots: None,
        }
    }

    fn alloc_vram_slot(&mut self, mm: &GpuMm<'_>, layout: VgpuVramLayout) -> Result<VgpuVramSlot> {
        let replace_empty_pool = match self.vram_slots.as_ref() {
            Some(allocator) if allocator.is_empty() => !allocator.matches_layout(layout)?,
            _ => false,
        };
        if replace_empty_pool {
            self.vram_slots = None;
        }

        if let Some(allocator) = self.vram_slots.as_mut() {
            return allocator.alloc(layout);
        }

        let mut allocator = VgpuVramSlotAllocator::new(mm, layout)?;
        let slot = allocator.alloc(layout)?;
        self.vram_slots = Some(allocator);
        Ok(slot)
    }

    /// Register host resources before submitting any firmware work.
    fn allocate_instance<'a>(
        &'a mut self,
        vgpu: &'a VgpuManager<'gpu>,
        bar0: Bar0<'gpu>,
        info: InstanceInfo,
    ) -> Result<PendingInstance<'a, 'gpu>> {
        let InstanceInfo {
            gfid,
            dbdf,
            vgpu_type,
            vm_pid,
        } = info;

        let instance_exists = self
            .instances
            .iter()
            .any(|instance| instance.gfid == gfid || instance.dbdf == dbdf);
        if instance_exists {
            return Err(EEXIST);
        }

        let type_id = vgpu_type.vgpu_type_id;
        let num_type_instances = self
            .instances
            .iter()
            .filter(|instance| instance.vgpu_type.vgpu_type_id == type_id)
            .count();
        let max_instances = usize::try_from(vgpu_type.max_instance).map_err(|_| EOVERFLOW)?;
        if max_instances == 0 || num_type_instances >= max_instances {
            return Err(ENOSPC);
        }

        // Reserve capacity before acquiring resources so registration cannot allocate.
        self.instances.reserve(1, GFP_KERNEL)?;

        let channels_per_instance = vgpu
            .total_channels
            .checked_div(vgpu_type.max_instance)
            .ok_or(EINVAL)?;
        if channels_per_instance <= 1 {
            return Err(EINVAL);
        }
        let channels_per_instance =
            usize::try_from(channels_per_instance).map_err(|_| EOVERFLOW)?;
        let channels_per_instance = NonZeroUsize::new(channels_per_instance).ok_or(EINVAL)?;
        let chids = vgpu
            .chid_pool
            .reserve_ids(channels_per_instance, Alignment::SZ_1)?;

        let vram_layout = VgpuVramLayout {
            type_id,
            max_slots: vgpu_type.max_instance,
            fb_size: vgpu_type.fb_length,
            heap_size: vgpu_type.gsp_heap_size,
            fb_align: vgpu.vmmu_segment_size,
        };
        let vram_slot = self.alloc_vram_slot(&vgpu.mm.lock(), vram_layout)?;
        let comm = CommBufferRegion::new(vgpu.bar_user, vgpu.mm, &vram_slot.mgmt_heap)?;

        let instance = VgpuInstance {
            gfid,
            dbdf,
            vgpu_type,
            vm_pid,
            num_plugin_channels: PLUGIN_CHANNELS_PER_ENGINE,
            plugin_rpc: PluginRpc::new(comm, bar0, gfid),
            vram_slot,
            chids,
            ceutils: None,
            needs_teardown: false,
            failure: None,
        };
        self.instances
            .push_within_capacity(instance)
            .map_err(|_| EIO)?;

        Ok(PendingInstance {
            instances: self,
            vgpu,
            gfid,
            committed: false,
        })
    }

    /// Remove an instance only after firmware and mapping teardown have succeeded.
    fn destroy_instance(&mut self, vgpu: &VgpuManager<'gpu>, gfid: Gfid) -> Result {
        let instance_index = self
            .instances
            .iter()
            .position(|instance| instance.gfid == gfid)
            .ok_or(ENOENT)?;
        let instance = self.instances.get_mut(instance_index).ok_or(EIO)?;
        instance.teardown(vgpu)?;

        let instance = self.instances.remove(instance_index).map_err(|_| EIO)?;
        drop(instance);
        Ok(())
    }

    /// Release the registry while the manager's MM and firmware dependencies remain available.
    pub(super) fn release_all(&mut self, vgpu: &VgpuManager<'gpu>) {
        for instance in &mut self.instances {
            // Failed operations retain their resources until this final device removal.
            // Do not resubmit commands whose firmware outcome was uncertain.
            if instance.failure.is_none() {
                if let Err(error) = instance.teardown(vgpu) {
                    dev_err!(
                        vgpu.dev,
                        "vGPU teardown failed for gfid={}: {:?}\n",
                        instance.gfid.get(),
                        error,
                    );
                }
            }
        }
        self.instances.clear();
    }
}

/// Rolls back this creation attempt unless ownership has passed to the caller.
struct PendingInstance<'a, 'gpu> {
    instances: &'a mut VgpuInstances<'gpu>,
    vgpu: &'a VgpuManager<'gpu>,
    gfid: Gfid,
    committed: bool,
}

#[expect(dead_code)]
impl PendingInstance<'_, '_> {
    fn activate(mut self) -> Result {
        let instance = self
            .instances
            .instances
            .iter_mut()
            .find(|instance| instance.gfid == self.gfid)
            .ok_or(EIO)?;
        instance.initialize(self.vgpu)?;
        instance.activate(self.vgpu)?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for PendingInstance<'_, '_> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        if let Err(error) = self.instances.destroy_instance(self.vgpu, self.gfid) {
            dev_err!(
                self.vgpu.dev,
                "vGPU creation could not release gfid={}: {:?}; resources retained until removal\n",
                self.gfid.get(),
                error,
            );
        }
    }
}
