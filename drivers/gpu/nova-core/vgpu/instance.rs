// SPDX-License-Identifier: GPL-2.0

use kernel::{
    device,
    devres::Devres,
    prelude::*,
    sync::{
        Arc,
        Mutex, //
    },
};

use crate::{
    driver::{
        Bar0,
        Bar1, //
    },
    gsp::{
        cmdq::Cmdq,
        nvkv, //
    },
    mm::{
        bar_user::BarUser,
        GpuMm, //
    },
    vgpu::{
        alloc_vram,
        bootload::{
            bootload,
            shutdown, //
        },
        consts::{
            gmcapi,
            vgpu_prop_keys, //
        },
        plugin_rpc::PluginRpc,
        Bar1Map,
        ChidAllocator,
        VgpuManager,
        VramBlock, //
    },
};

/// Guest Function ID. GFID 0 is reserved for PF, VFs start at 1.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Gfid(pub u32);

/// PCI address encoding: domain[31:16] bus[15:8] devfn[7:0].
#[repr(transparent)]
#[derive(Clone, Copy)]
pub(crate) struct Dbdf(pub u32);

/// vGPU type descriptor, populated from QUERY_VGPU_PROPERTIES NVKV response.
pub(crate) struct VgpuType {
    pub name: [u8; 64],
    pub class: [u8; 64],
    pub vgpu_type_id: u32,
    pub bar1_length: u64,
    pub max_instance: u32,
    pub ecc_supported: u32,
    pub profile_size: u64,
    pub max_fps: u32,
    pub num_heads: u32,
    pub max_res_x: u32,
    pub max_res_y: u32,
    pub pci_dev_id: u32,
    pub pci_subsys_id: u32,
    pub fb_length: u64,
    pub gsp_heap_size: u64,
    pub fb_reservation: u64,
}

impl Default for VgpuType {
    fn default() -> Self {
        Self {
            name: [0u8; 64],
            class: [0u8; 64],
            vgpu_type_id: 0,
            bar1_length: 0,
            max_instance: 0,
            ecc_supported: 0,
            profile_size: 0,
            max_fps: 0,
            num_heads: 0,
            max_res_x: 0,
            max_res_y: 0,
            pci_dev_id: 0,
            pci_subsys_id: 0,
            fb_length: 0,
            gsp_heap_size: 0,
            fb_reservation: 0,
        }
    }
}

/// A live vGPU instance with allocated resources.
#[expect(dead_code)]
pub(crate) struct VgpuInstance {
    pub id: i32,
    pub gfid: Gfid,
    pub dbdf: Dbdf,
    pub vgpu_type: VgpuType,
    pub vm_pid: u32,
    pub chid_offset: u32,
    pub num_chid: u32,
    pub num_plugin_channels: u32,
    pub fbmem_heap: Option<VramBlock>,
    pub mgmt_heap: Option<VramBlock>,
    pub plugin_rpc: Option<PluginRpc>,
    pub active: bool,
}

/// Query the vGPU type assigned to a VF by its DBDF.
#[expect(dead_code)]
pub(crate) fn query_assigned_vf_type(cmdq: &Cmdq, bar: &Bar0, dbdf: Dbdf) -> Result<u32> {
    let in_params = u64::from(dbdf.0).to_le_bytes();
    let resp =
        cmdq.send_gmc_and_receive(bar, gmcapi::VGPU_MGMT_QUERY_ASSIGNED_VF, &in_params, 64)?;
    if resp.status != 0 {
        return Err(EIO);
    }
    if resp.payload.len() < 4 {
        return Err(ENODEV);
    }
    Ok(u32::from_le_bytes(
        resp.payload[..4].try_into().map_err(|_| EINVAL)?,
    ))
}

/// Query vGPU type properties and decode NVKV response.
#[expect(dead_code)]
pub(crate) fn query_vgpu_type(cmdq: &Cmdq, bar: &Bar0, type_id: u32) -> Result<VgpuType> {
    let in_params = type_id.to_le_bytes();
    let resp =
        cmdq.send_gmc_and_receive(bar, gmcapi::VGPU_MGMT_QUERY_PROPERTIES, &in_params, 4096)?;
    if resp.status != 0 {
        return Err(EIO);
    }

    let mut vt = VgpuType::default();
    nvkv::nvkv_decode(&resp.payload, |key, _index, value| match key {
        vgpu_prop_keys::TYPE_NAME => nvkv::nvkv_read_string8(&value, &mut vt.name),
        vgpu_prop_keys::CLASS => nvkv::nvkv_read_string8(&value, &mut vt.class),
        vgpu_prop_keys::TYPE_ID => vt.vgpu_type_id = nvkv::nvkv_read_u32(&value),
        vgpu_prop_keys::BAR1_LENGTH => vt.bar1_length = nvkv::nvkv_read_u64(&value),
        vgpu_prop_keys::MAX_INSTANCE => vt.max_instance = nvkv::nvkv_read_u32(&value),
        vgpu_prop_keys::ECC => vt.ecc_supported = nvkv::nvkv_read_u32(&value),
        vgpu_prop_keys::PROFILE_SIZE => vt.profile_size = nvkv::nvkv_read_u64(&value),
        vgpu_prop_keys::MAX_FPS => vt.max_fps = nvkv::nvkv_read_u32(&value),
        vgpu_prop_keys::NUM_HEADS => vt.num_heads = nvkv::nvkv_read_u32(&value),
        vgpu_prop_keys::MAX_RES_X => vt.max_res_x = nvkv::nvkv_read_u32(&value),
        vgpu_prop_keys::MAX_RES_Y => vt.max_res_y = nvkv::nvkv_read_u32(&value),
        vgpu_prop_keys::DEV_ID => vt.pci_dev_id = nvkv::nvkv_read_u32(&value),
        vgpu_prop_keys::SUBSYSTEM_ID => vt.pci_subsys_id = nvkv::nvkv_read_u32(&value),
        vgpu_prop_keys::FB_LENGTH => vt.fb_length = nvkv::nvkv_read_u64(&value),
        vgpu_prop_keys::GSP_HEAP_SIZE => vt.gsp_heap_size = nvkv::nvkv_read_u64(&value),
        vgpu_prop_keys::FB_RESERVATION => vt.fb_reservation = nvkv::nvkv_read_u64(&value),
        _ => {}
    })?;

    Ok(vt)
}

impl VgpuManager {
    /// Allocate resources for a new vGPU instance without activating it.
    ///
    /// Returns an inactive instance with VRAM and channel IDs allocated.
    /// The caller must invoke [`activate_instance`] afterwards (outside the
    /// manager lock) to bootload the GSP plugin and negotiate the RPC
    /// channel.
    #[expect(dead_code)]
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn allocate_instance(
        &mut self,
        dev: &device::Device<device::Bound>,
        mm: &GpuMm,
        bar_user: &Arc<Mutex<BarUser>>,
        bar1: &Arc<Devres<Bar1>>,
        chid_alloc: &mut ChidAllocator,
        gfid: Gfid,
        dbdf: Dbdf,
        vgpu_type: VgpuType,
        vm_pid: u32,
    ) -> Result<VgpuInstance> {
        pr_debug!(
            "allocate_instance: gfid={} dbdf={:#x} vgpu type id={} vm_pid={}\n",
            gfid.0,
            dbdf.0,
            vgpu_type.vgpu_type_id,
            vm_pid
        );

        let num_chid = self.total_avail_chids / vgpu_type.max_instance.max(1);
        let chid_offset = chid_alloc.alloc(num_chid)?;

        pr_debug!(
            "allocate_instance: gfid={} chid_offset={} num_chid={}\n",
            gfid.0,
            chid_offset,
            num_chid
        );

        let fb_size = vgpu_type.fb_length;
        let fb_align = self.vmmu_segment_size;
        let fbmem = alloc_vram(mm, fb_size, fb_align)
            .inspect_err(|_| chid_alloc.free(chid_offset, num_chid))?;

        pr_debug!(
            "allocate_instance: gfid={} guest fbmem addr={:#x} size={:#x}\n",
            gfid.0,
            fbmem.addr,
            fbmem.size
        );

        let mgmt = alloc_vram(mm, vgpu_type.gsp_heap_size, 4096)
            .inspect_err(|_| chid_alloc.free(chid_offset, num_chid))?;

        pr_debug!(
            "allocate_instance: gfid={} mgmt_heap fbmem addr={:#x} size={:#x}\n",
            gfid.0,
            mgmt.addr,
            mgmt.size
        );

        let bar1_map = Bar1Map::new(bar_user, bar1, dev, mm, mgmt.addr, mgmt.size)?;
        let plugin_rpc = PluginRpc::new(bar1_map);

        Ok(VgpuInstance {
            id: self.next_id(),
            gfid,
            dbdf,
            vgpu_type,
            vm_pid,
            chid_offset,
            num_chid,
            num_plugin_channels: 3,
            fbmem_heap: Some(fbmem),
            mgmt_heap: Some(mgmt),
            plugin_rpc: Some(plugin_rpc),
            active: false,
        })
    }

    /// Destroy a vGPU instance by GFID: send GSP shutdown sequence, then free resources.
    #[expect(dead_code)]
    pub(crate) fn destroy_instance(
        &mut self,
        cmdq: &Cmdq,
        bar: &Bar0,
        chid_alloc: &mut ChidAllocator,
        gfid: Gfid,
    ) -> Result {
        pr_debug!("destroy_instance: gfid={}\n", gfid.0);

        let idx = self
            .instances
            .iter()
            .position(|i| i.gfid == gfid)
            .ok_or(ENOENT)?;

        shutdown(cmdq, bar, self.instances[idx].gfid)?;

        let instance = self.instances.remove(idx).map_err(|_| EINVAL)?;
        chid_alloc.free(instance.chid_offset, instance.num_chid);

        pr_debug!("destroy_instance: gfid={} done\n", gfid.0);

        Ok(())
    }
}

/// Bootload the GSP plugin for an allocated instance.
/// Called **without** holding the `VgpuManager` or `ChidAllocator` locks.
#[expect(dead_code)]
pub(crate) fn activate_instance(
    dev: &device::Device<device::Bound>,
    cmdq: &Cmdq,
    bar: &Bar0,
    instance: &mut VgpuInstance,
    engine_masks: &super::GmcEngineMasks,
) -> Result {
    bootload(dev, cmdq, bar, instance, engine_masks)?;
    instance.active = true;
    Ok(())
}
