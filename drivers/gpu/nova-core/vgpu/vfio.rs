// SPDX-License-Identifier: GPL-2.0

//! FFI exports for the VFIO variant driver.

use kernel::{
    device,
    pci,
    prelude::*, //
};

use crate::{
    driver::NovaCore,
    vgpu::{
        activate_instance,
        consts::plugin_rpc::RpcMsg,
        instance::VgpuType,
        query_assigned_vf_type,
        query_vgpu_type,
        scrubber,
        Dbdf,
        Gfid, //
    },
};

/// Transparent wrapper over the C `struct nvidia_vgpu_type_info`.
#[repr(transparent)]
pub(crate) struct VgpuTypeInfo(kernel::bindings::nvidia_vgpu_type_info);

impl VgpuTypeInfo {
    fn from_vgpu_type(vt: &VgpuType) -> Self {
        Self(kernel::bindings::nvidia_vgpu_type_info {
            pci_dev_id: vt.pci_dev_id,
            pci_subsys_id: vt.pci_subsys_id,
            bar1_length: vt.bar1_length,
        })
    }
}

/// Obtain `Pin<&NovaCore>` and `&device::Device<Bound>` from a PF's raw pci_dev.
///
/// # Safety
///
/// `pf_pdev` must be a valid PF `pci_dev` bound to nova-core.
unsafe fn pf_to_nova_core<'a>(
    pf_pdev: *mut kernel::bindings::pci_dev,
) -> Result<(Pin<&'a NovaCore>, &'a device::Device<device::Bound>)> {
    // SAFETY: `pci::Device` is `#[repr(transparent)]` over `bindings::pci_dev`.
    let pf: &pci::Device<device::Bound> = unsafe { &*pf_pdev.cast() };
    let pf_dev: &device::Device<device::Bound> = pf.as_ref();
    let nova_core: Pin<&NovaCore> = pf_dev.drvdata::<NovaCore>()?;
    Ok((nova_core, pf_dev))
}

fn nvidia_vgpu_open_inner(
    pf_pdev: *mut kernel::bindings::pci_dev,
    gfid: u32,
    dbdf: u32,
) -> Result<VgpuTypeInfo> {
    pr_debug!("vgpu_open: gfid={} dbdf={:#x}\n", gfid, dbdf);

    // SAFETY: The C caller (vfio open_device) guarantees `pf_pdev` is the PF
    // bound to nova-core and remains valid for the duration of this call.
    let (nova_core, dev) = unsafe { pf_to_nova_core(pf_pdev)? };
    let gpu = &nova_core.gpu;
    let gfid = Gfid(gfid);
    let dbdf = Dbdf(dbdf);

    let bar = gpu.bar.access(dev)?;
    let cmdq = &gpu.gsp.cmdq;
    let bar1 = gpu.bar1.as_ref().ok_or(ENODEV)?;

    let type_id = query_assigned_vf_type(cmdq, bar, dbdf)?;
    pr_debug!("vgpu_open: gfid={} assigned type_id={}\n", gfid.0, type_id);

    let vgpu_type = query_vgpu_type(cmdq, bar, type_id)?;
    pr_debug!("vgpu_open: gfid={} vgpu_type={} fb_length={:#x}\n",
        gfid.0, vgpu_type.vgpu_type_id, vgpu_type.fb_length);

    let type_info = VgpuTypeInfo::from_vgpu_type(&vgpu_type);

    // Phase 1: allocate resources (locks held).
    let (mut instance, engine_masks) = {
        let mut mgr = gpu.vgpu.lock();
        let mut chids = gpu.chid_allocator.lock();
        let inst = mgr.allocate_instance(
            dev, &gpu.mm, &gpu.bar_user, bar1, &mut chids, gfid, dbdf, vgpu_type, 0,
            gpu.spec.chipset(), gpu.build_id.as_ref(),
        )?;
        let masks = mgr.engine_masks;
        (inst, masks)
    };

    pr_debug!("vgpu_open: gfid={} instance allocated, chid_offset={}\n",
        gfid.0, instance.chid_offset);

    // Phase 1.5: allocate per-VM CeUtils and scrub guest FB before bootload.
    let (sema_phys, _sema_aperture) =
        scrubber::alloc_ceutils(cmdq, bar, gfid, instance.ceutils_chid, 0)?;
    instance.sema_phys_addr = sema_phys;
    pr_debug!(
        "vgpu_open: gfid={} ceutils sema_phys={:#x}\n",
        gfid.0, sema_phys
    );

    if let Some(fb) = instance.fbmem_heap.as_ref() {
        if let Err(e) = scrubber::write_magic(
            &gpu.bar_user, bar1, dev, &gpu.mm, fb.addr, fb.size,
        ) {
            pr_warn!("vgpu_open: gfid={} write_magic failed: {:?}\n", gfid.0, e);
        }

        match scrubber::scrub_guest_fb(cmdq, bar, gfid, fb.addr, fb.size) {
            Ok(work_id) => {
                if let Err(e) = scrubber::wait_scrub_complete(
                    &gpu.bar_user, bar1, dev, &gpu.mm, sema_phys, work_id,
                ) {
                    pr_warn!("vgpu_open: gfid={} wait_scrub timed out: {:?}\n", gfid.0, e);
                }
            }
            Err(e) => {
                pr_warn!("vgpu_open: gfid={} pre-boot scrub failed: {:?}\n", gfid.0, e);
            }
        }

        if let Err(e) = scrubber::verify_scrub(
            &gpu.bar_user, bar1, dev, &gpu.mm, fb.addr, fb.size,
        ) {
            pr_warn!("vgpu_open: gfid={} verify_scrub failed: {:?}\n", gfid.0, e);
        }
    }

    // Phase 2: bootload + RPC negotiation (no locks held).
    activate_instance(dev, cmdq, bar, &mut instance, &engine_masks)?;

    // Phase 3: register the active instance.
    gpu.vgpu.lock().instances.push(instance, GFP_KERNEL)?;

    Ok(type_info)
}

fn nvidia_vgpu_close_inner(pf_pdev: *mut kernel::bindings::pci_dev, gfid: u32) -> Result {
    pr_debug!("vgpu_close: gfid={}\n", gfid);

    // SAFETY: The C caller (vfio close_device) guarantees `pf_pdev` is valid.
    let (nova_core, dev) = unsafe { pf_to_nova_core(pf_pdev)? };
    let gpu = &nova_core.gpu;
    let gfid = Gfid(gfid);

    let bar = gpu.bar.access(dev)?;
    let cmdq = &gpu.gsp.cmdq;
    let bar1 = gpu.bar1.as_ref().ok_or(ENODEV)?;
    let mut mgr = gpu.vgpu.lock();
    let mut chids = gpu.chid_allocator.lock();

    let result = mgr.destroy_instance(
        dev, &gpu.mm, cmdq, bar, &gpu.bar_user, bar1, &mut chids, gfid,
    );
    pr_debug!("vgpu_close: gfid={} result={:?}\n", gfid.0, result);
    result
}

fn nvidia_vgpu_reset_inner(pf_pdev: *mut kernel::bindings::pci_dev, gfid: u32) -> Result {
    pr_debug!("vgpu_reset: gfid={}\n", gfid);

    // SAFETY: The C caller (vfio ioctl) guarantees `pf_pdev` is valid.
    let (nova_core, dev) = unsafe { pf_to_nova_core(pf_pdev)? };
    let gpu = &nova_core.gpu;
    let gfid = Gfid(gfid);
    let bar = gpu.bar.access(dev)?;

    let mut mgr = gpu.vgpu.lock();
    let instance = mgr
        .instances
        .iter_mut()
        .find(|i| i.gfid == gfid)
        .ok_or(ENOENT)?;

    let plugin_rpc = instance.plugin_rpc.as_mut().ok_or(EINVAL)?;
    plugin_rpc.rpc_call(dev, bar, gfid, RpcMsg::Reset, &[])?;

    if let Some(fb) = instance.fbmem_heap.as_ref() {
        let cmdq = &gpu.gsp.cmdq;
        let sema_phys = instance.sema_phys_addr;
        match scrubber::scrub_guest_fb(cmdq, bar, gfid, fb.addr, fb.size) {
            Ok(work_id) => {
                let bar1 = gpu.bar1.as_ref().ok_or(ENODEV)?;
                if let Err(e) = scrubber::wait_scrub_complete(
                    &gpu.bar_user, bar1, dev, &gpu.mm, sema_phys, work_id,
                ) {
                    pr_warn!("vgpu_reset: gfid={} wait_scrub timed out: {:?}\n", gfid.0, e);
                }
            }
            Err(e) => {
                pr_warn!("vgpu_reset: gfid={} scrub failed: {:?}\n", gfid.0, e);
            }
        }
    }

    pr_debug!("vgpu_reset: gfid={} done\n", gfid.0);
    Ok(())
}

/// # Safety
///
/// `pf_pdev` must be a valid pointer to the PF's `struct pci_dev` bound to
/// nova-core. `gfid` is the Guest Function ID (VF index + 1). `dbdf` is the
/// VF's Domain:Bus:Device.Function encoded as `(domain << 16) | (bus << 8) | devfn`.
/// `type_info` must be a valid pointer to a `struct nvidia_vgpu_type_info`.
#[no_mangle]
#[allow(unreachable_pub)]
pub unsafe extern "C" fn nvidia_vgpu_open(
    pf_pdev: *mut kernel::bindings::pci_dev,
    gfid: core::ffi::c_uint,
    dbdf: core::ffi::c_uint,
    type_info: *mut VgpuTypeInfo,
) -> core::ffi::c_int {
    match nvidia_vgpu_open_inner(pf_pdev, gfid, dbdf) {
        Ok(info) => {
            // SAFETY: caller guarantees `type_info` is a valid, writable pointer.
            unsafe { type_info.write(info) };
            0
        }
        Err(e) => e.to_errno(),
    }
}

/// # Safety
///
/// `pf_pdev` must be a valid pointer to the PF's `struct pci_dev`.
#[no_mangle]
#[allow(unreachable_pub)]
pub unsafe extern "C" fn nvidia_vgpu_close(
    pf_pdev: *mut kernel::bindings::pci_dev,
    gfid: core::ffi::c_uint,
) {
    let _ = nvidia_vgpu_close_inner(pf_pdev, gfid);
}

/// # Safety
///
/// `pf_pdev` must be a valid pointer to the PF's `struct pci_dev`.
#[no_mangle]
#[allow(unreachable_pub)]
pub unsafe extern "C" fn nvidia_vgpu_reset(
    pf_pdev: *mut kernel::bindings::pci_dev,
    gfid: core::ffi::c_uint,
) -> core::ffi::c_int {
    match nvidia_vgpu_reset_inner(pf_pdev, gfid) {
        Ok(()) => 0,
        Err(e) => e.to_errno(),
    }
}
