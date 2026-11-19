// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

//! PF-owned lifecycle operations for Nova VF drivers.

use core::num::NonZero;

use kernel::{
    device,
    pci,
    prelude::*,
    types::ForLt, //
};

use crate::{
    gpu::Gpu,
    gsp::cmdq::Cmdq,
    vgpu::{
        commands::query_vgpu_properties,
        fw::commands::VgpuProperties,
        instance::{
            Gfid,
            InstanceInfo, //
        },
    },
};

use super::{
    commands::query_assigned_vf_type,
    Dbdf,
    VgpuManager, //
};

/// Guest PCI properties of an activated vGPU instance.
pub struct VgpuTypeInfo {
    /// PCI device ID to present to the guest.
    pub pci_dev_id: u32,
    /// PCI subsystem ID to present to the guest.
    pub pci_subsys_id: u32,
    /// BAR1 aperture size in MiB.
    pub bar1_length: u64,
}

impl From<&VgpuProperties> for VgpuTypeInfo {
    fn from(vgpu_properties: &VgpuProperties) -> Self {
        Self {
            pci_dev_id: vgpu_properties.dev_id,
            pci_subsys_id: vgpu_properties.subsystem_id,
            bar1_length: vgpu_properties.bar1_length,
        }
    }
}

/// PF-owned operations available while a VF driver is bound.
#[pin_data]
pub struct NovaCoreVfApi<'gpu>
where
    // The GPU resources are invariant in their lifetime, which outlives this API borrow.
    exists<'gpu_resources>: 'gpu,
{
    pdev: &'gpu pci::Device<device::Bound>,
    #[uses('gpu_resources: invariant)]
    cmdq: &'gpu Cmdq<'gpu_resources>,
    #[uses('gpu_resources: invariant)]
    vgpu: Option<&'gpu VgpuManager<'gpu_resources>>,
    total_vfs: Option<NonZero<u16>>,
}

impl<'gpu> NovaCoreVfApi<'gpu> {
    /// Creates the PF services that VF drivers access through the VF registration.
    pub(crate) fn new<'res: 'gpu>(
        pdev: &'gpu pci::Device<device::Bound>,
        gpu: Pin<&'gpu Gpu<'res>>,
    ) -> impl PinInit<Self, Error> + use<'gpu, 'res> {
        let gpu = gpu.get_ref();
        try_pin_init!(Self {
            pdev,
            cmdq: gpu.cmdq(),
            vgpu: gpu.vgpu_manager(),
            total_vfs: gpu.vgpu_total_vfs(),
        })
    }

    /// Returns whether the PF booted with vGPU support enabled.
    pub fn is_available(&self) -> bool {
        self.with_vgpu(|vgpu| vgpu.is_some())
    }

    fn gfid(&self, gfid: u32) -> Result<Gfid> {
        let total_vfs = self.total_vfs.ok_or(ENODEV)?;
        Gfid::new(gfid, total_vfs)
    }
}

impl NovaCoreVfApi<'_> {
    /// Obtains the enabled PF services for a bound VF.
    pub fn handle(vf: &pci::Device<device::Bound>) -> Result<NovaCoreVfApiHandle<'_>> {
        let handle = NovaCoreVfApiHandle { vf };
        handle.with(|api| {
            if api.is_available() {
                Ok(())
            } else {
                Err(ENODEV)
            }
        })?;
        Ok(handle)
    }

    /// Creates and boots an instance for a one-based VF ID.
    ///
    /// `sbdf` encodes the VF address as `(segment << 16) | (bus << 8) | devfn`.
    /// `vm_pid` identifies the VM process's thread group. This call may sleep.
    pub fn open_instance(
        self: Pin<&Self>,
        gfid: u32,
        sbdf: u32,
        vm_pid: u32,
    ) -> Result<VgpuTypeInfo> {
        let dev = self.pdev.as_ref();
        let gfid = self.gfid(gfid)?;
        let dbdf = Dbdf::from_raw(sbdf);

        dev_dbg!(
            dev,
            "vgpu_open: gfid={} sbdf={:#x}\n",
            gfid.get(),
            dbdf.into_raw()
        );

        self.with_project_ref(|this| {
            let cmdq = *this.cmdq;
            let vgpu = this.vgpu.ok_or(ENODEV)?;

            let type_id = query_assigned_vf_type(dev, cmdq, dbdf)?;
            dev_dbg!(
                dev,
                "vgpu_open: gfid={} assigned type_id={}\n",
                gfid.get(),
                type_id
            );

            let vgpu_properties = query_vgpu_properties(dev, cmdq, type_id)?;
            dev_dbg!(
                dev,
                "vgpu_open: gfid={} vgpu_type={} fb_length={:#x}\n",
                gfid.get(),
                vgpu_properties.type_id,
                vgpu_properties.fb_length
            );

            vgpu.create_instance(InstanceInfo::new(gfid, dbdf, &vgpu_properties, vm_pid))?;

            Ok(VgpuTypeInfo::from(&*vgpu_properties))
        })
    }

    /// Tears down an instance, retaining resources if firmware cleanup fails.
    ///
    /// This call may sleep.
    pub fn close_instance(&self, gfid: u32) -> Result {
        let dev = self.pdev.as_ref();
        let gfid = self.gfid(gfid)?;

        dev_dbg!(dev, "vgpu_close: gfid={}\n", gfid.get());

        self.with_vgpu(|vgpu| {
            let result = vgpu.ok_or(ENODEV)?.close_instance(gfid);
            if let Err(error) = result {
                dev_err!(dev, "vgpu_close: gfid={} failed: {:?}\n", gfid.get(), error);
            }
            result
        })
    }

    /// Resets an instance and scrubs its guest VRAM. This call may sleep.
    pub fn reset_instance(&self, gfid: u32) -> Result {
        let dev = self.pdev.as_ref();
        let gfid = self.gfid(gfid)?;

        dev_dbg!(dev, "vgpu_reset: gfid={}\n", gfid.get());

        self.with_vgpu(|vgpu| vgpu.ok_or(ENODEV)?.reset_instance(gfid))?;

        dev_dbg!(dev, "vgpu_reset: gfid={} done\n", gfid.get());
        Ok(())
    }
}

/// Access to PF services for the lifetime of a VF driver binding.
///
/// The PF's registration keeps these services available until VF removal
/// completes and is dropped before the GPU resources they borrow.
pub struct NovaCoreVfApiHandle<'vf> {
    vf: &'vf pci::Device<device::Bound>,
}

impl<'vf> NovaCoreVfApiHandle<'vf> {
    /// Borrows the typed PF services for a single operation.
    fn with<R>(
        &self,
        f: impl for<'borrow, 'data> FnOnce(Pin<&'borrow NovaCoreVfApi<'data>>) -> Result<R>,
    ) -> Result<R> {
        self.vf
            .vf_registration_data_with::<ForLt!(NovaCoreVfApi<'_>), _>(f)?
    }

    /// Creates and boots an instance that closes when dropped.
    ///
    /// The arguments have the same meaning as in [`NovaCoreVfApi::open_instance`].
    pub fn open(&self, gfid: u32, dbdf: u32, vm_pid: u32) -> Result<VgpuInstance<'vf>> {
        let type_info = self.with(|api| api.open_instance(gfid, dbdf, vm_pid))?;
        Ok(VgpuInstance {
            api: Self { vf: self.vf },
            gfid,
            type_info,
        })
    }

    /// Resets an active instance and scrubs its guest VRAM.
    pub fn reset(&self, gfid: u32) -> Result {
        self.with(|api| api.reset_instance(gfid))
    }
}

/// An active instance whose teardown runs while its VF driver remains bound.
///
/// Dropping this guard may sleep while firmware teardown completes.
pub struct VgpuInstance<'vf> {
    api: NovaCoreVfApiHandle<'vf>,
    gfid: u32,
    type_info: VgpuTypeInfo,
}

impl VgpuInstance<'_> {
    /// Returns the assigned PCI IDs and BAR1 aperture size.
    #[inline]
    pub fn type_info(&self) -> &VgpuTypeInfo {
        &self.type_info
    }
}

impl Drop for VgpuInstance<'_> {
    fn drop(&mut self) {
        let _ = self.api.with(|api| api.close_instance(self.gfid));
    }
}
