// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

//! PF-owned lifecycle operations for Nova VF drivers.

use core::num::NonZero;

use kernel::{
    device,
    pci,
    prelude::*, //
};

#[cfg(CONFIG_PCI_IOV)]
use kernel::types::ForLt;

use crate::{
    driver::Bar0,
    gpu::Gpu,
    gsp::cmdq::Cmdq,
    vgpu::instance::{
        query_vgpu_type,
        Gfid,
        InstanceInfo,
        VgpuType, //
    },
};

use super::{
    commands::{
        query_assigned_vf_type,
        Dbdf, //
    },
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

impl VgpuTypeInfo {
    fn from_vgpu_type(vgpu_type: &VgpuType) -> Self {
        Self {
            pci_dev_id: vgpu_type.pci_dev_id,
            pci_subsys_id: vgpu_type.pci_subsys_id,
            bar1_length: vgpu_type.bar1_length,
        }
    }
}

/// PF-owned operations available while a VF driver is bound.
pub struct NovaCoreVfApi<'gpu> {
    pdev: &'gpu pci::Device<device::Bound>,
    cmdq: &'gpu Cmdq<'gpu>,
    bar: Bar0<'gpu>,
    vgpu: Option<&'gpu VgpuManager<'gpu>>,
    total_vfs: Option<NonZero<u16>>,
}

impl<'gpu> NovaCoreVfApi<'gpu> {
    pub(crate) fn new(gpu: &'gpu Gpu<'gpu>, pdev: &'gpu pci::Device<device::Bound>) -> Self {
        Self {
            pdev,
            cmdq: gpu.cmdq(),
            bar: gpu.bar0(),
            vgpu: gpu.vgpu_manager(),
            total_vfs: gpu.vgpu_total_vfs(),
        }
    }

    /// Returns whether the PF booted with vGPU support enabled.
    pub fn is_available(&self) -> bool {
        self.vgpu.is_some()
    }

    fn gfid(&self, gfid: u32) -> Result<Gfid> {
        let total_vfs = self.total_vfs.ok_or(ENODEV)?;
        Gfid::new(gfid, total_vfs)
    }
}

impl NovaCoreVfApi<'_> {
    /// Obtains the enabled PF services for a bound VF.
    #[cfg(CONFIG_PCI_IOV)]
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
    pub fn open_instance(&self, gfid: u32, sbdf: u32, vm_pid: u32) -> Result<VgpuTypeInfo> {
        let dev = self.pdev.as_ref();
        let gfid = self.gfid(gfid)?;
        let dbdf = Dbdf::from_raw(sbdf);

        dev_dbg!(
            dev,
            "vgpu_open: gfid={} sbdf={:#x}\n",
            gfid.get(),
            dbdf.into_raw()
        );

        let bar = self.bar;
        let cmdq = self.cmdq;
        let vgpu = self.vgpu.ok_or(ENODEV)?;

        let type_id = query_assigned_vf_type(dev, cmdq, dbdf)?;
        dev_dbg!(
            dev,
            "vgpu_open: gfid={} assigned type_id={}\n",
            gfid.get(),
            type_id
        );

        let vgpu_type = query_vgpu_type(dev, cmdq, type_id)?;
        dev_dbg!(
            dev,
            "vgpu_open: gfid={} vgpu_type={} fb_length={:#x}\n",
            gfid.get(),
            vgpu_type.vgpu_type_id,
            vgpu_type.fb_length
        );

        let type_info = VgpuTypeInfo::from_vgpu_type(&vgpu_type);
        vgpu.create_instance(bar, InstanceInfo::new(gfid, dbdf, vgpu_type, vm_pid))?;

        Ok(type_info)
    }

    /// Tears down an instance, retaining resources if firmware cleanup fails.
    ///
    /// This call may sleep.
    pub fn close_instance(&self, gfid: u32) -> Result {
        let dev = self.pdev.as_ref();
        let gfid = self.gfid(gfid)?;

        dev_dbg!(dev, "vgpu_close: gfid={}\n", gfid.get());

        let result = self.vgpu.ok_or(ENODEV)?.close_instance(gfid);
        if let Err(error) = result {
            dev_err!(dev, "vgpu_close: gfid={} failed: {:?}\n", gfid.get(), error);
        }
        result
    }

    /// Resets an instance and scrubs its guest VRAM. This call may sleep.
    pub fn reset_instance(&self, gfid: u32) -> Result {
        let dev = self.pdev.as_ref();
        let gfid = self.gfid(gfid)?;

        dev_dbg!(dev, "vgpu_reset: gfid={}\n", gfid.get());

        self.vgpu.ok_or(ENODEV)?.reset_instance(gfid)?;

        dev_dbg!(dev, "vgpu_reset: gfid={} done\n", gfid.get());
        Ok(())
    }
}

/// Access to PF services for the lifetime of a VF driver binding.
///
/// The PF's registration keeps these services available until VF removal
/// completes and is dropped before the GPU resources they borrow.
#[cfg(CONFIG_PCI_IOV)]
pub struct NovaCoreVfApiHandle<'vf> {
    vf: &'vf pci::Device<device::Bound>,
}

#[cfg(CONFIG_PCI_IOV)]
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
#[cfg(CONFIG_PCI_IOV)]
pub struct VgpuInstance<'vf> {
    api: NovaCoreVfApiHandle<'vf>,
    gfid: u32,
    type_info: VgpuTypeInfo,
}

#[cfg(CONFIG_PCI_IOV)]
impl VgpuInstance<'_> {
    /// Returns the assigned PCI IDs and BAR1 aperture size.
    #[inline]
    pub fn type_info(&self) -> &VgpuTypeInfo {
        &self.type_info
    }
}

#[cfg(CONFIG_PCI_IOV)]
impl Drop for VgpuInstance<'_> {
    fn drop(&mut self) {
        let _ = self.api.with(|api| api.close_instance(self.gfid));
    }
}
