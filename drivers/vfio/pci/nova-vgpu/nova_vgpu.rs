// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

//! NVIDIA vGPU VFIO variant driver.
//!
//! Nova VFs own an instance between the first VFIO open and last close. Other
//! matching devices use ordinary PCI passthrough.

use kernel::{
    bindings,
    device::{
        Bound,
        Core, //
    },
    io::resource::Flags,
    pci,
    prelude::*,
    sync::{
        CondVar,
        Mutex,
        MutexGuard, //
    },
    vfio::{
        self,
        pci::{
            GetRegionInfo,
            Ioctl,
            Mapping,
            Mmap,
            Open,
            Position,
            Read,
            Write, //
        }, //
    }, //
};
use nova_core::{
    NovaCoreVfApi,
    NovaCoreVfApiHandle,
    VgpuInstance, //
};

struct NovaVgpuOps;

struct VgpuRegistration<'a> {
    api: NovaCoreVfApiHandle<'a>,
    gfid: u32,
}

#[derive(Default)]
struct InstanceState {
    active: bool,
    resetting: bool,
    reset_error: Option<Error>,
}

#[pin_data]
struct NovaVgpuRegData<'a> {
    pdev: &'a pci::Device<Bound>,
    vgpu: Option<VgpuRegistration<'a>>,
    fb_bar: u32,
    #[pin]
    state: Mutex<InstanceState>,
    #[pin]
    reset_done: CondVar,
}

impl NovaVgpuRegData<'_> {
    fn lock_instance(&self) -> MutexGuard<'_, InstanceState> {
        let mut state = self.state.lock();
        while state.resetting {
            self.reset_done.wait(&mut state);
        }
        state
    }
}

struct NovaVgpuOpenData<'a, 'data> {
    registration: &'a NovaVgpuRegData<'data>,
    instance: Option<VgpuInstance<'data>>,
}

impl<'a, 'data> NovaVgpuOpenData<'a, 'data> {
    fn new(
        dev: &vfio::pci::Device<NovaVgpuOps, Open>,
        rd: &'a NovaVgpuRegData<'data>,
    ) -> Result<Self> {
        let mut state = rd.lock_instance();
        let instance = match &rd.vgpu {
            Some(vgpu) => {
                // The firmware DBDF format has a 16-bit PCI segment field.
                let segment = u16::try_from(rd.pdev.domain_nr()).map_err(|_| EOVERFLOW)?;
                let dbdf = (u32::from(segment) << 16) | u32::from(rd.pdev.dev_id());
                let vm_pid = kernel::current!().tgid().try_into()?;
                let instance = vgpu.api.open(vgpu.gfid, dbdf, vm_pid)?;
                let info = instance.type_info();
                dev.set_device_id(info.pci_dev_id as u16);
                state.active = true;
                state.reset_error = None;
                Some(instance)
            }
            None => None,
        };
        Ok(Self {
            registration: rd,
            instance,
        })
    }

    fn bar1_size(&self) -> Result<Option<u64>> {
        let Some(instance) = &self.instance else {
            return Ok(None);
        };
        let physical = self
            .registration
            .pdev
            .resource_len(self.registration.fb_bar)?;
        let size = instance
            .type_info()
            .bar1_length
            .checked_mul(1 << 20)
            .ok_or(EOVERFLOW)?;
        Ok(Some(if size == 0 {
            physical
        } else {
            size.min(physical)
        }))
    }

    fn limit_bar1(&self, buf: &mut vfio::UserBuf, position: &Position<'_>) -> Result {
        if buf.is_empty() || position.region_index() != self.registration.fb_bar {
            return Ok(());
        }
        if let Some(size) = self.bar1_size()? {
            let offset = position.region_offset();
            if offset >= size {
                return Err(EINVAL);
            }
            if size - offset < buf.len() as u64 {
                buf.truncate((size - offset) as usize);
            }
        }
        Ok(())
    }
}

impl Drop for NovaVgpuOpenData<'_, '_> {
    fn drop(&mut self) {
        let mut state = self.registration.lock_instance();
        state.active = false;
        drop(self.instance.take());
    }
}

impl vfio::pci::Operations for NovaVgpuOps {
    const NAME: &'static CStr = c"nova-vgpu-vfio-pci";
    type RegistrationData<'a> = NovaVgpuRegData<'a>;
    type OpenData<'a, 'data: 'a> = NovaVgpuOpenData<'a, 'data>;

    fn open_device<'a, 'data: 'a>(
        dev: &'a vfio::pci::Device<Self, Open>,
        rd: &'a NovaVgpuRegData<'data>,
    ) -> impl PinInit<Self::OpenData<'a, 'data>, Error> + 'a {
        NovaVgpuOpenData::new(dev, rd)
    }

    fn ioctl<'data>(
        dev: &vfio::pci::Device<Self, Ioctl>,
        rd: &NovaVgpuRegData<'data>,
        _open_data: Pin<&Self::OpenData<'_, 'data>>,
        cmd: u32,
        arg: usize,
    ) -> Result<isize> {
        let reset = cmd == vfio::DEVICE_RESET || cmd == vfio::pci::DEVICE_PCI_HOT_RESET;
        if reset {
            if let Some(error) = rd.state.lock().reset_error {
                return Err(error);
            }
        }
        let result = dev.core_ioctl(cmd, arg)?;
        if reset {
            if let Some(error) = rd.state.lock().reset_error {
                return Err(error);
            }
        }
        Ok(result)
    }

    fn read<'data>(
        dev: &vfio::pci::Device<Self, Read>,
        _rd: &NovaVgpuRegData<'data>,
        open_data: Pin<&Self::OpenData<'_, 'data>>,
        buf: &mut vfio::UserBuf,
        position: &mut Position<'_>,
    ) -> Result<isize> {
        if let Some(instance) = &open_data.instance {
            if position.region_index() == vfio::pci::CONFIG_REGION_INDEX {
                let offset = position.region_offset();
                return position.with_temporary(|next| {
                    let count = dev.core_read(buf, next)?;
                    buf.write_overlapping(
                        offset,
                        count.try_into()?,
                        u64::from(bindings::PCI_SUBSYSTEM_ID),
                        &(instance.type_info().pci_subsys_id as u16).to_le_bytes(),
                    )?;
                    Ok(count)
                });
            }
        }
        open_data.limit_bar1(buf, position)?;
        dev.core_read(buf, position)
    }

    fn write<'data>(
        dev: &vfio::pci::Device<Self, Write>,
        _rd: &NovaVgpuRegData<'data>,
        open_data: Pin<&Self::OpenData<'_, 'data>>,
        buf: &mut vfio::UserBuf,
        position: &mut Position<'_>,
    ) -> Result<isize> {
        open_data.limit_bar1(buf, position)?;
        dev.core_write(buf, position)
    }

    fn mmap<'data>(
        dev: &vfio::pci::Device<Self, Mmap>,
        rd: &NovaVgpuRegData<'data>,
        open_data: Pin<&Self::OpenData<'_, 'data>>,
        mapping: &mut Mapping<'_>,
    ) -> Result {
        if mapping.region_index() == rd.fb_bar {
            if let Some(size) = open_data.bar1_size()? {
                if mapping.region_end()? > size {
                    return Err(EINVAL);
                }
            }
        }
        dev.core_mmap(mapping)
    }

    fn get_region_info<'data>(
        dev: &vfio::pci::Device<Self, GetRegionInfo>,
        rd: &NovaVgpuRegData<'data>,
        open_data: Pin<&Self::OpenData<'_, 'data>>,
        info: &mut bindings::vfio_region_info,
        caps: &mut vfio::InfoCap<'_>,
    ) -> Result {
        dev.core_get_region_info(info, caps)?;
        if info.index == rd.fb_bar && info.size != 0 {
            if let Some(size) = open_data.bar1_size()? {
                info.size = info.size.min(size);
            }
        }
        Ok(())
    }

    fn reset_prepare(rd: &NovaVgpuRegData<'_>) {
        // PCI holds the device lock, and VFIO may hold its memory lock. Open and
        // close release this mutex before entering the VFIO core.
        let mut state = rd.state.lock();
        state.resetting = true;
        if state.active {
            if let Some(vgpu) = &rd.vgpu {
                if let Err(error) = vgpu.api.reset(vgpu.gfid) {
                    state.reset_error.get_or_insert(error);
                }
            }
        }
    }

    fn reset_done(rd: &NovaVgpuRegData<'_>) -> Result {
        let mut state = rd.state.lock();
        let error = state.reset_error;
        state.resetting = false;
        rd.reset_done.notify_all();
        error.map_or(Ok(()), Err)
    }
}

struct NovaVgpuDriver;

kernel::pci_device_table!(
    PCI_TABLE,
    <NovaVgpuDriver as pci::Driver>::IdInfo,
    [(
        pci::DeviceId::from_id_vfio_override(pci::Vendor::NVIDIA, bindings::PCI_ANY_ID as u32),
        (),
    ),]
);

#[pin_data]
struct NovaVgpuData<'bound> {
    reg: vfio::pci::Registration<'bound, NovaVgpuOps>,
}

#[vtable]
impl pci::Driver for NovaVgpuDriver {
    type IdInfo = ();
    type Data<'bound> = NovaVgpuData<'bound>;
    const ID_TABLE: pci::IdTable<Self::IdInfo> = &PCI_TABLE;
    // SAFETY: VFIO hands the DMA ownership of the device to userspace. This driver never uses
    // the kernel DMA API for its devices, directly or through helpers.
    const DRIVER_MANAGED_DMA: Option<pci::DriverManagedDma<Self>> =
        Some(unsafe { pci::DriverManagedDma::new() });

    fn probe<'bound>(
        pdev: &'bound pci::Device<Core<'_>>,
        _info: Option<&'bound Self::IdInfo>,
    ) -> impl PinInit<Self::Data<'bound>, Error> + 'bound {
        try_pin_init!(NovaVgpuData {
            reg: {
                let vgpu = match NovaCoreVfApi::handle(pdev) {
                    Ok(api) => Some(VgpuRegistration {
                        api,
                        gfid: u32::from(pdev.vf_id()?) + 1,
                    }),
                    Err(_) => None,
                };
                let dev = if vgpu.is_some() {
                    vfio::pci::Device::<NovaVgpuOps>::new(pdev)?
                } else {
                    vfio::pci::Device::<NovaVgpuOps>::new_passthrough(pdev)?
                };
                let fb_bar = if pdev.resource_flags(0)?.contains(Flags::IORESOURCE_MEM_64) {
                    2
                } else {
                    1
                };
                // SAFETY: The registration is stored in `NovaVgpuData`, this driver's bus
                // device private data, which is never leaked.
                unsafe {
                    vfio::pci::Registration::new::<Self>(
                        pdev,
                        &dev,
                        try_pin_init!(NovaVgpuRegData {
                            pdev,
                            vgpu,
                            fb_bar,
                            state <- kernel::new_mutex!(InstanceState::default()),
                            reset_done <- kernel::new_condvar!(),
                        }),
                    )?
                }
            },
        })
    }

    fn reset_prepare<'bound>(_pdev: &pci::Device<Core<'_>>, data: Pin<&Self::Data<'bound>>) {
        data.reg.reset_prepare();
    }

    fn reset_done<'bound>(pdev: &pci::Device<Core<'_>>, data: Pin<&Self::Data<'bound>>) {
        if let Err(error) = data.reg.reset_done() {
            dev_err!(pdev, "VFIO device reset failed: {:?}\n", error);
            data.reg.error_detected();
        }
    }

    fn error_detected<'bound>(
        _pdev: &pci::Device<Core<'_>>,
        data: Pin<&Self::Data<'bound>>,
        _state: pci::ChannelState,
    ) -> pci::ErsResult {
        data.reg.error_detected()
    }
}

kernel::module_pci_driver! {
    type: NovaVgpuDriver,
    name: "nova-vgpu-vfio-pci",
    authors: ["NVIDIA"],
    description: "NVIDIA vGPU VFIO variant driver",
    license: "GPL v2",
}
