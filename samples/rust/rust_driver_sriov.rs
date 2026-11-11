// SPDX-License-Identifier: GPL-2.0

//! Rust SR-IOV driver sample based on QEMU's 82576 ([igb]) emulation.
//!
//! To make these drivers probe, QEMU must be run with `-device igb`.
//!
//! Further, enable [vIOMMU] with interrupt remapping using, e.g.,
//!
//! `-M q35,accel=kvm,kernel-irqchip=split -device intel-iommu,intremap=on,caching-mode=on`
//!
//! and append `intel_iommu=on` to the guest kernel arguments.
//!
//! This module registers both the PF driver and the VF driver.
//!
//! [igb]: https://www.qemu.org/docs/master/system/devices/igb.html
//! [vIOMMU]: https://wiki.qemu.org/Features/VT-d

use kernel::{
    device::{
        Bound,
        Core, //
    },
    driver,
    pci,
    prelude::*,
    sync::atomic::{
        Atomic,
        Relaxed, //
    },
    types::CovariantForLt,
    InPlaceModule, //
};

#[path = "rust_driver_sriov_vf.rs"]
mod vf;

struct SamplePfDriver;

#[pin_data]
struct PfApi<'bound> {
    pdev: &'bound pci::Device<Bound>,
    next_request: Atomic<u64>,
}

type PfApiForLt = CovariantForLt!(PfApi<'_>);

impl PfApi<'_> {
    fn submit(self: Pin<&Self>, vf: &pci::Device<Bound>) -> u64 {
        let request = self.next_request.fetch_add(1, Relaxed);

        dev_info!(
            self.pdev,
            "Handle PF request {} from VF devfn {:#x}.\n",
            request,
            vf.dev_id()
        );

        request
    }
}

/// Submits a request to the PF associated with a bound VF.
fn submit_request(vf: &pci::Device<Bound>) -> Result<u64> {
    Ok(vf.vf_registration_data::<PfApiForLt>()?.submit(vf))
}

#[pin_data]
struct PfDriverData<'bound> {
    #[pin]
    _registration: pci::VfRegistration<'bound, PfApiForLt>,
    pdev: &'bound pci::Device<Bound>,
}

kernel::pci_device_table!(
    PF_TABLE,
    <SamplePfDriver as pci::Driver>::IdInfo,
    [(
        // E1000_DEV_ID_82576
        pci::DeviceId::from_id(pci::Vendor::INTEL, 0x10c9),
        ()
    )]
);

#[vtable]
impl pci::Driver for SamplePfDriver {
    type IdInfo = ();
    type Data<'bound> = PfDriverData<'bound>;

    const ID_TABLE: pci::IdTable<Self::IdInfo> = &PF_TABLE;

    fn probe<'bound>(
        pdev: &'bound pci::Device<Core<'_>>,
        _info: Option<&'bound Self::IdInfo>,
    ) -> impl PinInit<Self::Data<'bound>, Error> + 'bound {
        try_pin_init!(PfDriverData {
            _: {
                dev_info!(
                    pdev,
                    "Probe Rust SR-IOV PF sample (PCI ID: {}, 0x{:x}).\n",
                    pdev.vendor_id(),
                    pdev.device_id()
                );
            },
            // SAFETY: The registration is embedded in the PF driver's private data.
            _registration <- unsafe {
                pci::VfRegistration::new(
                    pdev,
                    try_pin_init!(PfApi {
                        pdev,
                        next_request: Atomic::new(1),
                    }),
                )
            },
            _: {
                pdev.enable_device_mem()?;
                pdev.set_master();
            },
            pdev,
        })
    }

    fn sriov_enable<'a>(
        _dev: &'a pci::Device<Core<'_>>,
        data: Pin<&Self::Data<'a>>,
        token: pci::SriovEnable<'a>,
    ) -> Result<pci::SriovEnabled<'a>> {
        dev_info!(
            data.pdev,
            "Enable SR-IOV (PCI ID: {}, 0x{:x}).\n",
            data.pdev.vendor_id(),
            data.pdev.device_id()
        );

        token.enable_all()
    }

    fn sriov_disable<'a>(
        _dev: &'a pci::Device<Core<'_>>,
        data: Pin<&Self::Data<'a>>,
        token: pci::SriovDisable<'a>,
    ) -> Result {
        dev_info!(
            data.pdev,
            "Disable SR-IOV (PCI ID: {}, 0x{:x}).\n",
            data.pdev.vendor_id(),
            data.pdev.device_id()
        );

        token.disable();
        Ok(())
    }
}

#[pin_data]
struct SampleModule {
    #[pin]
    _pf_driver: driver::Registration<pci::Adapter<SamplePfDriver>>,
    #[pin]
    _vf_driver: driver::Registration<pci::Adapter<vf::SampleVfDriver>>,
}

impl InPlaceModule for SampleModule {
    fn init(module: &'static kernel::ThisModule) -> impl PinInit<Self, Error> {
        // Both drivers are on the PCI bus, so they need distinct driver names.
        try_pin_init!(Self {
            _pf_driver <- driver::Registration::new(c"rust_driver_sriov_pf", module),
            _vf_driver <- driver::Registration::new(c"rust_driver_sriov_vf", module),
        })
    }
}

module! {
    type: SampleModule,
    name: "rust_driver_sriov",
    authors: ["Peter Colberg"],
    description: "Rust SR-IOV PF and VF driver sample",
    license: "GPL v2",
}
