// SPDX-License-Identifier: GPL-2.0

use kernel::{
    device,
    devres::Devres,
    dma::{
        Device,
        DmaMask, //
    },
    fmt,
    fwctl,
    gpu::buddy::GpuBuddyParams,
    io::Io,
    num::Bounded,
    pci,
    prelude::*,
    ptr::Alignment,
    sizes::SZ_4K,
    sync::{
        new_mutex, //
        Arc,
        Mutex,
    }, //
};

use crate::{
    bounded_enum,
    driver::{
        Bar0, //
    },
    falcon::{
        gsp::Gsp as GspFalcon,
        sec2::Sec2 as Sec2Falcon,
        Falcon, //
    },
    fb::SysmemFlush,
    firmware,
    fsp::FspCotVersion,
    gsp::{
        commands::GetGspStaticInfoReply,
        Gsp,
        GspBootContext, //
    },
    mm::{
        bar_user::BarUser,
        pagetable::MmuVersion,
        GpuMm,
        VramAddress, //
    },
    regs,
    vgpu::VgpuManager,
};

mod hal;

macro_rules! define_chipset {
    ({ $($variant:ident = $value:expr),* $(,)* }) =>
    {
        /// Enum representation of the GPU chipset.
        #[derive(fmt::Debug, Copy, Clone, PartialOrd, Ord, PartialEq, Eq)]
        pub(crate) enum Chipset {
            $($variant = $value),*,
        }

        impl Chipset {
            pub(crate) const ALL: &'static [Chipset] = &[
                $( Chipset::$variant, )*
            ];

            ::kernel::macros::paste!(
            /// Returns the name of this chipset, in lowercase.
            ///
            /// # Examples
            ///
            /// ```
            /// let chipset = Chipset::GA102;
            /// assert_eq!(chipset.name(), "ga102");
            /// ```
            pub(crate) const fn name(&self) -> &'static str {
                match *self {
                $(
                    Chipset::$variant => stringify!([<$variant:lower>]),
                )*
                }
            }
            );
        }

        // TODO[FPRI]: replace with something like derive(FromPrimitive)
        impl TryFrom<u32> for Chipset {
            type Error = kernel::error::Error;

            fn try_from(value: u32) -> Result<Self, Self::Error> {
                match value {
                    $( $value => Ok(Chipset::$variant), )*
                    _ => Err(ENODEV),
                }
            }
        }
    }
}

define_chipset!({
    // Turing
    TU102 = 0x162,
    TU104 = 0x164,
    TU106 = 0x166,
    TU117 = 0x167,
    TU116 = 0x168,
    // Ampere
    GA100 = 0x170,
    GA102 = 0x172,
    GA103 = 0x173,
    GA104 = 0x174,
    GA106 = 0x176,
    GA107 = 0x177,
    // Hopper
    GH100 = 0x180,
    // Ada
    AD102 = 0x192,
    AD103 = 0x193,
    AD104 = 0x194,
    AD106 = 0x196,
    AD107 = 0x197,
    // Blackwell
    GB100 = 0x1a0,
    GB102 = 0x1a2,
    GB202 = 0x1b2,
    GB203 = 0x1b3,
    GB205 = 0x1b5,
    GB206 = 0x1b6,
    GB207 = 0x1b7,
});

impl Chipset {
    pub(crate) const fn arch(self) -> Architecture {
        match self {
            Self::TU102 | Self::TU104 | Self::TU106 | Self::TU117 | Self::TU116 => {
                Architecture::Turing
            }
            Self::GA100 | Self::GA102 | Self::GA103 | Self::GA104 | Self::GA106 | Self::GA107 => {
                Architecture::Ampere
            }
            Self::GH100 => Architecture::Hopper,
            Self::AD102 | Self::AD103 | Self::AD104 | Self::AD106 | Self::AD107 => {
                Architecture::Ada
            }
            Self::GB100 | Self::GB102 => Architecture::BlackwellGB10x,
            Self::GB202 | Self::GB203 | Self::GB205 | Self::GB206 | Self::GB207 => {
                Architecture::BlackwellGB20x
            }
        }
    }

    /// Returns `true` if this chipset requires the PIO-loaded bootloader in order to boot FWSEC.
    ///
    /// This includes all chipsets < GA102.
    pub(crate) const fn needs_fwsec_bootloader(self) -> bool {
        matches!(self.arch(), Architecture::Turing) || matches!(self, Self::GA100)
    }

    /// Returns the MMU version for this chipset.
    pub(crate) fn mmu_version(self) -> MmuVersion {
        MmuVersion::from(self.arch())
    }

    /// Returns the FSP Chain of Trust (COT) protocol version for this chipset.
    ///
    /// Hopper (GH100) uses version 1, Blackwell uses version 2.
    /// Returns `None` for architectures that do not use FSP.
    pub(crate) const fn fsp_cot_version(&self) -> Option<FspCotVersion> {
        match self.arch() {
            Architecture::Hopper => Some(FspCotVersion::new(1)),
            Architecture::BlackwellGB10x | Architecture::BlackwellGB20x => {
                Some(FspCotVersion::new(2))
            }
            _ => None,
        }
    }
}

// TODO
//
// The resulting strings are used to generate firmware paths, hence the
// generated strings have to be stable.
//
// Hence, replace with something like strum_macros derive(Display).
//
// For now, redirect to fmt::Debug for convenience.
impl fmt::Display for Chipset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

bounded_enum! {
    /// Enum representation of the GPU generation.
    #[derive(fmt::Debug, Copy, Clone)]
    pub(crate) enum Architecture with TryFrom<Bounded<u32, 6>> {
        Turing = 0x16,
        Ampere = 0x17,
        Hopper = 0x18,
        Ada = 0x19,
        BlackwellGB10x = 0x1a,
        BlackwellGB20x = 0x1b,
    }
}

impl Architecture {
    /// Returns the DMA mask supported by this architecture.
    pub(crate) const fn dma_mask(&self) -> DmaMask {
        match self {
            Self::Turing | Self::Ampere | Self::Ada => DmaMask::new::<47>(),
            Self::Hopper | Self::BlackwellGB10x | Self::BlackwellGB20x => DmaMask::new::<52>(),
        }
    }

    /// Returns true for architectures that support vGPU (RTX PRO 6000 Blackwell Server Edition).
    pub(crate) const fn supports_vgpu(&self) -> bool {
        matches!(self, Self::BlackwellGB20x)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Revision {
    major: Bounded<u8, 4>,
    minor: Bounded<u8, 4>,
}

impl From<regs::NV_PMC_BOOT_42> for Revision {
    fn from(boot0: regs::NV_PMC_BOOT_42) -> Self {
        Self {
            major: boot0.major_revision().cast(),
            minor: boot0.minor_revision().cast(),
        }
    }
}

impl fmt::Display for Revision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:x}.{:x}", self.major, self.minor)
    }
}

/// Structure holding a basic description of the GPU: `Chipset` and `Revision`.
#[derive(Clone, Copy)]
pub(crate) struct Spec {
    chipset: Chipset,
    revision: Revision,
}

impl Spec {
    fn new(dev: &device::Device, bar: &Bar0) -> Result<Spec> {
        // Some brief notes about boot0 and boot42, in chronological order:
        //
        // NV04 through NV50:
        //
        //    Not supported by Nova. boot0 is necessary and sufficient to identify these GPUs.
        //    boot42 may not even exist on some of these GPUs.
        //
        // Fermi through Volta:
        //
        //     Not supported by Nova. boot0 is still sufficient to identify these GPUs, but boot42
        //     is also guaranteed to be both present and accurate.
        //
        // Turing and later:
        //
        //     Supported by Nova. Identified by first checking boot0 to ensure that the GPU is not
        //     from an earlier (pre-Fermi) era, and then using boot42 to precisely identify the GPU.
        //     Somewhere in the Rubin timeframe, boot0 will no longer have space to add new GPU IDs.

        let boot0 = bar.read(regs::NV_PMC_BOOT_0);

        if boot0.is_older_than_fermi() {
            return Err(ENODEV);
        }

        let boot42 = bar.read(regs::NV_PMC_BOOT_42);
        Spec::try_from(boot42).inspect_err(|_| {
            dev_err!(dev, "Unsupported chipset: {}\n", boot42);
        })
    }

    /// Returns this GPU's chipset.
    pub(crate) fn chipset(self) -> Chipset {
        self.chipset
    }
}

impl TryFrom<regs::NV_PMC_BOOT_42> for Spec {
    type Error = Error;

    fn try_from(boot42: regs::NV_PMC_BOOT_42) -> Result<Self> {
        Ok(Self {
            chipset: boot42.chipset()?,
            revision: boot42.into(),
        })
    }
}

impl fmt::Display for Spec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(fmt!(
            "Chipset: {}, Architecture: {:?}, Revision: {}",
            self.chipset,
            self.chipset.arch(),
            self.revision
        ))
    }
}

/// Structure holding the resources required to operate the GPU.
#[pin_data]
pub(crate) struct Gpu {
    spec: Spec,
    /// MMIO mapping of PCI BAR 0
    pub(crate) bar: Arc<Devres<Bar0>>,
    /// System memory page required for flushing all pending GPU-side memory writes done through
    /// PCIE into system memory, via sysmembar (A GPU-initiated HW memory-barrier operation).
    sysmem_flush: SysmemFlush,
    /// GSP falcon instance, used for GSP boot up and cleanup.
    gsp_falcon: Falcon<GspFalcon>,
    /// SEC2 falcon instance, used for GSP boot up and cleanup.
    sec2_falcon: Falcon<Sec2Falcon>,
    /// vGPU state (SR-IOV / FSP PRC), behind Mutex for FFI concurrency.
    #[pin]
    pub(crate) vgpu: Mutex<VgpuManager>,
    /// GSP runtime data.
    #[pin]
    pub(crate) gsp: Gsp,
    /// Static GPU information from GSP.
    gsp_static_info: GetGspStaticInfoReply,
    /// GPU memory manager owning memory management resources.
    #[pin]
    pub(crate) mm: GpuMm,
    /// BAR1 user interface for CPU access to GPU virtual memory.
    bar_user: BarUser,
    /// fwctl device registration for GMC API pass-through.
    #[pin]
    _fwctl_reg: Devres<fwctl::Registration<crate::fwctl::NovaCoreFwCtl>>,
}

impl Gpu {
    pub(crate) fn new<'a>(
        pdev: &'a pci::Device<device::Core>,
        devres_bar: Arc<Devres<Bar0>>,
        bar: &'a Bar0,
    ) -> impl PinInit<Self, Error> + 'a {
        pin_init::pin_init_scope(move || {
            let spec = Spec::new(pdev.as_ref(), bar).inspect(|spec| {
                dev_info!(pdev, "NVIDIA ({})\n", spec);
            })?;
            let chipset = spec.chipset();

            // SAFETY: `Gpu` owns all DMA allocations for this device, and we are
            // still constructing it, so no concurrent DMA allocations can exist.
            unsafe { pdev.dma_set_mask_and_coherent(chipset.arch().dma_mask())? };

            let build_id = firmware::request_firmware(pdev.as_ref(), chipset, "gsp-buildid")
                .ok()
                .and_then(|(_, fw)| firmware::BuildId::from_raw(fw.data()));
            if build_id.is_none() {
                dev_warn!(
                    pdev,
                    "GSP firmware build ID not found, log buffer headers omitted\n"
                );
            }

            let cmdq_cell = core::cell::Cell::new(core::ptr::null::<crate::gsp::cmdq::Cmdq>());

            Ok(try_pin_init!(Self {
                _: {
                    hal::gpu_hal(chipset).wait_gfw_boot_completion(bar)
                        .inspect_err(|_| dev_err!(pdev, "GFW boot did not complete\n"))?;
                },

                sysmem_flush: SysmemFlush::register(pdev.as_ref(), bar, chipset)?,

                gsp_falcon: Falcon::new(
                    pdev.as_ref(),
                    chipset,
                )
                .inspect(|falcon| falcon.clear_swgen0_intr(bar))?,

                sec2_falcon: Falcon::new(pdev.as_ref(), chipset)?,

                vgpu <- new_mutex!(VgpuManager::new(pdev, chipset.arch())?, "vgpu_manager"),

                gsp <- Gsp::new(pdev, chipset, build_id.as_ref()),

                gsp_static_info: {
                    cmdq_cell.set(core::ptr::addr_of!(gsp.as_ref().get_ref().cmdq));
                    let mut mgr = vgpu.lock();
                    let mut ctx = GspBootContext {
                        pdev,
                        bar,
                        chipset,
                        gsp_falcon,
                        sec2_falcon,
                        fsp_falcon: None,
                        vgpu_requested: mgr.vgpu_requested,
                        total_vfs: mgr.total_vfs,
                    };
                    let info = gsp.boot(&mut ctx)?;
                    mgr.set_vgpu_enabled(ctx.vgpu_requested);

                    dev_info!(
                        pdev.as_ref(),
                        "Total physical VRAM: {} MiB\n",
                        info.total_fb_end >> 20
                    );

                    dev_info!(
                        pdev.as_ref(),
                        "Using FB region: {:#x}..{:#x}\n",
                        info.usable_fb_region.start,
                        info.usable_fb_region.end
                    );

                    info
                },

                bar_user: {
                    let pdb_addr = VramAddress::new(gsp_static_info.bar1_pde_base);
                    let bar1_size = pdev.resource_len(2)?;
                    BarUser::new(pdb_addr, spec.chipset, bar1_size)?
                },

                mm <- {
                    let usable_vram = &gsp_static_info.usable_fb_region;

                    let pramin_vram_region = 0..gsp_static_info.total_fb_end;
                    GpuMm::new(devres_bar.clone(), pdev.as_ref(), GpuBuddyParams {
                        base_offset: usable_vram.start,
                        size: usable_vram.end - usable_vram.start,
                        chunk_size: Alignment::new::<SZ_4K>(),
                    }, spec.chipset, pramin_vram_region)?
                },

                _fwctl_reg <- {
                    // SAFETY: `devres_bar` is a valid `Devres<Bar0>` and `cmdq_cell`
                    // holds a pointer to a pinned `cmdq` that outlives this fwctl
                    // registration (torn down by `Devres` before `Gsp` is dropped).
                    let fwctl_data = unsafe {
                        crate::fwctl::NovaCoreFwCtlData::new(
                            devres_bar.clone(),
                            cmdq_cell.get(),
                        )
                    };
                    let fwctl_dev = fwctl::Device::<crate::fwctl::NovaCoreFwCtl>::new(
                        pdev.as_ref(),
                        // SAFETY: The closure writes a fully initialized
                        // `NovaCoreFwCtlData`.
                        unsafe {
                            pin_init::pin_init_from_closure(
                                move |slot: *mut crate::fwctl::NovaCoreFwCtlData| {
                                    slot.write(fwctl_data);
                                    Ok(())
                                },
                            )
                        },
                    )?;
                    fwctl::Registration::new_owned(pdev.as_ref(), fwctl_dev)
                },

                bar: devres_bar,
                spec,
            }))
        })
    }

    /// Called when the corresponding [`Device`](device::Device) is unbound.
    ///
    /// Note: This method must only be called from `Driver::unbind`.
    pub(crate) fn unbind(&self, dev: &device::Device<device::Core>) {
        kernel::warn_on!(self
            .bar
            .access(dev)
            .inspect(|bar| self.sysmem_flush.unregister(bar))
            .is_err());
    }

    /// Run selftests on the constructed [`Gpu`].
    pub(crate) fn run_selftests(
        mut self: Pin<&mut Self>,
        pdev: &pci::Device<device::Bound>,
    ) -> Result {
        self.as_mut().run_mm_selftests(pdev)?;
        Ok(())
    }

    #[cfg(CONFIG_NOVA_MM_SELFTESTS)]
    fn run_mm_selftests(self: Pin<&mut Self>, pdev: &pci::Device<device::Bound>) -> Result {
        // PRAMIN aperture self-tests.
        crate::mm::pramin::run_self_test(pdev.as_ref(), self.mm.pramin(), self.spec.chipset)?;

        // BAR1 self-tests.
        let bar1 = Arc::pin_init(
            pdev.iomap_region_sized::<BAR1_SIZE>(2, c"nova-core/bar1"),
            GFP_KERNEL,
        )?;
        let bar1_access = bar1.access(pdev.as_ref())?;

        crate::mm::bar_user::run_self_test(
            pdev.as_ref(),
            &self.mm,
            bar1_access,
            self.gsp_static_info.bar1_pde_base,
            self.spec.chipset,
        )?;

        Ok(())
    }

    #[cfg(not(CONFIG_NOVA_MM_SELFTESTS))]
    fn run_mm_selftests(self: Pin<&mut Self>, _pdev: &pci::Device<device::Bound>) -> Result {
        Ok(())
    }
}
