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
    sync::Arc, //
};

use crate::{
    bounded_enum,
    driver::{Bar0, Bar1},
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
        GspBootContext,
        MAX_PARTITIONS_WITH_GFID,
        MAX_PARTITIONS_WITH_GFID_32VM, //
    },
    mm::{
        bar_user::BarUser,
        pagetable::MmuVersion,
        GpuMm,
        VramAddress, //
    },
    regs,
    vgpu::Vgpu, //
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
    /// Whether this architecture uses SEC2 for GSP boot (vs FSP Chain of Trust).
    pub(crate) const fn uses_sec2_boot(&self) -> bool {
        matches!(self, Self::Turing | Self::Ampere | Self::Ada)
    }

    /// Whether this architecture supports vGPU.
    pub(crate) const fn supports_vgpu(&self) -> bool {
        matches!(
            self,
            Self::Ada | Self::BlackwellGB10x | Self::BlackwellGB20x
        )
    }

    /// Returns the DMA mask supported by this architecture.
    pub(crate) const fn dma_mask(&self) -> DmaMask {
        match self {
            Self::Turing | Self::Ampere | Self::Ada => DmaMask::new::<47>(),
            Self::Hopper | Self::BlackwellGB10x | Self::BlackwellGB20x => DmaMask::new::<52>(),
        }
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
    /// GPU memory manager owning memory management resources.
    #[pin]
    mm: GpuMm,
    /// GSP runtime data. Temporarily an empty placeholder.
    #[pin]
    pub(crate) gsp: Gsp,
    vgpu: Vgpu,
    /// Static GPU information from GSP.
    gsp_static_info: GetGspStaticInfoReply,
    /// MMIO mapping of PCI BAR 1
    pub(crate) bar1: Arc<Devres<Bar1>>,
    /// BAR1 user interface for CPU access to GPU virtual memory.
    bar_user: Option<BarUser>,
    /// fwctl device registration for GMC API pass-through.
    #[pin]
    _fwctl_reg: Devres<fwctl::Registration<crate::fwctl::NovaCoreFwCtl>>,
}

impl Gpu {
    pub(crate) fn new<'a>(
        pdev: &'a pci::Device<device::Core>,
        devres_bar: Arc<Devres<Bar0>>,
        devres_bar1: Arc<Devres<Bar1>>,
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

            let (gsp_fw_path, gsp_fw_blob) = firmware::request_firmware(
                pdev.as_ref(),
                chipset,
                "gsp",
                firmware::FIRMWARE_VERSION,
            )?;
            let build_id = firmware::elf_build_id(gsp_fw_blob.data());
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

                vgpu: Vgpu::new(pdev, chipset)?,

                gsp <- Gsp::new(pdev, chipset, build_id.as_ref()),

                _: {
                    // SAFETY: `gsp.cmdq` is pinned inside `Gsp` which is owned by
                    // this `Gpu`, and the fwctl `Devres` registration is torn down
                    // before `Gsp` is dropped. Use `get_unchecked_mut` + `addr_of!`
                    // to extract the pointer without consuming the pin-init field
                    // reference. Store it in `cmdq_cell` because pin-init blocks
                    // capture by move and cannot share mutable locals.
                    // SAFETY: We do not move the pinned `Gsp`; `addr_of!` only
                    // computes the address.
                    cmdq_cell.set(core::ptr::addr_of!(gsp.as_ref().get_ref().cmdq));
                },

                gsp_static_info: {
                    let mut ctx = GspBootContext {
                        pdev,
                        bar,
                        chipset,
                        gsp_falcon,
                        sec2_falcon,
                        fsp_falcon: None,
                        vgpu_requested: vgpu.vgpu_requested,
                        vmmu_segment_size: 0,
                        vf_partition_count: if vgpu.vgpu_requested {
                            if vgpu.total_vfs > u16::from(MAX_PARTITIONS_WITH_GFID_32VM) {
                                MAX_PARTITIONS_WITH_GFID
                            } else {
                                MAX_PARTITIONS_WITH_GFID_32VM
                            }
                        } else {
                            0
                        },
                    };
                    let info = gsp.boot(
                        &mut ctx,
                        &gsp_fw_blob, gsp_fw_path,
                    )?;
                    vgpu.vgpu_enabled = ctx.vgpu_requested;
                    vgpu.gsp_config.vmmu_segment_size = ctx.vmmu_segment_size;

                    dev_info!(
                        pdev.as_ref(),
                        "Using FB region: {:#x}..{:#x}\n",
                        info.usable_fb_region.start,
                        info.usable_fb_region.end
                    );

                    info
                },

                mm <- {
                    let usable_vram = &gsp_static_info.usable_fb_region;
                    let pramin_vram_region = 0..gsp_static_info.total_fb_end;

                    let mut vram_start = usable_vram.start;
                    let mut vram_size = usable_vram.end - usable_vram.start;

                    // Align buddy base to VMMU segment size so that
                    // power-of-2 buddy allocations produce VMMU-aligned
                    // absolute addresses without over-allocation.
                    let seg = vgpu.gsp_config.vmmu_segment_size;
                    if seg > 0 {
                        let aligned = (vram_start + seg - 1) & !(seg - 1);
                        vram_size -= aligned - vram_start;
                        vram_start = aligned;
                    }

                    GpuMm::new(devres_bar.clone(), GpuBuddyParams {
                        base_offset: vram_start,
                        size: vram_size,
                        chunk_size: Alignment::new::<SZ_4K>(),
                    }, pramin_vram_region)?
                },

                bar_user: {
                    let mmu_version = MmuVersion::from(spec.chipset.arch());
                    // TODO: Extend BarUser/Vmm to support MMU V3 for Hopper/Blackwell.
                    if mmu_version == MmuVersion::V2 {
                        let pdb_addr = VramAddress::new(gsp_static_info.bar1_pde_base);
                        let bar1_size = pdev.resource_len(1)?;
                        Some(BarUser::new(pdb_addr, mmu_version, bar1_size)?)
                    } else {
                        None
                    }
                },

                _fwctl_reg <- {
                    // SAFETY: `cmdq_cell.get()` returns a valid pointer to
                    // the pinned `Cmdq` inside `Gsp`, which outlives the fwctl
                    // registration (see `NovaCoreFwCtlData::new` safety docs).
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
                bar1: devres_bar1,
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
        let mmu_version = MmuVersion::from(self.spec.chipset.arch());

        // PRAMIN aperture self-tests.
        crate::mm::pramin::run_self_test(pdev.as_ref(), self.mm.pramin(), self.spec.chipset)?;

        // BAR1 self-tests using the persistent BAR1 mapping — only on V2 MMU.
        if self.bar_user.is_some() {
            let bar1_access = self.bar1.access(pdev.as_ref())?;

            crate::mm::bar_user::run_self_test(
                pdev.as_ref(),
                &self.mm,
                bar1_access,
                self.gsp_static_info.bar1_pde_base,
                mmu_version,
            )?;
        }

        Ok(())
    }

    #[cfg(not(CONFIG_NOVA_MM_SELFTESTS))]
    fn run_mm_selftests(self: Pin<&mut Self>, _pdev: &pci::Device<device::Bound>) -> Result {
        Ok(())
    }
}
