// SPDX-License-Identifier: GPL-2.0

use core::cell::Cell;

use kernel::{
    device,
    devres::Devres,
    dma::{
        Device,
        DmaMask, //
    },
    fmt,
    gpu::buddy::GpuBuddyParams,
    pci,
    prelude::*,
    sizes::SZ_4K,
    sync::Arc, //
};

use crate::{
    driver::{Bar0, Bar1},
    falcon::{
        gsp::Gsp as GspFalcon,
        sec2::Sec2 as Sec2Falcon,
        Falcon, //
    },
    fb::SysmemFlush,
    fsp::FspCotVersion,
    gfw,
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
    num::IntoSafeCast,
    regs,
    vgpu::Vgpu, //
};

/// Parameters extracted from GSP boot for initializing memory subsystems.
#[derive(Clone, Copy)]
struct BootParams {
    usable_vram_start: u64,
    usable_vram_size: u64,
    bar1_pde_base: u64,
}

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
            Self::GB100
            | Self::GB102
            | Self::GB202
            | Self::GB203
            | Self::GB205
            | Self::GB206
            | Self::GB207 => Architecture::Blackwell,
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
    pub(crate) const fn fsp_cot_version(self) -> Option<FspCotVersion> {
        match self.arch() {
            Architecture::Hopper => Some(FspCotVersion::new(1)),
            Architecture::Blackwell => Some(FspCotVersion::new(2)),
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

/// Enum representation of the GPU generation.
///
/// TODO: remove the `Default` trait implementation, and the `#[default]`
/// attribute, once the register!() macro (which creates Architecture items) no
/// longer requires it for read-only fields.
#[derive(fmt::Debug, Default, Copy, Clone)]
#[repr(u8)]
pub(crate) enum Architecture {
    #[default]
    Turing = 0x16,
    Ampere = 0x17,
    Hopper = 0x18,
    Ada = 0x19,
    Blackwell = 0x1b,
}

impl Architecture {
    /// Returns the DMA mask supported by this architecture.
    ///
    /// Hopper and Blackwell support 52-bit DMA addresses, while earlier architectures
    /// (Turing, Ampere, Ada) support 47-bit DMA addresses.
    pub(crate) const fn dma_mask(&self) -> DmaMask {
        match self {
            Self::Turing | Self::Ampere | Self::Ada => DmaMask::new::<47>(),
            Self::Hopper | Self::Blackwell => DmaMask::new::<52>(),
        }
    }

    /// Returns whether the GPU uses GFW_BOOT for firmware loading.
    ///
    /// Pre-Hopper architectures (Turing, Ampere, Ada) require waiting for GFW_BOOT completion
    /// before any significant GPU setup. Hopper and later use the FSP Chain of Trust boot path
    /// instead.
    pub(crate) const fn needs_gfw_boot(&self) -> bool {
        matches!(self, Self::Turing | Self::Ampere | Self::Ada)
    }

    /// Returns true for architectures that support vGPU (Ada and later).
    pub(crate) const fn supports_vgpu(&self) -> bool {
        matches!(self, Self::Ada | Self::Blackwell)
    }
}

impl TryFrom<u8> for Architecture {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0x16 => Ok(Self::Turing),
            0x17 => Ok(Self::Ampere),
            0x18 => Ok(Self::Hopper),
            0x19 => Ok(Self::Ada),
            0x1b => Ok(Self::Blackwell),
            _ => Err(ENODEV),
        }
    }
}

impl From<Architecture> for u8 {
    fn from(value: Architecture) -> Self {
        // CAST: `Architecture` is `repr(u8)`, so this cast is always lossless.
        value as u8
    }
}

pub(crate) struct Revision {
    major: u8,
    minor: u8,
}

impl From<regs::NV_PMC_BOOT_42> for Revision {
    fn from(boot0: regs::NV_PMC_BOOT_42) -> Self {
        Self {
            major: boot0.major_revision(),
            minor: boot0.minor_revision(),
        }
    }
}

impl fmt::Display for Revision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:x}.{:x}", self.major, self.minor)
    }
}

/// Structure holding a basic description of the GPU: `Chipset` and `Revision`.
pub(crate) struct Spec {
    chipset: Chipset,
    revision: Revision,
}

impl Spec {
    pub(crate) fn new(dev: &device::Device, bar: &Bar0) -> Result<Spec> {
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

        let boot0 = regs::NV_PMC_BOOT_0::read(bar);

        if boot0.is_older_than_fermi() {
            return Err(ENODEV);
        }

        let boot42 = regs::NV_PMC_BOOT_42::read(bar);
        Spec::try_from(boot42).inspect_err(|_| {
            dev_err!(dev, "Unsupported chipset: {}\n", boot42);
        })
    }

    pub(crate) fn chipset(&self) -> Chipset {
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
    pub bar: Arc<Devres<Bar0>>,
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
    /// vGPU state (module param + SR-IOV / FSP PRC).
    vgpu: Vgpu,
    /// GSP runtime data. Temporarily an empty placeholder.
    #[pin]
    pub(crate) gsp: Gsp,
    /// Static GPU information from GSP.
    gsp_static_info: GetGspStaticInfoReply,
    /// MMIO mapping of PCI BAR 1
    pub(crate) bar1: Arc<Devres<Bar1>>,
    /// BAR1 user interface for CPU access to GPU virtual memory.
    bar_user: BarUser,
}

impl Gpu {
    pub(crate) fn new<'a>(
        pdev: &'a pci::Device<device::Core>,
        devres_bar: Arc<Devres<Bar0>>,
        devres_bar1: Arc<Devres<Bar1>>,
        bar: &'a Bar0,
    ) -> impl PinInit<Self, Error> + 'a {
        let boot_params: Cell<BootParams> = Cell::new(BootParams {
            usable_vram_start: 0,
            usable_vram_size: 0,
            bar1_pde_base: 0,
        });

        pin_init::pin_init_scope(move || {
            let spec = Spec::new(pdev.as_ref(), bar)?;
            dev_info!(pdev, "NVIDIA ({})\n", spec);

            // SAFETY: No concurrent DMA allocations or mappings can be made because
            // the device is still being probed and therefore isn't being used by
            // other threads of execution.
            unsafe { pdev.dma_set_mask_and_coherent(spec.chipset().arch().dma_mask())? };

            let chipset = spec.chipset();

            Ok(try_pin_init!(Self {
                _: {
                    if chipset.arch().needs_gfw_boot() {
                        gfw::wait_gfw_boot_completion(bar)
                            .inspect_err(|_| dev_err!(pdev, "GFW boot did not complete\n"))?;
                    }
                },

                sysmem_flush: SysmemFlush::register(pdev.as_ref(), bar, chipset)?,

                gsp_falcon: Falcon::new(
                    pdev.as_ref(),
                    chipset,
                )
                .inspect(|falcon| falcon.clear_swgen0_intr(bar))?,

                sec2_falcon: Falcon::new(pdev.as_ref(), chipset)?,

                vgpu <- Vgpu::new(pdev, chipset)?,

                gsp <- Gsp::new(pdev),

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
                    let (info, fb_layout) = gsp.boot(&mut ctx)?;
                    vgpu.set_vgpu_enabled(ctx.vgpu_requested);

                    let usable_vram = fb_layout.usable_vram.as_ref().ok_or_else(|| {
                        dev_err!(pdev, "No usable FB regions found from GSP\n");
                        ENODEV
                    })?;

                    dev_info!(
                        pdev,
                        "Using FB region: {:#x}..{:#x}\n",
                        usable_vram.start,
                        usable_vram.end
                    );

                    let mut vram_start = usable_vram.start;
                    let mut vram_size = usable_vram.end - usable_vram.start;

                    // Align buddy base to VMMU segment size so that
                    // power-of-2 buddy allocations produce VMMU-aligned
                    // absolute addresses without over-allocation.
                    let seg = ctx.vmmu_segment_size;
                    if seg > 0 {
                        let aligned = (vram_start + seg - 1) & !(seg - 1);
                        vram_size -= aligned - vram_start;
                        vram_start = aligned;
                    }

                    vgpu.gsp_config.vmmu_segment_size = seg;

                    boot_params.set(BootParams {
                        usable_vram_start: vram_start,
                        usable_vram_size: vram_size,
                        bar1_pde_base: info.bar1_pde_base(),
                    });

                    info
                },

                mm <- {
                    let params = boot_params.get();
                    GpuMm::new(devres_bar.clone(), GpuBuddyParams {
                        base_offset: params.usable_vram_start,
                        physical_memory_size: params.usable_vram_size,
                        chunk_size: SZ_4K.into_safe_cast(),
                    })?
                },

                bar_user: {
                    let params = boot_params.get();
                    let pdb_addr = VramAddress::new(params.bar1_pde_base);
                    let mmu_version = MmuVersion::from(spec.chipset.arch());
                    let bar1_size = pdev.resource_len(crate::driver::bar1_region(pdev)?)?;
                    dev_info!(pdev.as_ref(), "BarUser: pdb_addr={:#x} mmu={:?} bar1_size={:#x}\n",
                        params.bar1_pde_base, mmu_version, bar1_size);

                    {
                        let mm_ref = mm.as_ref().get_ref();
                        let mut w = mm_ref.pramin().window()?;
                        let base: usize = params.bar1_pde_base as usize;
                        for i in 0..4usize {
                            let lo = w.try_read64(base + i * 8)?;
                            dev_info!(pdev.as_ref(), "  PDB[{}] = {:#018x}\n", i, lo);
                        }
                    }

                    {
                        let bar1_ref = devres_bar1.try_access().ok_or(ENODEV)?;
                        let v0 = bar1_ref.try_read32(0).unwrap_or(0xDEAD);
                        let v4 = bar1_ref.try_read32(4).unwrap_or(0xDEAD);
                        dev_info!(pdev.as_ref(), "raw BAR1[0]={:#x} BAR1[4]={:#x}\n", v0, v4);
                    }

                    BarUser::new(pdb_addr, mmu_version, bar1_size)?
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


    /// Run mock vGPU bootload test if vGPU is enabled.
    ///
    /// Uploads a hardcoded L40-1Q vGPU type, initializes post-boot state,
    /// creates a test instance, and verifies the GSP plugin writes the
    /// bootload completion marker.
    pub(crate) fn mock_bootload(
        mut self: Pin<&mut Self>,
        pdev: &pci::Device<device::Bound>,
    ) -> Result {
        // Force mock_bootload to run regardless of vgpu_enabled.
        // if !self.vgpu.vgpu_enabled {
        //     return Ok(());
        // }

        let proj = self.as_mut().project();
        let bar0 = proj.bar.access(pdev.as_ref())?;
        let bar1 = proj.bar1.access(pdev.as_ref())?;
        let gsp = proj.gsp.as_ref().get_ref();
        let mm = proj.mm.as_ref().get_ref();
        let total_vram = mm.buddy().size();

        if let Err(_) = proj.vgpu.upload_vgpu_type(
            &gsp.cmdq, bar0, gsp.h_client(), gsp.h_subdevice(),
        ) {
            dev_err!(pdev, "failed to upload vGPU type to GSP (continuing)\n");
        }

        proj.vgpu.init_post_gsp_boot(
            total_vram, &gsp.engine_caps(),
        ).inspect_err(|_| dev_err!(pdev, "vGPU post-boot init failed\n"))?;

        let comm_size = proj.vgpu.comm_layout.total_size;

        dev_info!(pdev, "mock_bootload: upload+init done, creating instance...\n");
        proj.vgpu.mock_create_instance(
            mm, &gsp.cmdq, bar0, gsp.h_client(), gsp.h_subdevice(), pdev,
        ).inspect_err(|_| dev_err!(pdev, "failed to create mock vGPU instance\n"))?;
        dev_info!(pdev, "mock_bootload: instance created, bootload RPC sent ok\n");

        let idx = proj.vgpu.instances.len() - 1;
        let instance = &proj.vgpu.instances[idx];
        Vgpu::wait_plugin_ready(instance, proj.bar_user, mm, bar1, comm_size)
            .inspect_err(|_| dev_err!(pdev, "vGPU plugin bootload timed out\n"))?;

        dev_info!(pdev, "vGPU plugin bootloaded successfully (magic {:#x})\n", 0x4E65_4A6Fu32);

        proj.vgpu.setup_plugin_rpc(
            instance, proj.bar_user, mm, bar0, bar1, gsp.h_client(),
        ).inspect_err(|e| dev_err!(pdev, "vGPU plugin RPC setup failed: {:?}\n", e))?;

        Ok(())
    }

    #[cfg(CONFIG_NOVA_MM_SELFTESTS)]
    fn run_mm_selftests(mut self: Pin<&mut Self>, pdev: &pci::Device<device::Bound>) -> Result {
        let mmu_version = MmuVersion::from(self.spec.chipset.arch());

        // PRAMIN aperture self-tests.
        crate::mm::pramin::run_self_test(pdev.as_ref(), self.bar.clone(), mmu_version)?;

        // BAR1 self-tests using the persistent BAR1 mapping.
        let proj = self.as_mut().project();
        let bar1_access = proj.bar1.access(pdev.as_ref())?;
        let bar1_pde_base = proj.gsp_static_info.bar1_pde_base();
        let mm = proj.mm.as_ref().get_ref();

        crate::mm::bar_user::run_self_test(
            pdev.as_ref(),
            mm,
            bar1_access,
            bar1_pde_base,
            mmu_version,
        )?;

        Ok(())
    }

    #[cfg(not(CONFIG_NOVA_MM_SELFTESTS))]
    fn run_mm_selftests(self: Pin<&mut Self>, _pdev: &pci::Device<device::Bound>) -> Result {
        Ok(())
    }
}
