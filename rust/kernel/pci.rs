// SPDX-License-Identifier: GPL-2.0

//! Abstractions for the PCI bus.
//!
//! C header: [`include/linux/pci.h`](srctree/include/linux/pci.h)

use crate::{
    bindings,
    container_of,
    device,
    device_id::{
        RawDeviceId,
        RawDeviceIdIndex, //
    },
    driver,
    error::{
        from_result,
        to_result, //
    },
    io::resource,
    prelude::*,
    str::CStr,
    types::Opaque,
    ThisModule, //
};
use core::{
    marker::PhantomData,
    mem::offset_of,
    num::NonZero,
    ptr::{
        addr_of_mut,
        NonNull, //
    },
};

mod cap;
mod id;
mod io;
#[cfg(CONFIG_PCI_IOV)]
mod iov;
mod irq;

pub use self::cap::{
    ExtCapId,
    ExtCapability,
    ExtSriovCapability,
    ExtSriovRegs,
    ExtSriovVfBar, //
};
pub use self::id::{
    Class,
    ClassMask,
    Vendor, //
};
pub use self::io::{
    Bar,
    ConfigSpace,
    ConfigSpaceSize,
    DevresBar,
    Extended,
    Normal, //
};
#[cfg(CONFIG_PCI_IOV)]
pub use self::iov::{
    SriovDisable,
    SriovEnable,
    SriovEnabled,
    VfRegistration, //
};
pub use self::irq::{
    IrqType,
    IrqTypes,
    IrqVector,
    IrqVectorRegistration, //
};

/// An adapter for the registration of PCI drivers.
pub struct Adapter<T: Driver>(T);

// SAFETY:
// - `bindings::pci_driver` is a C type declared as `repr(C)`.
// - `T::Data` is the type of the driver's device private data.
// - `struct pci_driver` embeds a `struct device_driver`.
// - `DEVICE_DRIVER_OFFSET` is the correct byte offset to the embedded `struct device_driver`.
unsafe impl<T: Driver> driver::DriverLayout for Adapter<T> {
    type DriverType = bindings::pci_driver;
    type DriverData<'bound> = T::Data<'bound>;
    const DEVICE_DRIVER_OFFSET: usize = core::mem::offset_of!(Self::DriverType, driver);
}

// SAFETY: A call to `unregister` for a given instance of `DriverType` is guaranteed to be valid if
// a preceding call to `register` has been successful.
unsafe impl<T: Driver> driver::RegistrationOps for Adapter<T> {
    unsafe fn register(
        pdrv: &Opaque<Self::DriverType>,
        name: &'static CStr,
        module: &'static ThisModule,
    ) -> Result {
        #[cfg(CONFIG_PCI_IOV)]
        build_assert!(
            T::HAS_SRIOV_ENABLE == T::HAS_SRIOV_DISABLE,
            "PCI drivers must implement both sriov_enable and sriov_disable"
        );

        // SAFETY: It's safe to set the fields of `struct pci_driver` on initialization.
        unsafe {
            (*pdrv.get()).name = name.as_char_ptr();
            (*pdrv.get()).probe = Some(Self::probe_callback);
            (*pdrv.get()).remove = Some(Self::remove_callback);
            (*pdrv.get()).id_table = T::ID_TABLE.as_ptr();
            (*pdrv.get()).driver_managed_dma = T::DRIVER_MANAGED_DMA;
            if T::HAS_ERROR_DETECTED || T::HAS_RESET_PREPARE || T::HAS_RESET_DONE {
                (*pdrv.get()).err_handler = &Self::ERROR_HANDLERS;
            }
            #[cfg(CONFIG_PM)]
            if let Some(ops) = T::PM_OPS {
                (*pdrv.get()).driver.pm = ops.raw;
            }
            #[cfg(CONFIG_PCI_IOV)]
            if T::HAS_SRIOV_ENABLE {
                (*pdrv.get()).sriov_configure = Some(Self::sriov_configure_callback);
            }
        }

        // SAFETY: `pdrv` is guaranteed to be a valid `DriverType`.
        to_result(unsafe {
            bindings::__pci_register_driver(pdrv.get(), module.as_ptr(), name.as_char_ptr())
        })
    }

    unsafe fn unregister(pdrv: &Opaque<Self::DriverType>) {
        // SAFETY: `pdrv` is guaranteed to be a valid `DriverType`.
        unsafe { bindings::pci_unregister_driver(pdrv.get()) }
    }
}

impl<T: Driver> Adapter<T> {
    extern "C" fn probe_callback(
        pdev: *mut bindings::pci_dev,
        id: *const bindings::pci_device_id,
    ) -> c_int {
        // SAFETY: The PCI bus only ever calls the probe callback with a valid pointer to a
        // `struct pci_dev`.
        //
        // INVARIANT: `pdev` is valid for the duration of `probe_callback()`.
        let pdev = unsafe { &*pdev.cast::<Device<device::CoreInternal<'_>>>() };

        // SAFETY: `DeviceId` is a `#[repr(transparent)]` wrapper of `struct pci_device_id` and
        // does not add additional invariants, so it's safe to transmute.
        let id = unsafe { &*id.cast::<DeviceId>() };

        // SAFETY: `id` comes from `T::ID_TABLE` which is of type `IdArray<_, T::IdInfo>` or
        // `pci_device_id_any` which has 0 as driver_data. It can also come from dynamic IDs, which
        // will ensure that `driver_data` exists in `T::ID_TABLE`.
        let info = unsafe { id.info_unchecked_opt::<T::IdInfo>() };

        from_result(|| {
            let data = T::probe(pdev, info);

            pdev.as_ref().set_drvdata(data)?;
            // SAFETY: `set_drvdata` installed the pinned `T::Data` above.
            let data = unsafe { pdev.as_ref().drvdata_borrow::<T::Data<'_>>() };
            if let Err(error) = T::bind(pdev, data) {
                // SAFETY: Failed bind leaves no callbacks borrowing the installed data.
                drop(unsafe { pdev.as_ref().drvdata_obtain::<T::Data<'_>>() });
                return Err(error);
            }
            Ok(0)
        })
    }

    extern "C" fn remove_callback(pdev: *mut bindings::pci_dev) {
        // SAFETY: The PCI bus only ever calls the remove callback with a valid pointer to a
        // `struct pci_dev`.
        //
        // INVARIANT: `pdev` is valid for the duration of `remove_callback()`.
        let pdev = unsafe { &*pdev.cast::<Device<device::CoreInternal<'_>>>() };

        // SAFETY: `remove_callback` is only ever called after a successful call to
        // `probe_callback`, hence it's guaranteed that `Device::set_drvdata()` has been called
        // and stored a `Pin<KBox<T::Data<'_>>>`.
        let data = unsafe { pdev.as_ref().drvdata_borrow::<T::Data<'_>>() };

        T::unbind(pdev, data);

        // SAFETY: The driver's unbind callback has returned, and no callbacks retain a borrow.
        let data = unsafe { pdev.as_ref().drvdata_obtain::<T::Data<'_>>() };

        // Drop private data before returning to PCI core, while its removal context is valid.
        drop(data);
    }

    /// # Safety
    ///
    /// `dev` must belong to `T`. Any installed driver data must remain installed until this
    /// call returns. The data lifetime is independent of the callback borrow.
    unsafe fn callback_data_with<R>(
        dev: *mut bindings::device,
        f: impl for<'borrow, 'data> FnOnce(Pin<&'borrow T::Data<'data>>) -> R,
    ) -> Option<R> {
        // SAFETY: The callback supplies a device bound to `T`.
        let data = unsafe { bindings::dev_get_drvdata(dev) };
        if data.is_null() {
            return None;
        }
        // SAFETY: PCI installed pinned `T::Data`; the caller excludes its removal.
        Some(f(unsafe {
            Pin::new_unchecked(&*data.cast::<T::Data<'_>>())
        }))
    }

    extern "C" fn reset_prepare_callback(pdev: *mut bindings::pci_dev) {
        // SAFETY: PCI holds the device lock throughout reset.
        let dev = unsafe { &*pdev.cast::<Device<device::CoreInternal<'_>>>() };
        // SAFETY: The device lock prevents removal of installed driver data.
        let _ = unsafe {
            Self::callback_data_with(&raw mut (*pdev).dev, |data| T::reset_prepare(dev, data))
        };
    }

    extern "C" fn reset_done_callback(pdev: *mut bindings::pci_dev) {
        // SAFETY: PCI holds the device lock throughout reset.
        let dev = unsafe { &*pdev.cast::<Device<device::CoreInternal<'_>>>() };
        // SAFETY: The device lock prevents removal of installed driver data.
        let _ = unsafe {
            Self::callback_data_with(&raw mut (*pdev).dev, |data| T::reset_done(dev, data))
        };
    }

    extern "C" fn error_detected_callback(
        pdev: *mut bindings::pci_dev,
        state: bindings::pci_channel_state_t,
    ) -> bindings::pci_ers_result_t {
        // SAFETY: PCI error recovery holds the device lock.
        let dev = unsafe { &*pdev.cast::<Device<device::CoreInternal<'_>>>() };
        // SAFETY: The device lock prevents removal of installed driver data.
        unsafe {
            Self::callback_data_with(&raw mut (*pdev).dev, |data| {
                T::error_detected(dev, data, state)
            })
        }
        .unwrap_or(bindings::pci_ers_result_PCI_ERS_RESULT_CAN_RECOVER)
    }

    const ERROR_HANDLERS: bindings::pci_error_handlers = bindings::pci_error_handlers {
        error_detected: if T::HAS_ERROR_DETECTED {
            Some(Self::error_detected_callback)
        } else {
            None
        },
        mmio_enabled: None,
        slot_reset: None,
        reset_prepare: if T::HAS_RESET_PREPARE {
            Some(Self::reset_prepare_callback)
        } else {
            None
        },
        reset_done: if T::HAS_RESET_DONE {
            Some(Self::reset_done_callback)
        } else {
            None
        },
        resume: None,
        cor_error_detected: None,
    };
}

#[cfg(CONFIG_PM)]
impl<T: Driver> Adapter<T>
where
    for<'a> T::Data<'a>: Sync,
{
    extern "C" fn runtime_suspend_callback(dev: *mut bindings::device) -> c_int {
        // SAFETY: PCI supplies its embedded device; runtime PM excludes driver-data removal.
        let pdev =
            unsafe { &*container_of!(dev, bindings::pci_dev, dev).cast::<Device<device::Bound>>() };
        // SAFETY: Runtime PM keeps installed data alive; `T::Data: Sync` permits shared borrows.
        let result =
            unsafe { Self::callback_data_with(dev, |data| T::runtime_suspend(pdev, data)) }
                .unwrap_or(Err(EBUSY));
        match result {
            Ok(()) => 0,
            Err(error) => error.to_errno(),
        }
    }

    extern "C" fn runtime_resume_callback(dev: *mut bindings::device) -> c_int {
        // SAFETY: PCI supplies its embedded device; runtime PM excludes driver-data removal.
        let pdev =
            unsafe { &*container_of!(dev, bindings::pci_dev, dev).cast::<Device<device::Bound>>() };
        // SAFETY: Runtime PM keeps installed data alive; `T::Data: Sync` permits shared borrows.
        let result = unsafe { Self::callback_data_with(dev, |data| T::runtime_resume(pdev, data)) }
            .unwrap_or(Err(ENODEV));
        match result {
            Ok(()) => 0,
            Err(error) => error.to_errno(),
        }
    }

    const PM_OPS: bindings::dev_pm_ops = bindings::dev_pm_ops {
        runtime_suspend: Some(Self::runtime_suspend_callback),
        runtime_resume: Some(Self::runtime_resume_callback),
        ..pin_init::zeroed()
    };
}

/// Runtime PM callbacks for PCI driver `T`, installed through [`Driver::PM_OPS`].
#[cfg(CONFIG_PM)]
pub struct PmOps<T: ?Sized> {
    raw: &'static bindings::dev_pm_ops,
    // Invariant in `T`, so subtyping can't pass one driver's callbacks off as another's.
    _driver: PhantomData<fn(&T) -> &T>,
}

#[cfg(CONFIG_PM)]
impl<T: Driver> PmOps<T>
where
    for<'a> T::Data<'a>: Sync,
{
    /// Returns the runtime PM callbacks for `T`, which must override both PM methods.
    #[expect(clippy::new_without_default)]
    pub const fn new() -> Self {
        const_assert!(
            T::HAS_RUNTIME_SUSPEND && T::HAS_RUNTIME_RESUME,
            "PCI runtime PM requires both runtime_suspend and runtime_resume"
        );
        Self {
            raw: &Adapter::<T>::PM_OPS,
            _driver: PhantomData,
        }
    }
}

/// Declares a kernel module that exposes a single PCI driver.
///
/// # Examples
///
///```ignore
/// kernel::module_pci_driver! {
///     type: MyDriver,
///     name: "Module name",
///     authors: ["Author name"],
///     description: "Description",
///     license: "GPL v2",
/// }
///```
#[macro_export]
macro_rules! module_pci_driver {
($($f:tt)*) => {
    $crate::module_driver!(<T>, $crate::pci::Adapter<T>, { $($f)* });
};
}

/// Abstraction for the PCI device ID structure ([`struct pci_device_id`]).
///
/// [`struct pci_device_id`]: https://docs.kernel.org/PCI/pci.html#c.pci_device_id
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct DeviceId(bindings::pci_device_id);

impl DeviceId {
    const PCI_ANY_ID: u32 = !0;

    /// Equivalent to C's `PCI_DRIVER_OVERRIDE_DEVICE_VFIO` macro.
    pub const fn from_id_vfio_override(vendor: Vendor, device: u32) -> Self {
        let mut id = Self::from_id(vendor, device);
        id.0.override_only = bindings::PCI_ID_F_VFIO_DRIVER_OVERRIDE;
        id
    }

    /// Equivalent to C's `PCI_DEVICE` macro.
    ///
    /// Create a new `pci::DeviceId` from a vendor and device ID.
    #[inline]
    pub const fn from_id(vendor: Vendor, device: u32) -> Self {
        Self(bindings::pci_device_id {
            vendor: vendor.as_raw() as u32,
            device,
            subvendor: DeviceId::PCI_ANY_ID,
            subdevice: DeviceId::PCI_ANY_ID,
            class: 0,
            class_mask: 0,
            driver_data: 0,
            override_only: 0,
        })
    }

    /// Equivalent to C's `PCI_DEVICE_CLASS` macro.
    ///
    /// Create a new `pci::DeviceId` from a class number and mask.
    #[inline]
    pub const fn from_class(class: u32, class_mask: u32) -> Self {
        Self(bindings::pci_device_id {
            vendor: DeviceId::PCI_ANY_ID,
            device: DeviceId::PCI_ANY_ID,
            subvendor: DeviceId::PCI_ANY_ID,
            subdevice: DeviceId::PCI_ANY_ID,
            class,
            class_mask,
            driver_data: 0,
            override_only: 0,
        })
    }

    /// Create a new [`DeviceId`] from a class number, mask, and specific vendor.
    ///
    /// This is more targeted than [`DeviceId::from_class`]: in addition to matching by [`Vendor`],
    /// it also matches the PCI [`Class`] (up to the entire 24 bits, depending on the
    /// [`ClassMask`]).
    #[inline]
    pub const fn from_class_and_vendor(
        class: Class,
        class_mask: ClassMask,
        vendor: Vendor,
    ) -> Self {
        Self(bindings::pci_device_id {
            vendor: vendor.as_raw() as u32,
            device: DeviceId::PCI_ANY_ID,
            subvendor: DeviceId::PCI_ANY_ID,
            subdevice: DeviceId::PCI_ANY_ID,
            class: class.as_raw(),
            class_mask: class_mask.as_raw(),
            driver_data: 0,
            override_only: 0,
        })
    }
}

// SAFETY: `DeviceId` is a `#[repr(transparent)]` wrapper of `pci_device_id` and does not add
// additional invariants, so it's safe to transmute to `RawType`.
unsafe impl RawDeviceId for DeviceId {
    type RawType = bindings::pci_device_id;
}

// SAFETY: `DRIVER_DATA_OFFSET` is the offset to the `driver_data` field.
unsafe impl RawDeviceIdIndex for DeviceId {
    const DRIVER_DATA_OFFSET: usize = core::mem::offset_of!(bindings::pci_device_id, driver_data);
}

/// `IdTable` type for PCI.
pub type IdTable<T> = &'static dyn kernel::device_id::IdTable<DeviceId, T>;

/// Create a PCI `IdTable` with its alias for modpost.
#[macro_export]
macro_rules! pci_device_table {
    ($($tt:tt)*) => {
        $crate::module_device_table!("pci", $crate::pci::DeviceId, $($tt)*);
    };
}

/// The PCI driver trait.
///
/// # Examples
///
///```
/// # use kernel::{bindings, device::Core, pci};
///
/// struct MyDriver;
///
/// kernel::pci_device_table!(
///     PCI_TABLE,
///     <MyDriver as pci::Driver>::IdInfo,
///     [
///         (
///             pci::DeviceId::from_id(pci::Vendor::REDHAT, bindings::PCI_ANY_ID as u32),
///             (),
///         )
///     ]
/// );
///
/// #[vtable]
/// impl pci::Driver for MyDriver {
///     type IdInfo = ();
///     type Data<'bound> = Self;
///     const ID_TABLE: pci::IdTable<Self::IdInfo> = &PCI_TABLE;
///
///     fn probe<'bound>(
///         _pdev: &'bound pci::Device<Core<'_>>,
///         _id_info: Option<&'bound Self::IdInfo>,
///     ) -> impl PinInit<Self::Data<'bound>, Error> + 'bound {
///         Err(ENODEV)
///     }
/// }
///```
/// Drivers must implement this trait in order to get a PCI driver registered. Please refer to the
/// `Adapter` documentation for an example.
#[vtable]
pub trait Driver {
    /// The type holding information about each device id supported by the driver.
    // TODO: Use `associated_type_defaults` once stabilized:
    //
    // ```
    // type IdInfo: 'static = ();
    // ```
    type IdInfo: 'static;

    /// The type of the driver's bus device private data.
    ///
    /// Callbacks access this data under the PCI device lock, except runtime PM callbacks,
    /// which additionally require [`Sync`].
    type Data<'bound>: Send + 'bound;

    /// The table of device ids supported by the driver.
    const ID_TABLE: IdTable<Self::IdInfo>;

    /// Whether the driver manages DMA ownership instead of using the default DMA domain.
    const DRIVER_MANAGED_DMA: bool = false;

    /// PCI driver probe.
    ///
    /// Called when a new pci device is added or discovered. Implementers should
    /// attempt to initialize the device here.
    fn probe<'bound>(
        dev: &'bound Device<device::Core<'_>>,
        id_info: Option<&'bound Self::IdInfo>,
    ) -> impl PinInit<Self::Data<'bound>, Error> + 'bound;

    /// Complete probe after the pinned driver data has been installed.
    ///
    /// Called with the PCI device lock held. On error, the data is dropped without calling
    /// [`Self::unbind`]. Any callbacks started here must be stopped before returning an error.
    fn bind<'bound>(dev: &Device<device::Core<'_>>, this: Pin<&Self::Data<'bound>>) -> Result {
        let _ = (dev, this);
        Ok(())
    }

    /// PCI driver unbind.
    ///
    /// Called when a [`Device`] is unbound from its bound [`Driver`]. Implementing this callback
    /// is optional.
    ///
    /// This callback serves as a place for drivers to perform teardown operations that require a
    /// `&Device<Core>` or `&Device<Bound>` reference. For instance, drivers may try to perform I/O
    /// operations to gracefully tear down the device.
    ///
    /// Otherwise, release operations for driver resources should be performed in `Drop`.
    ///
    /// PCI has drained runtime-PM work before this callback. Driver data remains installed
    /// until this callback returns.
    fn unbind<'bound>(dev: &'bound Device<device::Core<'_>>, this: Pin<&Self::Data<'bound>>) {
        let _ = (dev, this);
    }

    /// Called before a PCI function reset, with the device lock held.
    fn reset_prepare<'bound>(dev: &Device<device::Core<'_>>, this: Pin<&Self::Data<'bound>>) {
        let _ = (dev, this);
        build_error!(crate::error::VTABLE_DEFAULT_ERROR)
    }

    /// Called after a PCI function reset, with the device lock held.
    fn reset_done<'bound>(dev: &Device<device::Core<'_>>, this: Pin<&Self::Data<'bound>>) {
        let _ = (dev, this);
        build_error!(crate::error::VTABLE_DEFAULT_ERROR)
    }

    /// Handle a PCI error and return the PCI recovery result, with the device lock held.
    fn error_detected<'bound>(
        dev: &Device<device::Core<'_>>,
        this: Pin<&Self::Data<'bound>>,
        state: bindings::pci_channel_state_t,
    ) -> bindings::pci_ers_result_t {
        let _ = (dev, this, state);
        build_error!(crate::error::VTABLE_DEFAULT_ERROR)
    }

    /// Prepare for runtime suspend, without holding the PCI device lock.
    #[cfg(CONFIG_PM)]
    fn runtime_suspend<'bound>(
        dev: &Device<device::Bound>,
        this: Pin<&Self::Data<'bound>>,
    ) -> Result
    where
        for<'a> Self::Data<'a>: Sync,
    {
        let _ = (dev, this);
        build_error!(crate::error::VTABLE_DEFAULT_ERROR)
    }

    /// Finish runtime resume, without holding the PCI device lock.
    #[cfg(CONFIG_PM)]
    fn runtime_resume<'bound>(dev: &Device<device::Bound>, this: Pin<&Self::Data<'bound>>) -> Result
    where
        for<'a> Self::Data<'a>: Sync,
    {
        let _ = (dev, this);
        build_error!(crate::error::VTABLE_DEFAULT_ERROR)
    }

    /// Runtime PM callbacks, if any. Construct with [`PmOps::new`].
    #[cfg(CONFIG_PM)]
    const PM_OPS: Option<PmOps<Self>> = None;

    /// Enables Single Root I/O Virtualization (SR-IOV) for a PF.
    ///
    /// Called when userspace writes a nonzero VF count to `sriov_numvfs`. The token carries the
    /// requested count and permission to enable VFs on this PF. Return the guard obtained from
    /// [`SriovEnable::enable()`] after completing setup; dropping it rolls back the enable.
    ///
    /// The PF must own a [`VfRegistration`] before enabling VFs. That registration disables
    /// SR-IOV when the driver data is destroyed, while resources needed by VF drivers remain live.
    /// Implement this callback and [`Self::sriov_disable()`] together, or implement neither.
    ///
    /// See [PCI Express I/O Virtualization].
    ///
    /// [PCI Express I/O Virtualization]: https://docs.kernel.org/PCI/pci-iov-howto.html
    ///
    /// # Examples
    ///
    /// ```
    /// # use kernel::{
    /// #     device::Core,
    /// #     pci,
    /// #     prelude::*, //
    /// # };
    /// # struct Data;
    /// fn sriov_enable<'a>(
    ///     _dev: &'a pci::Device<Core<'_>>,
    ///     _data: Pin<&Data>,
    ///     token: pci::SriovEnable<'a>,
    /// ) -> Result<pci::SriovEnabled<'a>> {
    ///     let num_vfs = token.num_vfs();
    ///     let enabled = token.enable(num_vfs)?;
    ///     // Complete any additional setup before returning the guard.
    ///     Ok(enabled)
    /// }
    /// ```
    #[cfg(CONFIG_PCI_IOV)]
    fn sriov_enable<'a>(
        dev: &'a Device<device::Core<'_>>,
        data: Pin<&Self::Data<'a>>,
        token: SriovEnable<'a>,
    ) -> Result<SriovEnabled<'a>> {
        let _ = (dev, data, token);
        build_error!(crate::error::VTABLE_DEFAULT_ERROR)
    }

    /// Disables all VFs of a PF in response to a userspace request.
    ///
    /// Called when userspace writes zero to `sriov_numvfs`. Call [`SriovDisable::disable()`]
    /// while resources used by VF drivers are still available. Returning an error before calling
    /// it leaves the VFs enabled. Returning success requires that all VFs have been disabled.
    ///
    /// This callback is not invoked during PF unbind; [`VfRegistration`] owns that teardown.
    /// Implement this callback and [`Self::sriov_enable()`] together, or implement neither.
    ///
    /// # Examples
    ///
    /// ```
    /// # use kernel::{
    /// #     device::Core,
    /// #     pci,
    /// #     prelude::*, //
    /// # };
    /// # struct Data;
    /// fn sriov_disable<'a>(
    ///     _dev: &'a pci::Device<Core<'_>>,
    ///     _data: Pin<&Data>,
    ///     token: pci::SriovDisable<'a>,
    /// ) -> Result {
    ///     token.disable();
    ///     Ok(())
    /// }
    /// ```
    #[cfg(CONFIG_PCI_IOV)]
    fn sriov_disable<'a>(
        dev: &'a Device<device::Core<'_>>,
        data: Pin<&Self::Data<'a>>,
        token: SriovDisable<'a>,
    ) -> Result {
        let _ = (dev, data, token);
        build_error!(crate::error::VTABLE_DEFAULT_ERROR)
    }
}

/// The PCI device representation.
///
/// This structure represents the Rust abstraction for a C `struct pci_dev`. The implementation
/// abstracts the usage of an already existing C `struct pci_dev` within Rust code that we get
/// passed from the C side.
///
/// # Invariants
///
/// A [`Device`] instance represents a valid `struct pci_dev` created by the C portion of the
/// kernel.
#[repr(transparent)]
pub struct Device<Ctx: device::DeviceContext = device::Normal>(
    Opaque<bindings::pci_dev>,
    PhantomData<Ctx>,
);

impl<Ctx: device::DeviceContext> Device<Ctx> {
    #[inline]
    fn as_raw(&self) -> *mut bindings::pci_dev {
        self.0.get()
    }
}

impl Device {
    /// Returns `true` if this device is a Virtual Function (VF).
    #[inline]
    pub fn is_virtfn(&self) -> bool {
        // SAFETY: `self.as_raw` is a valid pointer to a `struct pci_dev`.
        unsafe { (*self.as_raw()).is_virtfn() != 0 }
    }

    /// Returns the PCI vendor ID as [`Vendor`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use kernel::{device::Core, pci::{self, Vendor}, prelude::*};
    /// fn log_device_info(pdev: &pci::Device<Core<'_>>) -> Result {
    ///     // Get an instance of `Vendor`.
    ///     let vendor = pdev.vendor_id();
    ///     dev_info!(
    ///         pdev,
    ///         "Device: Vendor={}, Device=0x{:x}\n",
    ///         vendor,
    ///         pdev.device_id()
    ///     );
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn vendor_id(&self) -> Vendor {
        // SAFETY: `self.as_raw` is a valid pointer to a `struct pci_dev`.
        let vendor_id = unsafe { (*self.as_raw()).vendor };
        Vendor::from_raw(vendor_id)
    }

    /// Returns the PCI device ID.
    #[inline]
    pub fn device_id(&self) -> u16 {
        // SAFETY: By its type invariant `self.as_raw` is always a valid pointer to a
        // `struct pci_dev`.
        unsafe { (*self.as_raw()).device }
    }

    /// Returns the PCI revision ID.
    #[inline]
    pub fn revision_id(&self) -> u8 {
        // SAFETY: By its type invariant `self.as_raw` is always a valid pointer to a
        // `struct pci_dev`.
        unsafe { (*self.as_raw()).revision }
    }

    /// Returns the PCI bus device/function.
    #[inline]
    pub fn dev_id(&self) -> u16 {
        // SAFETY: By its type invariant `self.as_raw` is always a valid pointer to a
        // `struct pci_dev`.
        unsafe { bindings::pci_dev_id(self.as_raw()) }
    }

    /// Returns the PCI domain number of the bus that this device is on.
    #[inline]
    pub fn domain_nr(&self) -> u32 {
        // SAFETY: By its type invariant `self.as_raw` is always a valid pointer to a
        // `struct pci_dev`.
        let domain_nr = unsafe { bindings::pci_domain_nr(self.as_raw()) };
        debug_assert!(domain_nr >= 0);

        // CAST: `pci_domain_nr` returns a non-negative domain number in an `int`.
        domain_nr as u32
    }

    /// Returns the PCI subsystem vendor ID.
    #[inline]
    pub fn subsystem_vendor_id(&self) -> u16 {
        // SAFETY: By its type invariant `self.as_raw` is always a valid pointer to a
        // `struct pci_dev`.
        unsafe { (*self.as_raw()).subsystem_vendor }
    }

    /// Returns the PCI subsystem device ID.
    #[inline]
    pub fn subsystem_device_id(&self) -> u16 {
        // SAFETY: By its type invariant `self.as_raw` is always a valid pointer to a
        // `struct pci_dev`.
        unsafe { (*self.as_raw()).subsystem_device }
    }

    /// Returns the start of the given PCI BAR resource.
    pub fn resource_start(&self, bar: u32) -> Result<bindings::resource_size_t> {
        if !Bar::index_is_valid(bar) {
            return Err(EINVAL);
        }

        // SAFETY:
        // - `bar` is a valid bar number, as guaranteed by the above call to `Bar::index_is_valid`,
        // - by its type invariant `self.as_raw` is always a valid pointer to a `struct pci_dev`.
        Ok(unsafe { bindings::pci_resource_start(self.as_raw(), bar.try_into()?) })
    }

    /// Returns the size of the given PCI BAR resource.
    pub fn resource_len(&self, bar: u32) -> Result<bindings::resource_size_t> {
        if !Bar::index_is_valid(bar) {
            return Err(EINVAL);
        }

        // SAFETY:
        // - `bar` is a valid bar number, as guaranteed by the above call to `Bar::index_is_valid`,
        // - by its type invariant `self.as_raw` is always a valid pointer to a `struct pci_dev`.
        Ok(unsafe { bindings::pci_resource_len(self.as_raw(), bar.try_into()?) })
    }

    /// Returns the resource flags (`IORESOURCE_*`) of the given PCI BAR.
    pub fn resource_flags(&self, bar: u32) -> Result<resource::Flags> {
        if !Bar::index_is_valid(bar) {
            return Err(EINVAL);
        }

        // SAFETY:
        // - `bar` is a valid bar number, as guaranteed by the above call to `Bar::index_is_valid`,
        // - by its type invariant `self.as_raw` is always a valid pointer to a `struct pci_dev`.
        let raw = unsafe { bindings::pci_resource_flags(self.as_raw(), bar.try_into()?) };
        Ok(resource::Flags::from_raw(raw))
    }

    /// Returns the PCI class as a `Class` struct.
    #[inline]
    pub fn pci_class(&self) -> Class {
        // SAFETY: `self.as_raw` is a valid pointer to a `struct pci_dev`.
        Class::from_raw(unsafe { (*self.as_raw()).class })
    }
}

impl<'a> Device<device::Core<'a>> {
    /// Returns the total number of VFs, or [`None`] if SR-IOV is not available.
    #[inline]
    pub fn sriov_get_totalvfs(&self) -> Option<NonZero<u16>> {
        // SAFETY: `self.as_raw()` is a valid pointer to a `struct pci_dev`.
        let total_vfs = unsafe { bindings::pci_sriov_get_totalvfs(self.as_raw()) };

        // CAST: The C function returns `unsigned int`, but the value originates
        // from TotalVFs/driver_max_VFs (which are defined as `u16`), so this cast
        // cannot truncate.
        NonZero::new(total_vfs as u16)
    }

    /// Enable memory resources for this device.
    pub fn enable_device_mem(&self) -> Result {
        // SAFETY: `self.as_raw` is guaranteed to be a pointer to a valid `struct pci_dev`.
        to_result(unsafe { bindings::pci_enable_device_mem(self.as_raw()) })
    }

    /// Enable bus-mastering for this device.
    #[inline]
    pub fn set_master(&self) {
        // SAFETY: `self.as_raw` is guaranteed to be a pointer to a valid `struct pci_dev`.
        unsafe { bindings::pci_set_master(self.as_raw()) };
    }
}

// SAFETY: `pci::Device` is a transparent wrapper of `struct pci_dev`.
// The offset is guaranteed to point to a valid device field inside `pci::Device`.
unsafe impl<Ctx: device::DeviceContext> device::AsBusDevice<Ctx> for Device<Ctx> {
    const OFFSET: usize = offset_of!(bindings::pci_dev, dev);
}

// SAFETY: `Device` is a transparent wrapper of a type that doesn't depend on `Device`'s generic
// argument.
kernel::impl_device_context_deref!(unsafe { Device });
kernel::impl_device_context_into_aref!(Device);

impl<'a> crate::dma::Device<'a> for Device<device::Core<'a>> {}

// SAFETY: Instances of `Device` are always reference-counted.
unsafe impl crate::sync::aref::AlwaysRefCounted for Device {
    #[inline]
    fn inc_ref(&self) {
        // SAFETY: The existence of a shared reference guarantees that the refcount is non-zero.
        unsafe { bindings::pci_dev_get(self.as_raw()) };
    }

    #[inline]
    unsafe fn dec_ref(obj: NonNull<Self>) {
        // SAFETY: The safety requirements guarantee that the refcount is non-zero.
        unsafe { bindings::pci_dev_put(obj.cast().as_ptr()) }
    }
}

impl<Ctx: device::DeviceContext> AsRef<device::Device<Ctx>> for Device<Ctx> {
    fn as_ref(&self) -> &device::Device<Ctx> {
        // SAFETY: By the type invariant of `Self`, `self.as_raw()` is a pointer to a valid
        // `struct pci_dev`.
        let dev = unsafe { addr_of_mut!((*self.as_raw()).dev) };

        // SAFETY: `dev` points to a valid `struct device`.
        unsafe { device::Device::from_raw(dev) }
    }
}

impl<Ctx: device::DeviceContext> TryFrom<&device::Device<Ctx>> for &Device<Ctx> {
    type Error = kernel::error::Error;

    fn try_from(dev: &device::Device<Ctx>) -> Result<Self, Self::Error> {
        // SAFETY: By the type invariant of `Device`, `dev.as_raw()` is a valid pointer to a
        // `struct device`.
        if !unsafe { bindings::dev_is_pci(dev.as_raw()) } {
            return Err(EINVAL);
        }

        // SAFETY: We've just verified that the bus type of `dev` equals `bindings::pci_bus_type`,
        // hence `dev` must be embedded in a valid `struct pci_dev` as guaranteed by the
        // corresponding C code.
        let pdev = unsafe { container_of!(dev.as_raw(), bindings::pci_dev, dev) };

        // SAFETY: `pdev` is a valid pointer to a `struct pci_dev`.
        Ok(unsafe { &*pdev.cast() })
    }
}

// SAFETY: A `Device` is always reference-counted and can be released from any thread.
unsafe impl Send for Device {}

// SAFETY: `Device` can be shared among threads because all methods of `Device`
// (i.e. `Device<Normal>) are thread safe.
unsafe impl Sync for Device {}

// SAFETY: Same as `Device<Normal>` -- the underlying `struct pci_dev` is the same;
// `Bound` is a zero-sized type-state marker that does not affect thread safety.
unsafe impl Sync for Device<device::Bound> {}
