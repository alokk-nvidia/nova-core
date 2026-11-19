// SPDX-License-Identifier: GPL-2.0

//! Virtual Function I/O (VFIO) devices and callback data.
//!
//! [`DeviceBase`] combines a C VFIO device structure with the data its callbacks use.
//! [`Device`] and the bus-specific devices, such as `pci::Device`, are aliases of it.
//!
//! C header: [`include/linux/vfio.h`](srctree/include/linux/vfio.h)

use crate::{
    bindings,
    prelude::*,
    sync::aref::AlwaysRefCounted,
    types::{
        NotThreadSafe,
        Opaque, //
    },
    uaccess::{
        UserPtr,
        UserSlice, //
    }, //
};
use core::{
    cell::UnsafeCell,
    marker::PhantomData,
    ptr::NonNull, //
};

#[cfg(CONFIG_VFIO_PCI_CORE)]
pub mod pci;

/// Reset the VFIO device.
pub const DEVICE_RESET: u32 = bindings::VFIO_DEVICE_RESET;

/// The context in which a VFIO device reference is valid.
///
/// Callback views can only be borrowed from the corresponding VFIO trampoline.
/// They cannot be refcounted or shared across threads.
pub trait DeviceContext: private::Sealed {}

/// A callback context in which VFIO keeps the device open.
///
/// Registration data and open data are available throughout such a callback.
pub trait IoContext: DeviceContext {}

/// An ordinary device reference, without callback-specific operations.
pub struct Normal;

/// The device is undergoing its exclusive first-open callback.
pub struct Open;

/// The device is borrowed for a VFIO ioctl callback on the current thread.
pub struct Ioctl;

/// The device is borrowed for a VFIO read callback on the current thread.
pub struct Read;

/// The device is borrowed for a VFIO write callback on the current thread.
pub struct Write;

/// The device is borrowed for a VFIO mmap callback on the current thread.
pub struct Mmap;

/// The device is borrowed for a VFIO region-info callback on the current thread.
pub struct GetRegionInfo;

mod private {
    pub trait Sealed {}

    impl Sealed for super::Normal {}
    impl Sealed for super::Open {}
    impl Sealed for super::Write {}
    impl Sealed for super::Mmap {}
    impl Sealed for super::Ioctl {}
    impl Sealed for super::Read {}
    impl Sealed for super::GetRegionInfo {}
}

impl DeviceContext for Normal {}
impl DeviceContext for Open {}
impl DeviceContext for Write {}
impl DeviceContext for Mmap {}
impl DeviceContext for Ioctl {}
impl DeviceContext for Read {}
impl DeviceContext for GetRegionInfo {}

impl IoContext for Write {}
impl IoContext for Mmap {}
impl IoContext for Ioctl {}
impl IoContext for Read {}
impl IoContext for GetRegionInfo {}

/// The data the driver of a [`Device`] shares with its callbacks.
///
/// Bus-specific devices take the data types from their own trait, such as `pci::Operations`.
pub trait Operations: Sized + Send + Sync + 'static {
    /// Data owned by the registration and passed to VFIO callbacks.
    ///
    /// It can borrow resources of the driver's binding for `'a`. Per-device state lives here.
    type RegistrationData<'a>: Send + Sync + 'a;

    /// Data shared by callbacks from the first successful open until the last close.
    ///
    /// It can borrow the registration data, of lifetime `'data`, for `'a`.
    type OpenData<'a, 'data: 'a>: Send + Sync + 'a;
}

/// Maps a C device structure to its embedded VFIO device.
///
/// # Safety
///
/// Implementations must return the address of the `vfio_device` at the beginning of `Self`,
/// preserving the input pointer's provenance.
pub unsafe trait RawDevice {
    /// Returns the `vfio_device` embedded in `raw`.
    ///
    /// # Safety
    ///
    /// `raw` must point to an initialized, live `Self`.
    unsafe fn raw_vfio_device(raw: *mut Self) -> *mut bindings::vfio_device;
}

// SAFETY: A `vfio_device` is its own VFIO device.
unsafe impl RawDevice for bindings::vfio_device {
    unsafe fn raw_vfio_device(raw: *mut Self) -> *mut bindings::vfio_device {
        raw
    }
}

/// Callback data stored after the C device structure.
///
/// `R` is the type of the registration data and `O` the type of the open data, both written
/// with the placeholder lifetime `'static`, because the device has no lifetime to name.
///
/// # Invariants
///
/// `reg_data` is `None` until a registration publishes its data, and is not changed again. The
/// data is pinned and remains valid while the device is registered. After unregistration or a
/// failed registration attempt, the pointer may dangle; its pointee must never again be
/// accessed through this device.
/// `open_data` is null outside a successful first open, otherwise it owns a pinned allocation
/// until the last close. The bus-specific layer drains callbacks before either is released.
struct CallbackData<R, O> {
    reg_data: UnsafeCell<Option<NonNull<R>>>,
    open_data: UnsafeCell<*mut O>,
}

/// A C VFIO device structure `Raw`, followed by the callback data of the driver `T`.
///
/// `raw` is the first field, so the `vfio_device` at the beginning of `Raw` is also at the
/// beginning of `Self`, as `vfio_alloc_device()` requires.
///
/// `R` and `O` are the types of the registration data and the open data of `T`, written with
/// the placeholder lifetime `'static`. Use the aliases [`Device`] and `pci::Device`, which
/// take both from the operations trait of `T`.
///
/// # Invariants
///
/// - `R` and `O` are the registration data and open data types of `T` with all lifetimes
///   replaced by `'static`. The data stored is only valid as described for `CallbackData`.
/// - `raw` is initialized and remains alive while borrowed. The reference count of its embedded
///   `vfio_device` owns the allocation.
/// - The device is registered with VFIO only after its registration data has been published.
/// - Callback contexts are borrowed only for the corresponding callback and thread; only
///   [`Normal`] references can be reference-counted.
#[repr(C)]
pub struct DeviceBase<Raw, R, O, T, Ctx: DeviceContext = Normal> {
    raw: Opaque<Raw>,
    data: CallbackData<R, O>,
    _context: PhantomData<(T, Ctx, NotThreadSafe)>,
}

/// A VFIO device, wrapping `struct vfio_device`.
pub type Device<T, Ctx = Normal> = DeviceBase<
    bindings::vfio_device,
    <T as Operations>::RegistrationData<'static>,
    <T as Operations>::OpenData<'static, 'static>,
    T,
    Ctx,
>;

#[cfg_attr(not(CONFIG_VFIO_PCI_CORE), expect(dead_code))]
impl<Raw: RawDevice, R, O, T, Ctx: DeviceContext> DeviceBase<Raw, R, O, T, Ctx> {
    /// Initializes the Rust fields of a newly allocated device.
    ///
    /// # Safety
    ///
    /// `this` must point to an allocation with the size and alignment of `Self`, whose `raw`
    /// has been initialized. Nothing else may access the allocation during this call.
    unsafe fn init(this: *mut Self) {
        // SAFETY: The caller provides exclusive access to a suitable allocation.
        unsafe {
            (&raw mut (*this).data).write(CallbackData {
                reg_data: UnsafeCell::new(None),
                open_data: UnsafeCell::new(core::ptr::null_mut()),
            });
            (&raw mut (*this)._context).write(PhantomData);
        }
    }

    /// Recovers `&Self` from the `vfio_device` VFIO passes to a callback.
    ///
    /// # Safety
    ///
    /// `vdev` must point to the `vfio_device` at the beginning of a `Self` that remains alive
    /// for `'a`. The caller must provide the guarantees of `Ctx` throughout the borrow.
    unsafe fn from_raw_vfio<'a>(vdev: *mut bindings::vfio_device) -> &'a Self {
        // SAFETY: `Raw` begins with its `vfio_device` and is the first field of `Self`. The
        // caller guarantees validity and the context.
        unsafe { &*vdev.cast() }
    }

    fn as_raw(&self) -> *mut Raw {
        self.raw.get()
    }

    fn as_raw_vfio(&self) -> *mut bindings::vfio_device {
        // SAFETY: By the type invariants, `raw` is initialized and alive.
        unsafe { Raw::raw_vfio_device(self.as_raw()) }
    }
}

#[cfg_attr(not(CONFIG_VFIO_PCI_CORE), expect(dead_code))]
impl<Raw: RawDevice, R, O, T> DeviceBase<Raw, R, O, T> {
    /// Publishes the registration data passed to callbacks.
    ///
    /// Returns [`EBUSY`] if registration data has been published before, so that a device can
    /// be registered at most once.
    ///
    /// # Safety
    ///
    /// The caller must serialize calls for this device. `data` must point to pinned data that
    /// remains valid until the device has been unregistered or, if registration fails, is not
    /// accessed through this device.
    unsafe fn publish_registration_data(&self, data: NonNull<R>) -> Result {
        // SAFETY: The caller serializes publication, and callbacks only read the slot.
        if unsafe { (*self.data.reg_data.get()).is_some() } {
            return Err(EBUSY);
        }
        // SAFETY: The slot is unpublished, so the device has not been registered and no
        // callback can read the slot.
        // INVARIANT: The slot is written once; the caller keeps `data` valid.
        unsafe { *self.data.reg_data.get() = Some(data) };
        Ok(())
    }

    /// Destroys the open data.
    ///
    /// # Safety
    ///
    /// Call once after a successful [`DeviceBase::open`], with all I/O callbacks drained and
    /// the next first open excluded. Registration data and the resources the open data borrows
    /// must remain alive.
    unsafe fn close(&self) {
        // SAFETY: The last close exclusively takes the allocation published by `open()`.
        let raw =
            unsafe { core::mem::replace(&mut *self.data.open_data.get(), core::ptr::null_mut()) };
        // SAFETY: The allocation came from `KBox::into_raw()` and is destroyed without moving
        // it. Lifetimes do not affect its layout.
        unsafe { drop(KBox::from_raw(raw)) };
    }
}

#[cfg_attr(not(CONFIG_VFIO_PCI_CORE), expect(dead_code))]
impl<Raw: RawDevice, R, O, T> DeviceBase<Raw, R, O, T, Open> {
    /// Creates the open data from the registration data.
    ///
    /// The [`Open`] context guarantees an exclusive first open of a registered device, whose
    /// registration data remains valid until the last close.
    ///
    /// # Safety
    ///
    /// - No open data may exist.
    /// - `OD` must be `O`, except for lifetimes.
    /// - `R` names the registration data with the placeholder lifetime `'static`, and the
    ///   caller chooses the lifetimes of `OD`; this function cannot tie either to the time the
    ///   data is valid. `init` must therefore use the registration data parametrically, as a
    ///   function generic over its lifetimes does: nothing borrowed from it may escape, other
    ///   than into the returned open data.
    /// - `init` must not capture references that are only valid for the callback and store
    ///   them in the open data, which outlives the callback.
    /// - The open data must be destroyed with [`DeviceBase::close`] at the last close.
    unsafe fn open<'a, OD, I>(&'a self, init: impl FnOnce(&'a R) -> I) -> Result
    where
        I: PinInit<OD, Error>,
    {
        // SAFETY: By the type invariants, an `Open` reference only exists for a registered
        // device, which has published registration data that nothing changes anymore. The
        // data remains valid until the last close.
        let reg_data = unsafe { (*self.data.reg_data.get()).unwrap_unchecked().as_ref() };

        // Allocate before calling the driver so allocation failure cannot follow open.
        let data = KBox::<OD>::new_uninit(GFP_KERNEL)?;
        let data = data.write_pin_init(init(reg_data))?;

        // SAFETY: Only the owning pointer moves; `close()` drops the allocation in place.
        let raw = KBox::into_raw(unsafe { Pin::into_inner_unchecked(data) });
        // SAFETY: The first open is exclusive, and no I/O callback runs before it returns.
        // The caller guarantees that `OD` and `O` only differ in lifetimes, which do not
        // affect the layout.
        // INVARIANT: The allocation is pinned and owned by the slot until `close()`.
        unsafe { *self.data.open_data.get() = raw.cast::<O>() };
        Ok(())
    }
}

#[cfg_attr(not(CONFIG_VFIO_PCI_CORE), expect(dead_code))]
impl<Raw: RawDevice, R, O, T, Ctx: IoContext> DeviceBase<Raw, R, O, T, Ctx> {
    /// Returns the registration data and the open data, for an I/O callback.
    ///
    /// # Safety
    ///
    /// `R` and `O` name the data with the placeholder lifetime `'static`, which is not the time
    /// the data is valid. The caller must use both parametrically, as a function generic over
    /// their lifetimes does: nothing borrowed from them may outlive the callback, and nothing
    /// borrowed for the callback may be stored in them.
    unsafe fn callback_data(&self) -> (&R, Pin<&O>) {
        // SAFETY: By the type invariants, an `IoContext` reference only exists during an I/O
        // callback on an open, registered device. VFIO excludes the last close and
        // unregistration for its duration, so by the type invariants of `CallbackData` both
        // slots hold valid data that nothing changes, and the open data is pinned.
        unsafe {
            let reg_data = (*self.data.reg_data.get()).unwrap_unchecked();
            let open_data = *self.data.open_data.get();
            (reg_data.as_ref(), Pin::new_unchecked(&*open_data))
        }
    }
}

// SAFETY: VFIO uses the embedded class device's refcount to own the full allocation.
unsafe impl<Raw: RawDevice, R, O, T> AlwaysRefCounted for DeviceBase<Raw, R, O, T> {
    fn inc_ref(&self) {
        // SAFETY: A shared reference keeps the initialized VFIO device alive.
        unsafe { bindings::get_device(&raw mut (*self.as_raw_vfio()).device) };
    }

    unsafe fn dec_ref(obj: NonNull<Self>) {
        // SAFETY: The caller owns a reference, so `obj` points to a live device.
        let raw = Opaque::cast_into(unsafe { &raw const (*obj.as_ptr()).raw });
        // SAFETY: `raw` is initialized and alive, as above.
        let vdev = unsafe { Raw::raw_vfio_device(raw) };
        // SAFETY: The caller owns the VFIO reference being released.
        unsafe { bindings::put_device(&raw mut (*vdev).device) };
    }
}

// SAFETY: VFIO serializes open/close transitions, and callback data is `Send + Sync`.
unsafe impl<Raw: RawDevice, R: Send + Sync, O: Send + Sync, T> Send for DeviceBase<Raw, R, O, T> {}

// SAFETY: Shared access only changes the embedded class device's reference count. I/O callbacks
// share `Sync` data; VFIO excludes them from mutations of `open_data` during first open and
// last close, and registration data follows the registration lifetime. Mutable C state is
// managed by the VFIO core.
unsafe impl<Raw: RawDevice, R: Send + Sync, O: Send + Sync, T> Sync for DeviceBase<Raw, R, O, T> {}

/// Capability buffer supplied by VFIO for a region-info callback.
pub struct InfoCap<'a> {
    #[cfg_attr(not(CONFIG_VFIO_PCI_CORE), expect(dead_code))]
    raw: &'a mut bindings::vfio_info_cap,
}

/// Opaque wrapper around a `char __user *` buffer from a VFIO read or write callback.
///
/// Provides bounds-checked copying of kernel data into the user-space buffer.
///
/// This type cannot be constructed by driver code — it is only created by the
/// callback trampoline.
pub struct UserBuf {
    ptr: UserPtr,
    count: usize,
}

impl UserBuf {
    /// Returns the buffer length in bytes.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` when the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Limit this callback's transfer to at most `count` bytes.
    pub fn truncate(&mut self, count: usize) {
        self.count = self.count.min(count);
    }

    /// Overwrite the part of `data` that overlaps a completed region read.
    ///
    /// `read_offset` is the starting region offset and `read_len` is the number of
    /// bytes returned by the read. `data_offset` locates `data` within the same region.
    pub fn write_overlapping(
        &self,
        read_offset: u64,
        read_len: usize,
        data_offset: u64,
        data: &[u8],
    ) -> Result {
        if read_len > self.len() {
            return Err(EINVAL);
        }
        let read_end = read_offset.checked_add(read_len as u64).ok_or(EOVERFLOW)?;
        let data_end = data_offset
            .checked_add(data.len() as u64)
            .ok_or(EOVERFLOW)?;
        let start = read_offset.max(data_offset);
        let end = read_end.min(data_end);
        if start >= end {
            return Ok(());
        }

        // These differences are bounded by `read_len` and `data.len()`.
        let buf_offset = (start - read_offset) as usize;
        let data_start = (start - data_offset) as usize;
        let data_end = (end - data_offset) as usize;
        self.write_at(buf_offset, &data[data_start..data_end])
    }

    /// Copy `data` to the user buffer at byte offset `offset`.
    ///
    /// Returns [`EFAULT`] if the copy fails, [`EINVAL`] if the write would
    /// exceed the buffer bounds.
    fn write_at(&self, offset: usize, data: &[u8]) -> Result {
        let end = offset.checked_add(data.len()).ok_or(EOVERFLOW)?;
        if end > self.count {
            return Err(EINVAL);
        }
        let dest = self.ptr.wrapping_byte_add(offset);
        let mut writer = UserSlice::new(dest, data.len()).writer();
        writer.write_slice(data)
    }
}
