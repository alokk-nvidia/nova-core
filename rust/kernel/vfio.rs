// SPDX-License-Identifier: GPL-2.0

//! Virtual Function I/O (VFIO) devices and callback data.
//!
//! Bus-specific device types borrow their embedded VFIO device through [`AsRef`].
//! C header: [`include/linux/vfio.h`](srctree/include/linux/vfio.h)

use crate::{
    bindings,
    prelude::*,
    sync::aref::AlwaysRefCounted,
    types::{
        CovariantForLt,
        ForLt,
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

/// A VFIO device, independent of its bus-specific wrapper.
///
/// # Invariants
///
/// The underlying `vfio_device` is initialized and remains alive while borrowed.
/// Callback contexts are borrowed only for the corresponding callback and thread;
/// only [`Normal`] references can be reference-counted.
#[repr(transparent)]
pub struct Device<Ctx: DeviceContext = Normal> {
    raw: Opaque<bindings::vfio_device>,
    _context: PhantomData<(Ctx, NotThreadSafe)>,
}

impl<Ctx: DeviceContext> Device<Ctx> {
    /// # Safety
    ///
    /// `raw` must point to an initialized `vfio_device` that remains alive for
    /// `'a`. The caller must provide the guarantees of `Ctx` throughout the borrow.
    #[cfg_attr(not(CONFIG_VFIO_PCI_CORE), expect(dead_code))]
    unsafe fn from_raw<'a>(raw: *mut bindings::vfio_device) -> &'a Self {
        // SAFETY: Self is a transparent wrapper; the caller guarantees validity and context.
        unsafe { &*raw.cast() }
    }

    fn as_raw(&self) -> *mut bindings::vfio_device {
        self.raw.get()
    }
}

// SAFETY: VFIO uses the embedded class device's refcount to own the full allocation.
unsafe impl AlwaysRefCounted for Device {
    fn inc_ref(&self) {
        // SAFETY: A shared reference keeps the initialized VFIO device alive.
        unsafe { bindings::get_device(&raw mut (*self.as_raw()).device) };
    }

    unsafe fn dec_ref(obj: NonNull<Self>) {
        // SAFETY: The caller owns the VFIO reference being released.
        unsafe { bindings::put_device(&raw mut (*obj.as_ref().as_raw()).device) };
    }
}

// SAFETY: Ordinary VFIO references expose no unsynchronized callback operations.
unsafe impl Send for Device {}
// SAFETY: Shared access only changes the embedded class device's reference count.
unsafe impl Sync for Device {}

/// Callback data stored after the bus-specific C device structure.
///
/// # Invariants
///
/// `reg_data` is dangling when unpublished; otherwise it borrows pinned data kept
/// alive until callback access has ended. `open_data` is null outside a successful
/// first open, otherwise owns a pinned allocation until the last close.
/// The bus-specific layer drains callbacks before releasing either allocation.
struct CallbackData<R: CovariantForLt, O: ForLt> {
    reg_data: UnsafeCell<NonNull<R::Of<'static>>>,
    open_data: UnsafeCell<*mut O::Of<'static>>,
}

#[cfg_attr(not(CONFIG_VFIO_PCI_CORE), expect(dead_code))]
impl<R: CovariantForLt, O: ForLt> CallbackData<R, O> {
    fn new() -> Self {
        Self {
            reg_data: UnsafeCell::new(NonNull::dangling()),
            open_data: UnsafeCell::new(core::ptr::null_mut()),
        }
    }

    /// Access registration data without allowing its erased lifetime to escape.
    ///
    /// # Safety
    ///
    /// Registration data must be published and remain valid through this call.
    /// The caller must exclude concurrent changes to `reg_data`.
    unsafe fn registration_data_with<V>(&self, f: impl for<'a> FnOnce(&'a R::Of<'a>) -> V) -> V {
        // SAFETY: The caller keeps registration data alive and excludes mutation.
        // Covariance permits shortening its erased lifetime to this borrow.
        let reg_data = unsafe { (*self.reg_data.get()).cast::<R::Of<'_>>().as_ref() };
        f(reg_data)
    }

    /// # Safety
    ///
    /// The caller must be an I/O callback while VFIO keeps the device open.
    /// Registration and open data must remain valid, with their pointer slots
    /// unchanged throughout the call.
    unsafe fn callback_data_with<V>(
        &self,
        f: impl for<'borrow, 'data> FnOnce(&'borrow R::Of<'data>, Pin<&'borrow O::Of<'data>>) -> V,
    ) -> V {
        // SAFETY: The caller excludes open/close and unregister while borrowing the data.
        unsafe {
            self.registration_data_with(|reg_data| {
                let ptr = (*self.open_data.get()).cast::<O::Of<'_>>();
                // The data lifetime stays independent of the callback borrow,
                // so borrowed OpenData fields cannot be stored back into OpenData.
                f(reg_data, Pin::new_unchecked(&*ptr))
            })
        }
    }

    /// # Safety
    ///
    /// Call only during exclusive first open, with no existing open data.
    /// All resources borrowed for `'a` must remain valid until `close()` returns.
    unsafe fn open<'a, I: PinInit<O::Of<'a>, Error>>(&self, init: impl FnOnce() -> I) -> Result {
        // Allocate before calling the driver so allocation failure cannot follow open.
        let data = KBox::<O::Of<'a>>::new_uninit(GFP_KERNEL)?;
        let data = data.write_pin_init(init())?;

        // SAFETY: Only the owning pointer moves; close drops the allocation in place.
        let raw = KBox::into_raw(unsafe { Pin::into_inner_unchecked(data) });
        // SAFETY: First open is exclusive; the caller keeps borrowed resources alive
        // through close. Lifetimes do not affect the allocation's layout.
        unsafe { *self.open_data.get() = raw.cast::<O::Of<'static>>() };
        Ok(())
    }

    /// # Safety
    ///
    /// Call once after a successful open, with all callbacks drained and the next
    /// open excluded. Registration data and borrowed resources must remain alive.
    unsafe fn close(&self) {
        // SAFETY: Last close exclusively takes the allocation published by open.
        let raw = unsafe { core::mem::replace(&mut *self.open_data.get(), core::ptr::null_mut()) };
        // SAFETY: The allocation came from KBox::into_raw and is destroyed without moving it.
        unsafe { drop(KBox::from_raw(raw)) };
    }
}

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
