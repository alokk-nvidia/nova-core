// SPDX-License-Identifier: GPL-2.0

//! VFIO PCI variant driver abstractions.
//!
//! Provides Rust abstractions for writing VFIO PCI variant drivers (also known
//! as "VFIO PCI core" drivers). A [`Registration`] struct owns the VFIO device
//! and its associated registration data, tying resource lifetimes to the PCI
//! driver's binding scope.
//!
//! C header: [`include/linux/vfio_pci_core.h`](srctree/include/linux/vfio_pci_core.h)

use super::{
    InfoCap,
    UserBuf, //
};

use crate::{
    alloc::allocator::Kmalloc,
    bindings,
    device,
    error::{
        from_err_ptr,
        to_result, //
    },
    mm::virt::VmaRef,
    pci,
    prelude::*,
    sync::aref::{
        ARef,
        AlwaysRefCounted, //
    },
    types::{
        CovariantForLt,
        ForLt,
        NotThreadSafe,
        Opaque, //
    },
    uaccess::UserPtr, //
};
use core::{
    alloc::Layout,
    cell::UnsafeCell,
    marker::PhantomData,
    ptr::NonNull, //
};

/// Index of the PCI configuration-space region.
pub const CONFIG_REGION_INDEX: u32 = bindings::VFIO_PCI_CONFIG_REGION_INDEX;

/// The context in which a VFIO PCI device reference is valid.
///
/// Callback views can only be borrowed from the corresponding VFIO trampoline.
/// They cannot be refcounted or shared across threads.
pub trait DeviceContext: private::Sealed {}

/// An ordinary device reference, without callback-specific operations.
pub struct Normal;

/// The device is enabled but has not yet completed its first-open callback.
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

/// The trait for VFIO PCI variant driver implementations.
///
/// Unoverridden callbacks delegate to the corresponding PCI core operation.
///
/// # Lifetime of `RegistrationData`
///
/// `RegistrationData` describes a covariant type family borrowing from the PCI
/// binding scope. [`Registration`] owns the data, and each callback borrows it.
/// Covariance permits shortening the binding lifetime to the callback's borrow
/// without allowing callback-local references to be stored in registration data.
pub trait Operations: Sized + Send + Sync + 'static {
    /// The name of the VFIO driver.
    const NAME: &'static CStr;

    /// Data owned by the [`Registration`] and passed to VFIO callbacks.
    ///
    /// Per-device state (e.g. the GFID, cached type info) lives here.
    type RegistrationData: for<'a> CovariantForLt<Of<'a>: Send + Sync> + 'static;

    /// Data shared by callbacks from the first successful open until the last close.
    /// May borrow registration data; dropped before PCI core close on the last close.
    type OpenData<'a>: Send + Sync + 'a;

    /// Called when the first file descriptor is opened for this device.
    ///
    /// `vfio_pci_core_enable()` has already succeeded; if this returns an
    /// error, `vfio_pci_core_disable()` is called automatically.
    fn open_device<'a>(
        dev: &'a Device<Self, Open>,
        reg_data: &'a <Self::RegistrationData as ForLt>::Of<'a>,
    ) -> impl PinInit<Self::OpenData<'a>, Error> + 'a;

    /// Handle a VFIO ioctl.
    ///
    /// The default `vfio_pci_core_ioctl()` is available via
    /// [`Device::core_ioctl()`] for delegation.
    fn ioctl<'a>(
        dev: &Device<Self, Ioctl>,
        reg_data: &<Self::RegistrationData as ForLt>::Of<'a>,
        open_data: Pin<&Self::OpenData<'a>>,
        cmd: u32,
        arg: usize,
    ) -> Result<isize>;

    /// Read from the device.
    ///
    /// `buf` is a user-space buffer provided by the VFIO core. Use
    /// [`Device::core_read()`] to delegate the default read, and
    /// [`UserBuf::write_overlapping()`] to override specific bytes afterwards.
    fn read<'a>(
        dev: &Device<Self, Read>,
        reg_data: &<Self::RegistrationData as ForLt>::Of<'a>,
        open_data: Pin<&Self::OpenData<'a>>,
        buf: &mut UserBuf,
        ppos: &mut Position<'_>,
    ) -> Result<isize>;

    /// Write to the device, delegating to [`Device::core_write()`] if appropriate.
    fn write<'a>(
        dev: &Device<Self, Write>,
        _reg_data: &<Self::RegistrationData as ForLt>::Of<'a>,
        _open_data: Pin<&Self::OpenData<'a>>,
        buf: &mut UserBuf,
        ppos: &mut Position<'_>,
    ) -> Result<isize> {
        dev.core_write(buf, ppos)
    }

    /// Map a device region, delegating to [`Device::core_mmap()`] if appropriate.
    fn mmap<'a>(
        dev: &Device<Self, Mmap>,
        _reg_data: &<Self::RegistrationData as ForLt>::Of<'a>,
        _open_data: Pin<&Self::OpenData<'a>>,
        mapping: &mut Mapping<'_>,
    ) -> Result {
        dev.core_mmap(mapping)
    }

    /// Prepare for PCI reset, including resets initiated during enable or close.
    fn reset_prepare(_reg_data: &<Self::RegistrationData as ForLt>::Of<'_>) {}

    /// Finish PCI reset; errors are reported through the VFIO error interrupt.
    fn reset_done(_reg_data: &<Self::RegistrationData as ForLt>::Of<'_>) -> Result {
        Ok(())
    }

    /// Fill in region info capabilities.
    ///
    /// The default `vfio_pci_ioctl_get_region_info()` is available via
    /// [`Device::core_get_region_info()`] for delegation.
    fn get_region_info<'a>(
        dev: &Device<Self, GetRegionInfo>,
        reg_data: &<Self::RegistrationData as ForLt>::Of<'a>,
        open_data: Pin<&Self::OpenData<'a>>,
        info: &mut bindings::vfio_region_info,
        caps: &mut InfoCap<'_>,
    ) -> Result;
}

/// A VFIO PCI file position, encoding a region index and an offset within that region.
pub struct Position<'a> {
    raw: &'a mut i64,
}

impl Position<'_> {
    /// Commit a temporary position only after the operation succeeds.
    ///
    /// This preserves the original position if a read succeeds but a subsequent
    /// user copy fails.
    pub fn with_temporary<R>(
        &mut self,
        operation: impl FnOnce(&mut Position<'_>) -> Result<R>,
    ) -> Result<R> {
        let mut next = *self.raw;
        let result = operation(&mut Position { raw: &mut next })?;
        *self.raw = next;
        Ok(result)
    }

    /// Returns the PCI region index.
    pub fn region_index(&self) -> u32 {
        ((*self.raw as u64) >> bindings::VFIO_PCI_OFFSET_SHIFT) as u32
    }

    /// Returns the byte offset within the current region.
    pub fn region_offset(&self) -> u64 {
        (*self.raw as u64) & ((1u64 << bindings::VFIO_PCI_OFFSET_SHIFT) - 1)
    }
}

/// Mapping metadata supplied by VFIO while the current thread holds the mmap lock.
///
/// The underlying VMA can only be passed back to the core mmap operation.
pub struct Mapping<'a> {
    vma: &'a VmaRef,
    _not_thread_safe: NotThreadSafe,
}

impl Mapping<'_> {
    fn page_offset(&self) -> u64 {
        // SAFETY: The mmap callback holds the mmap lock for this VMA.
        unsafe { (*self.vma.as_ptr()).vm_pgoff as u64 }
    }

    /// Returns the PCI region index encoded in the mapping offset.
    pub fn region_index(&self) -> u32 {
        (self.page_offset() >> (bindings::VFIO_PCI_OFFSET_SHIFT as usize - crate::page::PAGE_SHIFT))
            as u32
    }

    /// Returns the byte offset within the PCI region.
    fn region_offset(&self) -> u64 {
        let mask = ((1u64 << bindings::VFIO_PCI_OFFSET_SHIFT) - 1) >> crate::page::PAGE_SHIFT;
        (self.page_offset() & mask) << crate::page::PAGE_SHIFT
    }

    /// Returns the requested mapping length in bytes.
    fn size(&self) -> Result<u64> {
        self.vma
            .end()
            .checked_sub(self.vma.start())
            .map(|size| size as u64)
            .ok_or(EOVERFLOW)
    }

    /// Returns the exclusive end of the requested mapping within the PCI region.
    pub fn region_end(&self) -> Result<u64> {
        self.region_offset()
            .checked_add(self.size()?)
            .ok_or(EOVERFLOW)
    }
}

/// A VFIO PCI variant device, wrapping `struct vfio_pci_core_device`.
///
/// The layout is `#[repr(C)]` with `core_device` first, so the
/// `vfio_device` embedded at offset 0 of `vfio_pci_core_device` is also at
/// offset 0 of `Self`, matching the `vfio_alloc_device` requirement.
/// Callback contexts expose only their matching core helper and cannot be
/// refcounted or shared across threads.
///
/// # Invariants
///
/// - Callback contexts are only borrowed for the corresponding callback and thread.
/// - `core_device` was initialized by `vfio_pci_core_init_dev()`.
/// - The VFIO device refcount owns the allocation lifetime.
/// - `reg_data` is dangling without an owner or points to a valid
///   `<T::RegistrationData as ForLt>::Of<'_>` owned by the enclosing [`Registration`].
/// - `open_data` is null outside a successful open, otherwise it owns a pinned
///   `KBox<T::OpenData<'_>>` until last close. Only open/close mutate it; VFIO
///   prevents I/O callbacks from racing those transitions. The data is dropped
///   before PCI core close and before registration data is freed.
#[repr(C)]
pub struct Device<T: Operations, Ctx: DeviceContext = Normal> {
    core_device: Opaque<bindings::vfio_pci_core_device>,
    reg_data: UnsafeCell<NonNull<<T::RegistrationData as ForLt>::Of<'static>>>,
    open_data: UnsafeCell<*mut T::OpenData<'static>>,
    _context: PhantomData<(Ctx, NotThreadSafe)>,
}

impl<T: Operations> Device<T> {
    /// Allocate a VFIO PCI device for subsequent registration.
    pub fn new(pdev: &pci::Device<device::Core<'_>>) -> Result<ARef<Self>> {
        Self::allocate(pdev, false)
    }

    /// Allocate a device whose physical BARs may also be exported as DMA buffers.
    pub fn new_passthrough(pdev: &pci::Device<device::Core<'_>>) -> Result<ARef<Self>> {
        Self::allocate(pdev, true)
    }

    fn allocate(pdev: &pci::Device<device::Core<'_>>, passthrough: bool) -> Result<ARef<Self>> {
        const_assert!(core::mem::offset_of!(Self, core_device) == 0);
        let size = Kmalloc::aligned_layout(Layout::new::<Self>()).size();

        // SAFETY: The bound PCI device and static ops are valid. The allocation
        // includes the full Rust wrapper with kmalloc-compatible size and alignment.
        let raw = from_err_ptr(unsafe {
            bindings::_vfio_alloc_device(size, pdev.as_ref().as_raw(), &Self::OPS)
        })?;
        let this = NonNull::new(raw.cast::<Self>()).ok_or(ENOMEM)?;

        // SAFETY: Initialise the Rust field in the newly allocated device.
        unsafe {
            (&raw mut (*this.as_ptr()).reg_data).write(UnsafeCell::new(NonNull::dangling()));
            (&raw mut (*this.as_ptr()).open_data).write(UnsafeCell::new(core::ptr::null_mut()));
            (&raw mut (*this.as_ptr())._context).write(PhantomData);
        }

        if passthrough {
            // SAFETY: No registration or callbacks can observe this allocation yet.
            unsafe { (*this.as_ref().core_device()).pci_ops = &Self::PASSTHROUGH_OPS };
        }

        // SAFETY: `this` owns the initial VFIO device reference.
        Ok(unsafe { ARef::from_raw(this) })
    }

    /// Access the registration data through a closure with an HRTB lifetime.
    ///
    /// The closure's `for<'a>` bound prevents the caller from smuggling
    /// references with a concrete short lifetime out of the closure.
    ///
    /// # Safety
    ///
    /// Registration data must be published and remain valid through this call.
    unsafe fn registration_data_with<R>(
        &self,
        f: impl for<'a> FnOnce(&'a <T::RegistrationData as ForLt>::Of<'a>) -> R,
    ) -> R {
        // SAFETY: Registration keeps the allocation and its borrowed resources
        // alive until unregistration completes. Covariance permits shortening
        // the erased binding lifetime to this callback's borrow.
        let reg_data: &<T::RegistrationData as ForLt>::Of<'_> = unsafe {
            (*self.reg_data.get())
                .cast::<<T::RegistrationData as ForLt>::Of<'_>>()
                .as_ref()
        };
        f(reg_data)
    }

    /// # Safety
    ///
    /// The caller must be an I/O callback while VFIO keeps the device open.
    unsafe fn callback_data_with<R>(
        &self,
        f: impl for<'borrow, 'data> FnOnce(
            &'borrow <T::RegistrationData as ForLt>::Of<'data>,
            Pin<&'borrow T::OpenData<'data>>,
        ) -> R,
    ) -> R {
        // SAFETY: VFIO excludes open/close and unregister while this callback runs.
        unsafe {
            self.registration_data_with(|reg_data| {
                let ptr = (*self.open_data.get()).cast::<T::OpenData<'_>>();
                // The data lifetime stays independent of the callback borrow,
                // so borrowed OpenData fields cannot be stored back into OpenData.
                f(reg_data, Pin::new_unchecked(&*ptr))
            })
        }
    }

    /// # Safety
    /// The returned view must only be used during the callback represented by `Ctx`.
    unsafe fn with_context<Ctx: DeviceContext>(&self) -> &Device<T, Ctx> {
        // SAFETY: All contexts have the same layout; the caller guarantees the context.
        unsafe { &*core::ptr::from_ref(self).cast() }
    }

    /// Recover `&Self` from a raw `*mut vfio_device` in a callback.
    ///
    /// # Safety
    ///
    /// `vdev` must point to the `vfio_device` at the start of a
    /// `Device<T>` allocated by `_vfio_alloc_device()` and remain valid for `'a`.
    unsafe fn from_vfio_device<'a>(vdev: *mut bindings::vfio_device) -> &'a Self {
        // SAFETY: `vfio_device` is at offset 0 of `vfio_pci_core_device`,
        // which is at offset 0 of `Device<T>`, so the pointer cast is
        // valid.
        unsafe { &*(vdev.cast()) }
    }
}

impl<T: Operations, Ctx: DeviceContext> Device<T, Ctx> {
    fn core_device(&self) -> *mut bindings::vfio_pci_core_device {
        self.core_device.get()
    }

    fn vfio_device(&self) -> *mut bindings::vfio_device {
        self.core_device.get().cast()
    }
}

impl<T: Operations> Device<T, Open> {
    /// Set the device ID exposed through VFIO's virtual PCI configuration space.
    pub fn set_device_id(&self, device_id: u16) {
        // SAFETY: core_enable allocated vconfig; the exclusive first-open callback
        // runs before finish_enable publishes the configuration to userspace.
        unsafe {
            (*self.core_device())
                .vconfig
                .add(bindings::PCI_DEVICE_ID as usize)
                .cast::<u16>()
                .write_unaligned(device_id.to_le());
        }
    }
}

impl<T: Operations> Device<T, Ioctl> {
    /// Delegate to `vfio_pci_core_ioctl()`.
    pub fn core_ioctl(&self, cmd: u32, arg: usize) -> Result<isize> {
        // SAFETY: The Ioctl context keeps this device open on the callback thread.
        let ret = unsafe { bindings::vfio_pci_core_ioctl(self.vfio_device(), cmd, arg) };
        if ret < 0 {
            Err(Error::from_errno(ret as i32))
        } else {
            Ok(ret)
        }
    }
}

impl<T: Operations> Device<T, Read> {
    /// Delegate to `vfio_pci_core_read()`.
    ///
    /// The VFIO core fills the user-space buffer with the default PCI data
    /// for the region identified by `ppos` and advances the position by the bytes read.
    pub fn core_read(&self, buf: &UserBuf, ppos: &mut Position<'_>) -> Result<isize> {
        // SAFETY: The Read context keeps this device open on the callback thread;
        // `buf.ptr` is a user-space pointer supplied by VFIO.
        let ret = unsafe {
            bindings::vfio_pci_core_read(
                self.vfio_device(),
                buf.ptr.as_mut_ptr().cast(),
                buf.count,
                ppos.raw,
            )
        };
        if ret < 0 {
            Err(Error::from_errno(ret as i32))
        } else {
            Ok(ret)
        }
    }
}

impl<T: Operations> Device<T, Write> {
    /// Delegate to `vfio_pci_core_write()` for this callback's bounded buffer.
    pub fn core_write(&self, buf: &UserBuf, ppos: &mut Position<'_>) -> Result<isize> {
        // SAFETY: The Write context keeps the device open on the callback thread;
        // the buffer is a userspace pointer supplied by VFIO.
        let ret = unsafe {
            bindings::vfio_pci_core_write(
                self.vfio_device(),
                buf.ptr.as_const_ptr().cast(),
                buf.count,
                ppos.raw,
            )
        };
        if ret < 0 {
            Err(Error::from_errno(ret as i32))
        } else {
            Ok(ret)
        }
    }
}

impl<T: Operations> Device<T, Mmap> {
    /// Delegate to `vfio_pci_core_mmap()` for the current callback's VMA.
    pub fn core_mmap(&self, mapping: &mut Mapping<'_>) -> Result {
        // SAFETY: The Mmap context and Mapping keep the open device and locked VMA
        // on the callback thread. The VMA cannot be replaced by driver code.
        to_result(unsafe { bindings::vfio_pci_core_mmap(self.vfio_device(), mapping.vma.as_ptr()) })
    }
}

impl<T: Operations> Device<T, GetRegionInfo> {
    /// Delegate to `vfio_pci_ioctl_get_region_info()`.
    pub fn core_get_region_info(
        &self,
        info: &mut bindings::vfio_region_info,
        caps: &mut InfoCap<'_>,
    ) -> Result {
        // SAFETY: GetRegionInfo keeps the device open on the callback thread;
        // `InfoCap` preserves the validity of VFIO's capability buffer.
        to_result(unsafe {
            bindings::vfio_pci_ioctl_get_region_info(self.vfio_device(), info, caps.raw)
        })
    }
}

// SAFETY: The embedded device reference count owns the VFIO allocation.
unsafe impl<T: Operations> AlwaysRefCounted for Device<T> {
    fn inc_ref(&self) {
        // SAFETY: `self` holds a live VFIO device reference.
        unsafe { bindings::get_device(&raw mut (*self.vfio_device()).device) };
    }

    unsafe fn dec_ref(obj: NonNull<Self>) {
        // SAFETY: The caller owns the reference being released.
        unsafe { bindings::put_device(&raw mut (*obj.as_ref().vfio_device()).device) };
    }
}

// SAFETY: VFIO serializes open/close transitions, and callback data is Send + Sync.
unsafe impl<T: Operations> Send for Device<T> {}

// SAFETY: I/O callbacks share Sync data. VFIO excludes them from mutations of
// `open_data` during first open and last close; registration data follows the
// registration lifetime. Mutable C state is managed by the VFIO core.
unsafe impl<T: Operations> Sync for Device<T> {}

/// # Safety
///
/// `vdev` must belong to a registered `Device<T>` in VFIO's first-open context.
/// VFIO must exclude other opens, closes and I/O callbacks for this call.
unsafe extern "C" fn open_device_cb<T: Operations>(
    vdev: *mut bindings::vfio_device,
) -> core::ffi::c_int {
    // SAFETY: `vdev` is valid; set by vfio_alloc_device.
    let dev = unsafe { Device::<T>::from_vfio_device(vdev) };

    // SAFETY: Enable the PCI-core side first.
    let ret = unsafe { bindings::vfio_pci_core_enable(dev.core_device()) };
    if ret != 0 {
        return ret;
    }

    // SAFETY: Registration data is initialized before the VFIO device is published.
    let result = unsafe {
        dev.registration_data_with(|rd| {
            // Allocate before calling the driver so allocation failure cannot follow open.
            let data = KBox::<T::OpenData<'_>>::new_uninit(GFP_KERNEL)?;
            let data = data.write_pin_init(T::open_device(dev.with_context::<Open>(), rd))?;

            // SAFETY: Only the owning pointer is moved; the allocation remains pinned.
            let raw =
                KBox::into_raw(Pin::into_inner_unchecked(data)).cast::<T::OpenData<'static>>();
            // SAFETY: First open is exclusive with I/O callbacks and last close.
            // Lifetimes do not affect layout; the device owns the allocation until close.
            *dev.open_data.get() = raw;
            Ok::<(), Error>(())
        })
    };
    match result {
        Ok(()) => {
            // SAFETY: open succeeded.
            unsafe { bindings::vfio_pci_core_finish_enable(dev.core_device()) };
            0
        }
        Err(e) => {
            // SAFETY: Undo the enable on failure.
            unsafe { bindings::vfio_pci_core_disable(dev.core_device()) };
            e.to_errno()
        }
    }
}

/// # Safety
///
/// `vdev` must belong to a registered `Device<T>` whose first open succeeded.
/// Call exactly once for the last close, after draining I/O callbacks and while
/// excluding the next first open.
unsafe extern "C" fn close_device_cb<T: Operations>(vdev: *mut bindings::vfio_device) {
    // SAFETY: `vdev` is valid.
    let dev = unsafe { Device::<T>::from_vfio_device(vdev) };
    // SAFETY: Last close is exclusive with I/O callbacks and the next first open.
    let raw = unsafe { core::mem::replace(&mut *dev.open_data.get(), core::ptr::null_mut()) };
    // Run the driver's destructor while registration data and PCI resources are valid.
    // SAFETY: Successful open transferred this allocation with `KBox::into_raw`.
    // Last close takes ownership exactly once and drops it without moving the pointee.
    unsafe { drop(KBox::from_raw(raw)) };
    // SAFETY: Matches the enable in open_device_cb.
    unsafe { bindings::vfio_pci_core_close_device(vdev) };
}

/// # Safety
///
/// `vdev` must belong to a registered, open `Device<T>` in VFIO's ioctl callback
/// context. Registration and open data must remain alive until the call returns.
unsafe extern "C" fn ioctl_cb<T: Operations>(
    vdev: *mut bindings::vfio_device,
    cmd: core::ffi::c_uint,
    arg: usize,
) -> isize {
    // SAFETY: `vdev` is valid.
    let dev = unsafe { Device::<T>::from_vfio_device(vdev) };
    // SAFETY: VFIO invokes this callback for an open device with valid arguments.
    match unsafe {
        dev.callback_data_with(|rd, od| T::ioctl(dev.with_context::<Ioctl>(), rd, od, cmd, arg))
    } {
        Ok(v) => v,
        Err(e) => e.to_errno() as isize,
    }
}

/// # Safety
///
/// `vdev` must belong to a registered, open `Device<T>` in VFIO's read context.
/// `buf` must be a userspace address and `ppos` must be exclusively accessible
/// for this call. Registration and open data must remain alive until it returns.
unsafe extern "C" fn read_cb<T: Operations>(
    vdev: *mut bindings::vfio_device,
    buf: *mut u8,
    count: usize,
    ppos: *mut i64,
) -> isize {
    // SAFETY: `vdev` is valid. `buf` is a valid user-space pointer provided by
    // the VFIO core. `ppos` is a valid kernel pointer.
    let dev = unsafe { Device::<T>::from_vfio_device(vdev) };
    let mut ubuf = UserBuf {
        ptr: UserPtr::from_ptr(buf.cast()),
        count,
    };
    let mut pos = Position {
        // SAFETY: `ppos` is a valid kernel pointer provided by the VFIO core.
        raw: unsafe { &mut *ppos },
    };
    // SAFETY: VFIO invokes this callback for an open device with valid arguments.
    match unsafe {
        dev.callback_data_with(|rd, od| {
            T::read(dev.with_context::<Read>(), rd, od, &mut ubuf, &mut pos)
        })
    } {
        Ok(n) => n,
        Err(e) => e.to_errno() as isize,
    }
}

/// # Safety
///
/// `vdev` must belong to a registered, open `Device<T>` in VFIO's region-info
/// context. `info` and `caps` must be valid, disjoint, exclusively accessible
/// objects; the capability buffer must satisfy the VFIO core's invariants.
/// Registration and open data must remain alive until the call returns.
unsafe extern "C" fn get_region_info_cb<T: Operations>(
    vdev: *mut bindings::vfio_device,
    info: *mut bindings::vfio_region_info,
    caps: *mut bindings::vfio_info_cap,
) -> core::ffi::c_int {
    // SAFETY: `vdev`, `info`, and `caps` are valid kernel pointers provided
    // by the VFIO core.
    let dev = unsafe { Device::<T>::from_vfio_device(vdev) };
    // SAFETY: `info` is a valid pointer provided by the VFIO core.
    let info = unsafe { &mut *info };
    let mut caps = InfoCap {
        // SAFETY: `caps` is a valid pointer provided by the VFIO core.
        raw: unsafe { &mut *caps },
    };
    // SAFETY: VFIO invokes this callback for an open device with valid arguments.
    match unsafe {
        dev.callback_data_with(|rd, od| {
            T::get_region_info(dev.with_context::<GetRegionInfo>(), rd, od, info, &mut caps)
        })
    } {
        Ok(()) => 0,
        Err(e) => e.to_errno(),
    }
}

/// # Safety
///
/// `vdev` must belong to a registered, open `Device<T>` in VFIO's write context.
/// `buf` must be a userspace address and `ppos` must be exclusively accessible
/// for this call. Registration and open data must remain alive until it returns.
unsafe extern "C" fn write_cb<T: Operations>(
    vdev: *mut bindings::vfio_device,
    buf: *const u8,
    count: usize,
    ppos: *mut i64,
) -> isize {
    // SAFETY: VFIO supplies an open device and a valid file position pointer.
    let dev = unsafe { Device::<T>::from_vfio_device(vdev) };
    let mut ubuf = UserBuf {
        ptr: UserPtr::from_ptr(buf.cast_mut().cast()),
        count,
    };
    let mut pos = Position {
        // SAFETY: VFIO holds the file-position lock when required.
        raw: unsafe { &mut *ppos },
    };
    // SAFETY: This callback runs while VFIO keeps both callback data allocations alive.
    match unsafe {
        dev.callback_data_with(|rd, od| {
            T::write(dev.with_context::<Write>(), rd, od, &mut ubuf, &mut pos)
        })
    } {
        Ok(n) => n,
        Err(e) => e.to_errno() as isize,
    }
}

/// # Safety
///
/// `vdev` must belong to a registered, open `Device<T>` in VFIO's mmap context.
/// `vma` must be the valid VMA being mapped, with the mmap write lock held on
/// this thread. Registration and open data must remain alive until return.
unsafe extern "C" fn mmap_cb<T: Operations>(
    vdev: *mut bindings::vfio_device,
    vma: *mut bindings::vm_area_struct,
) -> core::ffi::c_int {
    // SAFETY: VFIO supplies an open device for this mmap callback.
    let dev = unsafe { Device::<T>::from_vfio_device(vdev) };
    let mut mapping = Mapping {
        // SAFETY: The mmap callback holds the mmap lock for this VMA.
        vma: unsafe { VmaRef::from_raw(vma) },
        _not_thread_safe: NotThreadSafe,
    };
    // SAFETY: This callback runs while VFIO keeps both callback data allocations alive.
    match unsafe {
        dev.callback_data_with(|rd, od| T::mmap(dev.with_context::<Mmap>(), rd, od, &mut mapping))
    } {
        Ok(()) => 0,
        Err(e) => e.to_errno(),
    }
}

/// Build the `vfio_device_ops` vtable for a variant driver `T`.
///
/// Open and I/O callbacks dispatch to `T`'s trait implementation.
/// All other callbacks use the `vfio_pci_core_*` defaults.
macro_rules! build_ops {
    ($T:ty) => {
        bindings::vfio_device_ops {
            #[allow(clippy::disallowed_methods)]
            name: <$T>::NAME.as_ptr().cast_mut().cast(),
            init: Some(bindings::vfio_pci_core_init_dev),
            release: Some(bindings::vfio_pci_core_release_dev),
            open_device: Some(open_device_cb::<$T>),
            close_device: Some(close_device_cb::<$T>),
            ioctl: Some(ioctl_cb::<$T>),
            read: Some(read_cb::<$T>),
            write: Some(write_cb::<$T>),
            mmap: Some(mmap_cb::<$T>),
            request: Some(bindings::vfio_pci_core_request),
            get_region_info_caps: Some(get_region_info_cb::<$T>),
            match_: Some(bindings::vfio_pci_core_match),
            match_token_uuid: Some(bindings::vfio_pci_core_match_token_uuid),
            device_feature: Some(bindings::vfio_pci_core_ioctl_feature),
            #[cfg(CONFIG_IOMMUFD)]
            bind_iommufd: Some(bindings::vfio_iommufd_physical_bind),
            #[cfg(not(CONFIG_IOMMUFD))]
            bind_iommufd: None,
            #[cfg(CONFIG_IOMMUFD)]
            unbind_iommufd: Some(bindings::vfio_iommufd_physical_unbind),
            #[cfg(not(CONFIG_IOMMUFD))]
            unbind_iommufd: None,
            #[cfg(CONFIG_IOMMUFD)]
            attach_ioas: Some(bindings::vfio_iommufd_physical_attach_ioas),
            #[cfg(not(CONFIG_IOMMUFD))]
            attach_ioas: None,
            #[cfg(CONFIG_IOMMUFD)]
            detach_ioas: Some(bindings::vfio_iommufd_physical_detach_ioas),
            #[cfg(not(CONFIG_IOMMUFD))]
            detach_ioas: None,
            #[cfg(CONFIG_IOMMUFD)]
            pasid_attach_ioas: Some(bindings::vfio_iommufd_physical_pasid_attach_ioas),
            #[cfg(not(CONFIG_IOMMUFD))]
            pasid_attach_ioas: None,
            #[cfg(CONFIG_IOMMUFD)]
            pasid_detach_ioas: Some(bindings::vfio_iommufd_physical_pasid_detach_ioas),
            #[cfg(not(CONFIG_IOMMUFD))]
            pasid_detach_ioas: None,
            dma_unmap: None,
        }
    };
}

/// The registration of a VFIO PCI variant device.
///
/// Owns both the VFIO device reference and the registration data, tying
/// resource lifetimes to the PCI binding scope `'a`.
///
/// Prepare this owner during probe, register it in [`pci::Driver::bind`], and unregister it
/// in [`pci::Driver::unbind`] before releasing any driver data.
pub struct Registration<'a, T: Operations> {
    dev: ARef<Device<T>>,
    _reg_data: Pin<KBox<<T::RegistrationData as ForLt>::Of<'a>>>,
}

impl<T: Operations> Device<T> {
    const OPS: bindings::vfio_device_ops = build_ops!(T);

    const PASSTHROUGH_OPS: bindings::vfio_pci_device_ops = bindings::vfio_pci_device_ops {
        #[cfg(CONFIG_VFIO_PCI_DMABUF)]
        get_dmabuf_phys: Some(bindings::vfio_pci_core_get_dmabuf_phys),
        #[cfg(not(CONFIG_VFIO_PCI_DMABUF))]
        get_dmabuf_phys: None,
    };
}

impl<'a, T: Operations> Registration<'a, T> {
    /// Prepare a previously allocated VFIO PCI device and its callback data.
    ///
    /// # Safety
    ///
    /// The device must have been allocated during this parent binding and never registered.
    /// No other `Registration` may own or publish callback data for this device until this
    /// owner is dropped.
    /// The caller must hold the PCI device lock. Store the owner in the driver's private data
    /// and keep it alive until unregistration has completed.
    pub unsafe fn new(
        pdev: &'a pci::Device<device::Core<'_>>,
        dev: &Device<T>,
        reg_data: impl PinInit<<T::RegistrationData as ForLt>::Of<'a>, Error>,
    ) -> Result<Self> {
        // SAFETY: `dev` holds a live VFIO allocation; only compare its parent pointer.
        if unsafe { (*dev.vfio_device()).dev } != pdev.as_ref().as_raw() {
            return Err(EINVAL);
        }

        let reg_data: Pin<KBox<<T::RegistrationData as ForLt>::Of<'a>>> =
            KBox::pin_init(reg_data, GFP_KERNEL)?;
        let ptr: NonNull<<T::RegistrationData as ForLt>::Of<'static>> =
            NonNull::from(Pin::get_ref(reg_data.as_ref())).cast();

        // SAFETY: No concurrent registration or callbacks; publish before registering.
        unsafe { *dev.reg_data.get() = ptr };

        Ok(Self {
            dev: dev.into(),
            _reg_data: reg_data,
        })
    }

    /// Register after the PCI adapter has installed the driver's private data.
    ///
    /// # Safety
    ///
    /// Call once from the bound PCI driver `D`'s `bind` callback, with `self` in its installed
    /// driver data. Its PCI reset, error and runtime-PM callbacks must delegate to this owner.
    /// After success, unregister before any later bind error or driver-data teardown.
    pub unsafe fn register<D: pci::Driver>(&self) -> Result {
        build_assert!(
            D::DRIVER_MANAGED_DMA,
            "VFIO drivers must manage DMA ownership"
        );
        #[cfg(CONFIG_PM)]
        build_assert!(
            D::PM_OPS.is_some(),
            "VFIO drivers must install runtime PM callbacks"
        );
        #[cfg(CONFIG_PCI_IOV)]
        build_assert!(
            !D::HAS_SRIOV_ENABLE && !D::HAS_SRIOV_DISABLE,
            "VFIO PCI variant drivers do not support SR-IOV configuration"
        );

        // SAFETY: The caller has published complete private data under the PCI device lock;
        // it remains accessible during runtime PM setup.
        to_result(unsafe { bindings::vfio_pci_core_register_device(self.dev.core_device()) })
    }

    /// Unregister while callbacks can still recover the PCI driver's private data.
    ///
    /// # Safety
    ///
    /// Call once after successful registration, with the PCI device lock held and all driver
    /// data still installed. Do not call after registration failed: that can disable VFs
    /// belonging to a PF which VFIO refused to bind.
    pub unsafe fn unregister(&self) {
        // SAFETY: The caller keeps private data installed while unregistration drains VFIO,
        // VGA and runtime PM callbacks. No field of the enclosing data is being destroyed.
        unsafe { bindings::vfio_pci_core_unregister_device(self.dev.core_device()) };
    }

    /// Prepare the variant driver's state for a PCI reset.
    pub fn reset_prepare(&self) {
        // SAFETY: This owner keeps the pinned registration data alive.
        unsafe { self.dev.registration_data_with(T::reset_prepare) };
    }

    /// Complete the variant driver's state update after a PCI reset.
    pub fn reset_done(&self) -> Result {
        // SAFETY: This owner keeps the pinned registration data alive.
        unsafe { self.dev.registration_data_with(T::reset_done) }
    }

    /// Signal the VFIO error interrupt and return the PCI recovery result.
    pub fn error_detected(&self) -> bindings::pci_ers_result_t {
        // SAFETY: This owner keeps the initialized VFIO core device alive.
        unsafe { bindings::vfio_pci_core_aer_err_detected(self.dev.core_device()) }
    }

    /// Prepare the VFIO core device for runtime suspend.
    ///
    /// # Safety
    ///
    /// Only call from the parent PCI driver's runtime-suspend callback.
    #[cfg(CONFIG_PM)]
    pub unsafe fn runtime_suspend(&self) -> Result {
        // SAFETY: The caller supplies the runtime-suspend callback context.
        to_result(unsafe { bindings::vfio_pci_core_runtime_suspend(self.dev.core_device()) })
    }

    /// Complete the VFIO core device's runtime resume.
    ///
    /// # Safety
    ///
    /// Only call from the parent PCI driver's runtime-resume callback.
    #[cfg(CONFIG_PM)]
    pub unsafe fn runtime_resume(&self) -> Result {
        // SAFETY: The caller supplies the runtime-resume callback context.
        to_result(unsafe { bindings::vfio_pci_core_runtime_resume(self.dev.core_device()) })
    }
}

impl<T: Operations> Drop for Registration<'_, T> {
    fn drop(&mut self) {
        // SAFETY: The driver unregistered in `unbind`, or registration did not succeed.
        unsafe { *self.dev.reg_data.get() = NonNull::dangling() };
    }
}
