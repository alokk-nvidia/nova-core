// SPDX-License-Identifier: GPL-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

//! Codec for NVKV, the binary key-value format of GMCAPI.
//!
//! Essentially, the format encodes a sequence of calls to some function f(key, index, value),
//! where value is a `[u8]`, `u32`, `u64`, `[u32]`, or a `[u64]`. The key is a `u16` and the index
//! is a 12 bit integer. The interpretation of these function calls is per GMCAPI. Generally
//! speaking, the function calls will map to some struct - for example, f(GPU_NAME_STRING_KEY, 0,
//! b"some gpu") naturally maps to storing a &str with the GPU name.

use core::{
    marker::PhantomData,
    ops::{
        Deref,
        DerefMut, //
    }, //
};

use kernel::{
    alloc::ArrayVec,
    bitfield,
    num::Bounded,
    prelude::*, //
};

use crate::bounded_enum;

use crate::sbuffer::SBufferIter;

/// Joins the two halves of a wrapped payload into the `u64` words an NVKV stream is made of.
///
/// # Errors
///
/// - `EINVAL` if the combined length is not a whole number of words.
/// - `ENOMEM` if the buffer cannot be allocated.
pub(crate) fn nvkv_words(payload_0: &[u8], payload_1: &[u8]) -> Result<KVVec<u64>> {
    const WORD_SIZE: usize = size_of::<u64>();

    // Each byte slice is at most `isize::MAX` bytes, so their sum fits in `usize`.
    let len = payload_0.len() + payload_1.len();
    if len % WORD_SIZE != 0 {
        return Err(EINVAL);
    }

    let mut words = KVVec::with_capacity(len / WORD_SIZE, GFP_KERNEL)?;
    let mut bytes = SBufferIter::new_reader([payload_0, payload_1]);
    for _ in 0..len / WORD_SIZE {
        let mut word = [0u8; WORD_SIZE];
        bytes.read_exact(&mut word)?;
        words.push(u64::from_le_bytes(word), GFP_KERNEL)?;
    }

    Ok(words)
}

mod encode;
pub(crate) use encode::*;

mod decode;
pub(crate) use decode::*;

/// The identifier of an NVKV key.
pub(crate) type KeyId = u16;

/// The index of an NVKV value.
pub(crate) type Index = Bounded<u64, 12>;

/// A static association between an NVKV key `KEY_ID` and the storage of its value.
///
/// Use with the encoder or decoder macros `nvkv_encode!` and `nvkv_decode!` to let them know how to
/// map the value `Key<T, KEY_ID, As>` to/from encoded data. For brevity, `As` inserts an additional
/// conversion ([`From`]) to avoid having to implement [`Encodable`] for many types. For example,
/// enums that are easily convertible to a u32 can have `As = u32` and rely on the existing encoding
/// for u32.
pub(crate) struct Key<T, const KEY_ID: KeyId, As = T>(T, PhantomData<As>);

impl<T, const KEY_ID: KeyId, As> Key<T, KEY_ID, As> {
    /// The maximum size in bytes we allow to materialize on the stack while
    /// encoding or decoding a [`Key`]. This is checked at compile time.
    const MAX_STACK_SIZE: usize = 32;
}

impl<T, const KEY_ID: KeyId, As> From<T> for Key<T, KEY_ID, As> {
    fn from(value: T) -> Self {
        Self(value, PhantomData)
    }
}

impl<'a, T, const KEY_ID: KeyId, As, const N: usize> From<&'a [T; N]> for Key<&'a [T], KEY_ID, As> {
    fn from(value: &'a [T; N]) -> Self {
        Self(&value[..], PhantomData)
    }
}

impl<T, const KEY_ID: KeyId, As> Deref for Key<T, KEY_ID, As> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const KEY_ID: KeyId, As> DerefMut for Key<T, KEY_ID, As> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Default, const KEY_ID: KeyId, As> Default for Key<T, KEY_ID, As> {
    fn default() -> Self {
        Self(T::default(), PhantomData)
    }
}

/// A value with a specific index that encodes under the NVKV key `KEY_ID`.
pub(crate) struct IndexedKey<T, const KEY_ID: KeyId> {
    index: Index,
    value: T,
}

impl<T, const KEY_ID: KeyId> IndexedKey<T, KEY_ID> {
    /// Creates a key with the given index and value.
    pub(crate) fn new(index: Index, value: T) -> Self {
        Self { index, value }
    }
}

/// A schema field for an array value under the NVKV key `KEY_ID`.
pub(crate) struct Array<T, const N: usize, const KEY_ID: KeyId> {
    vec: ArrayVec<T, N>,
}

impl<T: Default + Copy, const N: usize, const KEY_ID: KeyId> Array<T, N, KEY_ID> {
    /// Creates an array filled from `values`.
    ///
    /// Fails with `EINVAL` if `values` is longer than `N`.
    pub(crate) fn new(values: &[T]) -> Result<Self> {
        let mut vec = ArrayVec::default();
        vec.extend_from_slice(values)?;
        Ok(Self { vec })
    }
}

bitfield! {
    /// The op word that starts each NVKV operation.
    struct Op(u64) {
        15:0 key => KeyId;
        27:16 index => Index;
        31:28 opcode ?=> Opcode;
        63:32 value => u32;
    }
}

bounded_enum! {
    /// Describes the format of the following NVKV operation.
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    #[repr(u8)]
    enum Opcode with TryFrom<Bounded<u64, 4>> {
        /// A 32-bit value in the op word.
        Imm32 = 0,
        /// 32-bit values for consecutive keys, starting at the op word's key.
        Seq32 = 1,
        /// 64-bit values for consecutive keys, starting at the op word's key.
        Seq64 = 2,
        /// An array of bytes.
        Array8 = 3,
        /// An array of 32-bit elements.
        Array32 = 4,
        /// An array of 64-bit elements.
        Array64 = 5,
    }
}
