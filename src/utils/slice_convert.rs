//! Zero-copy helpers for viewing flattened buffers as fixed-size array slices.
//!
//! This is primarily intended for FFI / interop use-cases where some callers
//! naturally work with flattened buffers (`[T]`, e.g. XYZXYZ...) while Rust APIs
//! often prefer `[[T; N]]`.
//!
//! These helpers are *safe* wrappers around the small amount of `unsafe`
//! required to reinterpret the slice.

use core::{mem, slice};

/// Error returned by [`flat_as_array_chunks`] and friends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceConvertError {
    /// `flat.len()` is not divisible by `N`.
    LengthNotMultiple { len: usize, n: usize },
    /// The reinterpreted slice would overflow `isize::MAX` bytes.
    ByteSizeOverflow,
}

/// Computes the flattened length for `chunks` items of `[T; N]`.
#[inline]
pub const fn flat_len_for_chunks<const N: usize>(chunks: usize) -> Option<usize> {
    // `chunks * N` can overflow.
    chunks.checked_mul(N)
}

/// View a flattened slice (`[T]`) as `[[T; N]]` without copying.
///
/// # Errors
/// - If `flat.len()` is not divisible by `N`.
/// - If the resulting byte size would exceed `isize::MAX`.
#[inline]
pub fn flat_as_array_chunks<T, const N: usize>(flat: &[T]) -> Result<&[[T; N]], SliceConvertError> {
    if N == 0 {
        // `[T; 0]` is allowed, but is rarely useful; treat as empty view.
        return Ok(&[]);
    }

    let len = flat.len();
    if !len.is_multiple_of(N) {
        return Err(SliceConvertError::LengthNotMultiple { len, n: N });
    }

    // Validate the byte size fits in `isize` as required by `slice::from_raw_parts`.
    let bytes = len
        .checked_mul(mem::size_of::<T>())
        .ok_or(SliceConvertError::ByteSizeOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(SliceConvertError::ByteSizeOverflow);
    }

    let out_len = len / N;
    // Safety:
    // - `[T; N]` is layout-compatible with `N` consecutive `T`s.
    // - `flat` is properly aligned for `T`; `[T; N]` has the same alignment.
    // - length checks ensure no out-of-bounds.
    let ptr = flat.as_ptr() as *const [T; N];
    // Additional debug-time alignment check (no-op in release).
    debug_assert_eq!((ptr as usize) % mem::align_of::<[T; N]>(), 0);
    Ok(unsafe { slice::from_raw_parts(ptr, out_len) })
}

/// Mutable variant of [`flat_as_array_chunks`].
#[inline]
pub fn flat_as_array_chunks_mut<T, const N: usize>(
    flat: &mut [T],
) -> Result<&mut [[T; N]], SliceConvertError> {
    if N == 0 {
        return Ok(&mut []);
    }

    let len = flat.len();
    if !len.is_multiple_of(N) {
        return Err(SliceConvertError::LengthNotMultiple { len, n: N });
    }

    let bytes = len
        .checked_mul(mem::size_of::<T>())
        .ok_or(SliceConvertError::ByteSizeOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(SliceConvertError::ByteSizeOverflow);
    }

    let out_len = len / N;
    let ptr = flat.as_mut_ptr() as *mut [T; N];
    debug_assert_eq!((ptr as usize) % mem::align_of::<[T; N]>(), 0);
    Ok(unsafe { slice::from_raw_parts_mut(ptr, out_len) })
}

/// View `[[T; N]]` as a flattened `[T]` slice without copying.
#[inline]
pub fn slice_as_flat<T, const N: usize>(chunks: &[[T; N]]) -> &[T] {
    if N == 0 {
        return &[];
    }

    // Safety:
    // - `[T; N]` is `N` consecutive `T`s.
    // - same alignment as `T`.
    // - length is `chunks.len() * N`.
    let len = chunks.len().saturating_mul(N);
    unsafe { slice::from_raw_parts(chunks.as_ptr() as *const T, len) }
}

/// Mutable variant of [`slice_as_flat`].
#[inline]
pub fn slice_as_flat_mut<T, const N: usize>(chunks: &mut [[T; N]]) -> &mut [T] {
    if N == 0 {
        return &mut [];
    }

    let len = chunks.len().saturating_mul(N);
    unsafe { slice::from_raw_parts_mut(chunks.as_mut_ptr() as *mut T, len) }
}
