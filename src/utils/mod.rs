//! Internal utilities shared across codecs.
//!
//! Some helpers are `pub` because they are useful for interop (e.g. viewing
//! flattened `[T]` buffers as `[[T; N]]` without allocations). Everything here
//! is intentionally small and dependency-free.

pub(crate) mod byte_cursor;

pub mod slice_convert;
