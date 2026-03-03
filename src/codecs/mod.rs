//! Concrete codec implementations.
//!
//! Each submodule exposes `encode_from_payload_into` / `decode_into_*` functions through
//! thin wrappers that integrate with the top-level [`encoder`](crate::encoder) and
//! [`decoder`](crate::decoder) APIs. The list ranges from lightweight integer codecs
//! (`quantize`, `bitcode`) to wrappers around external libraries (`draco`, `tmf`,
//! `gzip`, …).

pub mod bitcode;
#[cfg(feature = "draco")]
pub mod draco;
pub mod gsplat16;
pub mod gzip;
pub mod lz4;
pub mod openzl;
pub mod ply;
pub mod quantize;
pub mod snappy;
pub mod sogp;
pub mod tmf;
pub mod zstd;
