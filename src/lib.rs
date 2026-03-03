//! High-level entry points for the `spatial_codecs` crate.
//!
//! The crate exposes a suite of point-cloud codecs (lossy and lossless) with a uniform
//! interface. Most users only ever interact with [`encoder`] and [`decoder`], while the
//! [`bench`] module provides tools for evaluating codec quality/performance.

pub mod bench;
pub mod bindings_generation;
pub mod codecs;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod ffi;
pub mod utils;
pub use error::CodecError;

pub type BasicResult = Result<(), Box<dyn std::error::Error>>;

pub use ffi::build_binding_inventory;
