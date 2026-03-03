//! Types and constants for the GSP16 codec.

use serde::{Deserialize, Serialize};

/// 3-byte magic for dispatch via `decoder::magic3`.
pub const MAGIC: &[u8; 3] = b"GSP";

/// Version byte immediately after MAGIC.
pub const VERSION: u8 = 1;

/// Total header length in bytes.
pub const HEADER_LEN: usize = 16;

/// Record length in bytes (fixed-width).
pub const RECORD_LEN: usize = 16;

/// Flags in the header.
pub const FLAG_SPLATS: u8 = 1 << 0;
pub const FLAG_HAS_ALPHA: u8 = 1 << 1;

/// Parameters for GSP16 (currently no knobs, kept for symmetry / future-proofing).
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Gsplat16Params {}
