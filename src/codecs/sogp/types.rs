//! Parameter + format types for SOGP.

use serde::{Deserialize, Serialize};

/// 3-byte magic for dispatch via `decoder::magic3`.
pub const MAGIC: &[u8; 3] = b"SGP";
pub const VERSION: u8 = 1;

/// Compression backend used inside the SOGP container.
///
/// Note: this is *not* the outer `EncodingParams::{Zstd,Lz4,...}` wrapper; this codec
/// compresses its own internal planes/streams.
#[repr(u8)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum SogpCompression {
    /// Store raw bytes (fast, large).
    None = 0,

    /// Zstd stream compression (fast + good ratio at low levels).
    Zstd { level: i32 },

    /// LZ4 frame (very fast, moderate ratio).
    Lz4,

    /// Snappy frame (fast, moderate ratio).
    Snappy,

    /// OpenZL serial compressor for byte payloads.
    OpenzlSerial,
}

impl Default for SogpCompression {
    fn default() -> Self {
        // Strong default for “fast but close to Draco”.
        Self::Zstd { level: 1 }
    }
}

/// Encoder parameters for SOGP.
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SogpParams {
    /// Quantization bits for positions per axis, in 1..=16.
    ///
    /// - 16 gives best quality.
    /// - 12..14 is often a good speed/ratio compromise.
    pub pos_bits: u8,

    /// Reorder points in Morton order before packing.
    ///
    /// Improves locality, usually improves compression, and often improves cache behavior
    /// in downstream renderers (at the cost of encoding-time sorting).
    #[serde(default = "d_true")]
    pub morton_order: bool,

    /// Include RGB plane.
    #[serde(default = "d_true")]
    pub include_color: bool,

    /// Pack all planes into a single compressed stream (recommended).
    ///
    /// This is a major fast-path:
    /// - one decompressor invocation instead of 2-3
    /// - no per-stream framing overhead
    /// - better cross-plane compression
    #[serde(default = "d_true")]
    pub packed_stream: bool,

    /// Internal compression backend.
    #[serde(default)]
    pub compression: SogpCompression,
}

fn d_true() -> bool {
    true
}

impl Default for SogpParams {
    fn default() -> Self {
        Self {
            pos_bits: 16,
            morton_order: true,
            include_color: true,
            packed_stream: true,
            compression: SogpCompression::default(),
        }
    }
}

/// Stream IDs used in the container.
pub(crate) mod stream_id {
    pub const POS_LO: u8 = 0;
    pub const POS_HI: u8 = 1;
    pub const RGB: u8 = 2;
    pub const PACKED: u8 = 255;
}

/// Flags in the header.
pub(crate) mod flags {
    pub const HAS_COLOR: u8 = 1 << 0;
    pub const HAS_POS_HI: u8 = 1 << 1;
    pub const MORTON_ORDERED: u8 = 1 << 2;
    pub const PACKED_STREAM: u8 = 1 << 3;
}

/// Compression IDs in header (stable ABI).
pub(crate) mod comp_id {
    pub const NONE: u8 = 0;
    pub const ZSTD: u8 = 1;
    pub const LZ4: u8 = 2;
    pub const SNAPPY: u8 = 3;
    pub const OPENZL_SERIAL: u8 = 4;
}
