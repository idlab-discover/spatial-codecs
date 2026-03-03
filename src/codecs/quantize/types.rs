use serde::{Deserialize, Serialize};

pub const MAGIC: &[u8; 3] = b"QNT";
pub const VERSION: u8 = 2;

// Layout (v2):
// magic (3) | version (1) | position_bits (1) | color_bits (1) | point_count (4) |
// position mins+maxs (6 * f32) | color mins (3 * u8) | color maxs (3 * u8)
// if palette flag set: palette_len (u16) + palette_len * 3 bytes
pub const HEADER_FIXED_SIZE: usize =
    MAGIC.len() + 1 + 1 + 4 + (6 * core::mem::size_of::<f32>()) + 3 + 3;
pub const HEADER_EXTENDED_SIZE: usize =
    MAGIC.len() + 1 + 2 + 4 + (6 * core::mem::size_of::<f32>()) + 3 + 3;
pub const PALETTE_LEN_FIELD_SIZE: usize = 2;

pub const COLOR_BITFLAG_PALETTE: u8 = 0x80;
pub const COLOR_BITFLAG_HAS_FLAGS: u8 = 0x40;
pub const COLOR_BITS_MASK: u8 = 0x3F;

pub const FLAG_DELTA_POSITIONS: u8 = 0x01;
pub const FLAG_DELTA_COLORS: u8 = 0x02;
pub const FLAG_FIXED_WIDTH_POSITIONS: u8 = 0x04;
pub const FLAG_FIXED_WIDTH_COLORS: u8 = 0x08;

fn default_true() -> bool {
    true
}

#[repr(C)]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct QuantizeParams {
    pub position_bits: u8,
    pub color_bits: u8,
    /// 0 disables palette. Otherwise we may auto-select grid = 5 or 6 bits/channel.
    #[serde(default)]
    pub max_palette_colors: u16,
    #[serde(default)]
    pub delta_positions: bool,
    #[serde(default)]
    pub delta_colors: bool,
    #[serde(default = "default_true")]
    pub pack_positions: bool,
    #[serde(default = "default_true")]
    pub pack_colors: bool,
}

impl Default for QuantizeParams {
    fn default() -> Self {
        Self {
            position_bits: 12,
            color_bits: 8,
            max_palette_colors: 0,
            delta_positions: false,
            delta_colors: false,
            pack_positions: true,
            pack_colors: true,
        }
    }
}

#[inline(always)]
pub fn clamp_bits(bits: u8) -> u8 {
    bits.clamp(1, 32)
}

#[inline(always)]
pub fn max_quant_value(bits: u8) -> u64 {
    if bits == 32 {
        u32::MAX as u64
    } else {
        (1u64 << bits) - 1
    }
}

#[inline(always)]
pub fn mask_for(bits: u8) -> u64 {
    if bits == 32 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

#[inline(always)]
pub fn bits_for_palette_len(len: usize) -> u8 {
    if len <= 1 {
        0
    } else {
        (usize::BITS - (len - 1).leading_zeros()) as u8
    }
}

#[inline(always)]
pub fn storage_bits_for(bits: u8, packed: bool) -> u8 {
    if bits == 0 {
        return 0;
    }
    if packed {
        bits
    } else if bits <= 8 {
        8
    } else if bits <= 16 {
        16
    } else {
        32
    }
}
