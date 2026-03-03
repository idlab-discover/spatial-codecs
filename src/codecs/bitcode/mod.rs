//! Lossless wrapper around the `bitcode` crate.
//!
//! This module simply serialises/deserialises `Vec<Point3D>` using `bitcode`’s compact
//! binary encoding. It is primarily used as a baseline for other codecs and for feeding
//! wrapper codecs (gzip/zstd/etc.).

use bitcode::{Decode, Encode};
use spatial_utils::{
    traits::{ColorKind, SpatialKind},
    utils::point_scalar::PointScalarKind,
};

pub mod decoder;
pub mod encoder;

pub(crate) const MAGIC: &[u8; 3] = b"BC1";
pub(crate) const MAX_HEADER_BYTES: usize = 1024;
pub(crate) const WIRE_HDR_V1_LEN: usize = 4;
pub(crate) const WIRE_HDR_V1_VERSION: u8 = 1;

/// Container stored inside the bitcode payload.
#[derive(Encode, Decode)]
pub struct BitcodeData<P> {
    pub payload: Vec<P>,
}

/// Header stored inside the *new* bitcode payload.
#[derive(Encode, Decode, Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct BitcodeHeader {
    pub spatial_kind: u8,
    pub color_kind: u8,
    pub scalar_kind: u8,
}

impl BitcodeHeader {
    #[inline(always)]
    pub(crate) const fn new(
        spatial_kind: SpatialKind,
        color_kind: ColorKind,
        scalar_kind: PointScalarKind,
    ) -> Self {
        Self {
            spatial_kind: spatial_kind as u8,
            color_kind: color_kind as u8,
            scalar_kind: scalar_kind as u8,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::{decoder, encoder};
    use spatial_utils::point::Point3RgbF32;

    #[test]
    fn bitcode_roundtrip_smoke() {
        let input: Vec<Point3RgbF32> = vec![
            Point3RgbF32::new(0.0, 0.0, 0.0, 255, 0, 0),
            Point3RgbF32::new(1.0, 2.0, 3.0, 0, 255, 0),
            Point3RgbF32::new(-4.0, 5.0, -6.0, 0, 0, 255),
            Point3RgbF32::new(10.0, -2.5, 0.125, 123, 45, 67),
        ];

        let params = encoder::BitcodeParams::default();

        let mut encoded = Vec::new();
        encoder::encode_from_payload_into(&input, &params, &mut encoded)
            .expect("encode must succeed");
        assert!(!encoded.is_empty());

        let mut out: Vec<Point3RgbF32> = Vec::new();
        decoder::decode_into(&encoded, &mut out).expect("decode must succeed");

        assert_eq!(out, input);
    }
}