//! SOGP: SOG-like codec for normal point clouds (XYZ + optional RGB).
//!
//! Goals:
//! - Very fast decode (few branches, bulk operations).
//! - Compact binary header (no JSON).
//! - Optional Morton ordering improves locality => better compression.
//! - No wasted alpha bytes (we store 3 bytes per point per plane, not RGBA).
//! - Multiple fast paths: single packed stream is the default.
//!
//! The format is self-contained and does not require external metadata.

pub mod decoder;
pub mod encoder;
pub mod types;

#[cfg(test)]
mod tests {
    use super::{decoder, encoder, types::SogpParams};
    use spatial_utils::{point::Point3RgbF32, traits::{HasPosition3, HasRgb8u}};

    fn sort_key(p: &Point3RgbF32) -> [u8; 3] {
        [p.r_u8(), p.g_u8(), p.b_u8()]
    }

    #[test]
    fn sogp_roundtrip_smoke_sorted() {
        // Unique colors so we can re-associate points after Morton re-ordering.
        let input: Vec<Point3RgbF32> = vec![
            Point3RgbF32::new(0.0, 0.0, 0.0, 1, 2, 3),
            Point3RgbF32::new(0.25, 0.5, 0.75, 10, 20, 30),
            Point3RgbF32::new(-0.1, 0.2, -0.3, 40, 50, 60),
            Point3RgbF32::new(1.0, -0.5, 0.125, 70, 80, 90),
        ];

        let params = SogpParams::default();

        let mut encoded = Vec::new();
        encoder::encode_from_payload_into::<Point3RgbF32, f32>(&input, &params, &mut encoded)
            .expect("encode must succeed");
        assert!(!encoded.is_empty());

        let mut out: Vec<Point3RgbF32> = Vec::new();
        decoder::decode_into(&encoded, &mut out).expect("decode must succeed");
        assert_eq!(out.len(), input.len());

        let mut input_sorted = input.clone();
        input_sorted.sort_by_key(sort_key);

        out.sort_by_key(sort_key);

        // Positions are quantized -> small tolerance.
        let eps = 1.0e-3_f32;
        for (a, b) in input_sorted.iter().zip(out.iter()) {
            assert_eq!(sort_key(a), sort_key(b), "color mismatch");

            assert!((a.x() - b.x()).abs() <= eps, "x mismatch: {a:?} vs {b:?}");
            assert!((a.y() - b.y()).abs() <= eps, "y mismatch: {a:?} vs {b:?}");
            assert!((a.z() - b.z()).abs() <= eps, "z mismatch: {a:?} vs {b:?}");
        }
    }
}