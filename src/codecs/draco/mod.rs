//! Draco codec bridge (via `spatial_codec_draco`).

pub mod decoder;
pub mod encoder;

#[cfg(test)]
mod tests {
    use super::{decoder, encoder};
    use spatial_utils::{point::Point3RgbF32, traits::{HasPosition3, HasRgb8u}};

    fn color_key(c: [u8; 3]) -> u32 {
        ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | (c[2] as u32)
    }

    fn coord_key(c: [f32; 3]) -> (u32, u32, u32) {
        // Stable ordering for our fixed finite test inputs.
        (c[0].to_bits(), c[1].to_bits(), c[2].to_bits())
    }

    fn sort_points(points: &mut [([f32; 3], [u8; 3])]) {
        points.sort_unstable_by(|(a_pos, a_col), (b_pos, b_col)| {
            let a_ck = color_key(*a_col);
            let b_ck = color_key(*b_col);
            match a_ck.cmp(&b_ck) {
                std::cmp::Ordering::Equal => coord_key(*a_pos).cmp(&coord_key(*b_pos)),
                other => other,
            }
        });
    }

    fn to_pairs(points: &[Point3RgbF32]) -> Vec<([f32; 3], [u8; 3])> {
        let mut out = Vec::with_capacity(points.len());
        for p in points {
            out.push((
                [p.x(), p.y(), p.z()],
                [p.r_u8(), p.g_u8(), p.b_u8()],
            ));
        }
        out
    }

    #[test]
    fn draco_roundtrip_smoke() {
        // Small, deterministic sample.
        let input: Vec<Point3RgbF32> = vec![
            Point3RgbF32::new(0.0, 0.0, 0.0, 255, 0, 0),
            Point3RgbF32::new(1.0, 2.0, 3.0, 0, 255, 0),
            Point3RgbF32::new(-4.0, 5.0, -6.0, 0, 0, 255),
            Point3RgbF32::new(10.0, -2.5, 0.125, 123, 45, 67),
        ];

        let mut params = encoder::DracoParams::default();
        if let Some(config) = &mut params.config {
            config.position_quantization_bits = 30; // Improve position accuracy for this small test set
        }


        let mut bytes = Vec::new();
        encoder::encode_from_payload_into(&input, &params, &mut bytes)
            .expect("encode must succeed");
        assert!(!bytes.is_empty());

        let mut out: Vec<Point3RgbF32> = Vec::new();
        decoder::decode_into(&bytes, &mut out).expect("decode must succeed");

        assert_eq!(out.len(), input.len());

        // Re-ordering of points is possible due to kd-tree encoding.
        // Compare as an order-independent multiset by sorting ourselves.
        let mut original = to_pairs(&input);
        let mut decoded = to_pairs(&out);

        sort_points(&mut original);
        sort_points(&mut decoded);

        // Colors should roundtrip exactly (we encode u8 RGB attributes).
        for (i, ((_, col_a), (_, col_b))) in original.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(col_a, col_b, "color mismatch at sorted index {i}");
        }

        // Coordinates are quantized -> compare with tolerance after sorting.
        let eps = 1e-4_f32;
        for (i, ((pos_a, _), (pos_b, _))) in original.iter().zip(decoded.iter()).enumerate() {
            for k in 0..3 {
                let da = pos_a[k];
                let db = pos_b[k];
                assert!(
                    (da - db).abs() <= eps,
                    "coord mismatch at sorted index {i}, axis {k}: {da} vs {db}"
                );
            }
        }
    }
}