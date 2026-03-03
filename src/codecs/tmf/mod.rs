//! TMF codec integration.

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
    fn tmf_roundtrip_smoke() {
        let input: Vec<Point3RgbF32> = vec![
            Point3RgbF32::new(0.0, 0.0, 0.0, 255, 0, 0),
            Point3RgbF32::new(1.0, 2.0, 3.0, 0, 255, 0),
            Point3RgbF32::new(-4.0, 5.0, -6.0, 0, 0, 255),
            Point3RgbF32::new(5.0, -2.5, 0.125, 123, 45, 67),
        ];

        let params = encoder::TmfParams::default();

        let mut encoded = Vec::new();
        encoder::encode_from_payload_into::<Point3RgbF32, f32>(&input, &params, &mut encoded)
            .expect("encode must succeed");
        assert!(!encoded.is_empty());

        let mut out: Vec<Point3RgbF32> = Vec::new();
        decoder::decode_into(&encoded, &mut out).expect("decode must succeed");

        assert_eq!(out.len(), input.len());

        let mut original = to_pairs(&input);
        let mut decoded = to_pairs(&out);

        sort_points(&mut original);
        sort_points(&mut decoded);

        // Compare with some tolerance
        let pos_eps = 1e-2_f32;
        let col_eps = 1; // Exact match for colors (u8), but allow for any ordering issues
        for (i, ((pos_a, col_a), (pos_b, col_b))) in original.iter().zip(decoded.iter()).enumerate() {
            for k in 0..3 {
                let da = pos_a[k];
                let db = pos_b[k];
                assert!(
                    (da - db).abs() <= pos_eps,
                    "coord mismatch at sorted index {i}, axis {k}: {da} vs {db}"
                );
                let ca = col_a[k];
                let cb = col_b[k];
                assert!(
                    (ca as i32 - cb as i32).abs() <= col_eps,
                    "color mismatch at sorted index {i}, channel {k}: {ca} vs {cb}"
                );
            }
        }
    }
}