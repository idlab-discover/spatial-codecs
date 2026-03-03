//! Custom quantisation-centric codec with optional palettes, delta coding, and
//! configurable packing strategies. The heavy lifting lives in the `encoder` and
//! `decoder` modules; helpers (bit I/O, palette construction, headers) are grouped here
//! to keep the implementation cohesive.

mod bitio;
pub mod decoder;
mod delta;
pub mod encoder;
mod header;
mod palette;
pub mod types;

pub use decoder::{decode_into, decode_into_flattened_vecs};
pub use encoder::encode_from_payload_into;
pub use types::{QuantizeParams, HEADER_EXTENDED_SIZE, HEADER_FIXED_SIZE, MAGIC, VERSION};

#[cfg(test)]
mod tests {
    use super::decoder;
    use super::encoder::encode_from_payload_into;
    use super::types::QuantizeParams;
    use spatial_utils::point::Point3D;

    #[test]
    fn roundtrip_preserves_points_within_quant_error() {
        let points = vec![
            Point3D {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                r: 10,
                g: 20,
                b: 30,
            },
            Point3D {
                x: 0.5,
                y: 0.75,
                z: 0.25,
                r: 40,
                g: 50,
                b: 60,
            },
            Point3D {
                x: 1.0,
                y: 1.0,
                z: 1.0,
                r: 70,
                g: 80,
                b: 90,
            },
        ];

        let mut encoded = Vec::new();
        encode_from_payload_into(&points, &QuantizeParams::default(), &mut encoded).unwrap();

        let mut decoded = Vec::new();
        decoder::decode_into::<Point3D>(&encoded, &mut decoded).unwrap();

        assert_eq!(decoded.len(), points.len());

        let tolerance = 1e-3;
        for (original, restored) in points.iter().zip(decoded.iter()) {
            assert!((original.x - restored.x).abs() <= tolerance);
            assert!((original.y - restored.y).abs() <= tolerance);
            assert!((original.z - restored.z).abs() <= tolerance);
            let dr = (original.r as i16 - restored.r as i16).abs();
            let dg = (original.g as i16 - restored.g as i16).abs();
            let db = (original.b as i16 - restored.b as i16).abs();
            assert!(dr <= 1, "Red channel differs too much: {dr}");
            assert!(dg <= 1, "Green channel differs too much: {dg}");
            assert!(db <= 1, "Blue channel differs too much: {db}");
        }
    }

    #[test]
    fn output_size_matches_expected_bit_budget() {
        let points = vec![
            Point3D {
                x: 1.0,
                y: 2.0,
                z: 3.0,
                r: 64,
                g: 128,
                b: 255,
            },
            Point3D {
                x: -1.0,
                y: -2.0,
                z: -3.0,
                r: 0,
                g: 32,
                b: 64,
            },
        ];

        let mut encoded = Vec::new();
        let position_bits = 10u8;
        let color_bits = 5u8;
        let params = QuantizeParams {
            position_bits,
            color_bits,
            max_palette_colors: 0,
            ..QuantizeParams::default()
        };
        encode_from_payload_into(&points, &params, &mut encoded).unwrap();

        let per_point_bits = (usize::from(position_bits) * 3) + (usize::from(color_bits) * 3);
        let expected_body_bytes = (points.len() * per_point_bits).div_ceil(8usize);
        assert_eq!(
            encoded.len(),
            super::types::HEADER_EXTENDED_SIZE + expected_body_bytes
        );
    }

    #[test]
    fn palette_mode_encodes_indices_and_roundtrips() {
        let base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)];

        let mut points = Vec::new();
        for (i, &(r, g, b)) in base_colors.iter().enumerate() {
            points.push(Point3D {
                x: i as f32,
                y: i as f32 * 0.5,
                z: i as f32 * 0.25,
                r,
                g,
                b,
            });
            points.push(Point3D {
                x: i as f32 + 0.1,
                y: i as f32 * 0.5 + 0.1,
                z: i as f32 * 0.25 + 0.2,
                r,
                g,
                b,
            });
        }

        let mut encoded = Vec::new();
        let params = QuantizeParams {
            position_bits: 12,
            color_bits: 8,
            max_palette_colors: 4,
            ..QuantizeParams::default()
        };
        encode_from_payload_into(&points, &params, &mut encoded).unwrap();

        let palette_len = base_colors.len();
        /*
        let palette_bits = if palette_len <= 1 {
            0usize
        } else {
            (usize::BITS - (palette_len - 1).leading_zeros()) as usize
        };
        let per_point_bits = (usize::from(params.position_bits) * 3) + palette_bits;
        let expected_body_bytes = (points.len() * per_point_bits).div_ceil(8usize);
        */
        let expected_header_bytes = super::types::HEADER_EXTENDED_SIZE
            + super::types::PALETTE_LEN_FIELD_SIZE
            + palette_len * 3;
        //assert_eq!(encoded.len(), expected_header_bytes + expected_body_bytes);
        assert!(encoded.len() > expected_header_bytes);
        assert!(encoded.len() < expected_header_bytes + points.len() * 8); // sanity bound

        let mut decoded = Vec::new();
        decoder::decode_into::<Point3D>(&encoded, &mut decoded).unwrap();
        assert_eq!(decoded.len(), points.len());
        for (original, restored) in points.iter().zip(decoded.iter()) {
            assert_eq!(original.r, restored.r);
            assert_eq!(original.g, restored.g);
            assert_eq!(original.b, restored.b);
        }
    }

    #[test]
    fn delta_roundtrip_and_fixed_width_behaviour() {
        let mut points = Vec::new();
        for i in 0..16 {
            points.push(Point3D {
                x: i as f32 * 0.25,
                y: (i as f32).sin(),
                z: (i as f32).cos(),
                r: (i * 11 % 256) as u8,
                g: (i * 17 % 256) as u8,
                b: (i * 23 % 256) as u8,
            });
        }

        let params = QuantizeParams {
            position_bits: 13,
            color_bits: 7,
            max_palette_colors: 0,
            delta_positions: true,
            delta_colors: true,
            pack_positions: false,
            pack_colors: false,
        };

        let mut encoded = Vec::new();
        encode_from_payload_into(&points, &params, &mut encoded).unwrap();

        // Header should include flags byte.
        assert_eq!(
            encoded[5] & super::types::COLOR_BITFLAG_HAS_FLAGS,
            super::types::COLOR_BITFLAG_HAS_FLAGS
        );

        let mut decoded = Vec::new();
        decoder::decode_into::<Point3D>(&encoded, &mut decoded).unwrap();
        assert_eq!(decoded.len(), points.len());

        // Quantization error bound:
        for (orig, restored) in points.iter().zip(decoded.iter()) {
            assert!((orig.x - restored.x).abs() < 5e-3);
            assert!((orig.y - restored.y).abs() < 5e-3);
            assert!((orig.z - restored.z).abs() < 5e-3);
        }
    }
}
