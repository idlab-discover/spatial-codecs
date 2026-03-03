//! PLY encoder/decoder implementation (ASCII + binary variants).

pub mod ascii;
pub mod binary;
pub mod decoder;
pub mod encoder;
pub mod header;
pub mod types;


// A small test
#[cfg(test)]
mod ply_tests {
    use spatial_utils::{color::Rgba8, point::{Point3RgbF32, Point3RgbaF32}, splat::GaussianSplatF32};

    use crate::codecs::ply::{decoder, encoder, types::{AsciiFloatMode, PlyEncoding, ScalarType}};


    fn params_for(enc: PlyEncoding) -> encoder::PlyParams {
        encoder::PlyParams {
            encoding: enc,
            coord_scalar: Some(ScalarType::Float),
            write_color: Some(true),
            write_alpha: None,
            comments: Some(vec![]),
            ascii_float_mode: Some(AsciiFloatMode::Shortest),
        }
    }
    
    #[test]
    fn ply_roundtrip_points_rgb() {
        let input: Vec<Point3RgbF32> = vec![
            Point3RgbF32::new(1.0, 2.5, -3.75, 10, 20, 30),
            Point3RgbF32::new(-4.0, 0.5, 6.0, 40, 50, 60),
        ];
    
        for enc in [PlyEncoding::Ascii, PlyEncoding::BinaryLittleEndian, PlyEncoding::BinaryBigEndian] {
            let p = params_for(enc);
            let mut bytes = Vec::new();
            encoder::encode_from_payload_into::<Point3RgbF32, f32>(&input, &p, &mut bytes).unwrap();
    
            let mut out: Vec<Point3RgbF32> = Vec::new();
            decoder::decode_into::<Point3RgbF32>(&bytes, &mut out).unwrap();
            assert_eq!(out, input);
        }
    }

    #[test]
    fn ply_roundtrip_points_rgba() {
        let input: Vec<Point3RgbaF32> = vec![
            Point3RgbaF32::new(1.0, 2.0, 3.0, 1, 2, 3, 4),
            Point3RgbaF32::new(-4.0, 0.5, 6.0, 254, 128, 0, 255),
        ];
    
        for enc in [PlyEncoding::Ascii, PlyEncoding::BinaryLittleEndian, PlyEncoding::BinaryBigEndian] {
            let p = params_for(enc);
            let mut bytes = Vec::new();
            encoder::encode_from_payload_into::<Point3RgbaF32, f32>(&input, &p, &mut bytes).unwrap();

            let mut out: Vec<Point3RgbaF32> = Vec::new();
            decoder::decode_into::<Point3RgbaF32>(&bytes, &mut out).unwrap();
            assert_eq!(out, input);
        }
   }
    fn assert_u8_close(a: u8, b: u8, tol: u8) {
        let da = a.abs_diff(b);
        assert!(da <= tol, "u8 mismatch: {a} vs {b} (tol={tol})");
    }

    fn assert_f32_close(a: f32, b: f32, eps: f32) {
        let d = (a - b).abs();
        assert!(d <= eps, "f32 mismatch: {a} vs {b} (|d|={d}, eps={eps})");
    }

    #[test]
    fn ply_roundtrip_splats() {
        let input: Vec<GaussianSplatF32> = vec![
            GaussianSplatF32::new(
                [1.0, 2.0, 3.0],
                Rgba8::new(10, 20, 30, 40),
                [0.10, 0.20, 0.30],
                [1.0, 0.0, 0.0, 0.0],
            ),
            GaussianSplatF32::new(
                [-4.0, 0.5, 6.0],
                Rgba8::new(254, 128, 0, 255),
                [1.0, 1.5, 2.0],
                [0.70710677, 0.70710677, 0.0, 0.0],
            ),
        ];

        for enc in [PlyEncoding::Ascii, PlyEncoding::BinaryLittleEndian, PlyEncoding::BinaryBigEndian] {
            let p = params_for(enc);
            let mut bytes = Vec::new();
            encoder::encode_from_payload_into::<GaussianSplatF32, f32>(&input, &p, &mut bytes).unwrap();

            let mut out: Vec<GaussianSplatF32> = Vec::new();
            decoder::decode_into::<GaussianSplatF32>(&bytes, &mut out).unwrap();
            assert_eq!(out.len(), input.len());

            for (a, b) in out.iter().zip(input.iter()) {
                // mean should be very close (exact for binary, close for ascii)
                assert_f32_close(a.mean[0], b.mean[0], 1e-4);
                assert_f32_close(a.mean[1], b.mean[1], 1e-4);
                assert_f32_close(a.mean[2], b.mean[2], 1e-4);

                // dc/logit mappings can round by ~1
                assert_u8_close(a.rgba.r, b.rgba.r, 1);
                assert_u8_close(a.rgba.g, b.rgba.g, 1);
                assert_u8_close(a.rgba.b, b.rgba.b, 1);
                assert_u8_close(a.rgba.a, b.rgba.a, 1);

                // scale is ln/exp roundtrip -> allow small relative-ish error
                assert_f32_close(a.scale[0], b.scale[0], 2e-3);
                assert_f32_close(a.scale[1], b.scale[1], 2e-3);
                assert_f32_close(a.scale[2], b.scale[2], 2e-3);

                // rotation should be close
                assert_f32_close(a.rotation[0], b.rotation[0], 1e-4);
                assert_f32_close(a.rotation[1], b.rotation[1], 1e-4);
                assert_f32_close(a.rotation[2], b.rotation[2], 1e-4);
                assert_f32_close(a.rotation[3], b.rotation[3], 1e-4);
            }
        }
    }
}