//! Wrappers around the OpenZL compressor (serial + columnar modes).

pub mod decoder;
pub mod encoder;

#[cfg(test)]
mod tests {
    use super::{decoder, encoder};
    use spatial_utils::point::Point3RgbF32;

    #[test]
    fn openzl_roundtrip_smoke() {
        let input: Vec<Point3RgbF32> = vec![
            Point3RgbF32::new(0.0, 0.0, 0.0, 255, 0, 0),
            Point3RgbF32::new(1.0, 2.0, 3.0, 0, 255, 0),
            Point3RgbF32::new(-4.0, 5.0, -6.0, 0, 0, 255),
            Point3RgbF32::new(10.0, -2.5, 0.125, 123, 45, 67),
        ];

        let params = encoder::OpenzlParams::columnar_default();

        let mut encoded = Vec::new();
        encoder::encode_from_payload_into::<Point3RgbF32, f32>(&input, &params, &mut encoded)
            .expect("encode must succeed");
        assert!(!encoded.is_empty());

        let mut out: Vec<Point3RgbF32> = Vec::new();
        decoder::decode_into(&encoded, &mut out).expect("decode must succeed");

        assert_eq!(out, input);
    }
}