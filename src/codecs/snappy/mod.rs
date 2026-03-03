//! Snappy wrapper codec (framed as `SNP`).

pub mod decoder;
pub mod encoder;

#[cfg(test)]
mod tests {
    use super::{decoder, encoder};
    use crate::codecs::bitcode::{encoder as bitcode_encoder};
    use spatial_utils::point::Point3RgbF32;

    #[test]
    fn snappy_wrap_roundtrip_smoke() {
        let input: Vec<Point3RgbF32> = vec![
            Point3RgbF32::new(0.0, 0.0, 0.0, 255, 0, 0),
            Point3RgbF32::new(1.0, 2.0, 3.0, 0, 255, 0),
            Point3RgbF32::new(-4.0, 5.0, -6.0, 0, 0, 255),
            Point3RgbF32::new(10.0, -2.5, 0.125, 123, 45, 67),
        ];

        let mut wrapped = Vec::new();
        encoder::wrap_snappy_into(|buf| {
            bitcode_encoder::encode_from_payload_into(&input, &bitcode_encoder::BitcodeParams::default(), buf)
            },&mut wrapped).expect("snappy wrap must succeed");
        assert!(!wrapped.is_empty());

        let mut out: Vec<Point3RgbF32> = Vec::new();
        decoder::decode_into(&wrapped, &mut out).expect("decode must succeed");

        assert_eq!(out, input);
    }
}