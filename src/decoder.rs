//! Codec-agnostic decoder entry points.
//!
//! The decoder mirrors [`crate::encoder`], dispatching to the correct codec by inspecting
//! the magic bytes at the start of the payload. Callers can request either a `Vec<P>`
//! or flattened position/color arrays depending on their downstream needs.

use crate::BasicResult;

use spatial_utils::{point::Point3D, traits::SpatialSink, utils::point_scalar::PointScalar};

use crate::codecs::{
    bitcode as bitcodeCodec, gsplat16, gzip, lz4, ply, quantize, snappy, sogp, tmf, zstd,
};
#[cfg(feature = "draco")]
use crate::codecs::draco;
#[cfg(feature = "openzl")]
use crate::codecs::openzl;

#[inline(always)]
fn magic3(data: &[u8]) -> Result<&[u8; 3], Box<dyn std::error::Error>> {
    let head = data.get(0..3).ok_or("Data too short")?;
    Ok(head.try_into()?)
}

/// Decode a payload into `Vec<P>` using the codec-specific decoder inferred
/// from the header magic.
pub fn decode_into<P>(data: &[u8], out: &mut Vec<P>) -> BasicResult
where
    P: SpatialSink + 'static,
{
    let mag = magic3(data)?;

    match mag {
        // The standalone codecs
        b"ply" => ply::decoder::decode_into(data, out),
        #[cfg(feature = "draco")]
        b"DRA" => draco::decoder::decode_into(data, out),
        #[cfg(not(feature = "draco"))]
        b"DRA" => Err("Draco decoding not available (feature disabled)".into()),
        b"GSP" => gsplat16::decoder::decode_into(data, out),
        b"TMF" => tmf::decoder::decode_into(data, out),
        bitcodeCodec::MAGIC => bitcodeCodec::decoder::decode_into(data, out),
        b"QNT" => quantize::decoder::decode_into(data, out),
        b"SGP" => sogp::decoder::decode_into(data, out),
        // Codecs that can be both standalone and a wrapper
        #[cfg(feature = "openzl")]
        b"OZL" => openzl::decoder::decode_into(data, out),
        #[cfg(not(feature = "openzl"))]
        b"OZL" => Err("OpenZL decoding not available (feature disabled)".into()),
        // The wrapper codecs
        b"GZP" => gzip::decoder::decode_into(data, out),
        b"ZST" => zstd::decoder::decode_into(data, out),
        b"LZ4" => lz4::decoder::decode_into(data, out),
        b"SNP" => snappy::decoder::decode_into(data, out),
        _ => Err("Unsupported data format".into()),
    }
}

/// Decode into flattened XYZ/colour arrays, preserving interleaving expectations
/// of codecs that naturally produce structure-of-arrays output.
pub fn decode_into_flattened_vecs<S>(
    data: &[u8],
    pos_out: &mut Vec<S>,
    color_out: &mut Vec<u8>,
) -> BasicResult
where
    S: PointScalar,
{
    let magic = magic3(data)?;
    match magic {
        // The standalone codecs
        b"ply" => ply::decoder::decode_into_flattened_vecs(data, pos_out, color_out),
        #[cfg(feature = "draco")]
        b"DRA" => draco::decoder::decode_into_flattened_vecs(data, pos_out, color_out),
        #[cfg(not(feature = "draco"))]
        b"DRA" => Err("Draco decoding not available (feature disabled)".into()),
        b"GSP" => gsplat16::decoder::decode_into_flattened_vecs(data, pos_out, color_out),
        b"TMF" => tmf::decoder::decode_into_flattened_vecs(data, pos_out, color_out),
        bitcodeCodec::MAGIC => {
            bitcodeCodec::decoder::decode_into_flattened_vecs(data, pos_out, color_out)
        }
        b"QNT" => quantize::decoder::decode_into_flattened_vecs(data, pos_out, color_out),
        b"SGP" => sogp::decoder::decode_into_flattened_vecs(data, pos_out, color_out),
        // Codecs that can be both standalone and a wrapper
        #[cfg(feature = "openzl")]
        b"OZL" => openzl::decoder::decode_into_flattened_vecs(data, pos_out, color_out),
        #[cfg(not(feature = "openzl"))]
        b"OZL" => Err("OpenZL decoding not available (feature disabled)".into()),
        // The wrapper codecs
        b"GZP" => gzip::decoder::decode_into_flattened_vecs(data, pos_out, color_out),
        b"ZST" => zstd::decoder::decode_into_flattened_vecs(data, pos_out, color_out),
        b"LZ4" => lz4::decoder::decode_into_flattened_vecs(data, pos_out, color_out),
        b"SNP" => snappy::decoder::decode_into_flattened_vecs(data, pos_out, color_out),
        _ => Err("Unsupported data format".into()),
    }
}

/// Convenience wrapper returning a freshly allocated `Vec<Point3D>`.
pub fn decode_to_points_vec(data: &[u8]) -> Result<Vec<Point3D>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    decode_into(data, &mut out)?;
    Ok(out)
}

/// Convenience wrapper returning freshly allocated flattened buffers.
pub fn decode_to_flattened_vecs(
    data: &[u8],
) -> Result<(Vec<f32>, Vec<u8>), Box<dyn std::error::Error>> {
    let mut pos_out = Vec::new();
    let mut color_out = Vec::new();
    decode_into_flattened_vecs(data, &mut pos_out, &mut color_out)?;
    Ok((pos_out, color_out))
}

#[cfg(test)]
mod tests {

    use super::*;
    use spatial_utils::point::Point3D;

    #[test]
    fn magic3_errors_on_short_input() {
        assert!(magic3(&[]).is_err());
        assert!(magic3(&[0x00, 0x01]).is_err());
    }

    #[test]
    fn decode_rejects_unknown_magic() {
        let mut out: Vec<Point3D> = Vec::new();
        let data = b"???not a real codec";
        assert!(decode_into::<Point3D>(data, &mut out).is_err());
    }

    #[test]
    fn decode_flattened_rejects_unknown_magic() {
        let mut pos_out = Vec::<f32>::new();
        let mut color_out = Vec::new();
        let data = b"???not a real codec";
        assert!(decode_into_flattened_vecs(data, &mut pos_out, &mut color_out).is_err());
    }
}
