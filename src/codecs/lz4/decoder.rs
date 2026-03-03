//! LZ4 wrapper decoder utilities.

use lz4_flex::frame::FrameDecoder;
use spatial_utils::{traits::SpatialSink, utils::point_scalar::PointScalar};
use crate::BasicResult;
use std::io::Read;

fn payload(data: &[u8]) -> Result<&[u8], Box<dyn std::error::Error>> {
    if data.len() < 3 || &data[0..3] != b"LZ4" {
        return Err("Invalid LZ4 header".into());
    }
    Ok(&data[3..])
}

/// Decompress into `Vec<Point3D>` using the inner codec.
pub fn decode_into<P>(data: &[u8], out: &mut Vec<P>) -> BasicResult
where
    P: SpatialSink + 'static,
{
    let mut dec = FrameDecoder::new(payload(data)?);
    let mut inner = Vec::new();
    dec.read_to_end(&mut inner)?;
    crate::decoder::decode_into(&inner, out)
}

/// Decompress into flattened buffers.
pub fn decode_into_flattened_vecs<S>(
    data: &[u8],
    pos_out: &mut Vec<S>,
    color_out: &mut Vec<u8>,
) -> BasicResult
where
    S: PointScalar,
{
    let mut dec = FrameDecoder::new(payload(data)?);
    let mut inner = Vec::new();
    dec.read_to_end(&mut inner)?;
    crate::decoder::decode_into_flattened_vecs(&inner, pos_out, color_out)
}
