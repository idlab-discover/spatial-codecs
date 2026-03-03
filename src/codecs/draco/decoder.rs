//! Draco decoder facade.
//!
//! The FFI helper returns flattened positions/colours; we expose convenience functions
//! mirroring the crate-wide decoder API.

pub use spatial_codec_draco::decode_draco;
use crate::BasicResult;

use spatial_utils::{traits::SpatialSink, utils::point_scalar::PointScalar};

/// Decode into `Vec<P>`.
pub fn decode_into<P>(data: &[u8], out: &mut Vec<P>) -> BasicResult
where
    P: SpatialSink,
{
    let mut pos_out = Vec::<P::Scalar>::new();
    let mut color_out = Vec::new();
    decode_into_flattened_vecs(data, &mut pos_out, &mut color_out)?;

    if pos_out.len() % 3 != 0 {
        return Err("Decoder: vertex buffer length is not a multiple of 3".into());
    }
    if color_out.len() != pos_out.len() {
        return Err("Decoder: color buffer length mismatch (expected 3 bytes per vertex)".into());
    }

    let n = pos_out.len() / 3;
    out.reserve(n);

    for i in 0..n {
        let x = pos_out[i * 3];
        let y = pos_out[i * 3 + 1];
        let z = pos_out[i * 3 + 2];
        let r = color_out[i * 3];
        let g = color_out[i * 3 + 1];
        let b = color_out[i * 3 + 2];

        P::push_xyz_rgb(out, x, y, z, r, g, b);
    }
    Ok(())
}

/// Decode into flattened buffers.
pub fn decode_into_flattened_vecs<S>(
    data: &[u8],
    pos_out: &mut Vec<S>,
    color_out: &mut Vec<u8>,
) -> BasicResult
where
    S: PointScalar,
{
    match decode_draco(data) {
        Ok((vertices, colors)) => {
            // Fast path when S is f32-like by size (safe here because S is either f32 or f64)
            if std::mem::size_of::<S>() == std::mem::size_of::<f32>() {
                // SAFETY: only sound when S has the same representation as f32 (true for S = f32)
                let slice: &[S] = unsafe {
                    std::slice::from_raw_parts(vertices.as_ptr() as *const S, vertices.len())
                };
                pos_out.extend_from_slice(slice);
            } else {
                pos_out.extend(vertices.iter().map(|&v| S::from_f32(v)));
            }
            color_out.extend_from_slice(&colors);
            Ok(())
        }
        Err(e) => Err(format!("Error decoding Draco data: {e}").into()),
    }
}
