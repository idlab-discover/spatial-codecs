//! Bitcode encoder: serialises `Point3D` losslessly using the `bitcode` crate.
//!
//! The implementation is intentionally simple – it exists so other codecs can be wrapped
//! without pulling in additional crates in higher layers.

use bitcode::encode as bt_encode;
use spatial_utils::{traits::SpatialOwnedFull, utils::point_scalar::PointScalar};
use serde::{Deserialize, Serialize};
use crate::BasicResult;

use crate::codecs::bitcode::{BitcodeData, BitcodeHeader, MAGIC};

/// Placeholder parameters (kept for symmetry with other codecs).
#[repr(C)]
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct BitcodeParams {
    // no knobs (lossless passthrough of your struct)
}

/// Encode a slice of points into the bitcode format.
pub fn encode_from_payload_into<P, S>(
    payload: &[P],
    _params: &BitcodeParams,
    out: &mut Vec<u8>,
) -> BasicResult
where
    P: SpatialOwnedFull<S>,
    S: PointScalar,
{
    let hdr = BitcodeHeader::new(P::KIND, P::COLOR_KIND, P::SCALAR_KIND);
    // Fast header bytes: [ver, spatial_kind, color_kind, scalar_kind]
    let hdr_raw: [u8; 4] = [
        crate::codecs::bitcode::WIRE_HDR_V1_VERSION,
        hdr.spatial_kind,
        hdr.color_kind,
        hdr.scalar_kind,
    ];

    let hdr_len_u32: u32 = crate::codecs::bitcode::WIRE_HDR_V1_LEN as u32;
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&hdr_len_u32.to_le_bytes());
    out.extend_from_slice(&hdr_raw);

    // body stays bitcode
    let body_raw = bt_encode(&BitcodeData::<P> {
        payload: payload.to_vec(),
    });
    out.extend_from_slice(&body_raw);
    Ok(())
}
