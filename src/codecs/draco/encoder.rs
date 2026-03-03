//! Thin wrapper around the `spatial_codec_draco` crate.
//!
//! We expose a minimal subset of Draco’s configuration surface: encoding method
//! (Kd-tree vs sequential) and quantisation/compression levels. The heavy lifting is
//! performed by the FFI helper crate.

use spatial_codec_draco::{
    encode_draco_with_config, PointCloudEncodingMethod,
};
use spatial_utils::{traits::PointTraits, utils::point_scalar::PointScalar};
use serde::{Deserialize, Serialize};

#[repr(u8)]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum DracoMethod {
    Sequential = 0,
    KdTree,
}

#[repr(C)]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DracoParams {
    pub method: DracoMethod,           // default: KdTree
    pub config: Option<spatial_codec_draco::EncodeConfig>,
}

impl Default for DracoParams {
    fn default() -> Self {
        let encoding_config = spatial_codec_draco::EncodeConfig::default();
        Self {
            method: DracoMethod::KdTree,
            config: Some(encoding_config),
        }
    }
}

/// Encode `points` with the requested Draco parameters, writing the `DRA` payload into `out`.
pub fn encode_from_payload_into<P, S>(
    points: &[P],
    params: &DracoParams,
    out: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>>
where
    P: PointTraits<S>,
    S: PointScalar,
{
    // Use draco compression library

    // Convert Vec<P> to Vec<[f32; 3]> and Vec<[u8; 3]> for colors
    let n = points.len();
    let mut v: Vec<[f32; 3]> = Vec::with_capacity(n);
    let mut c: Vec<[u8; 3]> = Vec::with_capacity(n);

    for point in points.iter() {
        v.push([
            point.x().to_f32(),
            point.y().to_f32(),
            point.z().to_f32()
        ]);

        c.push([
            point.r_u8(),
            point.g_u8(),
            point.b_u8()
        ]);
    }

    let encoding_method = match params.method {
        DracoMethod::Sequential => PointCloudEncodingMethod::Sequential,
        DracoMethod::KdTree => PointCloudEncodingMethod::KdTree,
    };

    let config = params.config.unwrap_or_default();

    let compressed_data = encode_draco_with_config(&v, &c, encoding_method, &config)?;

    // The compressed data already includes the "DRA" magic header
    out.extend_from_slice(&compressed_data);

    Ok(())
}
