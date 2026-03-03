//! Integration with the TMF point-cloud format.
//! The encoder maps `Point3D` into a TMF mesh with vertex colours and writes the TMF stream
//! with default precision parameters.

use spatial_utils::{traits::PointTraits, utils::point_scalar::PointScalar};
use serde::{Deserialize, Serialize};
use tmf::{FloatType, TMFMesh, TMFPrecisionInfo};

#[repr(C)]
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct TmfParams {
    // keep for future precision knobs; default TMFPrecisionInfo today
}

/// Encode points into the TMF format, appending to `out`.
pub fn encode_from_payload_into<P, S>(
    points: &[P],
    _params: &TmfParams,
    out: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>>
where
    P: PointTraits<S>,
    S: PointScalar,
{
    let mut mesh = TMFMesh::empty();
    // Pre-allocate capacity
    let n = points.len();
    let mut vertices = Vec::with_capacity(n);
    let mut colors_rgba = Vec::with_capacity(n);

    // Convert each point into a Vector3 for the mesh and a u32 for the color
    for p in points {
        // Positions go in mesh vertices
        vertices.push((
            p.x().to_f32() as FloatType,
            p.y().to_f32() as FloatType,
            p.z().to_f32() as FloatType,
        ));

        // Normalize to 0..1 and include alpha=1.0
        colors_rgba.push((
            (p.r_u8() as FloatType) / 255.0,
            (p.g_u8() as FloatType) / 255.0,
            (p.b_u8() as FloatType) / 255.0,
            1.0 as FloatType,
        ));
    }

    // Assign them to the mesh
    mesh.set_vertices(vertices);

    // Add the RGBA array as custom data
    // The name "point_colors" can be any nonempty string
    mesh.add_custom_data(colors_rgba[..].into(), "colors_rgba")
        .expect("Could not add colors to tmf mesh!");

    let precision_info = TMFPrecisionInfo::default();

    // Reserve at least 10 bytes per point as a guess
    // The 10 bytes is a rough estimate obtained from previous experiments
    // In those experiments, TMF produced around 77 bits per point, or 9.6 bytes per point
    out.reserve_exact(n * 10);
    mesh.write_tmf_one(out, &precision_info, "pc")?;
    Ok(())
}
