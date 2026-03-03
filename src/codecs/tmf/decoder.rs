//! TMF decoder helpers.

use spatial_utils::{traits::SpatialSink, utils::point_scalar::PointScalar};
use crate::BasicResult;
use tmf::TMFMesh;

#[inline]
fn to_u8(v: tmf::FloatType) -> u8 {
    // clamp to [0,1], scale, round
    #[allow(clippy::unnecessary_cast)] // TMF FloatType can be f32 or f64
    let v = (v as f32).clamp(0.0, 1.0);
    (v * 255.0).round() as u8
}

/// Decode a TMF payload into `Vec<P>`.
pub fn decode_into<P>(data: &[u8], out: &mut Vec<P>) -> BasicResult
where
    P: SpatialSink,
{
    // Read one TMF mesh from memory
    let (mesh, _name) = TMFMesh::read_tmf_one(&mut &data[..])
        .map_err(|e| format!("Error decoding TMF data: {e}"))?;

    let verts = mesh.get_vertices().unwrap_or(&[]);
    if verts.is_empty() {
        return Ok(());
    }
    let n = verts.len();

    out.reserve_exact(n);

    // Detect underlying float width once
    let is_f32 = if let Some(&(x0, _, _)) = verts.first() {
        std::mem::size_of_val(&x0) == std::mem::size_of::<f32>()
    } else {
        true
    };

    // Prefer float RGBA colors if present; fall back to zeros.
    let rgba_opt = mesh
        .lookup_custom_data("colors_rgba")
        .and_then(|cdata| cdata.as_color_rgba().map(|(rgba, _prec)| rgba));

    match rgba_opt {
        Some(rgba) => {
            let m = rgba.len();
            if is_f32 {
                #[allow(clippy::unnecessary_cast)]
                for (i, &(x, y, z)) in verts.iter().enumerate() {
                    let (r, g, b) = if i < m {
                        let (rf, gf, bf, _) = rgba[i];
                        (to_u8(rf), to_u8(gf), to_u8(bf))
                    } else {
                        (0, 0, 0)
                    };
                    P::push_xyz_rgb(
                        out,
                        P::Scalar::from_f32(x as f32),
                        P::Scalar::from_f32(y as f32),
                        P::Scalar::from_f32(z as f32),
                        r,
                        g,
                        b,
                    );
                }
            } else {
                #[allow(clippy::unnecessary_cast)]
                for (i, &(x, y, z)) in verts.iter().enumerate() {
                    let (r, g, b) = if i < m {
                        let (rf, gf, bf, _) = rgba[i];
                        (to_u8(rf), to_u8(gf), to_u8(bf))
                    } else {
                        (0, 0, 0)
                    };
                    P::push_xyz_rgb(
                        out,
                        P::Scalar::from_f64(x as f64),
                        P::Scalar::from_f64(y as f64),
                        P::Scalar::from_f64(z as f64),
                        r,
                        g,
                        b,
                    );
                }
            }
        }
        None => {
            if is_f32 {
                #[allow(clippy::unnecessary_cast)]
                for &(x, y, z) in verts.iter() {
                    P::push_xyz_rgb(
                        out,
                        P::Scalar::from_f32(x as f32),
                        P::Scalar::from_f32(y as f32),
                        P::Scalar::from_f32(z as f32),
                        0,
                        0,
                        0,
                    );
                }
            } else {
                #[allow(clippy::unnecessary_cast)]
                for &(x, y, z) in verts.iter() {
                    P::push_xyz_rgb(
                        out,
                        P::Scalar::from_f64(x as f64),
                        P::Scalar::from_f64(y as f64),
                        P::Scalar::from_f64(z as f64),
                        0,
                        0,
                        0,
                    );
                }
            }
        }
    }
    Ok(())
}

/// Decode a TMF payload into flattened buffers (`pos_out`, `color_out`).
pub fn decode_into_flattened_vecs<S>(
    data: &[u8],
    pos_out: &mut Vec<S>,
    color_out: &mut Vec<u8>,
) -> BasicResult
where
    S: PointScalar,
{
    //=== Step 1: Try reading one mesh
    let (mesh, _name) = TMFMesh::read_tmf_one(&mut &data[..])
        .map_err(|e| format!("Error decoding TMF data: {}", e))?;

    //=== Step 2: Extract the vertex array and flatten into f32 coords
    let verts = mesh.get_vertices().unwrap_or(&[]);
    if verts.is_empty() {
        return Ok(());
    }

    let vertex_count = verts.len();
    pos_out.reserve_exact(vertex_count * 3);
    color_out.reserve_exact(vertex_count * 3);

    // Detect underlying float width once
    let is_f32 = if let Some(&(x0, _, _)) = verts.first() {
        std::mem::size_of_val(&x0) == std::mem::size_of::<f32>()
    } else {
        true
    };

    if is_f32 {
        #[allow(clippy::unnecessary_cast)]
        for &(x, y, z) in verts.iter() {
            pos_out.push(S::from_f32(x as f32));
            pos_out.push(S::from_f32(y as f32));
            pos_out.push(S::from_f32(z as f32));
        }
    } else {
        #[allow(clippy::unnecessary_cast)]
        for &(x, y, z) in verts.iter() {
            pos_out.push(S::from_f64(x as f64));
            pos_out.push(S::from_f64(y as f64));
            pos_out.push(S::from_f64(z as f64));
        }
    }

    //=== Step 3: Extract color data and flatten into u8s in [r1,g1,b1, r2,g2,b2, ...]
    if let Some(cdata) = mesh.lookup_custom_data("colors_rgba") {
        if let Some((rgba, _prec)) = cdata.as_color_rgba() {
            let common = rgba.len().min(vertex_count);
            for &(r, g, b, _a) in rgba.iter().take(common) {
                // clamp + round back to 0..255
                color_out.push(to_u8(r));
                color_out.push(to_u8(g));
                color_out.push(to_u8(b));
            }
            // pad if fewer colors than vertices
            for _ in common..vertex_count {
                color_out.extend_from_slice(&[0, 0, 0]);
            }
        } else {
            // wrong type under this key -> zero-fill
            for _ in 0..vertex_count {
                color_out.extend_from_slice(&[0, 0, 0]);
            }
        }
    } else {
        // No color stream -> zero-fill
        for _ in 0..vertex_count {
            color_out.extend_from_slice(&[0, 0, 0]);
        }
    }

    Ok(())
}
