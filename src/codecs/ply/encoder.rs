//! High‑performance writer. Minimal header + contiguous body. Supports ASCII, BLE, BBE.
use crate::codecs::ply::header::{write_minimal_header, WriteHeaderOpts};
use crate::codecs::ply::types::*;
use spatial_utils::traits::{ColorKind, SpatialKind, SpatialMeta, SpatialOwnedFull};
use spatial_utils::utils::point_scalar::PointScalar;
use serde::{Deserialize, Serialize};
use std::io;

const DEFAULT_COORD_SCALAR: ScalarType = ScalarType::Float;
const DEFAULT_WRITE_COLOR: bool = true;
const DEFAULT_COMMENTS: Vec<String> = vec![];
const DEFAULT_ASCII_MODE: AsciiFloatMode = AsciiFloatMode::Shortest;

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlyParams {
    /// ascii | binary_little_endian | binary_big_endian
    pub encoding: PlyEncoding,
    /// Coordinate scalar type for x/y/z when writing (Float|Double). Defaults to Float.
    pub coord_scalar: Option<ScalarType>,
    /// Include rgb as uchar properties when true.
    pub write_color: Option<bool>,
    /// Include alpha as uchar when writing point payloads.
    /// When `None`, alpha is written automatically if the payload color kind is RGBA.
    pub write_alpha: Option<bool>,
    /// Comment lines to include in header.
    pub comments: Option<Vec<String>>,
    /// ASCII float should have a fixed precision, or shortest representation.
    pub ascii_float_mode: Option<AsciiFloatMode>,
}

impl Default for PlyParams {
    fn default() -> Self {
        Self {
            encoding: PlyEncoding::BinaryLittleEndian,
            coord_scalar: Some(DEFAULT_COORD_SCALAR),
            write_color: Some(DEFAULT_WRITE_COLOR),
            write_alpha: None,
            comments: Some(DEFAULT_COMMENTS),
            ascii_float_mode: Some(DEFAULT_ASCII_MODE),
        }
    }
}

pub fn encode_from_payload_into<P, S>(
    points: &[P],
    params: &PlyParams,
    out: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>>
where
    P: SpatialOwnedFull<S>,
    S: PointScalar,
{
    // Collect views avoiding copies.
    let n = points.len();
    let coord_scalar = params.coord_scalar.unwrap_or(DEFAULT_COORD_SCALAR);
    if !matches!(coord_scalar, ScalarType::Float | ScalarType::Double) {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidInput,
            "coord_scalar must be Float or Double for PLY encoding",
        )));
    }
    let coords_are_double = matches!(coord_scalar, ScalarType::Double);

    // Points: respect write_color + optional alpha.
    // Splats: always write splat fields (color comes from f_dc + opacity).
    let write_color = params.write_color.unwrap_or(DEFAULT_WRITE_COLOR);
    let write_alpha = params
        .write_alpha
        .unwrap_or(matches!(<P as SpatialMeta>::COLOR_KIND, ColorKind::Rgba8));
    // Header
     let mut vprops: Vec<PropertyDef> = Vec::new();
    match <P as SpatialMeta>::KIND {
        SpatialKind::Points => {
            // Canonical write order:
            // x y z red green blue (alpha)
            vprops.push(PropertyDef { name: "x".into(), ty: PropertyType::Scalar(coord_scalar) });
            vprops.push(PropertyDef { name: "y".into(), ty: PropertyType::Scalar(coord_scalar) });
            vprops.push(PropertyDef { name: "z".into(), ty: PropertyType::Scalar(coord_scalar) });
            if write_color {
                vprops.push(PropertyDef { name: "red".into(), ty: PropertyType::Scalar(ScalarType::UChar) });
                vprops.push(PropertyDef { name: "green".into(), ty: PropertyType::Scalar(ScalarType::UChar) });
                vprops.push(PropertyDef { name: "blue".into(), ty: PropertyType::Scalar(ScalarType::UChar) });
                if write_alpha {
                    vprops.push(PropertyDef { name: "alpha".into(), ty: PropertyType::Scalar(ScalarType::UChar) });
                }
            }
        }
        SpatialKind::Splats => {
            // Canonical write order:
            // x y z f_dc_0 f_dc_1 f_dc_2 opacity rot_0..3 scale_0..2
            vprops.push(PropertyDef { name: "x".into(), ty: PropertyType::Scalar(coord_scalar) });
            vprops.push(PropertyDef { name: "y".into(), ty: PropertyType::Scalar(coord_scalar) });
            vprops.push(PropertyDef { name: "z".into(), ty: PropertyType::Scalar(coord_scalar) });
            vprops.push(PropertyDef { name: "f_dc_0".into(), ty: PropertyType::Scalar(coord_scalar) });
            vprops.push(PropertyDef { name: "f_dc_1".into(), ty: PropertyType::Scalar(coord_scalar) });
            vprops.push(PropertyDef { name: "f_dc_2".into(), ty: PropertyType::Scalar(coord_scalar) });
            vprops.push(PropertyDef { name: "opacity".into(), ty: PropertyType::Scalar(coord_scalar) });
            vprops.push(PropertyDef { name: "rot_0".into(), ty: PropertyType::Scalar(coord_scalar) });
            vprops.push(PropertyDef { name: "rot_1".into(), ty: PropertyType::Scalar(coord_scalar) });
            vprops.push(PropertyDef { name: "rot_2".into(), ty: PropertyType::Scalar(coord_scalar) });
            vprops.push(PropertyDef { name: "rot_3".into(), ty: PropertyType::Scalar(coord_scalar) });
            vprops.push(PropertyDef { name: "scale_0".into(), ty: PropertyType::Scalar(coord_scalar) });
            vprops.push(PropertyDef { name: "scale_1".into(), ty: PropertyType::Scalar(coord_scalar) });
            vprops.push(PropertyDef { name: "scale_2".into(), ty: PropertyType::Scalar(coord_scalar) });
        }
    }

    // Reserve: header is tiny; body dominates. Use a good stride estimate to avoid reallocs.
    let scalar_sz = if coords_are_double { 8usize } else { 4usize };
    let est_body_bytes = match <P as SpatialMeta>::KIND {
        SpatialKind::Points => {
            let mut stride = 3 * scalar_sz;
            if write_color {
                stride += 3;
                if write_alpha {
                    stride += 1;
                }
            }
            n * stride
        }
        SpatialKind::Splats => {
            // 14 scalar fields.
            n * (14 * scalar_sz)
        }
    };
    out.reserve(est_body_bytes + 256);

    {
        let mut cursor = io::Cursor::new(out);
        write_minimal_header(
            &mut cursor,
            WriteHeaderOpts {
                encoding: params.encoding,
                version: "1.0",
                comments: params.comments.clone().unwrap_or(DEFAULT_COMMENTS).as_ref(),
                vertex_properties: &vprops,
                vertex_count: n,
            },
        )?;
        // Body
        match params.encoding {
            PlyEncoding::Ascii => {
                match <P as SpatialMeta>::KIND {
                    SpatialKind::Points => super::ascii::write_vertices_from_payload_points(
                        &mut cursor,
                        coords_are_double,
                        points,
                        write_color,
                        write_alpha,
                        params.ascii_float_mode.unwrap_or(DEFAULT_ASCII_MODE),
                    )?,
                    SpatialKind::Splats => super::ascii::write_vertices_from_payload_splats(
                        &mut cursor,
                        coords_are_double,
                        points,
                        params.ascii_float_mode.unwrap_or(DEFAULT_ASCII_MODE),
                    )?,
                }
            }
            PlyEncoding::BinaryLittleEndian | PlyEncoding::BinaryBigEndian => {
                match <P as SpatialMeta>::KIND {
                    SpatialKind::Points => super::binary::write_vertices_from_payload_points(
                        &mut cursor,
                        params.encoding,
                        coords_are_double,
                        points,
                        write_color,
                        write_alpha,
                    )?,
                    SpatialKind::Splats => super::binary::write_vertices_from_payload_splats(
                        &mut cursor,
                        params.encoding,
                        coords_are_double,
                        points,
                    )?,
                }
            }
        }
    }
    Ok(())
}
