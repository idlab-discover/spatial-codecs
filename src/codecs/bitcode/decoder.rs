use core::any::TypeId;

use crate::BasicResult;

use spatial_utils::{
    point::{Point3Rgb, Point3Rgba},
    splat::GaussianSplatF32,
    traits::{
        map_color_kind, map_spatial_kind, ColorKind, SpatialSink, HasPosition3, HasRgb8u,
        HasRgba8u, SpatialKind,
    },
    utils::point_scalar::{map_scalar_kind, PointScalar, PointScalarKind},
};

use crate::codecs::bitcode::{
    BitcodeData, BitcodeHeader, MAGIC, MAX_HEADER_BYTES, WIRE_HDR_V1_LEN, WIRE_HDR_V1_VERSION,
};

/// Conservative safety cap for body size; tweak if you need.
/// Keeps you safe from allocating insane Vecs on hostile inputs.
const MAX_BODY_BYTES: usize = 1_500_000_000;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum WireFormat {
    PointsRgbF32,
    PointsRgbaF32,
    PointsRgbF64,
    PointsRgbaF64,
    SplatsRgbaF32,
}

#[inline(always)]
fn parse_u32_le_at(data: &[u8], off: usize) -> Result<u32, Box<dyn std::error::Error>> {
    let slice = data.get(off..off + 4).ok_or("bitcode: truncated u32")?;
    let mut b = [0u8; 4];
    b.copy_from_slice(slice);
    Ok(u32::from_le_bytes(b))
}

#[inline(always)]
fn decode_header_bytes(hdr_bytes: &[u8]) -> Result<BitcodeHeader, Box<dyn std::error::Error>> {
    if hdr_bytes.len() == WIRE_HDR_V1_LEN {
        if hdr_bytes[0] != WIRE_HDR_V1_VERSION {
            return Err("bitcode: unsupported header version".into());
        }
        return Ok(BitcodeHeader {
            spatial_kind: hdr_bytes[1],
            color_kind: hdr_bytes[2],
            scalar_kind: hdr_bytes[3],
        });
    }

    // fallback path for legacy/forward-compat headers
    bitcode::decode::<BitcodeHeader>(hdr_bytes)
        .map_err(|e| format!("bitcode: failed to decode header: {e}").into())
}

#[inline(always)]
fn parse_frame(
    data: &[u8],
) -> Result<(BitcodeHeader, &[u8]), Box<dyn std::error::Error>> {
    // Need MAGIC + u32(header_len) at least.
    if data.len() < MAGIC.len() + 4 {
        return Err("bitcode: truncated BC1 frame".into());
    }
    if &data[..MAGIC.len()] != MAGIC {
        return Err("bitcode: invalid BC1 magic".into());
    }

    let hdr_len = parse_u32_le_at(data, MAGIC.len())? as usize;
    if hdr_len == 0 || hdr_len > MAX_HEADER_BYTES {
        return Err("bitcode: invalid header length".into());
    }

    let hdr_start = MAGIC.len() + 4;
    let hdr_end = hdr_start
        .checked_add(hdr_len)
        .ok_or("bitcode: header length overflow")?;
    let hdr_bytes = data
        .get(hdr_start..hdr_end)
        .ok_or("bitcode: truncated header")?;

    let body = data.get(hdr_end..).ok_or("bitcode: truncated body")?;
    if body.len() > MAX_BODY_BYTES {
        return Err("bitcode: body too large".into());
    }

    let hdr = decode_header_bytes(hdr_bytes)?;
    Ok((hdr, body))
}

#[inline(always)]
fn wire_format_from_header(hdr: BitcodeHeader) -> Result<WireFormat, Box<dyn std::error::Error>> {
    let pk = map_spatial_kind(hdr.spatial_kind).ok_or("bitcode: unknown spatial kind")?;
    let ck = map_color_kind(hdr.color_kind).ok_or("bitcode: unknown color kind")?;
    let sk = map_scalar_kind(hdr.scalar_kind).ok_or("bitcode: unknown scalar kind")?;

    match (pk, ck, sk) {
        (SpatialKind::Points, ColorKind::Rgb8, PointScalarKind::F32) => {
            Ok(WireFormat::PointsRgbF32)
        }
        (SpatialKind::Points, ColorKind::Rgba8, PointScalarKind::F32) => {
            Ok(WireFormat::PointsRgbaF32)
        }
        (SpatialKind::Points, ColorKind::Rgb8, PointScalarKind::F64) => {
            Ok(WireFormat::PointsRgbF64)
        }
        (SpatialKind::Points, ColorKind::Rgba8, PointScalarKind::F64) => {
            Ok(WireFormat::PointsRgbaF64)
        }
        (SpatialKind::Splats, ColorKind::Rgba8, PointScalarKind::F32) => {
            Ok(WireFormat::SplatsRgbaF32)
        }

        (SpatialKind::Splats, _, PointScalarKind::F64) => {
            Err("bitcode: splats(f64) not supported on-wire yet".into())
        }
        (SpatialKind::Splats, ColorKind::Rgb8, _) => {
            Err("bitcode: splats must be RGBA on-wire".into())
        }
    }
}

#[inline(always)]
fn decode_body<T>(body: &[u8]) -> Result<BitcodeData<T>, Box<dyn std::error::Error>>
where
    for<'a> T: bitcode::Decode<'a>,
{
    bitcode::decode::<BitcodeData<T>>(body)
        .map_err(|e| format!("bitcode: decode body failed: {e}").into())
}

/// If `P` is exactly `T`, extend `out` with `payload` without per-element conversion.
///
/// Safety: we only cast `Vec<P>` to `Vec<T>` when `TypeId` matches exactly.
#[inline(always)]
fn extend_if_exact<P, T>(out: &mut Vec<P>, payload: Vec<T>) -> Option<Vec<T>>
where
    P: 'static,
    T: 'static,
{
    if TypeId::of::<P>() == TypeId::of::<T>() {
        // SAFETY: TypeId equality implies P == T. Vec layouts identical.
        let out_t: &mut Vec<T> = unsafe { &mut *(out as *mut Vec<P> as *mut Vec<T>) };
        out_t.extend(payload);
        None
    } else {
        Some(payload)
    }
}

#[inline(always)]
fn emit_points_to_target<POut, SIn, VWire>(body: &[u8], out: &mut Vec<POut>) -> BasicResult
where
    POut: SpatialSink + 'static,
    SIn: PointScalar,
    VWire: HasPosition3<SIn> + HasRgba8u + for<'a> bitcode::Decode<'a> + 'static,
{
    let decoded = decode_body::<VWire>(body)?;
    out.reserve(decoded.payload.len());

    // exact-type shortcut (no rebuild, no conversion loop)
    if let Some(payload) = extend_if_exact::<POut, VWire>(out, decoded.payload) {
        out.reserve(payload.len());
        for v in payload {
            // Always use RGBA push: output types that don’t store alpha will drop it,
            // and RGB wire types return a=255 via HasRgba8u.
            POut::push_xyz_rgba(
                out,
                POut::Scalar::from_scalar(v.x()),
                POut::Scalar::from_scalar(v.y()),
                POut::Scalar::from_scalar(v.z()),
                v.r_u8(),
                v.g_u8(),
                v.b_u8(),
                v.a_u8(),
            );
        }
    }
    Ok(())
}

#[inline(always)]
fn emit_points_to_flat<SOut, SIn, VWire>(
    body: &[u8],
    pos_out: &mut Vec<SOut>,
    color_out: &mut Vec<u8>,
) -> BasicResult
where
    SOut: PointScalar,
    SIn: PointScalar,
    VWire: HasPosition3<SIn> + HasRgb8u + for<'a> bitcode::Decode<'a>,
{
    let decoded = decode_body::<VWire>(body)?;
    pos_out.reserve(decoded.payload.len() * 3);
    color_out.reserve(decoded.payload.len() * 3);

    for v in decoded.payload {
        pos_out.push(SOut::from_scalar(v.x()));
        pos_out.push(SOut::from_scalar(v.y()));
        pos_out.push(SOut::from_scalar(v.z()));
        color_out.push(v.r_u8());
        color_out.push(v.g_u8());
        color_out.push(v.b_u8());
    }
    Ok(())
}

#[inline(always)]
fn emit_splats_f32_to_target<POut>(body: &[u8], out: &mut Vec<POut>) -> BasicResult
where
    POut: SpatialSink + 'static,
{
    let decoded = bitcode::decode::<BitcodeData<GaussianSplatF32>>(body)
        .map_err(|e| format!("bitcode: decode body failed: {e}"))?;

    // exact-type shortcut (no rebuild, no conversion loop)
    if let Some(payload) = extend_if_exact::<POut, GaussianSplatF32>(out, decoded.payload) {
        out.reserve(payload.len());
        for s in payload {
            let mean = [
                POut::Scalar::from_scalar(s.mean[0]),
                POut::Scalar::from_scalar(s.mean[1]),
                POut::Scalar::from_scalar(s.mean[2]),
            ];
            let scale = [
                POut::Scalar::from_scalar(s.scale[0]),
                POut::Scalar::from_scalar(s.scale[1]),
                POut::Scalar::from_scalar(s.scale[2]),
            ];
            let rot = [
                POut::Scalar::from_scalar(s.rotation[0]),
                POut::Scalar::from_scalar(s.rotation[1]),
                POut::Scalar::from_scalar(s.rotation[2]),
                POut::Scalar::from_scalar(s.rotation[3]),
            ];
            let rgba = [s.rgba.r, s.rgba.g, s.rgba.b, s.rgba.a];
            POut::push_splat(out, mean, rgba, scale, rot);
        }
    }
    Ok(())
}

#[inline(always)]
fn emit_splats_f32_to_flat<SOut>(
    body: &[u8],
    pos_out: &mut Vec<SOut>,
    color_out: &mut Vec<u8>,
) -> BasicResult
where
    SOut: PointScalar,
{
    let decoded = decode_body::<GaussianSplatF32>(body)?;
    pos_out.reserve(decoded.payload.len() * 3);
    color_out.reserve(decoded.payload.len() * 3);

    for s in decoded.payload {
        pos_out.push(SOut::from_f32(s.mean[0]));
        pos_out.push(SOut::from_f32(s.mean[1]));
        pos_out.push(SOut::from_f32(s.mean[2]));
        color_out.push(s.rgba.r);
        color_out.push(s.rgba.g);
        color_out.push(s.rgba.b);
    }
    Ok(())
}

/// Decode into `SpatialSink` instances, appending to `out`.
pub fn decode_into<P>(data: &[u8], out: &mut Vec<P>) -> BasicResult
where
    P: SpatialSink + 'static,
{
    let (hdr, body) = parse_frame(data)?;
    let wf = wire_format_from_header(hdr)?;

    match wf {
        WireFormat::PointsRgbF32 => emit_points_to_target::<P, f32, Point3Rgb<f32>>(body, out),
        WireFormat::PointsRgbaF32 => emit_points_to_target::<P, f32, Point3Rgba<f32>>(body, out),
        WireFormat::PointsRgbF64 => emit_points_to_target::<P, f64, Point3Rgb<f64>>(body, out),
        WireFormat::PointsRgbaF64 => emit_points_to_target::<P, f64, Point3Rgba<f64>>(body, out),
        WireFormat::SplatsRgbaF32 => emit_splats_f32_to_target::<P>(body, out),
    }
}

/// Decode into flattened position/colour buffers (RGB only).
pub fn decode_into_flattened_vecs<S>(
    data: &[u8],
    pos_out: &mut Vec<S>,
    color_out: &mut Vec<u8>,
) -> BasicResult
where
    S: PointScalar,
{
    let (hdr, body) = parse_frame(data)?;
    let wf = wire_format_from_header(hdr)?;

    match wf {
        WireFormat::PointsRgbF32 => {
            emit_points_to_flat::<S, f32, Point3Rgb<f32>>(body, pos_out, color_out)
        }
        WireFormat::PointsRgbaF32 => {
            emit_points_to_flat::<S, f32, Point3Rgba<f32>>(body, pos_out, color_out)
        }
        WireFormat::PointsRgbF64 => {
            emit_points_to_flat::<S, f64, Point3Rgb<f64>>(body, pos_out, color_out)
        }
        WireFormat::PointsRgbaF64 => {
            emit_points_to_flat::<S, f64, Point3Rgba<f64>>(body, pos_out, color_out)
        }
        WireFormat::SplatsRgbaF32 => emit_splats_f32_to_flat::<S>(body, pos_out, color_out),
    }
}
