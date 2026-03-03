//! Binary payload reader/writer for little & big endian.
use spatial_utils::traits::{SpatialSink, SpatialView};
use spatial_utils::{splat::SH_C0, utils::point_scalar::PointScalar};

use crate::codecs::ply::types::*;
use std::io::{self, Read, Write};

#[inline]
fn rd_exact<const N: usize, R: Read>(r: &mut R) -> io::Result<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b)?;
    Ok(b)
}

#[inline]
fn skip_bytes<R: Read>(r: &mut R, mut n: usize) -> io::Result<()> {
    // Stack buffer avoids per-skip allocations; amortizes large skips.
    let mut buf = [0u8; 4096];
    while n > 0 {
        let take = n.min(buf.len());
        r.read_exact(&mut buf[..take])?;
        n -= take;
    }
    Ok(())
}

// TODO: Can we reduce the overhead caused by f64 conversions here?
#[inline]
fn read_scalar<R: Read>(r: &mut R, ty: ScalarType, be: bool) -> io::Result<f64> {
    use ScalarType::*;
    let v = match ty {
        Char => i8::from_ne_bytes([rd_exact::<1, _>(r)?[0]]) as f64,
        UChar => u8::from_ne_bytes([rd_exact::<1, _>(r)?[0]]) as f64,
        Short => {
            let b = rd_exact::<2, _>(r)?;
            if be {
                i16::from_be_bytes(b) as f64
            } else {
                i16::from_le_bytes(b) as f64
            }
        }
        UShort => {
            let b = rd_exact::<2, _>(r)?;
            if be {
                u16::from_be_bytes(b) as f64
            } else {
                u16::from_le_bytes(b) as f64
            }
        }
        Int => {
            let b = rd_exact::<4, _>(r)?;
            if be {
                i32::from_be_bytes(b) as f64
            } else {
                i32::from_le_bytes(b) as f64
            }
        }
        UInt => {
            let b = rd_exact::<4, _>(r)?;
            if be {
                u32::from_be_bytes(b) as f64
            } else {
                u32::from_le_bytes(b) as f64
            }
        }
        Float => {
            let b = rd_exact::<4, _>(r)?;
            if be {
                f32::from_be_bytes(b) as f64
            } else {
                f32::from_le_bytes(b) as f64
            }
        }
        Double => {
            let b = rd_exact::<8, _>(r)?;
            if be {
                f64::from_be_bytes(b)
            } else {
                f64::from_le_bytes(b)
            }
        }
    };
    Ok(v)
}

#[inline]
fn skip_scalar<R: Read>(r: &mut R, ty: ScalarType) -> io::Result<()> {
    skip_bytes(r, ty.size_of())
}

#[inline]
fn u8_from_scalar_value(v: f64, ty: ScalarType) -> u8 {
    if !v.is_finite() {
        return 0;
    }
    let vv = match ty {
        ScalarType::Float | ScalarType::Double => {
            if (0.0..=1.0).contains(&v) {
                v * 255.0
            } else {
                v
            }
        }
        _ => v,
    };
    if vv <= 0.0 {
        0
    } else if vv >= 255.0 {
        255
    } else {
        vv.round() as u8
    }
}

#[inline]
fn sigmoid01(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

#[inline]
fn sh_dc_to_u8(dc: f64) -> u8 {
    u8_from_scalar_value(SH_C0 * dc + 0.5, ScalarType::Double)
}

pub fn read_vertices_into_flat<R: Read, S>(
    mut r: R,
    elem: &ElementDef,
    be: bool,
    pos_out: &mut Vec<S>,
    rgb_out: &mut Vec<u8>,
) -> io::Result<()>
where
    S: PointScalar,
{
    if let Some(schema) = detect_fast_schema(elem) {
        pos_out.reserve(elem.count * 3);
        if matches!(schema, FastSchema::F32XYZ_U8RGB | FastSchema::F64XYZ_U8RGB) {
            rgb_out.reserve(elem.count * 3);
        }
        for _ in 0..elem.count {
            match schema {
                FastSchema::F32XYZ => {
                    let mut buf = [0u8; 12];
                    r.read_exact(&mut buf)?;
                    let (x, y, z) = if be {
                        (
                            S::from_be_bytes(buf[0..4].try_into().unwrap()),
                            S::from_be_bytes(buf[4..8].try_into().unwrap()),
                            S::from_be_bytes(buf[8..12].try_into().unwrap()),
                        )
                    } else {
                        (
                            S::from_le_bytes(buf[0..4].try_into().unwrap()),
                            S::from_le_bytes(buf[4..8].try_into().unwrap()),
                            S::from_le_bytes(buf[8..12].try_into().unwrap()),
                        )
                    };
                    pos_out.extend_from_slice(&[x, y, z]);
                }
                FastSchema::F64XYZ => {
                    let mut buf = [0u8; 24];
                    r.read_exact(&mut buf)?;
                    let (x, y, z) = if be {
                        (
                            S::from_be_bytes_f64(buf[0..8].try_into().unwrap()),
                            S::from_be_bytes_f64(buf[8..16].try_into().unwrap()),
                            S::from_be_bytes_f64(buf[16..24].try_into().unwrap()),
                        )
                    } else {
                        (
                            S::from_le_bytes_f64(buf[0..8].try_into().unwrap()),
                            S::from_le_bytes_f64(buf[8..16].try_into().unwrap()),
                            S::from_le_bytes_f64(buf[16..24].try_into().unwrap()),
                        )
                    };
                    pos_out.extend_from_slice(&[x, y, z]);
                }
                FastSchema::F32XYZ_U8RGB => {
                    let mut buf = [0u8; 15];
                    r.read_exact(&mut buf)?;
                    let (x, y, z) = if be {
                        (
                            S::from_be_bytes(buf[0..4].try_into().unwrap()),
                            S::from_be_bytes(buf[4..8].try_into().unwrap()),
                            S::from_be_bytes(buf[8..12].try_into().unwrap()),
                        )
                    } else {
                        (
                            S::from_le_bytes(buf[0..4].try_into().unwrap()),
                            S::from_le_bytes(buf[4..8].try_into().unwrap()),
                            S::from_le_bytes(buf[8..12].try_into().unwrap()),
                        )
                    };
                    pos_out.extend_from_slice(&[x, y, z]);
                    rgb_out.extend_from_slice(&buf[12..15]);
                }
                FastSchema::F64XYZ_U8RGB => {
                    let mut buf = [0u8; 27];
                    r.read_exact(&mut buf)?;
                    let (x, y, z) = if be {
                        (
                            S::from_be_bytes_f64(buf[0..8].try_into().unwrap()),
                            S::from_be_bytes_f64(buf[8..16].try_into().unwrap()),
                            S::from_be_bytes_f64(buf[16..24].try_into().unwrap()),
                        )
                    } else {
                        (
                            S::from_le_bytes_f64(buf[0..8].try_into().unwrap()),
                            S::from_le_bytes_f64(buf[8..16].try_into().unwrap()),
                            S::from_le_bytes_f64(buf[16..24].try_into().unwrap()),
                        )
                    };
                    pos_out.extend_from_slice(&[x, y, z]);
                    rgb_out.extend_from_slice(&buf[24..27]);
                }
            }
        }
        return Ok(());
    }

    let map = VertexPropIndex::from_element(elem);
    if !map.has_xyz() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "vertex has no x/y/z",
        ));
    }
    pos_out.reserve(elem.count * 3);
    if map.ir.is_some() && map.ig.is_some() && map.ib.is_some() {
        rgb_out.reserve(elem.count * 3);
    }

    for _ in 0..elem.count {
        for (pi, p) in elem.properties.iter().enumerate() {
            match &p.ty {
                PropertyType::Scalar(sty) => {
                    let v = read_scalar(&mut r, *sty, be)?;
                    if Some(pi) == map.ix {
                        pos_out.push(S::from_f64(v));
                    } else if Some(pi) == map.iy {
                        pos_out.push(S::from_f64(v));
                    } else if Some(pi) == map.iz {
                        pos_out.push(S::from_f64(v));
                    } else if Some(pi) == map.ir {
                        rgb_out.push(v as u8);
                    } else if Some(pi) == map.ig {
                        rgb_out.push(v as u8);
                    } else if Some(pi) == map.ib {
                        rgb_out.push(v as u8);
                    } else if Some(pi) == map.ia {
                        // Alpha channel, ignore for now
                        let _ = v as u8;
                    }
                }
                PropertyType::List { count, item } => {
                    let cnt = read_scalar(&mut r, *count, be)? as usize;
                    // Skip `cnt` items quickly by discarding bytes
                    let to_skip = cnt.checked_mul(item.size_of()).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "list size overflow")
                    })?;
                    skip_bytes(&mut r, to_skip)?;
                }
            }
        }
    }
    Ok(())
}

pub fn skip_element<R: Read>(mut r: R, elem: &ElementDef, be: bool) -> io::Result<()> {
    for _ in 0..elem.count {
        for p in &elem.properties {
            match &p.ty {
                PropertyType::Scalar(sty) => {
                    skip_scalar(&mut r, *sty)?;
                }
                PropertyType::List { count, item } => {
                    let cnt = read_scalar(&mut r, *count, be)? as usize;
                    let to_skip = cnt.checked_mul(item.size_of()).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "list size overflow")
                    })?;
                    skip_bytes(&mut r, to_skip)?;
                }
            }
        }
    }
    Ok(())
}

/// Decode binary vertices directly into a `SpatialSink` output buffer.
///
/// Dispatches:
/// - fast splat schema (float) -> bulk reads + `P::push_splat` (skips trailing `f_rest_*` in one go)
/// - fast point schemas -> bulk reads + `P::push_xyz_rgb`
/// - dynamic splat/point schemas -> name-indexed decode with an early cutoff + fast tail skip
pub fn read_vertices_into_target<R: Read, P: SpatialSink>(
    mut r: R,
    elem: &ElementDef,
    be: bool,
    out: &mut Vec<P>,
) -> io::Result<()> {
    // 1) Fast splat schema (most important for throughput).
    if let Some(fs) = detect_fast_splat_schema(elem) {
        out.reserve(elem.count);
        let base_len = if fs.has_normals { 68usize } else { 56usize };
        let mut buf = [0u8; 68];

        for _ in 0..elem.count {
            r.read_exact(&mut buf[..base_len])?;
            let f32_at = |off: usize| -> f32 {
                let b: [u8; 4] = buf[off..off + 4].try_into().unwrap();
                if be {
                    f32::from_be_bytes(b)
                } else {
                    f32::from_le_bytes(b)
                }
            };

            let x = f32_at(0) as f64;
            let y = f32_at(4) as f64;
            let z = f32_at(8) as f64;

            let dc_off = if fs.has_normals { 24 } else { 12 };
            let dc0 = f32_at(dc_off) as f64;
            let dc1 = f32_at(dc_off + 4) as f64;
            let dc2 = f32_at(dc_off + 8) as f64;
            let op = f32_at(dc_off + 12) as f64;

            let tail = dc_off + 16;
            let (r0, r1, r2, r3, s0, s1, s2) = if fs.rot_first {
                (
                    f32_at(tail) as f64,
                    f32_at(tail + 4) as f64,
                    f32_at(tail + 8) as f64,
                    f32_at(tail + 12) as f64,
                    f32_at(tail + 16) as f64,
                    f32_at(tail + 20) as f64,
                    f32_at(tail + 24) as f64,
                )
            } else {
                (
                    f32_at(tail + 12) as f64,
                    f32_at(tail + 16) as f64,
                    f32_at(tail + 20) as f64,
                    f32_at(tail + 24) as f64,
                    f32_at(tail) as f64,
                    f32_at(tail + 4) as f64,
                    f32_at(tail + 8) as f64,
                )
            };

            let rgba = [
                sh_dc_to_u8(dc0),
                sh_dc_to_u8(dc1),
                sh_dc_to_u8(dc2),
                u8_from_scalar_value(sigmoid01(op), ScalarType::Double),
            ];
            let mean = [
                P::Scalar::from_f64(x),
                P::Scalar::from_f64(y),
                P::Scalar::from_f64(z),
            ];
            let scale = [
                P::Scalar::from_f64(s0.exp()),
                P::Scalar::from_f64(s1.exp()),
                P::Scalar::from_f64(s2.exp()),
            ];
            let rot = [
                P::Scalar::from_f64(r0),
                P::Scalar::from_f64(r1),
                P::Scalar::from_f64(r2),
                P::Scalar::from_f64(r3),
            ];
            P::push_splat(out, mean, rgba, scale, rot);

            if fs.trailing_bytes != 0 {
                skip_bytes(&mut r, fs.trailing_bytes)?;
            }
        }
        return Ok(());
    }

    // 2) Existing fast point schemas (XYZ / XYZ+U8RGB).
    if let Some(schema) = detect_fast_schema(elem) {
        out.reserve(elem.count);
        for _ in 0..elem.count {
            match schema {
                FastSchema::F32XYZ => {
                    let b = rd_exact::<12, _>(&mut r)?;
                    let f = |off| {
                        let bb: [u8; 4] = b[off..off + 4].try_into().unwrap();
                        if be {
                            f32::from_be_bytes(bb) as f64
                        } else {
                            f32::from_le_bytes(bb) as f64
                        }
                    };
                    P::push_xyz_rgb(
                        out,
                        P::Scalar::from_f64(f(0)),
                        P::Scalar::from_f64(f(4)),
                        P::Scalar::from_f64(f(8)),
                        0,
                        0,
                        0,
                    );
                }
                FastSchema::F64XYZ => {
                    let b = rd_exact::<24, _>(&mut r)?;
                    let f = |off| {
                        let bb: [u8; 8] = b[off..off + 8].try_into().unwrap();
                        if be {
                            f64::from_be_bytes(bb)
                        } else {
                            f64::from_le_bytes(bb)
                        }
                    };
                    P::push_xyz_rgb(
                        out,
                        P::Scalar::from_f64(f(0)),
                        P::Scalar::from_f64(f(8)),
                        P::Scalar::from_f64(f(16)),
                        0,
                        0,
                        0,
                    );
                }
                FastSchema::F32XYZ_U8RGB => {
                    let b = rd_exact::<15, _>(&mut r)?;
                    let f = |off| {
                        let bb: [u8; 4] = b[off..off + 4].try_into().unwrap();
                        if be {
                            f32::from_be_bytes(bb) as f64
                        } else {
                            f32::from_le_bytes(bb) as f64
                        }
                    };
                    P::push_xyz_rgb(
                        out,
                        P::Scalar::from_f64(f(0)),
                        P::Scalar::from_f64(f(4)),
                        P::Scalar::from_f64(f(8)),
                        b[12],
                        b[13],
                        b[14],
                    );
                }
                FastSchema::F64XYZ_U8RGB => {
                    let b = rd_exact::<27, _>(&mut r)?;
                    let f = |off| {
                        let bb: [u8; 8] = b[off..off + 8].try_into().unwrap();
                        if be {
                            f64::from_be_bytes(bb)
                        } else {
                            f64::from_le_bytes(bb)
                        }
                    };
                    P::push_xyz_rgb(
                        out,
                        P::Scalar::from_f64(f(0)),
                        P::Scalar::from_f64(f(8)),
                        P::Scalar::from_f64(f(16)),
                        b[24],
                        b[25],
                        b[26],
                    );
                }
            }
        }
        return Ok(());
    }

    // 3) Dynamic splat (name-indexed) with early cutoff and tail skip.
    let splat = SplatPropIndex::from_element(elem);
    if let Some(max_prop) = splat.max_required_prop_index() {
        out.reserve(elem.count);

        // Precompute trailing fixed bytes if tail is scalar-only.
        let mut trailing_fixed_bytes = 0usize;
        let mut tail_is_scalar_only = true;
        for p in elem.properties.iter().skip(max_prop + 1) {
            match &p.ty {
                PropertyType::Scalar(sty) => trailing_fixed_bytes += sty.size_of(),
                PropertyType::List { .. } => {
                    tail_is_scalar_only = false;
                    break;
                }
            }
        }

        for _ in 0..elem.count {
            let mut x = 0.0f64;
            let mut y = 0.0f64;
            let mut z = 0.0f64;
            let mut dc0 = 0.0f64;
            let mut dc1 = 0.0f64;
            let mut dc2 = 0.0f64;
            let mut op = 0.0f64;
            let mut r0 = 0.0f64;
            let mut r1 = 0.0f64;
            let mut r2 = 0.0f64;
            let mut r3 = 0.0f64;
            let mut s0 = 0.0f64;
            let mut s1 = 0.0f64;
            let mut s2 = 0.0f64;

            for (pi, p) in elem.properties.iter().enumerate().take(max_prop + 1) {
                match &p.ty {
                    PropertyType::Scalar(sty) => {
                        // Only convert when we actually need the value.
                        let need = Some(pi) == splat.ix
                            || Some(pi) == splat.iy
                            || Some(pi) == splat.iz
                            || Some(pi) == splat.if_dc0
                            || Some(pi) == splat.if_dc1
                            || Some(pi) == splat.if_dc2
                            || Some(pi) == splat.iopacity
                            || Some(pi) == splat.irot0
                            || Some(pi) == splat.irot1
                            || Some(pi) == splat.irot2
                            || Some(pi) == splat.irot3
                            || Some(pi) == splat.iscale0
                            || Some(pi) == splat.iscale1
                            || Some(pi) == splat.iscale2;

                        if need {
                            let v = read_scalar(&mut r, *sty, be)?;
                            if Some(pi) == splat.ix {
                                x = v;
                            } else if Some(pi) == splat.iy {
                                y = v;
                            } else if Some(pi) == splat.iz {
                                z = v;
                            } else if Some(pi) == splat.if_dc0 {
                                dc0 = v;
                            } else if Some(pi) == splat.if_dc1 {
                                dc1 = v;
                            } else if Some(pi) == splat.if_dc2 {
                                dc2 = v;
                            } else if Some(pi) == splat.iopacity {
                                op = v;
                            } else if Some(pi) == splat.irot0 {
                                r0 = v;
                            } else if Some(pi) == splat.irot1 {
                                r1 = v;
                            } else if Some(pi) == splat.irot2 {
                                r2 = v;
                            } else if Some(pi) == splat.irot3 {
                                r3 = v;
                            } else if Some(pi) == splat.iscale0 {
                                s0 = v;
                            } else if Some(pi) == splat.iscale1 {
                                s1 = v;
                            } else if Some(pi) == splat.iscale2 {
                                s2 = v;
                            }
                        } else {
                            skip_scalar(&mut r, *sty)?;
                        }
                    }
                    PropertyType::List { count, item } => {
                        let cnt = read_scalar(&mut r, *count, be)? as usize;
                        let to_skip = cnt.checked_mul(item.size_of()).ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidData, "list size overflow")
                        })?;
                        skip_bytes(&mut r, to_skip)?;
                    }
                }
            }

            if tail_is_scalar_only && trailing_fixed_bytes != 0 {
                skip_bytes(&mut r, trailing_fixed_bytes)?;
            } else {
                // Conservative: fully parse tail if it contains lists.
                for p in elem.properties.iter().skip(max_prop + 1) {
                    match &p.ty {
                        PropertyType::Scalar(sty) => skip_scalar(&mut r, *sty)?,
                        PropertyType::List { count, item } => {
                            let cnt = read_scalar(&mut r, *count, be)? as usize;
                            let to_skip = cnt.checked_mul(item.size_of()).ok_or_else(|| {
                                io::Error::new(io::ErrorKind::InvalidData, "list size overflow")
                            })?;
                            skip_bytes(&mut r, to_skip)?;
                        }
                    }
                }
            }

            let rgba = [
                sh_dc_to_u8(dc0),
                sh_dc_to_u8(dc1),
                sh_dc_to_u8(dc2),
                u8_from_scalar_value(sigmoid01(op), ScalarType::Double),
            ];
            let mean = [
                P::Scalar::from_f64(x),
                P::Scalar::from_f64(y),
                P::Scalar::from_f64(z),
            ];
            let scale = [
                P::Scalar::from_f64(s0.exp()),
                P::Scalar::from_f64(s1.exp()),
                P::Scalar::from_f64(s2.exp()),
            ];
            let rot = [
                P::Scalar::from_f64(r0),
                P::Scalar::from_f64(r1),
                P::Scalar::from_f64(r2),
                P::Scalar::from_f64(r3),
            ];
            P::push_splat(out, mean, rgba, scale, rot);
        }
        return Ok(());
    }

    // 4) Dynamic points (XYZ + optional RGB + optional alpha/opacity).
    let map = VertexPropIndex::from_element(elem);
    if !map.has_xyz() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "vertex has no x/y/z",
        ));
    }
    out.reserve(elem.count);

    let want_rgb = map.has_rgb();
    let want_alpha = map.has_alpha();
    let max_prop = {
        let mut m = map.ix.unwrap().max(map.iy.unwrap()).max(map.iz.unwrap());
        if let Some(i) = map.ir {
            m = m.max(i);
        }
        if let Some(i) = map.ig {
            m = m.max(i);
        }
        if let Some(i) = map.ib {
            m = m.max(i);
        }
        if let Some(i) = map.ia {
            m = m.max(i);
        }
        m
    };

    // trailing bytes fast-skip (scalar-only tail)
    let mut trailing_fixed_bytes = 0usize;
    let mut tail_is_scalar_only = true;
    for p in elem.properties.iter().skip(max_prop + 1) {
        match &p.ty {
            PropertyType::Scalar(sty) => trailing_fixed_bytes += sty.size_of(),
            PropertyType::List { .. } => {
                tail_is_scalar_only = false;
                break;
            }
        }
    }

    for _ in 0..elem.count {
        let mut x = 0.0f64;
        let mut y = 0.0f64;
        let mut z = 0.0f64;
        let mut rr = 0u8;
        let mut gg = 0u8;
        let mut bb = 0u8;
        let mut aa = 255u8;

        for (pi, p) in elem.properties.iter().enumerate().take(max_prop + 1) {
            match &p.ty {
                PropertyType::Scalar(sty) => {
                    let need = Some(pi) == map.ix
                        || Some(pi) == map.iy
                        || Some(pi) == map.iz
                        || (want_rgb
                            && (Some(pi) == map.ir || Some(pi) == map.ig || Some(pi) == map.ib))
                        || (want_alpha && Some(pi) == map.ia);
                    if need {
                        let v = read_scalar(&mut r, *sty, be)?;
                        if Some(pi) == map.ix {
                            x = v;
                        } else if Some(pi) == map.iy {
                            y = v;
                        } else if Some(pi) == map.iz {
                            z = v;
                        } else if want_rgb && Some(pi) == map.ir {
                            rr = u8_from_scalar_value(v, *sty);
                        } else if want_rgb && Some(pi) == map.ig {
                            gg = u8_from_scalar_value(v, *sty);
                        } else if want_rgb && Some(pi) == map.ib {
                            bb = u8_from_scalar_value(v, *sty);
                        } else if want_alpha && Some(pi) == map.ia {
                            aa = u8_from_scalar_value(v, *sty);
                        }
                    } else {
                        skip_scalar(&mut r, *sty)?;
                    }
                }
                PropertyType::List { count, item } => {
                    let cnt = read_scalar(&mut r, *count, be)? as usize;
                    let to_skip = cnt.checked_mul(item.size_of()).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "list size overflow")
                    })?;
                    skip_bytes(&mut r, to_skip)?;
                }
            }
        }

        if tail_is_scalar_only && trailing_fixed_bytes != 0 {
            skip_bytes(&mut r, trailing_fixed_bytes)?;
        } else {
            for p in elem.properties.iter().skip(max_prop + 1) {
                match &p.ty {
                    PropertyType::Scalar(sty) => skip_scalar(&mut r, *sty)?,
                    PropertyType::List { count, item } => {
                        let cnt = read_scalar(&mut r, *count, be)? as usize;
                        let to_skip = cnt.checked_mul(item.size_of()).ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidData, "list size overflow")
                        })?;
                        skip_bytes(&mut r, to_skip)?;
                    }
                }
            }
        }

        if want_alpha {
            P::push_xyz_rgba(
                out,
                P::Scalar::from_f64(x),
                P::Scalar::from_f64(y),
                P::Scalar::from_f64(z),
                rr,
                gg,
                bb,
                aa,
            );
        } else {
            P::push_xyz_rgb(
                out,
                P::Scalar::from_f64(x),
                P::Scalar::from_f64(y),
                P::Scalar::from_f64(z),
                rr,
                gg,
                bb,
            );
        }
    }
    Ok(())
}

#[derive(Copy, Clone, Debug)]
struct FastSplatSchema {
    has_normals: bool,
    rot_first: bool,
    trailing_bytes: usize,
}

fn detect_fast_splat_schema(elem: &ElementDef) -> Option<FastSplatSchema> {
    use ScalarType::*;
    let props = &elem.properties;
    if props.len() < 14 {
        return None;
    }
    // Must start with: x y z (float)
    if props[0].name != "x" || props[1].name != "y" || props[2].name != "z" {
        return None;
    }
    match (&props[0].ty, &props[1].ty, &props[2].ty) {
        (PropertyType::Scalar(Float), PropertyType::Scalar(Float), PropertyType::Scalar(Float)) => {
        }
        _ => return None,
    }

    let mut i = 3usize;
    let mut has_normals = false;
    if props.len() >= 17
        && props[i].name == "nx"
        && props[i + 1].name == "ny"
        && props[i + 2].name == "nz"
        && matches!(props[i].ty, PropertyType::Scalar(Float))
        && matches!(props[i + 1].ty, PropertyType::Scalar(Float))
        && matches!(props[i + 2].ty, PropertyType::Scalar(Float))
    {
        has_normals = true;
        i += 3;
    }

    // f_dc_0..2 opacity
    if props.get(i)?.name != "f_dc_0"
        || props.get(i + 1)?.name != "f_dc_1"
        || props.get(i + 2)?.name != "f_dc_2"
    {
        return None;
    }
    if !matches!(props[i].ty, PropertyType::Scalar(Float))
        || !matches!(props[i + 1].ty, PropertyType::Scalar(Float))
        || !matches!(props[i + 2].ty, PropertyType::Scalar(Float))
    {
        return None;
    }
    i += 3;
    if props.get(i)?.name != "opacity" || !matches!(props[i].ty, PropertyType::Scalar(Float)) {
        return None;
    }
    i += 1;

    // rot/scale ordering
    let rot_first = if props.get(i)?.name == "rot_0" {
        // rot_0..3 then scale_0..2
        if props.get(i + 3)?.name != "rot_3" {
            return None;
        }
        for k in 0..4 {
            if props.get(i + k)?.name != format!("rot_{k}") {
                return None;
            }
            if !matches!(props[i + k].ty, PropertyType::Scalar(Float)) {
                return None;
            }
        }
        i += 4;
        for k in 0..3 {
            if props.get(i + k)?.name != format!("scale_{k}") {
                return None;
            }
            if !matches!(props[i + k].ty, PropertyType::Scalar(Float)) {
                return None;
            }
        }
        i += 3;
        true
    } else if props.get(i)?.name == "scale_0" {
        // scale_0..2 then rot_0..3
        for k in 0..3 {
            if props.get(i + k)?.name != format!("scale_{k}") {
                return None;
            }
            if !matches!(props[i + k].ty, PropertyType::Scalar(Float)) {
                return None;
            }
        }
        i += 3;
        for k in 0..4 {
            if props.get(i + k)?.name != format!("rot_{k}") {
                return None;
            }
            if !matches!(props[i + k].ty, PropertyType::Scalar(Float)) {
                return None;
            }
        }
        i += 4;
        false
    } else {
        return None;
    };

    // trailing bytes (e.g. f_rest_*) must all be scalar to allow a single skip.
    let mut trailing = 0usize;
    for p in props.iter().skip(i) {
        match &p.ty {
            PropertyType::Scalar(sty) => trailing += sty.size_of(),
            PropertyType::List { .. } => return None,
        }
    }
    Some(FastSplatSchema {
        has_normals,
        rot_first,
        trailing_bytes: trailing,
    })
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
enum FastSchema {
    F32XYZ,
    F64XYZ,
    F32XYZ_U8RGB,
    F64XYZ_U8RGB,
}

fn detect_fast_schema(elem: &ElementDef) -> Option<FastSchema> {
    use ScalarType::*;
    // exact property sequence only
    let props = &elem.properties;
    let s3 = props.len() == 3;
    let s6 = props.len() == 6;
    let is_xyz = |i: usize| props[i].name.as_str() == ["x", "y", "z"][i];
    let is_rgb = |i: usize| props[i + 3].name.as_str() == ["red", "green", "blue"][i];

    if s3 && is_xyz(0) {
        if let (
            PropertyType::Scalar(Float),
            PropertyType::Scalar(Float),
            PropertyType::Scalar(Float),
        ) = (&props[0].ty, &props[1].ty, &props[2].ty)
        {
            return Some(FastSchema::F32XYZ);
        }
        if let (
            PropertyType::Scalar(Double),
            PropertyType::Scalar(Double),
            PropertyType::Scalar(Double),
        ) = (&props[0].ty, &props[1].ty, &props[2].ty)
        {
            return Some(FastSchema::F64XYZ);
        }
    }
    if s6 && is_xyz(0) && is_rgb(0) {
        if let (
            PropertyType::Scalar(Float),
            PropertyType::Scalar(Float),
            PropertyType::Scalar(Float),
            PropertyType::Scalar(UChar),
            PropertyType::Scalar(UChar),
            PropertyType::Scalar(UChar),
        ) = (
            &props[0].ty,
            &props[1].ty,
            &props[2].ty,
            &props[3].ty,
            &props[4].ty,
            &props[5].ty,
        ) {
            return Some(FastSchema::F32XYZ_U8RGB);
        }
        if let (
            PropertyType::Scalar(Double),
            PropertyType::Scalar(Double),
            PropertyType::Scalar(Double),
            PropertyType::Scalar(UChar),
            PropertyType::Scalar(UChar),
            PropertyType::Scalar(UChar),
        ) = (
            &props[0].ty,
            &props[1].ty,
            &props[2].ty,
            &props[3].ty,
            &props[4].ty,
            &props[5].ty,
        ) {
            return Some(FastSchema::F64XYZ_U8RGB);
        }
    }
    None
}

pub fn write_vertices<W: Write, S>(
    mut w: W,
    enc: PlyEncoding,
    coords_are_double: bool,
    pos: &[[S; 3]],
    rgb: Option<&[[u8; 3]]>,
) -> io::Result<()>
where
    S: PointScalar,
{
    let be = matches!(enc, PlyEncoding::BinaryBigEndian);
    for (i, p) in pos.iter().enumerate() {
        // TODO: This is readable, but all these branches could be optimized better.
        if coords_are_double {
            let xs = if be {
                p[0].to_be_bytes_f64()
            } else {
                p[0].to_le_bytes_f64()
            };
            let ys = if be {
                p[1].to_be_bytes_f64()
            } else {
                p[1].to_le_bytes_f64()
            };
            let zs = if be {
                p[2].to_be_bytes_f64()
            } else {
                p[2].to_le_bytes_f64()
            };
            w.write_all(&xs)?;
            w.write_all(&ys)?;
            w.write_all(&zs)?;
        } else {
            let xs = if be {
                p[0].to_be_bytes()
            } else {
                p[0].to_le_bytes()
            };
            let ys = if be {
                p[1].to_be_bytes()
            } else {
                p[1].to_le_bytes()
            };
            let zs = if be {
                p[2].to_be_bytes()
            } else {
                p[2].to_le_bytes()
            };
            w.write_all(&xs)?;
            w.write_all(&ys)?;
            w.write_all(&zs)?;
        }
        if let Some(rgb) = rgb {
            let c = rgb[i];
            w.write_all(&c)?;
        }
    }
    Ok(())
}
 
// --- Gaussian splat encoding helpers (3DGS-compatible) ---
const INV_255: f64 = 1.0 / 255.0;
const OPACITY_EPS: f64 = 1e-6;

#[inline(always)]
fn dc_from_u8(c: u8) -> f64 {
    ((c as f64) * INV_255 - 0.5) / SH_C0
}

#[inline(always)]
fn logit_from_u8_alpha(a: u8) -> f64 {
    let p = (a as f64) * INV_255;
    let p = p.clamp(OPACITY_EPS, 1.0 - OPACITY_EPS);
    (p / (1.0 - p)).ln()
}

#[inline(always)]
fn safe_ln_scale(v: f64) -> f64 {
    if v.is_finite() && v > 0.0 { v.ln() } else { 0.0 }
}

#[inline(always)]
fn finite_or(v: f64, fallback: f64) -> f64 {
    if v.is_finite() { v } else { fallback }
}

#[inline(always)]
fn put_scalar4<S: PointScalar>(dst: &mut [u8], off: &mut usize, v: S, be: bool) {
    let b = if be { v.to_be_bytes() } else { v.to_le_bytes() };
    dst[*off..*off + 4].copy_from_slice(&b);
    *off += 4;
}

#[inline(always)]
fn put_scalar8<S: PointScalar>(dst: &mut [u8], off: &mut usize, v: S, be: bool) {
    let b = if be { v.to_be_bytes_f64() } else { v.to_le_bytes_f64() };
    dst[*off..*off + 8].copy_from_slice(&b);
    *off += 8;
}

/// Write binary PLY points directly from payload slice (no intermediate buffers).
///
/// Supports:
/// - xyz
/// - xyzrgb (uchar)
/// - xyzrgba (uchar)
pub fn write_vertices_from_payload_points<W: Write, P, S>(
    mut w: W,
    enc: PlyEncoding,
    coords_are_double: bool,
    points: &[P],
    write_color: bool,
    write_alpha: bool,
) -> io::Result<()>
where
    P: SpatialView<S>,
    S: PointScalar,
{
    let be = matches!(enc, PlyEncoding::BinaryBigEndian);
    let write_alpha = write_color && write_alpha;

    match (coords_are_double, write_color, write_alpha) {
        (false, false, _) => {
            let mut buf = [0u8; 12];
            for p in points {
                let mut off = 0usize;
                put_scalar4(&mut buf, &mut off, p.x(), be);
                put_scalar4(&mut buf, &mut off, p.y(), be);
                put_scalar4(&mut buf, &mut off, p.z(), be);
                w.write_all(&buf)?;
            }
        }
        (false, true, false) => {
            let mut buf = [0u8; 15];
            for p in points {
                let mut off = 0usize;
                put_scalar4(&mut buf, &mut off, p.x(), be);
                put_scalar4(&mut buf, &mut off, p.y(), be);
                put_scalar4(&mut buf, &mut off, p.z(), be);
                buf[12] = p.r_u8();
                buf[13] = p.g_u8();
                buf[14] = p.b_u8();
                w.write_all(&buf)?;
            }
        }
        (false, true, true) => {
            let mut buf = [0u8; 16];
            for p in points {
                let mut off = 0usize;
                put_scalar4(&mut buf, &mut off, p.x(), be);
                put_scalar4(&mut buf, &mut off, p.y(), be);
                put_scalar4(&mut buf, &mut off, p.z(), be);
                buf[12] = p.r_u8();
                buf[13] = p.g_u8();
                buf[14] = p.b_u8();
                buf[15] = p.a_u8();
                w.write_all(&buf)?;
            }
        }
        (true, false, _) => {
            let mut buf = [0u8; 24];
            for p in points {
                let mut off = 0usize;
                put_scalar8(&mut buf, &mut off, p.x(), be);
                put_scalar8(&mut buf, &mut off, p.y(), be);
                put_scalar8(&mut buf, &mut off, p.z(), be);
                w.write_all(&buf)?;
            }
        }
        (true, true, false) => {
            let mut buf = [0u8; 27];
            for p in points {
                let mut off = 0usize;
                put_scalar8(&mut buf, &mut off, p.x(), be);
                put_scalar8(&mut buf, &mut off, p.y(), be);
                put_scalar8(&mut buf, &mut off, p.z(), be);
                buf[24] = p.r_u8();
                buf[25] = p.g_u8();
                buf[26] = p.b_u8();
                w.write_all(&buf)?;
            }
        }
        (true, true, true) => {
            let mut buf = [0u8; 28];
            for p in points {
                let mut off = 0usize;
                put_scalar8(&mut buf, &mut off, p.x(), be);
                put_scalar8(&mut buf, &mut off, p.y(), be);
                put_scalar8(&mut buf, &mut off, p.z(), be);
                buf[24] = p.r_u8();
                buf[25] = p.g_u8();
                buf[26] = p.b_u8();
                buf[27] = p.a_u8();
                w.write_all(&buf)?;
            }
        }
    }
    Ok(())
}

/// Write binary PLY Gaussian splats directly from payload slice (no intermediate buffers).
///
/// Schema written (canonical order):
/// x y z f_dc_0 f_dc_1 f_dc_2 opacity rot_0 rot_1 rot_2 rot_3 scale_0 scale_1 scale_2
pub fn write_vertices_from_payload_splats<W: Write, P, S>(
    mut w: W,
    enc: PlyEncoding,
    coords_are_double: bool,
    points: &[P],
) -> io::Result<()>
where
    P: SpatialView<S>,
    S: PointScalar,
{
    let be = matches!(enc, PlyEncoding::BinaryBigEndian);
    if coords_are_double {
        // 14 * 8 bytes
        let mut buf = [0u8; 112];
        for p in points {
            let dc0 = S::from_f64(dc_from_u8(p.r_u8()));
            let dc1 = S::from_f64(dc_from_u8(p.g_u8()));
            let dc2 = S::from_f64(dc_from_u8(p.b_u8()));
            let op = S::from_f64(logit_from_u8_alpha(p.a_u8()));
            let rw = S::from_f64(finite_or(p.rot_w().to_f64(), 1.0));
            let rx = S::from_f64(finite_or(p.rot_x().to_f64(), 0.0));
            let ry = S::from_f64(finite_or(p.rot_y().to_f64(), 0.0));
            let rz = S::from_f64(finite_or(p.rot_z().to_f64(), 0.0));
            let sx = S::from_f64(safe_ln_scale(p.scale_x().to_f64()));
            let sy = S::from_f64(safe_ln_scale(p.scale_y().to_f64()));
            let sz = S::from_f64(safe_ln_scale(p.scale_z().to_f64()));

            let mut off = 0usize;
            put_scalar8(&mut buf, &mut off, p.x(), be);
            put_scalar8(&mut buf, &mut off, p.y(), be);
            put_scalar8(&mut buf, &mut off, p.z(), be);
            put_scalar8(&mut buf, &mut off, dc0, be);
            put_scalar8(&mut buf, &mut off, dc1, be);
            put_scalar8(&mut buf, &mut off, dc2, be);
            put_scalar8(&mut buf, &mut off, op, be);
            put_scalar8(&mut buf, &mut off, rw, be);
            put_scalar8(&mut buf, &mut off, rx, be);
            put_scalar8(&mut buf, &mut off, ry, be);
            put_scalar8(&mut buf, &mut off, rz, be);
            put_scalar8(&mut buf, &mut off, sx, be);
            put_scalar8(&mut buf, &mut off, sy, be);
            put_scalar8(&mut buf, &mut off, sz, be);
            w.write_all(&buf)?;
        }
    } else {
        // 14 * 4 bytes
        let mut buf = [0u8; 56];
        for p in points {
            let dc0 = S::from_f64(dc_from_u8(p.r_u8()));
            let dc1 = S::from_f64(dc_from_u8(p.g_u8()));
            let dc2 = S::from_f64(dc_from_u8(p.b_u8()));
            let op = S::from_f64(logit_from_u8_alpha(p.a_u8()));
            let rw = S::from_f64(finite_or(p.rot_w().to_f64(), 1.0));
            let rx = S::from_f64(finite_or(p.rot_x().to_f64(), 0.0));
            let ry = S::from_f64(finite_or(p.rot_y().to_f64(), 0.0));
            let rz = S::from_f64(finite_or(p.rot_z().to_f64(), 0.0));
            let sx = S::from_f64(safe_ln_scale(p.scale_x().to_f64()));
            let sy = S::from_f64(safe_ln_scale(p.scale_y().to_f64()));
            let sz = S::from_f64(safe_ln_scale(p.scale_z().to_f64()));

            let mut off = 0usize;
            put_scalar4(&mut buf, &mut off, p.x(), be);
            put_scalar4(&mut buf, &mut off, p.y(), be);
            put_scalar4(&mut buf, &mut off, p.z(), be);
            put_scalar4(&mut buf, &mut off, dc0, be);
            put_scalar4(&mut buf, &mut off, dc1, be);
            put_scalar4(&mut buf, &mut off, dc2, be);
            put_scalar4(&mut buf, &mut off, op, be);
            put_scalar4(&mut buf, &mut off, rw, be);
            put_scalar4(&mut buf, &mut off, rx, be);
            put_scalar4(&mut buf, &mut off, ry, be);
            put_scalar4(&mut buf, &mut off, rz, be);
            put_scalar4(&mut buf, &mut off, sx, be);
            put_scalar4(&mut buf, &mut off, sy, be);
            put_scalar4(&mut buf, &mut off, sz, be);
            w.write_all(&buf)?;
        }
    }
    Ok(())
}