//! ASCII payload reader/writer. Optimized for predictable per‑vertex lines.
use spatial_utils::traits::{SpatialSink, SpatialView};
use spatial_utils::{splat::SH_C0, utils::point_scalar::PointScalar};

use crate::codecs::ply::types::*;
use std::io::{self, BufRead, Write};

// TODO: Can we reduce the overhead caused by f64 conversions here?
#[inline]
fn parse_scalar(tok: &str, ty: ScalarType) -> io::Result<f64> {
    // Return as f64 for a uniform path; caller decides to cast.
    match ty {
        ScalarType::Float | ScalarType::Double => tok
            .parse::<f64>()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad float")),
        ScalarType::Char | ScalarType::Short | ScalarType::Int => tok
            .parse::<i64>()
            .map(|v| v as f64)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad int")),
        ScalarType::UChar | ScalarType::UShort | ScalarType::UInt => tok
            .parse::<u64>()
            .map(|v| v as f64)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad uint")),
    }
}

#[inline]
fn u8_from_scalar_value(v: f64, ty: ScalarType) -> u8 {
    if !v.is_finite() {
        return 0;
    }
    // Heuristic: floats in [0,1] are treated as normalized.
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
    // numerically stable sigmoid
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

#[inline]
fn sh_dc_to_u8(dc: f64) -> u8 {
    // Common 3DGS decode: rgb = clamp(0,1, SH_C0 * dc + 0.5)
    u8_from_scalar_value(SH_C0 * dc + 0.5, ScalarType::Double)
}

/// Decode ASCII vertices directly into a `SpatialSink` output buffer.
///
/// Dispatches:
/// - splat schema -> `P::push_splat`
/// - point schema with alpha/opacity -> `P::push_xyz_rgba`
/// - otherwise -> `P::push_xyz_rgb`
pub fn read_vertices_into_target<R: BufRead, P: SpatialSink>(
    mut r: R,
    elem: &ElementDef,
    out: &mut Vec<P>,
) -> io::Result<()> {
    // Prefer splat detection first (it has an "opacity" prop too).
    let splat = SplatPropIndex::from_element(elem);
    if let Some(max_prop) = splat.max_required_prop_index() {
        out.reserve(elem.count);
        let mut line = String::new();
        for _ in 0..elem.count {
            line.clear();
            if r.read_line(&mut line)? == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "vertex ascii EOF",
                ));
            }
            let mut tok_it = line.split_ascii_whitespace();

            // Defaults are fine: header guarantees indices; EOF/truncation will error on token fetch.
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

            // Only walk properties up to the largest needed index; ignore trailing `f_rest_*` quickly.
            for (pi, p) in elem.properties.iter().enumerate().take(max_prop + 1) {
                match &p.ty {
                    PropertyType::Scalar(sty) => {
                        let tok = tok_it.next().ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidData, "too few tokens")
                        })?;
                        let v = parse_scalar(tok, *sty)?;
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
                    }
                    PropertyType::List { count, .. } => {
                        // Rare for splat vertices; but keep stream correctness.
                        let tokc = tok_it.next().ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidData, "missing list count")
                        })?;
                        let cnt = parse_scalar(tokc, *count)? as usize;
                        for _ in 0..cnt {
                            let _ = tok_it.next().ok_or_else(|| {
                                io::Error::new(io::ErrorKind::InvalidData, "list too short")
                            })?;
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
            // Typical 3DGS stores log-scales in PLY.
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

    // Classic points (XYZ + optional RGB + optional alpha/opacity).
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

    let mut line = String::new();
    for _ in 0..elem.count {
        line.clear();
        if r.read_line(&mut line)? == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "vertex ascii EOF",
            ));
        }
        let mut tok_it = line.split_ascii_whitespace();

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
                    let tok = tok_it.next().ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "too few tokens")
                    })?;
                    let v = parse_scalar(tok, *sty)?;
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
                }
                PropertyType::List { count, .. } => {
                    let tokc = tok_it.next().ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "missing list count")
                    })?;
                    let cnt = parse_scalar(tokc, *count)? as usize;
                    for _ in 0..cnt {
                        let _ = tok_it.next().ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidData, "list too short")
                        })?;
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

pub fn read_vertices_into_flat<R: BufRead, S>(
    mut r: R,
    elem: &ElementDef,
    pos_out: &mut Vec<S>,
    rgb_out: &mut Vec<u8>,
) -> io::Result<()>
where
    S: PointScalar,
{
    let map = VertexPropIndex::from_element(elem);
    if !(map.ix.is_some() && map.iy.is_some() && map.iz.is_some()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "vertex has no x/y/z",
        ));
    }

    // reserve once
    pos_out.reserve(elem.count * 3);
    let want_rgb = map.ir.is_some() && map.ig.is_some() && map.ib.is_some();
    if want_rgb {
        rgb_out.reserve(elem.count * 3);
    }

    let mut line = String::new();
    for _ in 0..elem.count {
        line.clear();
        if r.read_line(&mut line)? == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "vertex ascii EOF",
            ));
        }

        // allocation-free token walk
        let mut tok_it = line.split_ascii_whitespace();

        // walk properties in header order; consume exactly what the header declares
        for (pi, p) in elem.properties.iter().enumerate() {
            match &p.ty {
                PropertyType::Scalar(sty) => {
                    let tok = tok_it.next().ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "too few tokens")
                    })?;
                    let v = parse_scalar(tok, *sty)?;
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
                    }
                    // unknown scalar -> consumed and ignored
                }
                PropertyType::List { count, item } => {
                    // count first
                    let tokc = tok_it.next().ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "missing list count")
                    })?;
                    let cnt = parse_scalar(tokc, *count)? as usize;
                    // skip exactly `cnt` item tokens
                    for _ in 0..cnt {
                        let _ = tok_it.next().ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidData, "list too short")
                        })?;
                    }
                    let _ = item; // silence lint; we don't need item values
                }
            }
        }
        // any extra tokens in the line are ignored (robustness)
    }
    Ok(())
}

pub fn skip_element_lines<R: BufRead>(mut r: R, elem: &ElementDef) -> io::Result<()> {
    let mut line = String::new();
    for _ in 0..elem.count {
        if r.read_line(&mut line)? == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "skip EOF"));
        }
        line.clear();
    }
    Ok(())
}

// --- Gaussian splat encoding helpers (3DGS-compatible) ---
//
// Decoder side used:
//   rgb = clamp01(SH_C0 * f_dc + 0.5)
//   alpha = sigmoid(opacity)
//
// Encoder side inverts that:
//   f_dc = (rgb01 - 0.5) / SH_C0
//   opacity = logit(alpha01)
//
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
    if v.is_finite() && v > 0.0 {
        v.ln()
    } else {
        0.0
    }
}

#[inline(always)]
fn finite_or(v: f64, fallback: f64) -> f64 {
    if v.is_finite() { v } else { fallback }
}

/// Write PLY ASCII vertices directly from a payload slice (no intermediate buffers).
///
/// Supports:
/// - xyz
/// - xyzrgb (uchar)
/// - xyzrgba (uchar) when `write_alpha` is true
pub fn write_vertices_from_payload_points<W: Write, P, S>(
    mut w: W,
    coords_are_double: bool,
    points: &[P],
    write_color: bool,
    write_alpha: bool,
    mode: AsciiFloatMode,
) -> io::Result<()>
where
    P: SpatialView<S>,
    S: PointScalar,
{
    let write_alpha = write_color && write_alpha;
    match mode {
        AsciiFloatMode::Fixed(prec) => {
            if coords_are_double {
                for p in points {
                    write!(
                        w,
                        "{:.prec$} {:.prec$} {:.prec$}",
                        p.x().to_f64(),
                        p.y().to_f64(),
                        p.z().to_f64(),
                        prec = prec
                    )?;
                    if write_color {
                        write!(w, " {} {} {}", p.r_u8(), p.g_u8(), p.b_u8())?;
                        if write_alpha {
                            write!(w, " {}", p.a_u8())?;
                        }
                    }
                    w.write_all(b"\n")?;
                }
            } else {
                for p in points {
                    write!(
                        w,
                        "{:.prec$} {:.prec$} {:.prec$}",
                        p.x().to_f32(),
                        p.y().to_f32(),
                        p.z().to_f32(),
                        prec = prec
                    )?;
                    if write_color {
                        write!(w, " {} {} {}", p.r_u8(), p.g_u8(), p.b_u8())?;
                        if write_alpha {
                            write!(w, " {}", p.a_u8())?;
                        }
                    }
                    w.write_all(b"\n")?;
                }
            }
        }
        AsciiFloatMode::Shortest => {
            if coords_are_double {
                for p in points {
                    write!(w, "{} {} {}", p.x().to_f64(), p.y().to_f64(), p.z().to_f64())?;
                    if write_color {
                        write!(w, " {} {} {}", p.r_u8(), p.g_u8(), p.b_u8())?;
                        if write_alpha {
                            write!(w, " {}", p.a_u8())?;
                        }
                    }
                    w.write_all(b"\n")?;
                }
            } else {
                for p in points {
                    write!(w, "{} {} {}", p.x().to_f32(), p.y().to_f32(), p.z().to_f32())?;
                    if write_color {
                        write!(w, " {} {} {}", p.r_u8(), p.g_u8(), p.b_u8())?;
                        if write_alpha {
                            write!(w, " {}", p.a_u8())?;
                        }
                    }
                    w.write_all(b"\n")?;
                }
            }
        }
    }
    Ok(())
}

/// Write PLY ASCII Gaussian splats directly from a payload slice (no intermediate buffers).
///
/// Schema written (canonical order):
/// x y z f_dc_0 f_dc_1 f_dc_2 opacity rot_0 rot_1 rot_2 rot_3 scale_0 scale_1 scale_2
pub fn write_vertices_from_payload_splats<W: Write, P, S>(
    mut w: W,
    coords_are_double: bool,
    points: &[P],
    mode: AsciiFloatMode,
) -> io::Result<()>
where
    P: SpatialView<S>,
    S: PointScalar,
{
    match mode {
        AsciiFloatMode::Fixed(prec) => {
            if coords_are_double {
                for p in points {
                    let x = p.x().to_f64();
                    let y = p.y().to_f64();
                    let z = p.z().to_f64();
                    let dc0 = dc_from_u8(p.r_u8());
                    let dc1 = dc_from_u8(p.g_u8());
                    let dc2 = dc_from_u8(p.b_u8());
                    let op = logit_from_u8_alpha(p.a_u8());
                    let rw = finite_or(p.rot_w().to_f64(), 1.0);
                    let rx = finite_or(p.rot_x().to_f64(), 0.0);
                    let ry = finite_or(p.rot_y().to_f64(), 0.0);
                    let rz = finite_or(p.rot_z().to_f64(), 0.0);
                    let sx = safe_ln_scale(p.scale_x().to_f64());
                    let sy = safe_ln_scale(p.scale_y().to_f64());
                    let sz = safe_ln_scale(p.scale_z().to_f64());
                    write!(
                        w,
                        "{:.prec$} {:.prec$} {:.prec$} {:.prec$} {:.prec$} {:.prec$} {:.prec$} \
{:.prec$} {:.prec$} {:.prec$} {:.prec$} {:.prec$} {:.prec$} {:.prec$}",
                        x, y, z,
                        dc0, dc1, dc2,
                        op,
                        rw, rx, ry, rz,
                        sx, sy, sz,
                        prec = prec
                    )?;
                    w.write_all(b"\n")?;
                }
            } else {
                for p in points {
                    let x = p.x().to_f32();
                    let y = p.y().to_f32();
                    let z = p.z().to_f32();
                    let dc0 = dc_from_u8(p.r_u8()) as f32;
                    let dc1 = dc_from_u8(p.g_u8()) as f32;
                    let dc2 = dc_from_u8(p.b_u8()) as f32;
                    let op = logit_from_u8_alpha(p.a_u8()) as f32;
                    let rw = finite_or(p.rot_w().to_f64(), 1.0) as f32;
                    let rx = finite_or(p.rot_x().to_f64(), 0.0) as f32;
                    let ry = finite_or(p.rot_y().to_f64(), 0.0) as f32;
                    let rz = finite_or(p.rot_z().to_f64(), 0.0) as f32;
                    let sx = safe_ln_scale(p.scale_x().to_f64()) as f32;
                    let sy = safe_ln_scale(p.scale_y().to_f64()) as f32;
                    let sz = safe_ln_scale(p.scale_z().to_f64()) as f32;
                    write!(
                        w,
                        "{:.prec$} {:.prec$} {:.prec$} {:.prec$} {:.prec$} {:.prec$} {:.prec$} \
{:.prec$} {:.prec$} {:.prec$} {:.prec$} {:.prec$} {:.prec$} {:.prec$}",
                        x, y, z,
                        dc0, dc1, dc2,
                        op,
                        rw, rx, ry, rz,
                        sx, sy, sz,
                        prec = prec
                    )?;
                    w.write_all(b"\n")?;
                }
            }
        }
        AsciiFloatMode::Shortest => {
            if coords_are_double {
                for p in points {
                    let x = p.x().to_f64();
                    let y = p.y().to_f64();
                    let z = p.z().to_f64();
                    let dc0 = dc_from_u8(p.r_u8());
                    let dc1 = dc_from_u8(p.g_u8());
                    let dc2 = dc_from_u8(p.b_u8());
                    let op = logit_from_u8_alpha(p.a_u8());
                    let rw = finite_or(p.rot_w().to_f64(), 1.0);
                    let rx = finite_or(p.rot_x().to_f64(), 0.0);
                    let ry = finite_or(p.rot_y().to_f64(), 0.0);
                    let rz = finite_or(p.rot_z().to_f64(), 0.0);
                    let sx = safe_ln_scale(p.scale_x().to_f64());
                    let sy = safe_ln_scale(p.scale_y().to_f64());
                    let sz = safe_ln_scale(p.scale_z().to_f64());
                    write!(
                        w,
                        "{} {} {} {} {} {} {} {} {} {} {} {} {} {}",
                        x, y, z,
                        dc0, dc1, dc2,
                        op,
                        rw, rx, ry, rz,
                        sx, sy, sz
                    )?;
                    w.write_all(b"\n")?;
                }
            } else {
                for p in points {
                    let x = p.x().to_f32();
                    let y = p.y().to_f32();
                    let z = p.z().to_f32();
                    let dc0 = dc_from_u8(p.r_u8()) as f32;
                    let dc1 = dc_from_u8(p.g_u8()) as f32;
                    let dc2 = dc_from_u8(p.b_u8()) as f32;
                    let op = logit_from_u8_alpha(p.a_u8()) as f32;
                    let rw = finite_or(p.rot_w().to_f64(), 1.0) as f32;
                    let rx = finite_or(p.rot_x().to_f64(), 0.0) as f32;
                    let ry = finite_or(p.rot_y().to_f64(), 0.0) as f32;
                    let rz = finite_or(p.rot_z().to_f64(), 0.0) as f32;
                    let sx = safe_ln_scale(p.scale_x().to_f64()) as f32;
                    let sy = safe_ln_scale(p.scale_y().to_f64()) as f32;
                    let sz = safe_ln_scale(p.scale_z().to_f64()) as f32;
                    write!(
                        w,
                        "{} {} {} {} {} {} {} {} {} {} {} {} {} {}",
                        x, y, z,
                        dc0, dc1, dc2,
                        op,
                        rw, rx, ry, rz,
                        sx, sy, sz
                    )?;
                    w.write_all(b"\n")?;
                }
            }
        }
    }
    Ok(())
}