//! GSP16 encoder (high-throughput, streaming into caller-provided `Vec<u8>`).
//!
//! This module provides a single public entry point (`encode_from_payload_into`) and
//! internally selects the fastest available float32->float16 conversion backend:
//! - x86/x86_64 with F16C
//! - aarch64 with FP16
//! - portable scalar fallback
//!
//! The selection is performed once per call (outside the hot loop). The hot loop is
//! monomorphized over a small `to_f16` closure, so there is no per-record feature
//! branching and no indirect calls.

use core::{f32::consts::PI, ptr};

use spatial_utils::{
    traits::{ColorKind, SpatialView, SpatialKind, SpatialOwnedFull},
    utils::point_scalar::PointScalar,
};
use crate::BasicResult;

use crate::codecs::gsplat16::types::{
    FLAG_HAS_ALPHA, FLAG_SPLATS, Gsplat16Params, HEADER_LEN, MAGIC, RECORD_LEN, VERSION,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::codecs::gsplat16::f16c;

#[cfg(target_arch = "aarch64")]
use crate::codecs::gsplat16::fp16;

/// Internal constants for scale quantization (matches shader-style decode).
const LN_SCALE_MIN: f32 = -12.0;
const LN_SCALE_MAX: f32 = 9.0;
const LN_SCALE_RANGE: f32 = LN_SCALE_MAX - LN_SCALE_MIN; // 21
const LN_SCALE_SCALE_ENC: f32 = 254.0 / LN_SCALE_RANGE; // encode multiplier
const SCALE_ZERO_CUTOFF: f32 = 9.357_623e-14; // e^-30 ≈ 9.3576e-14

#[inline(always)]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

#[inline(always)]
fn quantize_u8_unit(x01: f32) -> u8 {
    // round-to-nearest
    (clamp01(x01) * 255.0 + 0.5).floor() as u8
}

#[inline(always)]
fn f32_to_f16_bits(v: f32) -> u16 {
    // IEEE754 f32 -> f16 (round-to-nearest-even), dependency-free.
    let f = v.to_bits();
    let sign = ((f >> 16) & 0x8000) as u16;
    let exp = ((f >> 23) & 0xFF) as i32;
    let mant: u32 = f & 0x7FFFFF;

    // NaN/Inf
    if exp == 255 {
        return if mant == 0 {
            sign | 0x7C00 // Inf
        } else {
            sign | 0x7E00 // qNaN
        };
    }

    // rebias exponent
    let e = exp - 127 + 15;

    if e >= 31 {
        // overflow => Inf
        return sign | 0x7C00;
    }

    if e <= 0 {
        // subnormal or underflow to zero
        if e < -10 {
            return sign;
        }
        let mant = mant | 0x800000;
        let shift = 14 - e; // 14..=24
        let mut half_m = (mant >> shift) as u16;

        // round-to-nearest-even
        let round_bit = 1u32 << (shift - 1);
        let rest = mant & (round_bit - 1);
        let lsb = (half_m & 1) as u32;
        if (mant & round_bit) != 0 && (rest != 0 || lsb != 0) {
            half_m = half_m.wrapping_add(1);
        }
        return sign | half_m;
    }

    // normal
    let mut half_e = (e as u16) << 10;
    let mut half_m = (mant >> 13) as u16;

    // round-to-nearest-even using bit 12
    let round_bit = 0x1000u32;
    let rest = mant & (round_bit - 1);
    let lsb = (half_m & 1) as u32;
    if (mant & round_bit) != 0 && (rest != 0 || lsb != 0) {
        half_m = half_m.wrapping_add(1);
        if half_m == 0x0400 {
            // mantissa overflow => bump exponent
            half_m = 0;
            half_e = half_e.wrapping_add(1 << 10);
            if half_e >= 0x7C00 {
                // overflow => Inf
                half_e = 0x7C00;
            }
        }
    }

    sign | half_e | (half_m & 0x03FF)
}

#[inline(always)]
unsafe fn write_u128_le(dst: *mut u8, w0: u32, w1: u32, w2: u32, w3: u32) {
    let v = (w0 as u128)
        | ((w1 as u128) << 32)
        | ((w2 as u128) << 64)
        | ((w3 as u128) << 96);
    ptr::write_unaligned(dst as *mut u128, v.to_le());
}

#[inline(always)]
fn encode_scale_u8(scale_linear: f32) -> u8 {
    if !scale_linear.is_finite() {
        return 0;
    }
    if scale_linear <= 0.0 || scale_linear < SCALE_ZERO_CUTOFF {
        return 0;
    }
    let ln = scale_linear.ln();
    // map ln(scale) from [LN_SCALE_MIN..LN_SCALE_MAX] into [1..255]
    let v = ((ln - LN_SCALE_MIN) * LN_SCALE_SCALE_ENC + 1.0).round();
    v.clamp(1.0, 255.0) as u8
}

#[inline(always)]
fn sign_not_zero(x: f32) -> f32 {
    if x >= 0.0 { 1.0 } else { -1.0 }
}

#[inline(always)]
fn oct_encode_axis(axis_x: f32, axis_y: f32, axis_z: f32) -> (u8, u8) {
    // Normalize L1
    let ax = axis_x.abs();
    let ay = axis_y.abs();
    let az = axis_z.abs();
    let denom = ax + ay + az;
    if denom <= 0.0 || !denom.is_finite() {
        // default axis => (1,0,0)
        return (255, 128);
    }

    let mut x = axis_x / denom;
    let mut y = axis_y / denom;
    let z = axis_z / denom;

    // fold if z < 0
    if z < 0.0 {
        let ox = x;
        let oy = y;
        x = (1.0 - oy.abs()) * sign_not_zero(ox);
        y = (1.0 - ox.abs()) * sign_not_zero(oy);
    }

    // map [-1..1] => [0..1]
    let u = x * 0.5 + 0.5;
    let v = y * 0.5 + 0.5;
    (quantize_u8_unit(u), quantize_u8_unit(v))
}

#[inline(always)]
fn quat_wxyz_to_oct_angle_u8(w: f32, x: f32, y: f32, z: f32) -> (u8, u8, u8) {
    // Ensure a stable hemisphere so theta stays in [0..pi]
    let mut w = w;
    let mut x = x;
    let mut y = y;
    let mut z = z;
    if w < 0.0 {
        w = -w;
        x = -x;
        y = -y;
        z = -z;
    }

    // Normalize defensively.
    let n2 = w * w + x * x + y * y + z * z;
    if !n2.is_finite() || n2 <= 0.0 {
        // identity
        return (255, 128, 0);
    }
    let inv_n = n2.sqrt().recip();
    w *= inv_n;
    x *= inv_n;
    y *= inv_n;
    z *= inv_n;

    // Clamp w to avoid NaNs from acos due to tiny drift.
    let w_clamped = w.clamp(-1.0, 1.0);

    // Axis-angle: theta = 2 acos(w), axis = xyz / sin(theta/2)
    let theta = 2.0 * w_clamped.acos(); // [0..pi]
    let s = (1.0 - w_clamped * w_clamped).sqrt(); // ~= sin(theta/2)

    let (axis_x, axis_y, axis_z) = if s > 1.0e-8 && s.is_finite() {
        (x / s, y / s, z / s)
    } else {
        (1.0, 0.0, 0.0)
    };

    let (ou, ov) = oct_encode_axis(axis_x, axis_y, axis_z);

    let mut ang = theta * (255.0 / PI);
    if !ang.is_finite() || ang < 0.0 {
        ang = 0.0;
    } else if ang > 255.0 {
        ang = 255.0;
    }
    let angle_u8 = (ang + 0.5).floor() as u8;

    (ou, ov, angle_u8)
}

#[inline(always)]
fn pack_word0_rgba(r: u8, g: u8, b: u8, a: u8) -> u32 {
    (r as u32) | ((g as u32) << 8) | ((b as u32) << 16) | ((a as u32) << 24)
}

#[inline(always)]
fn pack_word1_xy(hx: u16, hy: u16) -> u32 {
    (hx as u32) | ((hy as u32) << 16)
}

#[inline(always)]
fn pack_word2_z_oct(hz: u16, ou: u8, ov: u8) -> u32 {
    (hz as u32) | ((ou as u32) << 16) | ((ov as u32) << 24)
}

#[inline(always)]
fn pack_word3_scale_angle(sx: u8, sy: u8, sz: u8, ang: u8) -> u32 {
    (sx as u32) | ((sy as u32) << 8) | ((sz as u32) << 16) | ((ang as u32) << 24)
}

#[inline(always)]
fn encode_loop_points_impl<P, S, F>(
    payload: &[P],
    buf: &mut [u8],
    def_scale: u8,
    def_ou: u8,
    def_ov: u8,
    def_ang: u8,
    mut to_f16: F,
) -> BasicResult
where
    P: SpatialOwnedFull<S> + SpatialView<S>,
    S: PointScalar,
    F: FnMut(f32, f32, f32) -> (u16, u16, u16),
{
    debug_assert_eq!(buf.len(), payload.len() * RECORD_LEN);

    for (i, p) in payload.iter().enumerate() {
        let x = p.x().to_f32();
        let y = p.y().to_f32();
        let z = p.z().to_f32();
        if !(x.is_finite() && y.is_finite() && z.is_finite()) {
            return Err(format!("gsplat16: non-finite position at index {i}").into());
        }

        let (hx, hy, hz) = to_f16(x, y, z);

        let r = p.r_u8();
        let g = p.g_u8();
        let b = p.b_u8();
        let a = p.a_u8();

        let word0 = pack_word0_rgba(r, g, b, a);
        let word1 = pack_word1_xy(hx, hy);
        let word2 = pack_word2_z_oct(hz, def_ou, def_ov);
        let word3 = pack_word3_scale_angle(def_scale, def_scale, def_scale, def_ang);

        let off = i * RECORD_LEN;
        unsafe {
            write_u128_le(buf.as_mut_ptr().add(off), word0, word1, word2, word3);
        }
    }
    Ok(())
}

#[inline(always)]
fn encode_loop_splats_impl<P, S, F>(
    payload: &[P],
    buf: &mut [u8],
    mut to_f16: F,
) -> BasicResult
where
    P: SpatialOwnedFull<S> + SpatialView<S>,
    S: PointScalar,
    F: FnMut(f32, f32, f32) -> (u16, u16, u16),
{
    debug_assert_eq!(buf.len(), payload.len() * RECORD_LEN);

    for (i, p) in payload.iter().enumerate() {
        let x = p.x().to_f32();
        let y = p.y().to_f32();
        let z = p.z().to_f32();
        if !(x.is_finite() && y.is_finite() && z.is_finite()) {
            return Err(format!("gsplat16: non-finite position at index {i}").into());
        }

        let (hx, hy, hz) = to_f16(x, y, z);

        let r = p.r_u8();
        let g = p.g_u8();
        let b = p.b_u8();
        let a = p.a_u8();

        // Scale is stored as linear in spatial_utils; quantize ln(scale).
        let sx_l = p.scale_x().to_f32();
        let sy_l = p.scale_y().to_f32();
        let sz_l = p.scale_z().to_f32();
        if !(sx_l.is_finite() && sy_l.is_finite() && sz_l.is_finite()) {
            return Err(format!("gsplat16: non-finite scale at index {i}").into());
        }

        // Rotation is stored as w,x,y,z in spatial_utils.
        let rw = p.rot_w().to_f32();
        let rx = p.rot_x().to_f32();
        let ry = p.rot_y().to_f32();
        let rz = p.rot_z().to_f32();
        if !(rw.is_finite() && rx.is_finite() && ry.is_finite() && rz.is_finite()) {
            return Err(format!("gsplat16: non-finite rotation at index {i}").into());
        }

        let (ou, ov, ang) = quat_wxyz_to_oct_angle_u8(rw, rx, ry, rz);
        let sx = encode_scale_u8(sx_l);
        let sy = encode_scale_u8(sy_l);
        let sz = encode_scale_u8(sz_l);

        let word0 = pack_word0_rgba(r, g, b, a);
        let word1 = pack_word1_xy(hx, hy);
        let word2 = pack_word2_z_oct(hz, ou, ov);
        let word3 = pack_word3_scale_angle(sx, sy, sz, ang);

        let off = i * RECORD_LEN;
        unsafe {
            write_u128_le(buf.as_mut_ptr().add(off), word0, word1, word2, word3);
        }
    }

    Ok(())
}

#[inline(always)]
fn encode_loop_scalar<P, S>(payload: &[P], buf: &mut [u8], def_scale: u8) -> BasicResult
where
    P: SpatialOwnedFull<S> + SpatialView<S>,
    S: PointScalar,
{
    // Defaults for point payloads.
    let (def_ou, def_ov, def_ang) = (255u8, 128u8, 0u8);

    if P::KIND == SpatialKind::Splats {
        encode_loop_splats_impl::<P, S, _>(payload, buf, |x, y, z| {
            (f32_to_f16_bits(x), f32_to_f16_bits(y), f32_to_f16_bits(z))
        })
    } else {
        encode_loop_points_impl::<P, S, _>(payload, buf, def_scale, def_ou, def_ov, def_ang, |x, y, z| {
            (f32_to_f16_bits(x), f32_to_f16_bits(y), f32_to_f16_bits(z))
        })
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "f16c")]
unsafe fn encode_loop_f16c<P, S>(payload: &[P], buf: &mut [u8], def_scale: u8) -> BasicResult
where
    P: SpatialOwnedFull<S> + SpatialView<S>,
    S: PointScalar,
{
    let (def_ou, def_ov, def_ang) = (255u8, 128u8, 0u8);

    if P::KIND == SpatialKind::Splats {
        encode_loop_splats_impl::<P, S, _>(payload, buf, |x, y, z| {
            let (hx, hy, hz, _) = f16c::f32x4_to_f16x4_bits(x, y, z, 0.0);
            (hx, hy, hz)
        })
    } else {
        encode_loop_points_impl::<P, S, _>(payload, buf, def_scale, def_ou, def_ov, def_ang, |x, y, z| {
            let (hx, hy, hz, _) = f16c::f32x4_to_f16x4_bits(x, y, z, 0.0);
            (hx, hy, hz)
        })
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "fp16")]
unsafe fn encode_loop_fp16<P, S>(payload: &[P], buf: &mut [u8], def_scale: u8) -> BasicResult
where
    P: SpatialOwnedFull<S> + SpatialView<S>,
    S: PointScalar,
{
    let (def_ou, def_ov, def_ang) = (255u8, 128u8, 0u8);

    if P::KIND == SpatialKind::Splats {
        encode_loop_splats_impl::<P, S, _>(payload, buf, |x, y, z| {
            let (hx, hy, hz, _) = fp16::f32x4_to_f16x4_bits(x, y, z, 0.0);
            (hx, hy, hz)
        })
    } else {
        encode_loop_points_impl::<P, S, _>(payload, buf, def_scale, def_ou, def_ov, def_ang, |x, y, z| {
            let (hx, hy, hz, _) = fp16::f32x4_to_f16x4_bits(x, y, z, 0.0);
            (hx, hy, hz)
        })
    }
}

/// Encode a payload into the GSP16 container.
///
/// Appends to `out` (does not clear it). This function performs a single backend selection
/// outside the hot loop, then encodes records with minimal branching and without extra
/// allocations (beyond the `out` growth).
pub fn encode_from_payload_into<P, S>(
    payload: &[P],
    _params: &Gsplat16Params,
    out: &mut Vec<u8>,
) -> BasicResult
where
    P: SpatialOwnedFull<S> + SpatialView<S>,
    S: PointScalar,
{
    if payload.len() > (u32::MAX as usize) {
        return Err("gsplat16: payload too large".into());
    }
    let n = payload.len();
    let body_len = n
        .checked_mul(RECORD_LEN)
        .ok_or("gsplat16: size overflow")?;

    // Header
    out.reserve(HEADER_LEN + body_len);
    out.extend_from_slice(MAGIC);
    out.push(VERSION);

    let mut flags = 0u8;
    if P::KIND == SpatialKind::Splats {
        flags |= FLAG_SPLATS;
    }
    if P::COLOR_KIND == ColorKind::Rgba8 {
        flags |= FLAG_HAS_ALPHA;
    }
    out.push(flags);
    out.push(0); // rsv0
    out.extend_from_slice(&0u16.to_le_bytes()); // rsv1
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // rsv2

    debug_assert_eq!(out.len() % 4, 0);

    // Body: single resize then write in-place.
    let start = out.len();
    out.resize(start + body_len, 0);
    let buf = &mut out[start..];

    // Precompute scale default used for points (avoid exp/log per record).
    let def_scale = encode_scale_u8(1.0);

    // Backend selection (single dispatch outside the loop).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("f16c") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { encode_loop_f16c::<P, S>(payload, buf, def_scale) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("fp16") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { encode_loop_fp16::<P, S>(payload, buf, def_scale) };
        }
    }

    // Portable fallback.
    encode_loop_scalar::<P, S>(payload, buf, def_scale)
}
