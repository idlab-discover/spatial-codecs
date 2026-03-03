use core::{f32::consts::PI, ptr};

use spatial_utils::{
    traits::{SpatialSink, SpatialKind},
    utils::point_scalar::PointScalar,
};
use crate::BasicResult;

use crate::codecs::gsplat16::types::{FLAG_SPLATS, HEADER_LEN, MAGIC, RECORD_LEN, VERSION};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::codecs::gsplat16::f16c;
#[cfg(target_arch = "aarch64")]
use crate::codecs::gsplat16::fp16;

const LN_SCALE_MIN: f32 = -12.0;
const LN_SCALE_MAX: f32 = 9.0;
const LN_SCALE_RANGE: f32 = LN_SCALE_MAX - LN_SCALE_MIN;
const LN_SCALE_SCALE_DEC: f32 = LN_SCALE_RANGE / 254.0;

#[derive(Clone, Copy, Debug)]
struct Header {
    file_is_splats: bool,
    count: u32,
}

#[inline(always)]
fn read_u32_le_at(data: &[u8], off: usize) -> Result<u32, Box<dyn std::error::Error>> {
    let s = data.get(off..off + 4).ok_or("gsplat16: truncated u32")?;
    let mut b = [0u8; 4];
    b.copy_from_slice(s);
    Ok(u32::from_le_bytes(b))
}

#[inline(always)]
fn parse_header(data: &[u8]) -> Result<Header, Box<dyn std::error::Error>> {
    if data.len() < HEADER_LEN {
        return Err("gsplat16: truncated header".into());
    }
    if &data[0..3] != MAGIC {
        return Err("gsplat16: bad magic".into());
    }
    if data[3] != VERSION {
        return Err("gsplat16: unsupported version".into());
    }
    let flags = data[4];
    let file_is_splats = (flags & FLAG_SPLATS) != 0;
    let count = read_u32_le_at(data, 8)?;
    Ok(Header { file_is_splats, count })
}

#[inline(always)]
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;

    let bits = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                // subnormal: normalize mantissa
                let mut m = mant;
                let mut e = -14i32;
                while (m & 0x0400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x03FF;
                let exp32 = (e + 127) as u32;
                sign | (exp32 << 23) | (m << 13)
            }
        }
        31 => sign | 0x7F80_0000 | (mant << 13), // Inf/NaN
        _ => {
            let exp32 = (exp as i32 - 15 + 127) as u32;
            sign | (exp32 << 23) | (mant << 13)
        }
    };

    f32::from_bits(bits)
}

#[inline(always)]
fn decode_scale_u8(u: u8) -> f32 {
    if u == 0 {
        return 0.0;
    }
    let ln = LN_SCALE_MIN + ((u as f32) - 1.0) * LN_SCALE_SCALE_DEC;
    ln.exp()
}

#[inline(always)]
fn oct_decode_axis(ou: u8, ov: u8) -> (f32, f32, f32) {
    // map [0..255] => [-1..1]
    let u = (ou as f32) * (1.0 / 255.0) * 2.0 - 1.0;
    let v = (ov as f32) * (1.0 / 255.0) * 2.0 - 1.0;

    let mut x = u;
    let mut y = v;
    let mut z = 1.0 - x.abs() - y.abs();

    // unfold if z < 0
    if z < 0.0 {
        let t = (-z).max(0.0);
        x += if x >= 0.0 { -t } else { t };
        y += if y >= 0.0 { -t } else { t };
        z = 1.0 - x.abs() - y.abs();
    }

    // normalize
    let n2 = x * x + y * y + z * z;
    if n2 <= 0.0 || !n2.is_finite() {
        return (1.0, 0.0, 0.0);
    }
    let inv = n2.sqrt().recip();
    (x * inv, y * inv, z * inv)
}

#[inline(always)]
fn decode_quat_wxyz(ou: u8, ov: u8, angle_u8: u8) -> (f32, f32, f32, f32) {
    let (ax, ay, az) = oct_decode_axis(ou, ov);

    let theta = (angle_u8 as f32) * (1.0 / 255.0) * PI;
    let half = 0.5 * theta;
    let (s, c) = half.sin_cos();

    let x = ax * s;
    let y = ay * s;
    let z = az * s;
    let w = c;

    // return WXYZ (spatial_utils convention)
    (w, x, y, z)
}

#[inline(always)]
unsafe fn read_u128_le(ptr_u8: *const u8) -> u128 {
    u128::from_le(ptr::read_unaligned(ptr_u8 as *const u128))
}

#[inline(always)]
fn words_from_record_u128(rec: u128) -> (u32, u32, u32, u32) {
    (
        (rec as u32),
        ((rec >> 32) as u32),
        ((rec >> 64) as u32),
        ((rec >> 96) as u32),
    )
}

#[inline(always)]
fn decode_points_impl<P, F>(
    body: &[u8],
    n: usize,
    out: &mut Vec<P>,
    mut from_f16: F,
) -> BasicResult
where
    P: SpatialSink + 'static,
    F: FnMut(u16, u16, u16) -> (f32, f32, f32),
{
    debug_assert_eq!(body.len(), n * RECORD_LEN);

    let out_start = out.len();
    out.reserve(n);

    for i in 0..n {
        let off = i * RECORD_LEN;
        unsafe {
            let base = body.as_ptr().add(off);
            let rec = read_u128_le(base);
            let (word0, word1, word2, _word3) = words_from_record_u128(rec);

            let r = (word0 & 0xFF) as u8;
            let g = ((word0 >> 8) & 0xFF) as u8;
            let b = ((word0 >> 16) & 0xFF) as u8;
            let a = ((word0 >> 24) & 0xFF) as u8;

            let hx = (word1 & 0xFFFF) as u16;
            let hy = ((word1 >> 16) & 0xFFFF) as u16;
            let hz = (word2 & 0xFFFF) as u16;

            let (x, y, z) = from_f16(hx, hy, hz);

            if !(x.is_finite() && y.is_finite() && z.is_finite()) {
                out.truncate(out_start);
                return Err(format!("gsplat16: non-finite position at index {i}").into());
            }

            P::push_xyz_rgba(
                out,
                <P::Scalar as PointScalar>::from_f32(x),
                <P::Scalar as PointScalar>::from_f32(y),
                <P::Scalar as PointScalar>::from_f32(z),
                r,
                g,
                b,
                a,
            );
        }
    }

    Ok(())
}

#[inline(always)]
fn decode_splats_impl<P, F>(
    body: &[u8],
    n: usize,
    out: &mut Vec<P>,
    mut from_f16: F,
) -> BasicResult
where
    P: SpatialSink + 'static,
    F: FnMut(u16, u16, u16) -> (f32, f32, f32),
{
    debug_assert_eq!(body.len(), n * RECORD_LEN);

    let out_start = out.len();
    out.reserve(n);

    for i in 0..n {
        let off = i * RECORD_LEN;
        unsafe {
            let base = body.as_ptr().add(off);
            let rec = read_u128_le(base);
            let (word0, word1, word2, word3) = words_from_record_u128(rec);

            let r = (word0 & 0xFF) as u8;
            let g = ((word0 >> 8) & 0xFF) as u8;
            let b = ((word0 >> 16) & 0xFF) as u8;
            let a = ((word0 >> 24) & 0xFF) as u8;

            let hx = (word1 & 0xFFFF) as u16;
            let hy = ((word1 >> 16) & 0xFFFF) as u16;
            let hz = (word2 & 0xFFFF) as u16;

            let (x, y, z) = from_f16(hx, hy, hz);
            if !(x.is_finite() && y.is_finite() && z.is_finite()) {
                out.truncate(out_start);
                return Err(format!("gsplat16: non-finite position at index {i}").into());
            }

            let mean = [
                <P::Scalar as PointScalar>::from_f32(x),
                <P::Scalar as PointScalar>::from_f32(y),
                <P::Scalar as PointScalar>::from_f32(z),
            ];
            let rgba = [r, g, b, a];

            let ou = ((word2 >> 16) & 0xFF) as u8;
            let ov = ((word2 >> 24) & 0xFF) as u8;
            let sx_u = (word3 & 0xFF) as u8;
            let sy_u = ((word3 >> 8) & 0xFF) as u8;
            let sz_u = ((word3 >> 16) & 0xFF) as u8;
            let ang = ((word3 >> 24) & 0xFF) as u8;

            let sx = decode_scale_u8(sx_u);
            let sy = decode_scale_u8(sy_u);
            let sz = decode_scale_u8(sz_u);
            let (rw, rx, ry, rz) = decode_quat_wxyz(ou, ov, ang);

            let scale = [
                <P::Scalar as PointScalar>::from_f32(sx),
                <P::Scalar as PointScalar>::from_f32(sy),
                <P::Scalar as PointScalar>::from_f32(sz),
            ];
            let rot = [
                <P::Scalar as PointScalar>::from_f32(rw),
                <P::Scalar as PointScalar>::from_f32(rx),
                <P::Scalar as PointScalar>::from_f32(ry),
                <P::Scalar as PointScalar>::from_f32(rz),
            ];

            P::push_splat(out, mean, rgba, scale, rot);
        }
    }

    Ok(())
}

#[inline(always)]
fn decode_points_scalar<P>(body: &[u8], n: usize, out: &mut Vec<P>) -> BasicResult
where
    P: SpatialSink + 'static,
{
    decode_points_impl::<P, _>(body, n, out, |hx, hy, hz| {
        (f16_bits_to_f32(hx), f16_bits_to_f32(hy), f16_bits_to_f32(hz))
    })
}

#[inline(always)]
fn decode_splats_scalar<P>(body: &[u8], n: usize, out: &mut Vec<P>) -> BasicResult
where
    P: SpatialSink + 'static,
{
    decode_splats_impl::<P, _>(body, n, out, |hx, hy, hz| {
        (f16_bits_to_f32(hx), f16_bits_to_f32(hy), f16_bits_to_f32(hz))
    })
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "f16c")]
unsafe fn decode_points_f16c<P>(body: &[u8], n: usize, out: &mut Vec<P>) -> BasicResult
where
    P: SpatialSink + 'static,
{
    decode_points_impl::<P, _>(body, n, out, |hx, hy, hz| unsafe {
        let (x, y, z, _) = f16c::f16x4_bits_to_f32x4(hx, hy, hz, 0);
        (x, y, z)
    })
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "f16c")]
unsafe fn decode_splats_f16c<P>(body: &[u8], n: usize, out: &mut Vec<P>) -> BasicResult
where
    P: SpatialSink + 'static,
{
    decode_splats_impl::<P, _>(body, n, out, |hx, hy, hz| unsafe {
        let (x, y, z, _) = f16c::f16x4_bits_to_f32x4(hx, hy, hz, 0);
        (x, y, z)
    })
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "fp16")]
unsafe fn decode_points_fp16<P>(body: &[u8], n: usize, out: &mut Vec<P>) -> BasicResult
where
    P: SpatialSink + 'static,
{
    decode_points_impl::<P, _>(body, n, out, |hx, hy, hz| unsafe {
        let (x, y, z, _) = fp16::f16x4_bits_to_f32x4(hx, hy, hz, 0);
        (x, y, z)
    })
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "fp16")]
unsafe fn decode_splats_fp16<P>(body: &[u8], n: usize, out: &mut Vec<P>) -> BasicResult
where
    P: SpatialSink + 'static,
{
    decode_splats_impl::<P, _>(body, n, out, |hx, hy, hz| unsafe {
        let (x, y, z, _) = fp16::f16x4_bits_to_f32x4(hx, hy, hz, 0);
        (x, y, z)
    })
}

/// Decode into `Vec<P>`.
pub fn decode_into<P>(data: &[u8], out: &mut Vec<P>) -> BasicResult
where
    P: SpatialSink + 'static,
{
    let h = parse_header(data)?;
    let n = h.count as usize;

    let body_len = n
        .checked_mul(RECORD_LEN)
        .ok_or("gsplat16: size overflow")?;

    if data.len() != HEADER_LEN + body_len {
        return Err("gsplat16: invalid length (header/count mismatch)".into());
    }

    let body = &data[HEADER_LEN..];

    // If caller wants Points, we *never* decode scale/quat (huge perf win).
    let out_kind = <P as spatial_utils::traits::SpatialMeta>::KIND;
    let emit_points_only = SpatialKind::Points == out_kind;

    // Also skip splat decoding if input isn't splats.
    let need_splat_math = !emit_points_only && h.file_is_splats;

    // Backend dispatch once per call.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("f16c") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe {
                if need_splat_math {
                    decode_splats_f16c::<P>(body, n, out)
                } else {
                    decode_points_f16c::<P>(body, n, out)
                }
            };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("fp16") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe {
                if need_splat_math {
                    decode_splats_fp16::<P>(body, n, out)
                } else {
                    decode_points_fp16::<P>(body, n, out)
                }
            };
        }
    }

    // Portable fallback.
    if need_splat_math {
        decode_splats_scalar::<P>(body, n, out)
    } else {
        decode_points_scalar::<P>(body, n, out)
    }
}

/// Decode into flattened buffers:
/// - `pos_out`: [x0,y0,z0,x1,y1,z1,...]
/// - `color_out`: [r0,g0,b0,r1,g1,b1,...]
pub fn decode_into_flattened_vecs<S>(
    data: &[u8],
    pos_out: &mut Vec<S>,
    color_out: &mut Vec<u8>,
) -> BasicResult
where
    S: PointScalar,
{
    let h = parse_header(data)?;
    let n = h.count as usize;

    let body_len = n
        .checked_mul(RECORD_LEN)
        .ok_or("gsplat16: size overflow")?;

    if data.len() != HEADER_LEN + body_len {
        return Err("gsplat16: invalid length (header/count mismatch)".into());
    }

    let body = &data[HEADER_LEN..];

    let add_pos = n.checked_mul(3).ok_or("gsplat16: pos size overflow")?;
    let add_col = n.checked_mul(3).ok_or("gsplat16: col size overflow")?;

    let pos_start = pos_out.len();
    let col_start = color_out.len();

    pos_out.reserve_exact(add_pos);
    color_out.reserve_exact(add_col);

    #[inline(always)]
    fn flat_impl<Ss, F>(
        body: &[u8],
        n: usize,
        pos_out: &mut Vec<Ss>,
        color_out: &mut Vec<u8>,
        pos_start: usize,
        col_start: usize,
        mut from_f16: F,
    ) -> BasicResult
    where
        Ss: PointScalar,
        F: FnMut(u16, u16, u16) -> (f32, f32, f32),
    {
        unsafe {
            pos_out.set_len(pos_start + n * 3);
            color_out.set_len(col_start + n * 3);

            let mut pptr = pos_out.as_mut_ptr().add(pos_start);
            let mut cptr = color_out.as_mut_ptr().add(col_start);

            for i in 0..n {
                let off = i * RECORD_LEN;
                let base = body.as_ptr().add(off);
                let rec = read_u128_le(base);
                let (word0, word1, word2, _word3) = words_from_record_u128(rec);

                let hx = (word1 & 0xFFFF) as u16;
                let hy = ((word1 >> 16) & 0xFFFF) as u16;
                let hz = (word2 & 0xFFFF) as u16;

                let (x, y, z) = from_f16(hx, hy, hz);

                if !(x.is_finite() && y.is_finite() && z.is_finite()) {
                    pos_out.set_len(pos_start);
                    color_out.set_len(col_start);
                    return Err(format!("gsplat16: non-finite position at index {i}").into());
                }

                ptr::write(pptr.add(0), Ss::from_f32(x));
                ptr::write(pptr.add(1), Ss::from_f32(y));
                ptr::write(pptr.add(2), Ss::from_f32(z));
                pptr = pptr.add(3);

                let r = (word0 & 0xFF) as u8;
                let g = ((word0 >> 8) & 0xFF) as u8;
                let b = ((word0 >> 16) & 0xFF) as u8;

                ptr::write(cptr.add(0), r);
                ptr::write(cptr.add(1), g);
                ptr::write(cptr.add(2), b);
                cptr = cptr.add(3);
            }
        }

        Ok(())
    }

    // Backend dispatch once per call.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("f16c") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe {
                #[target_feature(enable = "f16c")]
                unsafe fn flat_f16c<Ss: PointScalar>(
                    body: &[u8],
                    n: usize,
                    pos_out: &mut Vec<Ss>,
                    color_out: &mut Vec<u8>,
                    pos_start: usize,
                    col_start: usize,
                ) -> BasicResult {
                    flat_impl(body, n, pos_out, color_out, pos_start, col_start, |hx, hy, hz| unsafe {
                        let (x, y, z, _) = f16c::f16x4_bits_to_f32x4(hx, hy, hz, 0);
                        (x, y, z)
                    })
                }
                flat_f16c::<S>(body, n, pos_out, color_out, pos_start, col_start)
            };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("fp16") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe {
                #[target_feature(enable = "fp16")]
                unsafe fn flat_fp16<Ss: PointScalar>(
                    body: &[u8],
                    n: usize,
                    pos_out: &mut Vec<Ss>,
                    color_out: &mut Vec<u8>,
                    pos_start: usize,
                    col_start: usize,
                ) -> BasicResult {
                    flat_impl(body, n, pos_out, color_out, pos_start, col_start, |hx, hy, hz| unsafe {
                        let (x, y, z, _) = fp16::f16x4_bits_to_f32x4(hx, hy, hz, 0);
                        (x, y, z)
                    })
                }
                flat_fp16::<S>(body, n, pos_out, color_out, pos_start, col_start)
            };
        }
    }

    // Portable fallback.
    flat_impl(body, n, pos_out, color_out, pos_start, col_start, |hx, hy, hz| {
        (f16_bits_to_f32(hx), f16_bits_to_f32(hy), f16_bits_to_f32(hz))
    })
}
