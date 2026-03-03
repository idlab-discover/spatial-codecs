//! SOGP decoder.
//!
//! Provides two entry points matching your crate conventions:
//! - decode_into
//! - decode_into_flattened_vecs
use std::io::Read;

use spatial_utils::{
    point::Point3Rgb,
    traits::SpatialSink,
    utils::{
        point_scalar::PointScalar,
        quant::{dequantize_f32, max_q_u32},
    },
};
use crate::BasicResult;

use crate::utils::byte_cursor::ByteCursor;

use super::types::{comp_id, flags as hdr_flags, stream_id, MAGIC, VERSION};

const MAX_DECOMPRESSED_BYTES: u64 = 2_000_000_000; // 2GB safety cap
const MAX_EXPANSION_RATIO: u64 = 512; // anti-decompression-bomb guard

struct Header {
    flags: u8,
    pos_bits: u8,
    comp: u8,
    n: u32,
    mins: [f32; 3],
    maxs: [f32; 3],
    stream_count: u8,
}

fn parse_header<'a>(data: &'a [u8]) -> Result<(Header, ByteCursor<'a>), String> {
    if data.len() < 3 + 1 + 1 + 1 + 1 + 1 + 4 + 24 {
        return Err("sogp: truncated header".into());
    }
    if &data[0..3] != MAGIC {
        return Err("sogp: bad magic".into());
    }
    if data[3] != VERSION {
        return Err("sogp: unsupported version".into());
    }

    let mut cur = ByteCursor::new(&data[4..]);
    let flags = cur.read_u8().map_err(|e| format!("sogp: {e}"))?;
    let pos_bits = cur.read_u8().map_err(|e| format!("sogp: {e}"))?;
    let comp = cur.read_u8().map_err(|e| format!("sogp: {e}"))?;
    let _reserved = cur.read_u8().map_err(|e| format!("sogp: {e}"))?;
    let n = cur.read_u32_le().map_err(|e| format!("sogp: {e}"))?;

    let mut mins = [0f32; 3];
    let mut maxs = [0f32; 3];
    for i in 0..3 {
        mins[i] = cur.read_f32_le().map_err(|e| format!("sogp: {e}"))?;
    }
    for i in 0..3 {
        maxs[i] = cur.read_f32_le().map_err(|e| format!("sogp: {e}"))?;
    }

    let stream_count = cur.read_u8().map_err(|e| format!("sogp: {e}"))?;

    Ok((
        Header {
            flags,
            pos_bits,
            comp,
            n,
            mins,
            maxs,
            stream_count,
        },
        cur,
    ))
}

fn read_streams<'a>(
    hdr: &Header,
    mut cur: ByteCursor<'a>,
    full_len: usize,
) -> Result<std::collections::HashMap<u8, (u32, &'a [u8])>, String> {
    let mut map = std::collections::HashMap::with_capacity(hdr.stream_count as usize);

    for _ in 0..hdr.stream_count {
        let sid = cur.read_u8().map_err(|e| format!("sogp: {e}"))?;
        let plain_len = cur.read_u32_le().map_err(|e| format!("sogp: {e}"))?;
        let comp_len = cur.read_u32_le().map_err(|e| format!("sogp: {e}"))? as usize;

        // Defensive: bound expansion before allocating
        let plain_len_u64 = plain_len as u64;
        if plain_len_u64 > MAX_DECOMPRESSED_BYTES {
            return Err("sogp: declared stream too large".into());
        }
        let remaining = cur.remaining() as u64;
        if plain_len_u64
            > (full_len as u64)
                .saturating_mul(MAX_EXPANSION_RATIO)
                .max(remaining)
        {
            return Err("sogp: suspicious expansion ratio".into());
        }

        let payload = cur.take(comp_len).map_err(|e| format!("sogp: {e}"))?;
        if map.insert(sid, (plain_len, payload)).is_some() {
            return Err("sogp: duplicate stream id".into());
        }
    }

    Ok(map)
}

fn decompress(comp: u8, payload: &[u8], expected: usize) -> Result<Vec<u8>, String> {
    if expected == 0 {
        return Ok(Vec::new());
    }

    match comp {
        comp_id::NONE => {
            if payload.len() != expected {
                return Err("sogp: raw stream length mismatch".into());
            }
            Ok(payload.to_vec())
        }
        comp_id::ZSTD => {
            let mut dec = zstd::stream::read::Decoder::new(payload).map_err(|e| e.to_string())?;
            let mut out = Vec::with_capacity(expected.min(1 << 20));
            dec.read_to_end(&mut out).map_err(|e| e.to_string())?;
            if out.len() != expected {
                return Err("sogp: zstd stream length mismatch".into());
            }
            Ok(out)
        }
        comp_id::LZ4 => {
            let mut dec = lz4_flex::frame::FrameDecoder::new(payload);
            let mut out = Vec::with_capacity(expected.min(1 << 20));
            dec.read_to_end(&mut out).map_err(|e| e.to_string())?;
            if out.len() != expected {
                return Err("sogp: lz4 stream length mismatch".into());
            }
            Ok(out)
        }
        comp_id::SNAPPY => {
            let mut dec = snap::read::FrameDecoder::new(payload);
            let mut out = Vec::with_capacity(expected.min(1 << 20));
            dec.read_to_end(&mut out).map_err(|e| e.to_string())?;
            if out.len() != expected {
                return Err("sogp: snappy stream length mismatch".into());
            }
            Ok(out)
        }
        comp_id::OPENZL_SERIAL => {
            let out = rust_openzl::decompress_serial(payload)
                .map_err(|_| "sogp: openzl decompress failed".to_string())?;
            if out.len() != expected {
                return Err("sogp: openzl stream length mismatch".into());
            }
            Ok(out)
        }
        _ => Err("sogp: unknown compression id".into()),
    }
}

fn decode_core<P>(
    data: &[u8],
    mut out_pos: Option<&mut Vec<P::Scalar>>,
    mut out_col: Option<&mut Vec<u8>>,
    mut out_pts: Option<&mut Vec<P>>,
) -> BasicResult
where
    P: SpatialSink,
{
    let (hdr, cur) = parse_header(data)?;

    let n = hdr.n as usize;
    let has_color = (hdr.flags & hdr_flags::HAS_COLOR) != 0;
    let has_pos_hi = (hdr.flags & hdr_flags::HAS_POS_HI) != 0;
    let packed = (hdr.flags & hdr_flags::PACKED_STREAM) != 0;

    let maxq = max_q_u32(hdr.pos_bits) as f32;

    // Prepare scaling (range/maxq)
    let rx = (hdr.maxs[0] - hdr.mins[0]).max(0.0);
    let ry = (hdr.maxs[1] - hdr.mins[1]).max(0.0);
    let rz = (hdr.maxs[2] - hdr.mins[2]).max(0.0);

    let kx = if maxq > 0.0 && rx > f32::EPSILON {
        rx / maxq
    } else {
        0.0
    };
    let ky = if maxq > 0.0 && ry > f32::EPSILON {
        ry / maxq
    } else {
        0.0
    };
    let kz = if maxq > 0.0 && rz > f32::EPSILON {
        rz / maxq
    } else {
        0.0
    };

    // Expected plane lengths (3 bytes per point)
    let pos_len = n.checked_mul(3).ok_or("sogp: size overflow")?;
    let hi_len = if has_pos_hi { pos_len } else { 0 };
    let rgb_len = if has_color { pos_len } else { 0 };

    // Handle empty
    if n == 0 {
        if let Some(p) = out_pos.as_mut() {
            (**p).clear();
        }
        if let Some(c) = out_col.as_mut() {
            (**c).clear();
        }
        return Ok(());
    }

    let streams = read_streams(&hdr, cur, data.len())?;

    // Fast-path: packed stream => decompress once, split by offsets.
    let (pos_lo_buf, pos_hi_buf_opt, rgb_buf_opt, _backing) = if packed {
        let Some((plain_len, payload)) = streams.get(&stream_id::PACKED) else {
            return Err("sogp: missing packed stream".into());
        };
        let expected = pos_len + hi_len + rgb_len;
        if (*plain_len as usize) != expected {
            return Err("sogp: packed uncompressed length mismatch".into());
        }
        let backing = decompress(hdr.comp, payload, expected)?;
        let mut off = 0usize;
        let pos_lo = &backing[off..off + pos_len];
        off += pos_len;
        let pos_hi = if has_pos_hi {
            let s = &backing[off..off + hi_len];
            off += hi_len;
            Some(s)
        } else {
            None
        };
        let rgb = if has_color {
            let s = &backing[off..off + rgb_len];
            Some(s)
        } else {
            None
        };
        (
            pos_lo.to_vec(),
            pos_hi.map(|s| s.to_vec()),
            rgb.map(|s| s.to_vec()),
            Some(backing),
        )
    } else {
        // Per-plane mode: decompress individually (2–3 streams).
        let (plain_lo, pay_lo) = streams
            .get(&stream_id::POS_LO)
            .ok_or("sogp: missing pos_lo stream")?;
        if (*plain_lo as usize) != pos_len {
            return Err("sogp: pos_lo length mismatch".into());
        }
        let pos_lo = decompress(hdr.comp, pay_lo, pos_len)?;

        let pos_hi = if has_pos_hi {
            let (plain_hi, pay_hi) = streams
                .get(&stream_id::POS_HI)
                .ok_or("sogp: missing pos_hi stream")?;
            if (*plain_hi as usize) != pos_len {
                return Err("sogp: pos_hi length mismatch".into());
            }
            Some(decompress(hdr.comp, pay_hi, pos_len)?)
        } else {
            None
        };

        let rgb = if has_color {
            let (plain_rgb, pay_rgb) = streams
                .get(&stream_id::RGB)
                .ok_or("sogp: missing rgb stream")?;
            if (*plain_rgb as usize) != pos_len {
                return Err("sogp: rgb length mismatch".into());
            }
            Some(decompress(hdr.comp, pay_rgb, pos_len)?)
        } else {
            None
        };

        (pos_lo, pos_hi, rgb, None)
    };

    // Emit flattened or AoS.
    if let Some(pos_out) = out_pos.as_mut() {
        (**pos_out).resize(n * 3, P::Scalar::ZERO); // TODO: perhaps we could reuse the buffer, so we don't have to resize each time?
        for i in 0..n {
            let o = i * 3;
            let xl = pos_lo_buf[o] as u16;
            let yl = pos_lo_buf[o + 1] as u16;
            let zl = pos_lo_buf[o + 2] as u16;
            let (xh, yh, zh) = if let Some(ref hi) = pos_hi_buf_opt {
                (hi[o] as u16, hi[o + 1] as u16, hi[o + 2] as u16)
            } else {
                (0u16, 0u16, 0u16)
            };

            let qx = xl | (xh << 8);
            let qy = yl | (yh << 8);
            let qz = zl | (zh << 8);

            (**pos_out)[o] = P::Scalar::from_f32(dequantize_f32(qx, hdr.mins[0], kx));
            (**pos_out)[o + 1] = P::Scalar::from_f32(dequantize_f32(qy, hdr.mins[1], ky));
            (**pos_out)[o + 2] = P::Scalar::from_f32(dequantize_f32(qz, hdr.mins[2], kz));
        }
    }

    if let Some(col_out) = out_col.as_mut() {
        if has_color {
            let rgb = rgb_buf_opt.as_ref().ok_or("sogp: missing rgb buffer")?;
            (**col_out).resize(n * 3, 0); // TODO: perhaps we could reuse the buffer, so we don't have to resize each time?
            (**col_out).copy_from_slice(rgb);
        } else {
            (**col_out).clear();
        }
    }

    if let Some(pts_out) = out_pts.as_mut() {
        pts_out.reserve(n);
        let rgb = rgb_buf_opt.as_deref();

        for i in 0..n {
            let o = i * 3;

            let xl = pos_lo_buf[o] as u16;
            let yl = pos_lo_buf[o + 1] as u16;
            let zl = pos_lo_buf[o + 2] as u16;
            let (xh, yh, zh) = if let Some(ref hi) = pos_hi_buf_opt {
                (hi[o] as u16, hi[o + 1] as u16, hi[o + 2] as u16)
            } else {
                (0u16, 0u16, 0u16)
            };

            let qx = xl | (xh << 8);
            let qy = yl | (yh << 8);
            let qz = zl | (zh << 8);

            let (r, g, b) = if let Some(rgb) = rgb {
                (rgb[o], rgb[o + 1], rgb[o + 2])
            } else {
                (0, 0, 0)
            };

            let x = P::Scalar::from_f32(dequantize_f32(qx, hdr.mins[0], kx));
            let y = P::Scalar::from_f32(dequantize_f32(qy, hdr.mins[1], ky));
            let z = P::Scalar::from_f32(dequantize_f32(qz, hdr.mins[2], kz));

            P::push_xyz_rgb(pts_out, x, y, z, r, g, b);
        }
    }

    Ok(())
}

pub fn decode_into<P>(data: &[u8], out: &mut Vec<P>) -> BasicResult
where
    P: SpatialSink,
{
    decode_core(data, None, None, Some(out))
}

pub fn decode_into_flattened_vecs<S>(
    data: &[u8],
    pos_out: &mut Vec<S>,
    color_out: &mut Vec<u8>,
) -> BasicResult
where
    S: PointScalar,
{
    decode_core::<Point3Rgb<S>>(data, Some(pos_out), Some(color_out), None)
}

#[cfg(test)]
mod tests {
    use spatial_utils::point::Point3D;

    use super::*;
    use crate::codecs::sogp::encoder::encode_from_payload_into;
    use crate::codecs::sogp::types::{SogpCompression, SogpParams};

    fn lcg(seed: &mut u32) -> u32 {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        *seed
    }

    #[test]
    fn sogp_roundtrip_sane() {
        let mut pts = Vec::new();
        let mut s = 1u32;
        for _ in 0..50_000 {
            let x = (lcg(&mut s) as f32 / u32::MAX as f32) * 10.0 - 5.0;
            let y = (lcg(&mut s) as f32 / u32::MAX as f32) * 3.0 - 1.5;
            let z = (lcg(&mut s) as f32 / u32::MAX as f32) * 100.0;
            let r = (lcg(&mut s) & 0xFF) as u8;
            let g = (lcg(&mut s) & 0xFF) as u8;
            let b = (lcg(&mut s) & 0xFF) as u8;
            pts.push(Point3D { x, y, z, r, g, b });
        }

        let params = SogpParams {
            pos_bits: 16,
            morton_order: true,
            include_color: true,
            packed_stream: true,
            compression: SogpCompression::Zstd { level: 1 },
        };

        let mut buf = Vec::new();
        encode_from_payload_into(&pts, &params, &mut buf).unwrap();

        let mut pos = Vec::new();
        let mut col = Vec::new();
        decode_into_flattened_vecs::<f32>(&buf, &mut pos, &mut col).unwrap();

        assert_eq!(pos.len(), pts.len() * 3);
        assert_eq!(col.len(), pts.len() * 3);
    }

    #[test]
    fn sogp_truncated_rejected() {
        let data = b"SGP\x01\x00\x10\x01\x00\x00";
        let mut out = Vec::new();
        assert!(decode_into::<Point3D>(data, &mut out).is_err());
    }
}
