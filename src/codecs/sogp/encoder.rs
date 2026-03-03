//! SOGP encoder.
//!
//! Layout (V1):
//!   magic[3]="SGP"
//!   version u8 = 1
//!   flags   u8 (HAS_COLOR, HAS_POS_HI, MORTON_ORDERED, PACKED_STREAM)
//!   pos_bits u8 (1..=16)
//!   comp_id u8 (NONE/ZSTD/LZ4/SNAPPY/OPENZL_SERIAL)
//!   reserved u8
//!   point_count u32
//!   mins[3] f32
//!   maxs[3] f32
//!   stream_count u8
//!   repeated stream:
//!     stream_id u8
//!     u32 uncompressed_len
//!     u32 compressed_len
//!     bytes[compressed_len]
use std::io::Write;

use spatial_utils::{
    traits::PointTraits,
    utils::{
        aabb::compute_aabb_generic,
        morton::morton_order,
        point_scalar::PointScalar,
        quant::{max_q_u32, quantize_u16},
    },
};

use crate::utils::byte_cursor::{write_scalar_le, write_u32_le, write_u8};

use super::types::{
    comp_id, flags as hdr_flags, stream_id, SogpCompression, SogpParams, MAGIC, VERSION,
};

#[inline]
fn comp_id_of(c: &SogpCompression) -> u8 {
    match c {
        SogpCompression::None => comp_id::NONE,
        SogpCompression::Zstd { .. } => comp_id::ZSTD,
        SogpCompression::Lz4 => comp_id::LZ4,
        SogpCompression::Snappy => comp_id::SNAPPY,
        SogpCompression::OpenzlSerial => comp_id::OPENZL_SERIAL,
    }
}

/// Encode into `out` (appends).
pub fn encode_from_payload_into<P, S>(
    points: &[P],
    params: &SogpParams,
    out: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>>
where
    P: PointTraits<S>,
    S: PointScalar,
{
    if points.len() > u32::MAX as usize {
        return Err("sogp: too many points for u32 header".into());
    }
    let pos_bits = params.pos_bits;
    let maxq = max_q_u32(pos_bits);

    let (mins, maxs) = compute_aabb_generic(points).map_err(|e| format!("sogp: {e}"))?;
    let n = points.len() as u32;

    let has_pos_hi = pos_bits > 8;
    let has_color = params.include_color;

    // Quantization scalars
    let rx = (maxs[0] - mins[0]).max(S::ZERO);
    let ry = (maxs[1] - mins[1]).max(S::ZERO);
    let rz = (maxs[2] - mins[2]).max(S::ZERO);

    let invx = if rx > S::EPS { S::ONE / rx } else { S::ZERO };
    let invy = if ry > S::EPS { S::ONE / ry } else { S::ZERO };
    let invz = if rz > S::EPS { S::ONE / rz } else { S::ZERO };
    // Planes are 3 bytes per point (no alpha waste)
    let n_usize = points.len();
    let pos_len = n_usize
        .checked_mul(3)
        .ok_or("sogp: size overflow (pos plane)")?;
    let rgb_len = pos_len;

    let mut pos_lo = vec![0u8; pos_len];
    let mut pos_hi = if has_pos_hi {
        Some(vec![0u8; pos_len])
    } else {
        None
    };
    let mut rgb = if has_color {
        Some(vec![0u8; rgb_len])
    } else {
        None
    };

    // Morton permutation (optional)
    if params.morton_order && points.len() > 1 {
        let order = morton_order(points, mins, maxs);
        // Pack in Morton order (single pass over permutation)
        for (i, &ix) in order.iter().enumerate() {
            let p = &points[ix as usize];

            // Quantize into u16 (even if bits<16); hi plane may be omitted for <=8 bits.
            let qx = quantize_u16(p.x(), mins[0], invx, maxq);
            let qy = quantize_u16(p.y(), mins[1], invy, maxq);
            let qz = quantize_u16(p.z(), mins[2], invz, maxq);

            let o = i * 3;
            pos_lo[o] = (qx & 0xFF) as u8;
            pos_lo[o + 1] = (qy & 0xFF) as u8;
            pos_lo[o + 2] = (qz & 0xFF) as u8;

            if let Some(ref mut hi) = pos_hi {
                hi[o] = (qx >> 8) as u8;
                hi[o + 1] = (qy >> 8) as u8;
                hi[o + 2] = (qz >> 8) as u8;
            }

            if let Some(ref mut c) = rgb {
                c[o] = p.r_u8();
                c[o + 1] = p.g_u8();
                c[o + 2] = p.b_u8();
            }
        }
    } else {
        // Pack in original order
        for (i, p) in points.iter().enumerate() {
            // Quantize into u16 (even if bits<16); hi plane may be omitted for <=8 bits.
            let qx = quantize_u16(p.x(), mins[0], invx, maxq);
            let qy = quantize_u16(p.y(), mins[1], invy, maxq);
            let qz = quantize_u16(p.z(), mins[2], invz, maxq);

            let o = i * 3;
            pos_lo[o] = (qx & 0xFF) as u8;
            pos_lo[o + 1] = (qy & 0xFF) as u8;
            pos_lo[o + 2] = (qz & 0xFF) as u8;

            if let Some(ref mut hi) = pos_hi {
                hi[o] = (qx >> 8) as u8;
                hi[o + 1] = (qy >> 8) as u8;
                hi[o + 2] = (qz >> 8) as u8;
            }

            if let Some(ref mut c) = rgb {
                c[o] = p.r_u8();
                c[o + 1] = p.g_u8();
                c[o + 2] = p.b_u8();
            }
        }
    };

    // ---------------- Header ----------------
    let mut flags = 0u8;
    if has_color {
        flags |= hdr_flags::HAS_COLOR;
    }
    if has_pos_hi {
        flags |= hdr_flags::HAS_POS_HI;
    }
    if params.morton_order {
        flags |= hdr_flags::MORTON_ORDERED;
    }
    if params.packed_stream {
        flags |= hdr_flags::PACKED_STREAM;
    }

    let cid = comp_id_of(&params.compression);

    // Reserve a rough upper bound to reduce reallocations.
    // (Header ~ 3+1+1+1+1+1 + 4 + 24 + 1 + stream framing)
    out.reserve(64);

    out.extend_from_slice(MAGIC);
    write_u8(out, VERSION);
    write_u8(out, flags);
    write_u8(out, pos_bits);
    write_u8(out, cid);
    write_u8(out, 0); // reserved
    write_u32_le(out, n);

    for v in mins {
        write_scalar_le(out, v);
    }
    for v in maxs {
        write_scalar_le(out, v);
    }

    // Stream(s)
    if n == 0 {
        write_u8(out, 0);
        return Ok(());
    }

    if params.packed_stream {
        // Single stream that concatenates all planes in a fixed order:
        //   pos_lo | (pos_hi?) | (rgb?)
        write_u8(out, 1);
        write_stream_multi(
            out,
            stream_id::PACKED,
            &params.compression,
            &[
                (&pos_lo[..], pos_len),
                (
                    pos_hi.as_deref().unwrap_or(&[]),
                    pos_hi.as_ref().map(|_| pos_len).unwrap_or(0),
                ),
                (
                    rgb.as_deref().unwrap_or(&[]),
                    rgb.as_ref().map(|_| rgb_len).unwrap_or(0),
                ),
            ],
        )?;
    } else {
        // Per-plane streams (more flexible, slightly more overhead)
        let mut sc = 1u8; // pos_lo always
        if has_pos_hi {
            sc += 1;
        }
        if has_color {
            sc += 1;
        }
        write_u8(out, sc);

        write_stream_single(out, stream_id::POS_LO, &params.compression, &pos_lo)?;
        if let Some(hi) = pos_hi.as_deref() {
            write_stream_single(out, stream_id::POS_HI, &params.compression, hi)?;
        }
        if let Some(c) = rgb.as_deref() {
            write_stream_single(out, stream_id::RGB, &params.compression, c)?;
        }
    }

    Ok(())
}

fn write_stream_single(
    out: &mut Vec<u8>,
    sid: u8,
    comp: &SogpCompression,
    plain: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    write_stream_multi(out, sid, comp, &[(plain, plain.len())])
}

fn write_stream_multi(
    out: &mut Vec<u8>,
    sid: u8,
    comp: &SogpCompression,
    parts: &[(&[u8], usize)],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut total_plain = 0usize;
    for &(_, len) in parts {
        total_plain = total_plain
            .checked_add(len)
            .ok_or("sogp: size overflow (packed len)")?;
    }

    write_u8(out, sid);
    write_u32_le(out, total_plain as u32);

    // placeholder compressed_len
    let comp_len_pos = out.len();
    write_u32_le(out, 0);

    let start = out.len();

    match comp {
        SogpCompression::None => {
            for &(p, _) in parts {
                out.extend_from_slice(p);
            }
        }
        SogpCompression::Zstd { level } => {
            let mut enc = zstd::stream::write::Encoder::new(&mut *out, *level)?;
            for &(p, _) in parts {
                enc.write_all(p)?;
            }
            enc.finish()?;
        }
        SogpCompression::Lz4 => {
            let mut enc = lz4_flex::frame::FrameEncoder::new(&mut *out);
            for &(p, _) in parts {
                enc.write_all(p)?;
            }
            enc.finish()?;
        }
        SogpCompression::Snappy => {
            let mut enc = snap::write::FrameEncoder::new(&mut *out);
            for &(p, _) in parts {
                enc.write_all(p)?;
            }
            enc.flush()?;
        }
        #[cfg(feature = "openzl")]
        SogpCompression::OpenzlSerial => {
            // OpenZL serial needs a contiguous buffer.
            let mut tmp = Vec::with_capacity(total_plain);
            for &(p, _) in parts {
                tmp.extend_from_slice(p);
            }
            let oz = rust_openzl::compress_serial(&tmp)?;
            out.extend_from_slice(&oz);
        }
        #[cfg(not(feature = "openzl"))]
        SogpCompression::OpenzlSerial => {
            return Err("OpenZL encoding not available (feature disabled)".into());
        }
    }

    let comp_len = (out.len() - start) as u32;
    out[comp_len_pos..comp_len_pos + 4].copy_from_slice(&comp_len.to_le_bytes());
    Ok(())
}
