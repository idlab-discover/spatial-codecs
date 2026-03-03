//! Decoder for the OpenZL wrapper format.
//!
//! The framing is shared with the encoder (`OZL` magic, version byte, flags). Depending
//! on the mode we either decompress an inner codec (serial) or numeric columns (columnar).

use crate::decoder::decode_into as decode_inner;
use spatial_utils::{traits::SpatialSink, utils::point_scalar::PointScalar};
use crate::BasicResult;

use rust_openzl::{decompress_numeric, decompress_serial};

const MAGIC: &[u8; 3] = b"OZL";
const V1: u8 = 1;

// ---------- small helpers ----------

#[inline]
fn read_u8(cur: &mut &[u8]) -> Result<u8, &'static str> {
    if cur.is_empty() {
        return Err("ozl: truncated (u8)");
    }
    let v = cur[0];
    *cur = &cur[1..];
    Ok(v)
}

#[inline]
fn read_u32(cur: &mut &[u8]) -> Result<u32, &'static str> {
    if cur.len() < 4 {
        return Err("ozl: truncated (u32)");
    }
    let v = u32::from_le_bytes(cur[0..4].try_into().unwrap());
    *cur = &cur[4..];
    Ok(v)
}

#[inline]
fn take<'a>(cur: &mut &'a [u8], len: usize) -> Result<&'a [u8], &'static str> {
    if cur.len() < len {
        return Err("ozl: truncated (take)");
    }
    let (h, t) = cur.split_at(len);
    *cur = t;
    Ok(h)
}

// We stored packed RGB per point as 0x00RRGGBB in the columnar mode when pack_rgb=true.
#[inline]
fn unpack_rgb(rgb: &[u32]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut r = Vec::with_capacity(rgb.len());
    let mut g = Vec::with_capacity(rgb.len());
    let mut b = Vec::with_capacity(rgb.len());
    for &v in rgb {
        r.push(((v >> 16) & 0xFF) as u8);
        g.push(((v >> 8) & 0xFF) as u8);
        b.push((v & 0xFF) as u8);
    }
    (r, g, b)
}

#[inline]
fn unpack_rgb_triplet(v: u32) -> (u8, u8, u8) {
    (
        ((v >> 16) & 0xFF) as u8,
        ((v >> 8) & 0xFF) as u8,
        (v & 0xFF) as u8,
    )
}

// ---------------------------------------------------------------------
// 1. AoS (Vec<Point3D>) decode
//
// Mirrors your Draco-style decode_into(), using flatten first.
// ---------------------------------------------------------------------

/// Decode an OpenZL payload into `Vec<P>`.
pub fn decode_into<P>(data: &[u8], out: &mut Vec<P>) -> BasicResult
where
    P: SpatialSink + 'static,
{
    if data.len() < 5 {
        return Err("ozl: truncated header".into());
    }
    if &data[0..3] != MAGIC {
        return Err("ozl: bad magic".into());
    }
    if data[3] != V1 {
        return Err("ozl: bad version".into());
    }
    let mut cur = &data[4..];

    let flags = read_u8(&mut cur).map_err(|e| e.to_string())?;
    let is_columnar = (flags & 1) != 0;

    // SERIAL MODE: decompress inner bytes and let the inner decoder fill `out` directly.
    if !is_columnar {
        let len = read_u32(&mut cur).map_err(|e| e.to_string())? as usize;
        let payload = take(&mut cur, len).map_err(|e| e.to_string())?;
        let inner_bytes =
            decompress_serial(payload).map_err(|_| "ozl: decompress_serial failed")?;
        decode_inner(&inner_bytes, out).map_err(|e| format!("ozl(inner): {e}"))?;
        return Ok(());
    }

    // COLUMNAR MODE: decompress numeric columns, then construct Point3D in one pass.
    let n = read_u32(&mut cur).map_err(|e| e.to_string())? as usize;
    let sc = read_u8(&mut cur).map_err(|e| e.to_string())? as usize;
    if sc != 4 && sc != 6 {
        return Err("ozl: bad stream_count".into());
    }

    let mut blobs = Vec::with_capacity(sc);
    for _ in 0..sc {
        let l = read_u32(&mut cur).map_err(|e| e.to_string())? as usize;
        blobs.push(take(&mut cur, l).map_err(|e| e.to_string())?);
    }

    let xs: Vec<P::Scalar> =
        decompress_numeric(blobs[0]).map_err(|_| "ozl: decompress_numeric(x) failed")?;
    let ys: Vec<P::Scalar> =
        decompress_numeric(blobs[1]).map_err(|_| "ozl: decompress_numeric(y) failed")?;
    let zs: Vec<P::Scalar> =
        decompress_numeric(blobs[2]).map_err(|_| "ozl: decompress_numeric(z) failed")?;

    out.reserve(n);

    if sc == 4 {
        // rgb packed u32: unpack on the fly to avoid extra buffers
        let rgb: Vec<u32> =
            decompress_numeric(blobs[3]).map_err(|_| "ozl: decompress_numeric(rgb) failed")?;
        let m = n.min(xs.len()).min(ys.len()).min(zs.len()).min(rgb.len());
        for i in 0..m {
            let (r, g, b) = unpack_rgb_triplet(rgb[i]);
            P::push_xyz_rgb(out, xs[i], ys[i], zs[i], r, g, b);
        }
    } else {
        // r,g,b separate
        let rs: Vec<u8> =
            decompress_numeric(blobs[3]).map_err(|_| "ozl: decompress_numeric(r) failed")?;
        let gs: Vec<u8> =
            decompress_numeric(blobs[4]).map_err(|_| "ozl: decompress_numeric(g) failed")?;
        let bs: Vec<u8> =
            decompress_numeric(blobs[5]).map_err(|_| "ozl: decompress_numeric(b) failed")?;
        let m = n
            .min(xs.len())
            .min(ys.len())
            .min(zs.len())
            .min(rs.len())
            .min(gs.len())
            .min(bs.len());
        for i in 0..m {
            P::push_xyz_rgb(out, xs[i], ys[i], zs[i], rs[i], gs[i], bs[i]);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------
// 2. Flattened decode
//
//    decode_into_flattened_vecs(&[u8], &mut pos_out, &mut color_out)
//
//    - pos_out = [x0,y0,z0, x1,y1,z1, ...]  (f32)
//    - color_out = [r0,g0,b0, r1,g1,b1, ...] (u8)
// ---------------------------------------------------------------------

/// Decode into flattened buffers (`pos_out`, `color_out`).
pub fn decode_into_flattened_vecs<S>(
    data: &[u8],
    pos_out: &mut Vec<S>,
    color_out: &mut Vec<u8>,
) -> BasicResult
where
    S: PointScalar,
{
    // 0. Basic header sanity
    if data.len() < 3 + 1 + 1 {
        return Err("ozl: truncated header".into());
    }
    if &data[0..3] != MAGIC {
        return Err("ozl: bad magic".into());
    }
    if data[3] != V1 {
        return Err("ozl: bad version".into());
    }

    // cursor to payload after magic(3) + version(1)
    let mut cur = &data[4..];

    // 1. flags
    let flags = read_u8(&mut cur).map_err(|e| e.to_string())?;
    let is_columnar = (flags & 1) != 0;

    if !is_columnar {
        // --- SERIAL MODE ---
        //
        // Layout:
        //   u32 len
        //   <len bytes> = openzl::compress_serial(inner_bytes)
        // After decompress_serial() we get `inner_bytes`, which themselves have a 3-byte magic
        // (QNT / BC1 / DRA / etc.). We now want a flattened version of THAT.
        //
        // We'll just decode_inner() into a temporary Vec<Point3Rgb<S>>,
        // then flatten into the requested output layout.

        let len = read_u32(&mut cur).map_err(|e| e.to_string())? as usize;
        let payload = take(&mut cur, len).map_err(|e| e.to_string())?;

        let inner_bytes =
            decompress_serial(payload).map_err(|_| "ozl: decompress_serial failed".to_string())?;

        // We need flattened, but decode_inner currently gives AoS (Vec<Point3Rgb<S>>) via
        // decode_into(&inner_bytes, &mut Vec<Point3Rgb<S>>).
        // We'll build points AoS and then flatten them.

        return crate::decoder::decode_into_flattened_vecs(&inner_bytes, pos_out, color_out);
    }

    // --- COLUMNAR MODE ---
    //
    // Layout:
    //   u32 n_points
    //   u8 stream_count  (4 if packed RGB into u32; 6 if separate r,g,b)
    //   loop stream_count:
    //       u32 blob_len
    //       [blob_len bytes of openzl::compress_numeric(...)]
    //
    // Order of streams:
    //   [x, y, z, rgb]            if stream_count == 4
    //   [x, y, z, r,  g,  b]      if stream_count == 6
    //
    // We'll decompress_numeric() each of these, reconstruct SoA,
    // then emit into the caller's pos_out/color_out in interleaved form.

    let n_points = read_u32(&mut cur).map_err(|e| e.to_string())? as usize;
    let sc = read_u8(&mut cur).map_err(|e| e.to_string())? as usize;

    if sc != 4 && sc != 6 {
        return Err("ozl: bad stream_count".into());
    }

    let mut blob_slices = Vec::with_capacity(sc);
    for _ in 0..sc {
        let l = read_u32(&mut cur).map_err(|e| e.to_string())? as usize;
        let slice = take(&mut cur, l).map_err(|e| e.to_string())?;
        blob_slices.push(slice);
    }

    // Decompress into typed vectors
    let xs: Vec<S> =
        decompress_numeric(blob_slices[0]).map_err(|_| "ozl: decompress_numeric(x) failed")?;
    let ys: Vec<S> =
        decompress_numeric(blob_slices[1]).map_err(|_| "ozl: decompress_numeric(y) failed")?;
    let zs: Vec<S> =
        decompress_numeric(blob_slices[2]).map_err(|_| "ozl: decompress_numeric(z) failed")?;

    let (rs, gs, bs) = if sc == 4 {
        let rgb: Vec<u32> = decompress_numeric(blob_slices[3])
            .map_err(|_| "ozl: decompress_numeric(rgb) failed")?;
        unpack_rgb(&rgb)
    } else {
        (
            decompress_numeric::<u8>(blob_slices[3])
                .map_err(|_| "ozl: decompress_numeric(r) failed")?,
            decompress_numeric::<u8>(blob_slices[4])
                .map_err(|_| "ozl: decompress_numeric(g) failed")?,
            decompress_numeric::<u8>(blob_slices[5])
                .map_err(|_| "ozl: decompress_numeric(b) failed")?,
        )
    };

    // Produce flattened output:
    // pos_out = [x0,y0,z0,x1,y1,z1,...]
    // color_out = [r0,g0,b0,r1,g1,b1,...]
    let m = n_points
        .min(xs.len())
        .min(ys.len())
        .min(zs.len())
        .min(rs.len())
        .min(gs.len())
        .min(bs.len());

    pos_out.reserve(m * 3);
    color_out.reserve(m * 3);
    for i in 0..m {
        pos_out.push(xs[i]);
        pos_out.push(ys[i]);
        pos_out.push(zs[i]);

        color_out.push(rs[i]);
        color_out.push(gs[i]);
        color_out.push(bs[i]);
    }

    Ok(())
}
