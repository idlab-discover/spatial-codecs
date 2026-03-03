//! Integrations with the OpenZL compressor.
//!
//! Two modes are supported:
//! - `Serial`: run an inner codec (Quantize/Draco/…) and feed the resulting bytes to
//!   OpenZL’s serial compressor.
//! - `Columnar`: build a structure-of-arrays layout and compress numeric columns with
//!   OpenZL’s columnar engine.

use spatial_utils::{traits::SpatialOwnedFull, utils::point_scalar::PointScalar};
use serde::{Deserialize, Serialize};

use rust_openzl::{compress_numeric, compress_serial}; // safe API

use crate::encoder::{encode_into as encode_inner, EncodingParams as InnerParams};

const MAGIC: &[u8; 3] = b"OZL";
const V1: u8 = 1;

#[repr(u8)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "kebab-case")]
pub enum OpenzlParams {
    /// Serial: run an inner codec first (e.g., Quantize/Draco/Bitcode), then
    /// compress the resulting bytes with OpenZL serial.
    Serial { inner: Box<InnerParams> } = 0,

    /// Columnar: split into SoA columns and compress with OpenZL numeric.
    /// If `pack_rgb` is true, pack (r,g,b) into u32=0x00RRGGBB to reduce streams.
    Columnar {
        #[serde(default = "d_true")]
        pack_rgb: bool,
    },
}

fn d_true() -> bool {
    true
}

impl OpenzlParams {
    /// Convenience constructor matching the default used by the top-level API.
    pub fn columnar_default() -> Self {
        OpenzlParams::Columnar { pack_rgb: true }
    }
}

// Header for V1:
// "OZL"(3) | v(1)=1 | flags(1)
//   flags bit0: 1 = columnar, 0 = serial
// when serial:  u32 payload_len | <serial-compressed bytes of inner>
// when columnar:
//   u32 n_points | u8 stream_count (4 or 6)
//   then for each stream: u32 len | <compressed stream bytes>
//   stream order (pack_rgb=true): [x,y,z,rgb]; else [x,y,z,r,g,b]
#[allow(dead_code)]
#[derive(Clone, Copy)]
struct HdrV1 {
    flags: u8,
}

#[inline]
fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn pack_rgb_u32(r: &[u8], g: &[u8], b: &[u8]) -> Vec<u32> {
    let n = r.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(((r[i] as u32) << 16) | ((g[i] as u32) << 8) | (b[i] as u32));
    }
    out
}

/// Encode points using the selected OpenZL mode.
pub fn encode_from_payload_into<P, S>(
    points: &[P],
    params: &OpenzlParams,
    out: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>>
where
    P: SpatialOwnedFull<S>,
    S: PointScalar,
{
    match params {
        OpenzlParams::Serial { inner } => {
            // 1) Produce inner bytes (e.g., Quantize) into a temp buffer
            let mut inner_buf = Vec::new();
            encode_inner(points, inner, &mut inner_buf)?;

            // 2) OpenZL serial-compress those bytes
            let oz = compress_serial(&inner_buf)?;

            // 3) Frame: magic + v + flags + len + payload
            out.reserve(3 + 1 + 1 + 4 + oz.len());
            out.extend_from_slice(MAGIC);
            out.push(V1);
            out.push(0); // flags: bit0=0 => serial
            write_u32(out, oz.len() as u32);
            out.extend_from_slice(&oz);
            Ok(())
        }

        OpenzlParams::Columnar { pack_rgb } => {
            let n = points.len();
            // 1) AoS->SoA
            let mut xs = Vec::with_capacity(n);
            let mut ys = Vec::with_capacity(n);
            let mut zs = Vec::with_capacity(n);
            let mut rs = Vec::with_capacity(n);
            let mut gs = Vec::with_capacity(n);
            let mut bs = Vec::with_capacity(n);

            for p in points {
                xs.push(p.x());
                ys.push(p.y());
                zs.push(p.z());
                rs.push(p.r_u8());
                gs.push(p.g_u8());
                bs.push(p.b_u8());
            }

            // 2) Compress columns
            let mut blobs: Vec<Vec<u8>> = Vec::with_capacity(if *pack_rgb { 4 } else { 6 });
            blobs.push(compress_numeric(&xs)?);
            blobs.push(compress_numeric(&ys)?);
            blobs.push(compress_numeric(&zs)?);

            if *pack_rgb {
                let rgb = pack_rgb_u32(&rs, &gs, &bs);
                blobs.push(compress_numeric(&rgb)?);
            } else {
                blobs.push(compress_numeric(&rs)?);
                blobs.push(compress_numeric(&gs)?);
                blobs.push(compress_numeric(&bs)?);
            }

            // 3) Frame: magic + v + flags + n_points + stream_count + [len+blob]...
            let sc = if *pack_rgb { 4u8 } else { 6u8 };
            let len_sum: usize = blobs.iter().map(|b| 4 + b.len()).sum();
            out.reserve(3 + 1 + 1 + 4 + 1 + len_sum);
            out.extend_from_slice(MAGIC);
            out.push(V1);
            out.push(1); // flags: bit0=1 => columnar
            write_u32(out, n as u32);
            out.push(sc);
            for b in blobs {
                write_u32(out, b.len() as u32);
                out.extend_from_slice(&b);
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::codecs::lz4::encoder::wrap_lz4_into;

    use lz4_flex::frame::FrameDecoder;
    use std::io::Read;

    #[test]
    fn lz4_wrapper_frames_and_decompresses_inner_bytes() {
        let mut out = Vec::new();
        wrap_lz4_into(
            |buf| {
                buf.extend_from_slice(b"hello");
                Ok(())
            },
            &mut out,
        )
        .unwrap();
        assert_eq!(&out[..3], b"LZ4");

        let mut dec = FrameDecoder::new(&out[3..]);
        let mut inner = Vec::new();
        dec.read_to_end(&mut inner).unwrap();
        assert_eq!(inner, b"hello");
    }
}
