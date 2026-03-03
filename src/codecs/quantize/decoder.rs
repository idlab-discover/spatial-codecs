//! Decoder for the custom quantize format.
//!
//! Mirrors the encoder features: optional palettes, delta reconstruction, configurable
//! bit widths. See the module docs for encoding details.

use spatial_utils::{point::Point3Rgb, traits::SpatialSink, utils::point_scalar::PointScalar};
use crate::BasicResult;

use super::{
    bitio::FastBitReader,
    delta,
    header::parse_header,
    types::{
        mask_for, max_quant_value, storage_bits_for, FLAG_DELTA_COLORS, FLAG_DELTA_POSITIONS,
        FLAG_FIXED_WIDTH_COLORS, FLAG_FIXED_WIDTH_POSITIONS,
    },
};

fn decode_point_cloud<P>(
    data: &[u8],
    mut out_positions: Option<&mut Vec<P::Scalar>>,
    mut out_colors: Option<&mut Vec<u8>>,
    mut out_points: Option<&mut Vec<P>>,
) -> BasicResult
where
    P: SpatialSink,
{
    let header = parse_header(data)?;

    let delta_positions = (header.flags & FLAG_DELTA_POSITIONS) != 0;
    let delta_colors = (header.flags & FLAG_DELTA_COLORS) != 0;
    let fixed_width_positions = (header.flags & FLAG_FIXED_WIDTH_POSITIONS) != 0;
    let fixed_width_colors = (header.flags & FLAG_FIXED_WIDTH_COLORS) != 0;

    let pos_storage_bits = storage_bits_for(header.position_bits, !fixed_width_positions);
    let color_storage_bits = storage_bits_for(header.color_bits, !fixed_width_colors);

    let pos_bits_total = usize::from(pos_storage_bits) * 3;
    let color_bits_total = if header.has_palette {
        usize::from(color_storage_bits)
    } else {
        usize::from(color_storage_bits) * 3
    };

    let per_point_bits = pos_bits_total
        .checked_add(color_bits_total)
        .ok_or("Quantize decoder: bit budget overflow")?;
    let total_bits = header
        .point_count
        .checked_mul(per_point_bits)
        .ok_or("Quantize decoder: bit budget overflow")?;
    let total_bytes = total_bits.div_ceil(8);

    if data.len() < header.offset + total_bytes {
        return Err("Quantize decoder: truncated payload".into());
    }

    let mut reader = FastBitReader::new(&data[header.offset..header.offset + total_bytes]);

    let pos_mask_u64 = mask_for(header.position_bits);
    let pos_mask_u32 = pos_mask_u64 as u32;

    let range_x = header.maxs[0] - header.mins[0];
    let range_y = header.maxs[1] - header.mins[1];
    let range_z = header.maxs[2] - header.mins[2];

    let position_max = max_quant_value(header.position_bits) as f32;
    let kx = if range_x > f32::EPSILON {
        range_x / position_max
    } else {
        0.0
    };
    let ky = if range_y > f32::EPSILON {
        range_y / position_max
    } else {
        0.0
    };
    let kz = if range_z > f32::EPSILON {
        range_z / position_max
    } else {
        0.0
    };

    let mut color_mask_u32 = 0u32;
    let mut palette_mask_u32 = 0u32;

    if header.has_palette {
        if header.color_bits > 0 {
            palette_mask_u32 = mask_for(header.color_bits) as u32;
        }
    } else {
        color_mask_u32 = mask_for(header.color_bits) as u32;
    }

    let raw_rgb = !header.has_palette
        && header.color_bits == 8
        && header.color_mins == [0, 0, 0]
        && header.color_maxs == [255, 255, 255];

    let color_max = if header.has_palette {
        0.0
    } else {
        max_quant_value(header.color_bits) as f32
    };

    let cr0 = (header.color_maxs[0] as f32) - (header.color_mins[0] as f32);
    let cg0 = (header.color_maxs[1] as f32) - (header.color_mins[1] as f32);
    let cb0 = (header.color_maxs[2] as f32) - (header.color_mins[2] as f32);

    let kr = if cr0 > f32::EPSILON {
        cr0 / color_max
    } else {
        0.0
    };
    let kg = if cg0 > f32::EPSILON {
        cg0 / color_max
    } else {
        0.0
    };
    let kb = if cb0 > f32::EPSILON {
        cb0 / color_max
    } else {
        0.0
    };

    let use_color_lut = !header.has_palette && !raw_rgb && header.color_bits <= 8;
    let (lut_r, lut_g, lut_b) = if use_color_lut {
        let max_q = ((1u32 << header.color_bits) - 1) as usize;
        let mut lr = vec![0u8; max_q + 1];
        let mut lg = vec![0u8; max_q + 1];
        let mut lb = vec![0u8; max_q + 1];
        let min_r = header.color_mins[0] as f32;
        let min_g = header.color_mins[1] as f32;
        let min_b = header.color_mins[2] as f32;
        for q in 0..=max_q {
            let fq = q as f32;
            lr[q] = (fq * kr + min_r).round().clamp(0.0, 255.0) as u8;
            lg[q] = (fq * kg + min_g).round().clamp(0.0, 255.0) as u8;
            lb[q] = (fq * kb + min_b).round().clamp(0.0, 255.0) as u8;
        }
        (lr, lg, lb)
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };

    if let Some(points) = out_points.as_mut() {
        (**points).reserve(header.point_count);
    }
    if let Some(pos) = out_positions.as_mut() {
        (**pos).reserve(header.point_count * 3);
    }
    if let Some(col) = out_colors.as_mut() {
        (**col).reserve(header.point_count * 3);
    }

    let mut prev_pos = [0u32; 3];
    let mut prev_cols = [0u32; 3];
    let mut prev_palette_idx = 0u32;

    for _ in 0..header.point_count {
        let rx = reader
            .read(pos_storage_bits)
            .ok_or("Quantize decoder: not enough data")? as u32;
        let ry = reader
            .read(pos_storage_bits)
            .ok_or("Quantize decoder: not enough data")? as u32;
        let rz = reader
            .read(pos_storage_bits)
            .ok_or("Quantize decoder: not enough data")? as u32;

        let mut qx = rx & pos_mask_u32;
        let mut qy = ry & pos_mask_u32;
        let mut qz = rz & pos_mask_u32;
        if delta_positions {
            qx = delta::decode_delta(qx, &mut prev_pos[0], pos_mask_u32);
            qy = delta::decode_delta(qy, &mut prev_pos[1], pos_mask_u32);
            qz = delta::decode_delta(qz, &mut prev_pos[2], pos_mask_u32);
        }

        let x = (qx as f32).mul_add(kx, header.mins[0]);
        let y = (qy as f32).mul_add(ky, header.mins[1]);
        let z = (qz as f32).mul_add(kz, header.mins[2]);

        let (r, g, b) = if header.has_palette {
            let palette_idx = if color_storage_bits == 0 {
                0
            } else {
                let raw = reader
                    .read(color_storage_bits)
                    .ok_or("Quantize decoder: not enough data")? as u32;
                let masked = raw & palette_mask_u32;
                if delta_colors {
                    delta::decode_delta(masked, &mut prev_palette_idx, palette_mask_u32)
                } else {
                    prev_palette_idx = masked;
                    masked
                }
            } as usize;
            let palette = header
                .palette
                .get(palette_idx)
                .ok_or("Quantize decoder: palette index out of range")?;
            (palette[0], palette[1], palette[2])
        } else if raw_rgb {
            let mut rr = reader
                .read(color_storage_bits)
                .ok_or("Quantize decoder: not enough data")? as u32
                & color_mask_u32;
            let mut gg = reader
                .read(color_storage_bits)
                .ok_or("Quantize decoder: not enough data")? as u32
                & color_mask_u32;
            let mut bb = reader
                .read(color_storage_bits)
                .ok_or("Quantize decoder: not enough data")? as u32
                & color_mask_u32;
            if delta_colors {
                rr = delta::decode_delta(rr, &mut prev_cols[0], color_mask_u32);
                gg = delta::decode_delta(gg, &mut prev_cols[1], color_mask_u32);
                bb = delta::decode_delta(bb, &mut prev_cols[2], color_mask_u32);
            } else {
                prev_cols[0] = rr;
                prev_cols[1] = gg;
                prev_cols[2] = bb;
            }
            (rr as u8, gg as u8, bb as u8)
        } else if use_color_lut {
            let mut qr = reader
                .read(color_storage_bits)
                .ok_or("Quantize decoder: not enough data")? as u32
                & color_mask_u32;
            let mut qg = reader
                .read(color_storage_bits)
                .ok_or("Quantize decoder: not enough data")? as u32
                & color_mask_u32;
            let mut qb = reader
                .read(color_storage_bits)
                .ok_or("Quantize decoder: not enough data")? as u32
                & color_mask_u32;
            if delta_colors {
                qr = delta::decode_delta(qr, &mut prev_cols[0], color_mask_u32);
                qg = delta::decode_delta(qg, &mut prev_cols[1], color_mask_u32);
                qb = delta::decode_delta(qb, &mut prev_cols[2], color_mask_u32);
            } else {
                prev_cols[0] = qr;
                prev_cols[1] = qg;
                prev_cols[2] = qb;
            }
            (lut_r[qr as usize], lut_g[qg as usize], lut_b[qb as usize])
        } else {
            let mut qr = reader
                .read(color_storage_bits)
                .ok_or("Quantize decoder: not enough data")? as f32;
            let mut qg = reader
                .read(color_storage_bits)
                .ok_or("Quantize decoder: not enough data")? as f32;
            let mut qb = reader
                .read(color_storage_bits)
                .ok_or("Quantize decoder: not enough data")? as f32;

            if delta_colors {
                let raw_r = delta::decode_delta(qr as u32, &mut prev_cols[0], color_mask_u32);
                let raw_g = delta::decode_delta(qg as u32, &mut prev_cols[1], color_mask_u32);
                let raw_b = delta::decode_delta(qb as u32, &mut prev_cols[2], color_mask_u32);
                qr = raw_r as f32;
                qg = raw_g as f32;
                qb = raw_b as f32;
            } else {
                prev_cols[0] = qr as u32;
                prev_cols[1] = qg as u32;
                prev_cols[2] = qb as u32;
            }

            let r = qr
                .mul_add(kr, header.color_mins[0] as f32)
                .round()
                .clamp(0.0, 255.0) as u8;
            let g = qg
                .mul_add(kg, header.color_mins[1] as f32)
                .round()
                .clamp(0.0, 255.0) as u8;
            let b = qb
                .mul_add(kb, header.color_mins[2] as f32)
                .round()
                .clamp(0.0, 255.0) as u8;
            (r, g, b)
        };

        let xs = P::Scalar::from_f32(x);
        let ys = P::Scalar::from_f32(y);
        let zs = P::Scalar::from_f32(z);

        if let Some(points) = out_points.as_mut() {
            P::push_xyz_rgb(&mut **points, xs, ys, zs, r, g, b);
        }
        if let Some(pos) = out_positions.as_mut() {
            (**pos).extend_from_slice(&[xs, ys, zs]);
        }
        if let Some(col) = out_colors.as_mut() {
            (**col).extend_from_slice(&[r, g, b]);
        }
    }

    Ok(())
}

pub fn decode_into<P>(data: &[u8], out: &mut Vec<P>) -> BasicResult
where
    P: SpatialSink,
{
    decode_point_cloud(data, None, None, Some(out))
}

pub fn decode_into_flattened_vecs<S>(
    data: &[u8],
    pos_out: &mut Vec<S>,
    color_out: &mut Vec<u8>,
) -> BasicResult
where
    S: PointScalar,
{
    decode_point_cloud::<Point3Rgb<S>>(data, Some(pos_out), Some(color_out), None)
}
