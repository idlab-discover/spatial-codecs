//! Custom quantize encoder implementation.
//!
//! The encoder supports optional palettes, delta coding, and configurable bit packing.
//! The format is described in detail in the crate README and the quantize module docs.

use spatial_utils::{traits::PointTraits, utils::point_scalar::PointScalar};
use crate::BasicResult;

use crate::codecs::quantize::palette::palette_wins_estimate;

use super::{
    bitio::TurboBitWriter,
    delta,
    header::write_header_v2,
    palette::build_palette_indices_grid,
    types::{
        bits_for_palette_len, clamp_bits, mask_for, max_quant_value, storage_bits_for,
        QuantizeParams, COLOR_BITFLAG_HAS_FLAGS, COLOR_BITFLAG_PALETTE, COLOR_BITS_MASK,
        FLAG_DELTA_COLORS, FLAG_DELTA_POSITIONS, FLAG_FIXED_WIDTH_COLORS,
        FLAG_FIXED_WIDTH_POSITIONS, HEADER_EXTENDED_SIZE, PALETTE_LEN_FIELD_SIZE,
    },
};

fn build_color_lut(min_c: u8, scale: f32, col_max: u64) -> [u32; 256] {
    let mut lut = [0u32; 256];
    let base = min_c as i32;
    for (i, slot) in lut.iter_mut().enumerate() {
        let d = (i as i32 - base) as f32;
        let y = (d * scale + 0.5).floor();
        let y = if y < 0.0 {
            0.0
        } else if y > col_max as f32 {
            col_max as f32
        } else {
            y
        };
        *slot = y as u32;
    }
    lut
}

pub fn encode_from_payload_into<P, S>(
    points: &[P],
    params: &QuantizeParams,
    out: &mut Vec<u8>,
) -> BasicResult
where
    P: PointTraits<S>,
    S: PointScalar,
{
    if points.len() > u32::MAX as usize {
        return Err("Quantize encoder: too many points for u32 header field".into());
    }

    let position_bits = clamp_bits(params.position_bits);
    let color_bits = clamp_bits(params.color_bits);

    let pos_max = max_quant_value(position_bits);
    let col_max = max_quant_value(color_bits);

    // Position bounds
    let (min_x, min_y, min_z, max_x, max_y, max_z) = if let Some(first) = points.first() {
        let (mut min_x, mut min_y, mut min_z) = (first.x(), first.y(), first.z());
        let (mut max_x, mut max_y, mut max_z) = (first.x(), first.y(), first.z());
        for p in &points[1..] {
            if p.x() < min_x {
                min_x = p.x();
            } else if p.x() > max_x {
                max_x = p.x();
            }
            if p.y() < min_y {
                min_y = p.y();
            } else if p.y() > max_y {
                max_y = p.y();
            }
            if p.z() < min_z {
                min_z = p.z();
            } else if p.z() > max_z {
                max_z = p.z();
            }
        }
        (min_x, min_y, min_z, max_x, max_y, max_z)
    } else {
        (S::ZERO, S::ZERO, S::ZERO, S::ZERO, S::ZERO, S::ZERO)
    };

    // Color bounds
    let (mut min_r, mut min_g, mut min_b) = (u8::MAX, u8::MAX, u8::MAX);
    let (mut max_r, mut max_g, mut max_b) = (u8::MIN, u8::MIN, u8::MIN);
    for p in points {
        if p.r_u8() < min_r {
            min_r = p.r_u8();
        } else if p.r_u8() > max_r {
            max_r = p.r_u8();
        }
        if p.g_u8() < min_g {
            min_g = p.g_u8();
        } else if p.g_u8() > max_g {
            max_g = p.g_u8();
        }
        if p.b_u8() < min_b {
            min_b = p.b_u8();
        } else if p.b_u8() > max_b {
            max_b = p.b_u8();
        }
    }
    if points.is_empty() {
        min_r = 0;
        min_g = 0;
        min_b = 0;
        max_r = 0;
        max_g = 0;
        max_b = 0;
    }

    let point_count = points.len() as u32;

    let delta_positions = params.delta_positions && point_count > 0;
    let delta_colors = params.delta_colors && point_count > 0;
    let fixed_width_positions = !params.pack_positions;
    let fixed_width_colors = !params.pack_colors;

    let mut header_flags = 0u8;
    if delta_positions {
        header_flags |= FLAG_DELTA_POSITIONS;
    }
    if delta_colors {
        header_flags |= FLAG_DELTA_COLORS;
    }
    if fixed_width_positions {
        header_flags |= FLAG_FIXED_WIDTH_POSITIONS;
    }
    if fixed_width_colors {
        header_flags |= FLAG_FIXED_WIDTH_COLORS;
    }

    let header_extra = if header_flags != 0 { 1 } else { 0 };

    let pos_storage_bits = storage_bits_for(position_bits, params.pack_positions);
    let mut color_storage_bits = storage_bits_for(color_bits, params.pack_colors);

    let pos_bits_total = usize::from(pos_storage_bits) * 3;
    let mut color_bits_total = usize::from(color_storage_bits) * 3;

    let mut use_palette = params.max_palette_colors > 0 && point_count > 0;
    if use_palette {
        // --- estimate whether a palette would reduce the bits per point ---

        // estimated palette length (use the configured cap here)
        let palette_len_est = params.max_palette_colors as usize;

        // bits to index that many palette entries, then *storage* bits for that index
        let palette_index_bits = bits_for_palette_len(palette_len_est);
        let palette_index_storage_bits_est =
            usize::from(storage_bits_for(palette_index_bits, params.pack_colors));
        let est_wins = palette_wins_estimate(
            point_count as usize,
            pos_bits_total,
            color_bits_total,
            palette_index_storage_bits_est,
            palette_len_est,
        );
        if !est_wins {
            use_palette = false;
        }
    }

    let (palette, palette_indices, palette_bits) = if use_palette {
        let (pal, indices) = build_palette_indices_grid(points, params.max_palette_colors as usize);
        if pal.len() > u16::MAX as usize {
            return Err("Quantize encoder: palette too large for header field".into());
        }
        let bits = bits_for_palette_len(pal.len());
        (pal, indices, bits)
    } else {
        (Vec::new(), Vec::new(), 0u8)
    };

    if use_palette {
        color_storage_bits = storage_bits_for(palette_bits, params.pack_colors);
        color_bits_total = usize::from(color_storage_bits);
    }

    let per_point_bits = pos_bits_total + color_bits_total;
    let body_bytes = (points.len() * per_point_bits).div_ceil(8);

    let palette_bytes = if use_palette {
        PALETTE_LEN_FIELD_SIZE + palette.len() * 3
    } else {
        0
    };

    out.reserve(HEADER_EXTENDED_SIZE + header_extra + palette_bytes + body_bytes + 8);

    let mut color_bits_field = if use_palette {
        palette_bits
    } else {
        color_bits
    };
    color_bits_field &= COLOR_BITS_MASK;
    if use_palette {
        color_bits_field |= COLOR_BITFLAG_PALETTE;
    }
    let flags_opt = if header_flags != 0 {
        color_bits_field |= COLOR_BITFLAG_HAS_FLAGS;
        Some(header_flags)
    } else {
        None
    };

    write_header_v2(
        out,
        position_bits,
        color_bits_field,
        flags_opt,
        point_count,
        [min_x, min_y, min_z],
        [max_x, max_y, max_z],
        [min_r, min_g, min_b],
        [max_r, max_g, max_b],
        if use_palette { Some(&palette) } else { None },
    );

    if point_count == 0 {
        return Ok(());
    }

    let body_start = out.len();
    unsafe {
        out.set_len(body_start + body_bytes);
    }

    let pos_mask_u64 = mask_for(position_bits);
    let pos_mask_u32 = pos_mask_u64 as u32;
    let col_mask_u64 = mask_for(color_bits);
    let col_mask_u32 = col_mask_u64 as u32;
    let palette_mask_u32 = if palette_bits == 0 {
        0
    } else {
        mask_for(palette_bits) as u32
    };
    let palette_mask_u64 = if palette_bits == 0 {
        0
    } else {
        mask_for(palette_bits)
    };

    let rx = max_x - min_x;
    let ry = max_y - min_y;
    let rz = max_z - min_z;

    let sx = if rx > S::EPS {
        S::from_f32(pos_max as f32) / rx
    } else {
        S::ZERO
    };
    let sy = if ry > S::EPS {
        S::from_f32(pos_max as f32) / ry
    } else {
        S::ZERO
    };
    let sz = if rz > S::EPS {
        S::from_f32(pos_max as f32) / rz
    } else {
        S::ZERO
    };

    let rr = (max_r as f32) - (min_r as f32);
    let rg = (max_g as f32) - (min_g as f32);
    let rb = (max_b as f32) - (min_b as f32);

    let sr = if rr > f32::EPSILON {
        (col_max as f32) / rr
    } else {
        0.0
    };
    let sg = if rg > f32::EPSILON {
        (col_max as f32) / rg
    } else {
        0.0
    };
    let sb = if rb > f32::EPSILON {
        (col_max as f32) / rb
    } else {
        0.0
    };

    let color_fast_path = !use_palette
        && color_bits == 8
        && min_r == 0
        && min_g == 0
        && min_b == 0
        && max_r == 255
        && max_g == 255
        && max_b == 255;

    let lut_r;
    let lut_g;
    let lut_b;
    let use_color_lut = !use_palette && !color_fast_path;
    if use_color_lut {
        lut_r = build_color_lut(min_r, sr, col_max);
        lut_g = build_color_lut(min_g, sg, col_max);
        lut_b = build_color_lut(min_b, sb, col_max);
    } else {
        lut_r = [0u32; 256];
        lut_g = [0u32; 256];
        lut_b = [0u32; 256];
    }

    let mut writer = unsafe { TurboBitWriter::new_at(out, body_start, body_bytes) };

    let mut prev_pos = [0u32; 3];
    let mut prev_cols = [0u32; 3];
    let mut prev_palette_idx = 0u32;

    for (idx, p) in points.iter().enumerate() {
        let pos_max = S::from_u64(pos_max);
        let qx = if sx == S::ZERO {
            0
        } else {
            ((p.x() - min_x) * sx + S::HALF)
                .floor()
                .clamp(S::ZERO, pos_max)
                .to_u32()
        };
        let qy = if sy == S::ZERO {
            0
        } else {
            ((p.y() - min_y) * sy + S::HALF)
                .floor()
                .clamp(S::ZERO, pos_max)
                .to_u32()
        };
        let qz = if sz == S::ZERO {
            0
        } else {
            ((p.z() - min_z) * sz + S::HALF)
                .floor()
                .clamp(S::ZERO, pos_max)
                .to_u32()
        };

        let mut vx = qx;
        let mut vy = qy;
        let mut vz = qz;
        if delta_positions {
            vx = delta::encode_delta(vx, &mut prev_pos[0], pos_mask_u32);
            vy = delta::encode_delta(vy, &mut prev_pos[1], pos_mask_u32);
            vz = delta::encode_delta(vz, &mut prev_pos[2], pos_mask_u32);
        }

        writer.write_masked(vx, pos_storage_bits, pos_mask_u64);
        writer.write_masked(vy, pos_storage_bits, pos_mask_u64);
        writer.write_masked(vz, pos_storage_bits, pos_mask_u64);

        if use_palette {
            if color_storage_bits > 0 {
                let mut idx_value = palette_indices[idx] as u32;
                if delta_colors {
                    idx_value =
                        delta::encode_delta(idx_value, &mut prev_palette_idx, palette_mask_u32);
                }
                writer.write_masked(idx_value, color_storage_bits, palette_mask_u64);
            }
        } else if color_fast_path {
            let mut rr_q = p.r_u8() as u32;
            let mut gg_q = p.g_u8() as u32;
            let mut bb_q = p.b_u8() as u32;
            if delta_colors {
                rr_q = delta::encode_delta(rr_q, &mut prev_cols[0], col_mask_u32);
                gg_q = delta::encode_delta(gg_q, &mut prev_cols[1], col_mask_u32);
                bb_q = delta::encode_delta(bb_q, &mut prev_cols[2], col_mask_u32);
            }
            writer.write_masked(rr_q, color_storage_bits, col_mask_u64);
            writer.write_masked(gg_q, color_storage_bits, col_mask_u64);
            writer.write_masked(bb_q, color_storage_bits, col_mask_u64);
        } else {
            let qr = lut_r[p.r_u8() as usize];
            let qg = lut_g[p.g_u8() as usize];
            let qb = lut_b[p.b_u8() as usize];
            let mut rr_q = qr;
            let mut gg_q = qg;
            let mut bb_q = qb;
            if delta_colors {
                rr_q = delta::encode_delta(rr_q, &mut prev_cols[0], col_mask_u32);
                gg_q = delta::encode_delta(gg_q, &mut prev_cols[1], col_mask_u32);
                bb_q = delta::encode_delta(bb_q, &mut prev_cols[2], col_mask_u32);
            }
            writer.write_masked(rr_q, color_storage_bits, col_mask_u64);
            writer.write_masked(gg_q, color_storage_bits, col_mask_u64);
            writer.write_masked(bb_q, color_storage_bits, col_mask_u64);
        }
    }

    writer.finish();

    Ok(())
}
