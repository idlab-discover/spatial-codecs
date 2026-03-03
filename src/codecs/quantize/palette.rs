use spatial_utils::{traits::PointTraits, utils::point_scalar::PointScalar};

/// Choose grid bits based on K; tweak as needed.
#[inline(always)]
pub fn choose_grid_bits(max_colors: usize) -> u32 {
    // 5 for K ≤ 4096, 6 otherwise. You can add 7 if you push K very high.
    if max_colors <= 4096 {
        5
    } else {
        6
    }
}

// Generic helpers for any GB in {5,6}
#[inline(always)]
fn qn<const GB: u32>(v: u8) -> u32 {
    (v as u32) >> (8 - GB)
}
#[inline(always)]
fn bucket<const GB: u32>(r: u8, g: u8, b: u8) -> usize {
    let m = qn::<GB>(r) | (qn::<GB>(g) << GB) | (qn::<GB>(b) << (2 * GB));
    m as usize
}
#[inline(always)]
fn bucket_center_rgb<const GB: u32>(bin: usize) -> [u8; 3] {
    // center ≈ scale*x + offset, where scale = 2^(8-GB), offset = scale/2
    let scale = 1u32 << (8 - GB);
    let offs = scale >> 1;
    let mask = (1usize << GB) - 1;
    let r = ((bin) & mask) as u32;
    let g = ((bin >> GB) & mask) as u32;
    let b = ((bin >> (2 * GB)) & mask) as u32;
    [
        (r * scale + offs) as u8,
        (g * scale + offs) as u8,
        (b * scale + offs) as u8,
    ]
}

#[inline(always)]
fn sq(a: i32) -> u32 {
    (a * a) as u32
}

/// Core palette builder on a GB×GB×GB grid (GB in {5,6}).
pub fn build_palette_indices_grid_gb<const GB: u32, P, S>(
    points: &[P],
    max_colors: usize,
) -> (Vec<[u8; 3]>, Vec<u16>)
where
    P: PointTraits<S>,
    S: PointScalar,
{
    debug_assert!(GB == 5 || GB == 6);
    const fn nbins<const GB: u32>() -> usize {
        1usize << (3 * GB)
    }
    let nb = nbins::<GB>();

    // 1) Histogram + sums
    let mut counts = vec![0u32; nb];
    let mut sum_r = vec![0u32; nb];
    let mut sum_g = vec![0u32; nb];
    let mut sum_b = vec![0u32; nb];
    let mut used: Vec<usize> = Vec::with_capacity(32768);

    for p in points {
        let bi = bucket::<GB>(p.r_u8(), p.g_u8(), p.b_u8());
        if counts[bi] == 0 {
            used.push(bi);
        }
        counts[bi] += 1;
        sum_r[bi] += p.r_u8() as u32;
        sum_g[bi] += p.g_u8() as u32;
        sum_b[bi] += p.b_u8() as u32;
    }
    if used.is_empty() {
        return (vec![[0, 0, 0]], vec![0u16; points.len()]);
    }

    // 2) Top-K by count (selection then deterministic sort)
    let k = max_colors.max(1).min(used.len());
    if used.len() > k {
        used.select_nth_unstable_by(k - 1, |&a, &b| counts[b].cmp(&counts[a]));
        used.truncate(k);
    }
    used.sort_unstable_by(|&a, &b| counts[b].cmp(&counts[a]).then_with(|| a.cmp(&b)));

    // 3) Palette = per-bucket means
    let mut palette = Vec::with_capacity(k);
    for &bi in &used {
        let c = counts[bi] as u32;
        let r = (sum_r[bi] + c / 2) / c;
        let g = (sum_g[bi] + c / 2) / c;
        let b = (sum_b[bi] + c / 2) / c;
        palette.push([r as u8, g as u8, b as u8]);
    }

    // 4) Bucket→palette index (u16::MAX = unmapped)
    let mut map = vec![u16::MAX; nb];
    for (i, &bi) in used.iter().enumerate() {
        map[bi] = i as u16;
    }

    // 5) Assign indices (cache miss: search neighbors once, then cache)
    let mut indices = Vec::with_capacity(points.len());
    let maxc = (1i32 << GB) - 1;
    for p in points {
        let bi = bucket::<GB>(p.r_u8(), p.g_u8(), p.b_u8());
        let mut idx = map[bi];
        if idx == u16::MAX {
            let br = (bi & ((1 << GB) - 1)) as i32;
            let bg = ((bi >> GB) & ((1 << GB) - 1)) as i32;
            let bb = ((bi >> (2 * GB)) & ((1 << GB) - 1)) as i32;

            let pr = p.r_u8() as i32;
            let pg = p.g_u8() as i32;
            let pb = p.b_u8() as i32;
            let mut best_idx = u16::MAX;
            let mut best_d = u32::MAX;

            'outer: for rad in 0..=2 {
                let r0 = (br - rad).max(0);
                let r1 = (br + rad).min(maxc);
                let g0 = (bg - rad).max(0);
                let g1 = (bg + rad).min(maxc);
                let b0 = (bb - rad).max(0);
                let b1 = (bb + rad).min(maxc);
                for rr in r0..=r1 {
                    for gg in g0..=g1 {
                        for bbk in b0..=b1 {
                            let nbk = (rr | (gg << GB) | (bbk << (2 * GB))) as usize;
                            let midx = map[nbk];
                            if midx != u16::MAX {
                                let q = palette[midx as usize];
                                let d = sq(q[0] as i32 - pr)
                                    + sq(q[1] as i32 - pg)
                                    + sq(q[2] as i32 - pb);
                                if d < best_d {
                                    best_d = d;
                                    best_idx = midx;
                                    if d == 0 {
                                        break 'outer;
                                    }
                                }
                            }
                        }
                    }
                }
                if best_idx != u16::MAX {
                    break;
                }
            }
            if best_idx == u16::MAX {
                // Fallback: closest by bucket center (rare)
                let c = bucket_center_rgb::<GB>(bi);
                let (cr, cg, cb) = (c[0] as i32, c[1] as i32, c[2] as i32);
                let mut best = (u32::MAX, 0u16);
                for (i, q) in palette.iter().enumerate() {
                    let d = sq(q[0] as i32 - cr) + sq(q[1] as i32 - cg) + sq(q[2] as i32 - cb);
                    if d < best.0 {
                        best = (d, i as u16);
                    }
                }
                best_idx = best.1;
            }
            map[bi] = best_idx;
            idx = best_idx;
        }
        indices.push(idx);
    }
    (palette, indices)
}

/// Wrapper that picks grid at runtime (still monomorphizes to two fast versions).
pub fn build_palette_indices_grid<P, S>(points: &[P], max_colors: usize) -> (Vec<[u8; 3]>, Vec<u16>)
where
    P: PointTraits<S>,
    S: PointScalar,
{
    let gb = choose_grid_bits(max_colors);
    if gb == 5 {
        build_palette_indices_grid_gb::<5, P, S>(points, max_colors)
    } else {
        build_palette_indices_grid_gb::<6, P, S>(points, max_colors)
    }
}

/// Quick estimator: will palette beat straight RGB (bits-wise)?
pub fn palette_wins_estimate(
    n_points: usize,
    position_bits_pp: usize,
    color_bits_plain_pp: usize,
    palette_bits: usize,
    palette_len: usize,
) -> bool {
    let plain = (n_points * (position_bits_pp + color_bits_plain_pp)).div_ceil(8);
    let pal = super::types::PALETTE_LEN_FIELD_SIZE
        + palette_len * 3
        + (n_points * (position_bits_pp + palette_bits)).div_ceil(8);
    // Margin avoids doing palette work when gains are negligible
    pal + 256 < plain
}

#[cfg(test)]
mod tests {

    use super::*;
    use spatial_utils::point::Point3D;

    #[test]
    fn choose_grid_bits_thresholds() {
        assert_eq!(choose_grid_bits(1), 5);
        assert_eq!(choose_grid_bits(4096), 5);
        assert_eq!(choose_grid_bits(4097), 6);
        assert_eq!(choose_grid_bits(10_000), 6);
    }

    #[test]
    fn palette_wins_estimator_prefers_smaller_payloads() {
        let n = 10_000usize;
        // Palette with small bits and small palette should win.
        assert!(palette_wins_estimate(n, 30, 24, 4, 16));
        // Huge palette overhead should lose.
        assert!(!palette_wins_estimate(n, 30, 24, 8, 8129));
    }

    #[test]
    fn build_palette_indices_is_deterministic_and_in_bounds() {
        let mut pts = Vec::new();
        // Color (10,20,30) dominates
        for i in 0..100 {
            pts.push(Point3D {
                x: i as f32,
                y: 0.0,
                z: 0.0,
                r: 10,
                g: 20,
                b: 30,
            });
        }
        // Secondary colors
        for i in 0..10 {
            pts.push(Point3D {
                x: i as f32,
                y: 1.0,
                z: 0.0,
                r: 200,
                g: 10,
                b: 10,
            });
        }
        for i in 0..10 {
            pts.push(Point3D {
                x: i as f32,
                y: 2.0,
                z: 0.0,
                r: 10,
                g: 200,
                b: 10,
            });
        }

        let (pal, idx) = build_palette_indices_grid(&pts, 2);
        assert_eq!(pal.len(), 2);
        assert_eq!(idx.len(), pts.len());

        // Most frequent color should be first due to deterministic sort by count then bin index.
        assert_eq!(pal[0], [10, 20, 30]);

        for &i in &idx {
            assert!((i as usize) < pal.len());
        }

        // Re-run and ensure identical palette order and indices.
        let (pal2, idx2) = build_palette_indices_grid(&pts, 2);
        assert_eq!(pal, pal2);
        assert_eq!(idx, idx2);
    }
}
