//! Colour quality metrics.
//!
//! We evaluate colour fidelity using PSNR in the ITU-R BT.709 YCbCr space. Converting to
//! YCbCr separates luminance from chroma, which provides a better proxy for perceived error
//! than raw RGB RMSE.

use super::ref_cache::RefCache;

/// Per-channel PSNR results. Channels are optional to make it painless to skip the colour
/// metric entirely when the benchmark is run in geometry-only mode.
pub struct ColorPsnr {
    pub y: Option<f64>,
    pub cb: Option<f64>,
    pub cr: Option<f64>,
}

/// Evaluate colour quality against the reference cache.
///
/// `test_pos` is paired with `test_rgb` so that we can reuse the same nearest-neighbour
/// mapping as the geometry metric. The reference side precomputes YCbCr values in the
/// cache, avoiding repeated RGB→YCbCr conversions across sweeps.
pub fn evaluate_color(
    ref_cache: &RefCache,
    test_pos: &[[f64; 3]],
    test_rgb: &[[u8; 3]],
) -> ColorPsnr {
    if test_pos.is_empty() || test_pos.len() != test_rgb.len() {
        return ColorPsnr {
            y: None,
            cb: None,
            cr: None,
        };
    }

    let nn_r = &ref_cache.nn_ref;
    let ref_pts = nn_r.points();
    if ref_pts.is_empty() {
        return ColorPsnr {
            y: None,
            cb: None,
            cr: None,
        };
    }

    let ref_ycbcr = match &ref_cache.ref_ycbcr {
        Some(v) if v.len() == ref_pts.len() => v.as_slice(),
        _ => {
            return ColorPsnr {
                y: None,
                cb: None,
                cr: None,
            }
        }
    };

    let (mut mse_y, mut mse_cb, mut mse_cr, mut n) = (0.0, 0.0, 0.0, 0usize);
    for (q, c2) in test_pos.iter().zip(test_rgb.iter()) {
        let (idx, _) = nn_r.nn_index(q);
        let ref_sample = ref_ycbcr[idx];
        let (y2, cb2, cr2) = ycbcr709(*c2);
        // MSE accumulates the squared error between the reconstructed sample and the nearest
        // reference sample in YCbCr space.
        let dy = ref_sample[0] - y2;
        let dcb = ref_sample[1] - cb2;
        let dcr = ref_sample[2] - cr2;
        mse_y += dy * dy;
        mse_cb += dcb * dcb;
        mse_cr += dcr * dcr;
        n += 1;
    }

    if n == 0 {
        return ColorPsnr {
            y: None,
            cb: None,
            cr: None,
        };
    }
    mse_y /= n as f64;
    mse_cb /= n as f64;
    mse_cr /= n as f64;
    ColorPsnr {
        y: Some(psnr255(mse_y)),
        cb: Some(psnr255(mse_cb)),
        cr: Some(psnr255(mse_cr)),
    }
}

#[inline]
pub fn ycbcr709(rgb: [u8; 3]) -> (f64, f64, f64) {
    // ITU-R BT.709 full-range transform (derived from the standard matrix, scaled for 8-bit data).
    // Y is luma (perceptual brightness), Cb/Cr are blue- and red-difference chroma.
    let r = rgb[0] as f64;
    let g = rgb[1] as f64;
    let b = rgb[2] as f64;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let cb = -0.1146 * r - 0.3854 * g + 0.5 * b + 128.0;
    let cr = 0.5 * r - 0.4542 * g - 0.0458 * b + 128.0;
    (y, cb, cr)
}
#[inline]
fn psnr255(mse: f64) -> f64 {
    if mse <= 0.0 {
        f64::INFINITY
    } else {
        // PSNR = 20 * log10(MAX_I / sqrt(MSE)), with MAX_I = 255 for 8-bit samples.
        20.0 * (255.0 / mse.sqrt()).log10()
    }
}
