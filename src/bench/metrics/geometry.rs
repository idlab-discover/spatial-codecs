//! Geometry-centric benchmark metrics.
//!
//! The routines in this module evaluate geometric fidelity between a decoded point set
//! and a reference set. The implementation favours re-use of scratch buffers – benchmarks
//! often run the same metric hundreds of times, and we want allocations to scale with the
//! largest point set we encounter rather than each individual sweep.
//!
//! Metrics are designed to be symmetric (reference ↔ test) when requested, which avoids
//! understating error when a codec adds or drops points.
//!
//! ## Geometric background
//!
//! When estimating surface-aware errors we need local neighbourhood information. We obtain
//! that neighbourhood by querying the *k-nearest neighbours* (k-NN) of each point: given a
//! query position **q**, the k-NN set consists of the `k` reference points with the smallest
//! Euclidean distance to **q**. These queries are accelerated by a KD-tree (see
//! [`metrics::nn`]), a space-partitioning data structure that recursively splits the point
//! set along coordinate axes so nearest-neighbour lookups avoid scanning every point.
//! KD-trees reduce the query cost from *O(n)* to roughly *O(log n)* in typical datasets.
//!
//! For the D2 (point-to-plane) metric we also estimate local surface normals. We use PCA
//! (Principal Component Analysis) on the k-NN cloud: PCA computes the covariance matrix
//! of the neighbour coordinates, the eigenvector with the smallest eigenvalue points along
//! the direction of least variance, i.e. the normal of the best-fitting plane. See
//! [`metrics::normals`] for the implementation.

use super::nn::Nn;
use super::normals::{estimate_normals_pca, normalize};
use super::ref_cache::RefCache;
use crate::bench::config::{MetricsConfig, OutlierThreshold};
use serde::Serialize;
use std::borrow::Cow;

/// High-level statistics produced by [`agg`].
///
/// * `rmse` - root-mean-square error in metres.
/// * `median` - 50th percentile of the distance distribution.
/// * `p95` - 95th percentile, useful for spotting heavy tails.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Aggregates {
    pub rmse: f64,
    pub median: f64,
    pub p95: f64,
}

/// Summary bundle returned by [`evaluate_geometry`].
///
/// Fields are optional when the corresponding calculation is disabled via
/// [`MetricsConfig`].
#[derive(Debug, Clone, Serialize)]
pub struct MetricSummary {
    pub d1: Aggregates,
    pub d2: Option<Aggregates>,
    pub n_ang_mean_deg: Option<f64>,
    pub n_ang_p95_deg: Option<f64>,
    pub outlier_ratio: f64,
}

/// Record representing a (test → reference) match.
#[derive(Default)]
struct PointMatch {
    idx: usize,
    dist: f64,
}

/// Reusable buffers shared across sweeps.
///
/// The benchmark harness may call [`evaluate_geometry`] repeatedly for clouds of
/// different sizes. Reusing buffers amortises allocation cost while ensuring we never
/// keep cross-run data alive - [`clear`](MetricsScratch::clear) resets vector lengths.
#[derive(Default)]
pub struct MetricsScratch {
    matches: Vec<PointMatch>,
    d_buffer: Vec<f64>,
    d2_buffer: Vec<f64>,
    mad_scratch: Vec<f64>,
}

impl MetricsScratch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(n: usize) -> Self {
        let mut s = Self::default();
        s.ensure_capacity(n);
        s
    }

    pub fn ensure_capacity(&mut self, n: usize) {
        ensure_capacity(&mut self.matches, n);
        ensure_capacity(&mut self.d_buffer, n);
        ensure_capacity(&mut self.d2_buffer, n);
        ensure_capacity(&mut self.mad_scratch, n);
    }

    fn clear(&mut self) {
        self.matches.clear();
        self.d_buffer.clear();
        self.d2_buffer.clear();
        self.mad_scratch.clear();
    }
}

/// Grow a vector's capacity without touching its length.
fn ensure_capacity<T>(v: &mut Vec<T>, n: usize) {
    if v.capacity() < n {
        v.reserve(n - v.capacity());
    }
}

pub fn evaluate_geometry(
    ref_cache: &RefCache,
    scratch: &mut MetricsScratch,
    test_pos: &[[f64; 3]],
    ref_normals_opt: Option<&[[f64; 3]]>,
    cfg: &MetricsConfig,
) -> MetricSummary {
    // Reference KD-tree is cached across sweeps by `RefCache`.
    let nn_r = &ref_cache.nn_ref;
    let ref_pos = nn_r.points();
    // Build a KD-tree for the test cloud if we need the symmetric pass.
    let nn_t = if cfg.symmetric {
        Some(Nn::build(test_pos))
    } else {
        None
    };

    let needed = test_pos.len().max(ref_pos.len());
    scratch.ensure_capacity(needed);
    scratch.clear();

    // Capture nearest-neighbour pairs once so subsequent metrics stay O(n).
    for q in test_pos {
        let (idx, dist) = nn_r.nn_index(q);
        scratch.matches.push(PointMatch { idx, dist });
    }

    if let Some(nn_t) = &nn_t {
        // Reuse d_buffer as temporary storage for the reverse (R → T) pass.
        scratch.d_buffer.clear();
        scratch.d_buffer.reserve(ref_pos.len());
        for q in ref_pos.iter() {
            scratch.d_buffer.push(nn_t.nn_index(q).1);
        }
        let len_r2t = scratch.d_buffer.len();
        if len_r2t != 0 {
            for (i, m) in scratch.matches.iter_mut().enumerate() {
                let r = scratch.d_buffer[i % len_r2t];
                m.dist = 0.5 * (m.dist + r);
            }
        }
        scratch.d_buffer.clear();
    }

    scratch
        .d_buffer
        .extend(scratch.matches.iter().map(|m| m.dist));
    let d1 = agg(scratch.d_buffer.as_mut_slice());
    let outlier_ratio = outl(
        scratch.d_buffer.as_slice(),
        &cfg.outlier_threshold,
        &mut scratch.mad_scratch,
    );

    // D2 (point-to-plane) and normal angle metrics require local normals. Either
    // use a pre-computed set or estimate on demand with PCA.
    let (d2, nang_mean, nang_p95) = if cfg.d2 || cfg.normal_angle {
        let ref_normals_cow: Cow<'_, [[f64; 3]]> = if let Some(n) = ref_normals_opt {
            Cow::Owned(normalize_owned(n))
        } else if let Some(n) = ref_cache.ref_normals.as_deref() {
            Cow::Borrowed(n)
        } else {
            Cow::Owned(estimate_normals_pca(ref_pos, nn_r, cfg.k_normals))
        };
        let ref_normals = ref_normals_cow.as_ref();

        scratch.d2_buffer.clear();
        let mut angles = Vec::<f64>::new(); // (optional) if you later supply test normals

        for (match_pair, q) in scratch.matches.iter().zip(test_pos.iter()) {
            let idx = match_pair.idx;
            let r = ref_pos[idx];
            let n = ref_normals[idx];
            let v = [q[0] - r[0], q[1] - r[1], q[2] - r[2]];
            let d2 = (v[0] * n[0] + v[1] * n[1] + v[2] * n[2]).abs();
            if cfg.d2 {
                scratch.d2_buffer.push(d2);
            }
        }

        // Aggregates over the L1 point-to-plane distance. Using the absolute projection
        // avoids penalising flipped normals twice; consumers can still inspect the sign
        // when they need oriented error.
        let d2_aggr = if cfg.d2 {
            Some(agg(scratch.d2_buffer.as_mut_slice()))
        } else {
            None
        };
        let (mean, p95) = if cfg.normal_angle && !angles.is_empty() {
            (
                Some(mean(&angles)),
                Some(quantile_unstable(angles.as_mut_slice(), 95.0)),
            )
        } else {
            (None, None)
        };

        (d2_aggr, mean, p95)
    } else {
        (None, None, None)
    };

    MetricSummary {
        d1,
        d2,
        n_ang_mean_deg: nang_mean,
        n_ang_p95_deg: nang_p95,
        outlier_ratio,
    }
}

fn agg(s: &mut [f64]) -> Aggregates {
    // The buffer is mutated in place in order to reuse memory between quantile calls.
    if s.is_empty() {
        return Aggregates {
            rmse: 0.0,
            median: 0.0,
            p95: 0.0,
        };
    }
    // RMSE = sqrt(mean squared error); robust to sign and useful for compatibility with
    // other codecs’ metrics.
    let rmse = (s.iter().map(|x| x * x).sum::<f64>() / (s.len() as f64)).sqrt();
    // `quantile_unstable` uses select_nth_unstable, so it does not fully sort the slice.
    // We call it twice (median + 95th percentile) and accept that the order changes between
    // calls - reserved buffers ensure the slice length stays constant.
    let median = quantile_unstable(s, 50.0);
    let p95 = quantile_unstable(s, 95.0);
    Aggregates { rmse, median, p95 }
}
fn normalize_owned(src: &[[f64; 3]]) -> Vec<[f64; 3]> {
    src.iter()
        .map(|n| {
            let mut out = [n[0], n[1], n[2]];
            normalize(&mut out);
            out
        })
        .collect()
}
fn outl(v: &[f64], rule: &OutlierThreshold, scratch: &mut Vec<f64>) -> f64 {
    match rule {
        OutlierThreshold::Abs { distance } => {
            if v.is_empty() {
                0.0
            } else {
                v.iter().filter(|&&x| x > *distance).count() as f64 / v.len() as f64
            }
        }
        OutlierThreshold::Mad3 => mad3_ratio(v, scratch),
    }
}
fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / (v.len().max(1) as f64)
}
#[inline]
fn quant_index(len: usize, p: f64) -> usize {
    (((p / 100.0) * ((len.saturating_sub(1)) as f64)).round() as usize).min(len.saturating_sub(1))
}

#[inline]
fn quantile_unstable(s: &mut [f64], p: f64) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    let idx = quant_index(s.len(), p);
    let (_, val, _) = s.select_nth_unstable_by(idx, |a, b| a.partial_cmp(b).unwrap());
    *val
}

#[inline]
fn mad3_ratio(v: &[f64], scratch: &mut Vec<f64>) -> f64 {
    // MAD = median absolute deviation; a robust scale estimator.
    if v.is_empty() {
        return 0.0;
    }
    ensure_capacity(scratch, v.len());
    scratch.clear();
    scratch.extend_from_slice(v);
    let med = quantile_unstable(scratch.as_mut_slice(), 50.0);
    // abs deviations reusing the same buffer
    for x in scratch.iter_mut() {
        *x = (*x - med).abs();
    }
    let mad = quantile_unstable(scratch.as_mut_slice(), 50.0);
    // `mad3` rule: flag points more than 3 MAD above the median.
    let thr = med + 3.0 * mad;
    v.iter().filter(|&&x| x > thr).count() as f64 / v.len() as f64
}
