//! Reference-side caches for benchmark metrics.
//!
//! Many metrics (geometry, colour) need KD-trees or derived per-point attributes
//! such as normals or YCbCr values. Building those on every sweep would dominate
//! runtime, so we memoise them per unique reference dataset.

use super::{nn::Nn, normals::estimate_normals_pca};
use crate::bench::config::MetricsConfig;
use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    hash::{Hash, Hasher},
};

/// Cached artefacts derived from the reference point cloud.
pub struct RefCache<'a> {
    pub nn_ref: Nn<'a>,
    pub ref_normals: Option<Vec<[f64; 3]>>,
    pub ref_ycbcr: Option<Vec<[f64; 3]>>,
}

#[derive(Default)]
pub struct RefCacheMap<'a> {
    map: HashMap<u64, RefCache<'a>>,
}

impl<'a> RefCacheMap<'a> {
    pub fn new() -> Self {
        RefCacheMap::default()
    }

    /// Obtain a [`RefCache`] for the specific reference position/colour arrays.
    ///
    /// The key is a lightweight fingerprint - good enough to avoid rebuilding the same
    /// KD-tree during a benchmark session but cheap enough that we can call this function
    /// before each sweep without worrying about collisions.
    pub fn get_or_build(
        &mut self,
        ref_pos: &'a [[f64; 3]],
        ref_rgb: &[[u8; 3]],
        cfg: &MetricsConfig,
    ) -> &RefCache<'a> {
        let key = fingerprint_ref(ref_pos, ref_rgb);
        let entry = self.map.entry(key).or_insert_with(|| RefCache {
            nn_ref: Nn::build(ref_pos),
            ref_normals: None,
            ref_ycbcr: None,
        });

        if entry.ref_normals.is_none() && (cfg.d2 || cfg.normal_angle) {
            entry.ref_normals = Some(estimate_normals_pca(ref_pos, &entry.nn_ref, cfg.k_normals));
        }
        if entry.ref_ycbcr.is_none() && cfg.color_psnr {
            entry.ref_ycbcr = Some(
                ref_rgb
                    .iter()
                    .map(|&c| {
                        let (y, cb, cr) = super::color::ycbcr709(c);
                        [y, cb, cr]
                    })
                    .collect(),
            );
        }
        entry
    }
}

fn fingerprint_ref(pos: &[[f64; 3]], rgb: &[[u8; 3]]) -> u64 {
    // Cheap & deterministic; collisions are extremely unlikely for typical dataset sizes
    // but we keep the hash short so the map stays fast.
    let mut h = DefaultHasher::new();
    pos.len().hash(&mut h);
    rgb.len().hash(&mut h);
    // Sample a subset to avoid O(n) hashing on huge clouds; adjust the stride if your
    // datasets demand stronger fingerprints.
    let stride_pos = (pos.len() / 1024).max(1);
    for p in pos.iter().step_by(stride_pos) {
        for v in p {
            v.to_bits().hash(&mut h);
        }
    }
    let stride_rgb = (rgb.len() / 1024).max(1);
    for c in rgb.iter().step_by(stride_rgb) {
        c.hash(&mut h);
    }
    h.finish()
}
