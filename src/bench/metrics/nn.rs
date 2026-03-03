//! KD-tree wrapper used by the geometry metrics.
//!
//! A *kd-tree* partitions 3D space by recursively splitting the point set along coordinate
//! axes, producing a balanced binary tree. During nearest-neighbour queries the tree lets us
//! discard large regions that cannot possibly contain a closer point, yielding logarithmic
//! behaviour on typical data instead of the linear scans we would otherwise need.
//!
//! kiddo (the KD-tree crate we depend on) requires the node bucket size `B` (the maximum
//! number of points stored in a leaf) to be known at compile time. Real-world datasets
//! frequently contain repeated coordinates along one axis (think quantised LiDAR scans), and
//! the library demands `B >= ties + 1`. Rather than picking a single pessimistic `B`, we
//! compile a small ladder of bucket sizes and choose the smallest that satisfies the dataset
//! we’re probing. When even that is insufficient we fall back to a linear scan: correctness is
//! more important than a rare slowdown.

use std::collections::HashMap;

// kiddo 5.x: use the concrete float kd-tree path + metric
use kiddo::float::kdtree::KdTree as RawKd;
use kiddo::SquaredEuclidean;

// Item type (index stored in the tree). u32 is fine for typical dataset sizes.
type Item = u32;

// Handy type alias for a concrete B
type KD<const B: usize> = RawKd<f64, Item, 3, B, Item>;

pub struct Nn<'a> {
    pts: &'a [[f64; 3]],
    tree: Tree,
}

enum Tree {
    B32(KD<32>),
    B128(KD<128>),
    B512(KD<512>),
    B1024(KD<1024>),
    B2048(KD<2048>),
    B8192(KD<8192>),
    Linear, // fallback for absurdly large per-axis ties
}

impl<'a> Nn<'a> {
    /// Build a KD-tree tuned to the observed number of axis ties.
    pub fn build(pts: &'a [[f64; 3]]) -> Self {
        let ties = max_axis_ties(pts);
        let tree = build_tree_with_bucket(pts, ties);
        Self { pts, tree }
    }

    #[inline]
    pub fn points(&self) -> &[[f64; 3]] {
        self.pts
    }

    #[inline]
    pub fn nn_index(&self, q: &[f64; 3]) -> (usize, f64) {
        match &self.tree {
            Tree::B32(t) => to_res(t.nearest_one::<SquaredEuclidean>(q)),
            Tree::B128(t) => to_res(t.nearest_one::<SquaredEuclidean>(q)),
            Tree::B512(t) => to_res(t.nearest_one::<SquaredEuclidean>(q)),
            Tree::B1024(t) => to_res(t.nearest_one::<SquaredEuclidean>(q)),
            Tree::B2048(t) => to_res(t.nearest_one::<SquaredEuclidean>(q)),
            Tree::B8192(t) => to_res(t.nearest_one::<SquaredEuclidean>(q)),
            Tree::Linear => lin_nn(self.pts, q),
        }
    }

    pub fn k_indices(&self, q: &[f64; 3], k: usize) -> Vec<usize> {
        match &self.tree {
            Tree::B32(t) => t
                .nearest_n::<SquaredEuclidean>(q, k)
                .into_iter()
                .map(|n| n.item as usize)
                .collect(),
            Tree::B128(t) => t
                .nearest_n::<SquaredEuclidean>(q, k)
                .into_iter()
                .map(|n| n.item as usize)
                .collect(),
            Tree::B512(t) => t
                .nearest_n::<SquaredEuclidean>(q, k)
                .into_iter()
                .map(|n| n.item as usize)
                .collect(),
            Tree::B1024(t) => t
                .nearest_n::<SquaredEuclidean>(q, k)
                .into_iter()
                .map(|n| n.item as usize)
                .collect(),
            Tree::B2048(t) => t
                .nearest_n::<SquaredEuclidean>(q, k)
                .into_iter()
                .map(|n| n.item as usize)
                .collect(),
            Tree::B8192(t) => t
                .nearest_n::<SquaredEuclidean>(q, k)
                .into_iter()
                .map(|n| n.item as usize)
                .collect(),
            Tree::Linear => lin_k(self.pts, q, k),
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub fn point(&self, i: usize) -> [f64; 3] {
        self.pts[i]
    }
}

#[inline]
fn to_res(nn: kiddo::nearest_neighbour::NearestNeighbour<f64, Item>) -> (usize, f64) {
    (nn.item as usize, nn.distance.sqrt())
}

// ------- constructors -------

fn build_tree_with_bucket(pts: &[[f64; 3]], max_ties: usize) -> Tree {
    // kiddo requires: B >= 1 + max items sharing the same coordinate on ANY ONE axis
    let need = max_ties + 1;

    // Precompiled choices (extend if you run into bigger ties)
    if need <= 32 {
        let mut t: KD<32> = RawKd::with_capacity(pts.len());
        add_all(&mut t, pts);
        Tree::B32(t)
    } else if need <= 128 {
        let mut t: KD<128> = RawKd::with_capacity(pts.len());
        add_all(&mut t, pts);
        Tree::B128(t)
    } else if need <= 512 {
        let mut t: KD<512> = RawKd::with_capacity(pts.len());
        add_all(&mut t, pts);
        Tree::B512(t)
    } else if need <= 1024 {
        let mut t: KD<1024> = RawKd::with_capacity(pts.len());
        add_all(&mut t, pts);
        Tree::B1024(t)
    } else if need <= 2048 {
        let mut t: KD<2048> = RawKd::with_capacity(pts.len());
        add_all(&mut t, pts);
        Tree::B2048(t)
    } else if need <= 8192 {
        let mut t: KD<8192> = RawKd::with_capacity(pts.len());
        add_all(&mut t, pts);
        Tree::B8192(t)
    } else {
        // Safety-first: correctness with linear scan if ties are enormous
        Tree::Linear
    }
}

fn add_all<const B: usize>(tree: &mut KD<B>, pts: &[[f64; 3]]) {
    // `with_capacity` performs the allocation upfront. Here we only insert points along
    // with their indices (stored as u32).
    for (i, p) in pts.iter().enumerate() {
        // item is a compact index
        tree.add(p, i as Item);
    }
}

/// Count exact-value ties per axis.
/// We’re safe using exact equality because the coordinates are quantized (PLY/integers in f64).
fn max_axis_ties(pts: &[[f64; 3]]) -> usize {
    let mut cx: HashMap<u64, usize> = HashMap::new();
    let mut cy: HashMap<u64, usize> = HashMap::new();
    let mut cz: HashMap<u64, usize> = HashMap::new();

    for p in pts {
        *cx.entry(p[0].to_bits()).or_insert(0) += 1;
        *cy.entry(p[1].to_bits()).or_insert(0) += 1;
        *cz.entry(p[2].to_bits()).or_insert(0) += 1;
    }

    cx.values()
        .chain(cy.values())
        .chain(cz.values())
        .copied()
        .max()
        .unwrap_or(1)
}

// ------- linear fallback (rare path) -------

fn lin_nn(pts: &[[f64; 3]], q: &[f64; 3]) -> (usize, f64) {
    let mut best_i = 0usize;
    let mut best_d2 = f64::INFINITY;
    // Keep squared distances until the end to avoid square roots in the hot loop.
    for (i, p) in pts.iter().enumerate() {
        let dx = p[0] - q[0];
        let dy = p[1] - q[1];
        let dz = p[2] - q[2];
        let d2 = dx * dx + dy * dy + dz * dz;
        if d2 < best_d2 {
            best_d2 = d2;
            best_i = i;
        }
    }
    (best_i, best_d2.sqrt())
}

fn lin_k(pts: &[[f64; 3]], q: &[f64; 3], k: usize) -> Vec<usize> {
    use std::cmp::Reverse;
    let mut heap: Vec<(Reverse<f64>, usize)> = Vec::with_capacity(k);
    for (i, p) in pts.iter().enumerate() {
        let dx = p[0] - q[0];
        let dy = p[1] - q[1];
        let dz = p[2] - q[2];
        let d2 = dx * dx + dy * dy + dz * dz;
        if heap.len() < k {
            heap.push((Reverse(d2), i));
            if heap.len() == k {
                // Sort using `partial_cmp` because `f64` has NaN. Nearest neighbour lookups
                // never produce NaNs, so the unwrap is safe.
                heap.sort_unstable_by(|a, b| a.0 .0.partial_cmp(&b.0 .0).unwrap());
            }
        } else if d2 < (heap[heap.len() - 1].0).0 {
            let pos = heap.len() - 1;
            heap[pos] = (Reverse(d2), i);
            heap.sort_unstable_by(|a, b| a.0 .0.partial_cmp(&b.0 .0).unwrap());
        }
    }
    heap.into_iter().map(|(_, i)| i).collect()
}
