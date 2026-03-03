//! Normal estimation helpers used by the geometry metrics.
//!
//! We perform a straightforward PCA (Principal Component Analysis) over the neighbourhood
//! of each point. Given the `k` closest neighbours, PCA builds the 3×3 covariance matrix of
//! their coordinates. The eigenvectors of this matrix describe the principal directions of
//! local variation; the eigenvector associated with the smallest eigenvalue is orthogonal to
//! the best-fitting plane through the neighbourhood and therefore serves as the surface
//! normal. Normals are normalised to unit length before being returned.

use super::nn::Nn;
use nalgebra as na;

/// Estimate normals via PCA over the `k` nearest neighbours of each point.
///
/// The PCA covariance matrix is accumulated explicitly so we can keep everything in
/// owned `f64` arrays (no temporary heap allocations per point). `k` is clamped to at least
/// three neighbours - fewer would make the covariance matrix singular.
pub fn estimate_normals_pca(pts: &[[f64; 3]], nn: &Nn, k: usize) -> Vec<[f64; 3]> {
    pts.iter()
        .map(|p| {
            let idxs = nn.k_indices(p, k.max(3));
            let mu = mean3(idxs.iter().map(|&i| pts[i]));
            let mut cov = na::Matrix3::<f64>::zeros();
            for &i in &idxs {
                let q = pts[i];
                let d = [q[0] - mu[0], q[1] - mu[1], q[2] - mu[2]];
                cov[(0, 0)] += d[0] * d[0];
                cov[(0, 1)] += d[0] * d[1];
                cov[(0, 2)] += d[0] * d[2];
                cov[(1, 0)] += d[1] * d[0];
                cov[(1, 1)] += d[1] * d[1];
                cov[(1, 2)] += d[1] * d[2];
                cov[(2, 0)] += d[2] * d[0];
                cov[(2, 1)] += d[2] * d[1];
                cov[(2, 2)] += d[2] * d[2];
            }
            // Convert sums to covariance by dividing with the number of points.
            cov /= idxs.len() as f64;
            let eig = cov.symmetric_eigen();
            let (mut vmin, mut min) = (eig.eigenvectors.column(0).into_owned(), eig.eigenvalues[0]);
            for j in 1..3 {
                if eig.eigenvalues[j] < min {
                    min = eig.eigenvalues[j];
                    vmin = eig.eigenvectors.column(j).into_owned();
                }
            }
            let mut n = [vmin[0], vmin[1], vmin[2]];
            normalize(&mut n);
            n
        })
        .collect()
}

#[inline]
fn mean3<I: Iterator<Item = [f64; 3]>>(it: I) -> [f64; 3] {
    let (mut x, mut y, mut z, mut c) = (0.0, 0.0, 0.0, 0.0);
    for p in it {
        x += p[0];
        y += p[1];
        z += p[2];
        c += 1.0;
    }
    if c == 0.0 {
        [0.0, 0.0, 0.0]
    } else {
        [x / c, y / c, z / c]
    }
}
#[inline]
pub fn normalize(n: &mut [f64; 3]) {
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if l > 0.0 {
        n[0] /= l;
        n[1] /= l;
        n[2] /= l;
    }
}
