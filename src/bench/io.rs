//! Dataset discovery and loading utilities for the benchmark harness.
//!
//! Benchmarks operate on collections of point-cloud files. These helpers scan the dataset
//! roots, pick files with known extensions, and decode them into the common `Point3D`
//! representation using the crate’s own decoders.

use std::{
    fs,
    path::{Path, PathBuf},
};

/// Recursively discover files with supported point-cloud extensions under `roots`.
///
/// The function accepts both files and directories. Directories are scanned one level deep
/// (sufficient for typical dataset layouts). Returned paths are sorted for reproducibility.
pub fn discover_pc_files(roots: &[PathBuf]) -> Vec<PathBuf> {
    let exts = ["ply", "pcd", "tmf", "dra", "gzp"];
    let mut out = Vec::new();
    for root in roots {
        if root.is_file() {
            out.push(root.clone());
            continue;
        }
        if let Ok(rd) = fs::read_dir(root) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension()
                    .and_then(|s| s.to_str())
                    .map(|s| exts.contains(&s))
                    .unwrap_or(false)
                {
                    out.push(p);
                }
            }
        }
    }
    out.sort();
    out
}

pub type Pt = spatial_utils::point::Point3D;

/// Decode a reference file into `Point3D` values using the crate’s generic decoder.
pub fn load_ref_points(path: &Path) -> Result<Vec<Pt>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    crate::decoder::decode_to_points_vec(&bytes)
}
