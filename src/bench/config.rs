//! Deserialisable configuration structures for the benchmarking harness.
//!
//! A `BenchConfig` typically comes from a TOML file (see `configs/bench.toml`). The
//! `serde`-annotated types below provide a strongly typed view over that file so the runner
//! can reason about datasets, codec sweeps, metric settings, and resampling policies.
//!
//! The guiding principles are:
//! - Keep the format human-editable.
//! - Avoid bespoke glue: wherever possible we forward directly to existing codec parameter
//!   structures (e.g. `EncodingParams`).

use crate::encoder::EncodingParams;
use serde::Deserialize;
use std::path::PathBuf; // Our serde-enabled enum

/// Top-level benchmark configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct BenchConfig {
    pub datasets: Datasets,
    pub sweeps: Vec<Sweep>,
    #[serde(default)]
    pub metrics: MetricsConfig,
}

/// Dataset sources (directories or individual files) plus optional resampling.
#[derive(Debug, Clone, Deserialize)]
pub struct Datasets {
    pub roots: Vec<PathBuf>,
    #[serde(default)]
    pub resample: Option<ResampleSpec>,
}

/// One codec configuration to test. `params` is fully deserialised into the
/// crate-wide [`EncodingParams`] enum, so individual codecs do not require extra glue.
#[derive(Debug, Clone, Deserialize)]
pub struct Sweep {
    pub name: Option<String>,
    pub params: EncodingParams, // ← zero glue code
}

/// Rules describing which distances count as outliers when summarising geometry error.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind")]
pub enum OutlierThreshold {
    #[serde(rename = "abs")]
    Abs { distance: f64 },
    #[serde(rename = "mad3")]
    Mad3,
}

/// Metric configuration toggles.
#[derive(Debug, Clone, Deserialize)]
pub struct MetricsConfig {
    #[serde(default = "d_true")]
    pub symmetric: bool,
    /// Number of neighbours to consider when estimating local surfaces (used by PCA for
    /// normals and by the symmetric Chamfer-style distance). Larger values smooth out
    /// noise but increase the spatial footprint of each query.
    #[serde(default = "d_k")]
    pub k_normals: usize,
    #[serde(default = "d_outl")]
    pub outlier_threshold: OutlierThreshold,
    #[serde(default = "d_true")]
    pub normal_angle: bool,
    #[serde(default = "d_true")]
    pub d2: bool,
    #[serde(default = "d_true")]
    pub color_psnr: bool,
}
impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            symmetric: true,
            k_normals: 16,
            outlier_threshold: OutlierThreshold::Mad3,
            normal_angle: true,
            d2: true,
            color_psnr: true,
        }
    }
}
fn d_true() -> bool {
    true
}
fn d_k() -> usize {
    16
}
fn d_outl() -> OutlierThreshold {
    OutlierThreshold::Mad3
}

/// Dataset-level resampling strategies.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "strategy")]
pub enum ResampleSpec {
    /// Keep each point with probability `rate` in (0,1]
    #[serde(rename = "rate")]
    Rate { rate: f64 },

    /// Choose exactly `count` points uniformly at random (without replacement)
    #[serde(rename = "count")]
    Count { count: usize },

    /// Weighted by proximity to ROIs
    #[serde(rename = "biased")]
    Biased {
        count: usize,
        roi_centers: Vec<RoiCenter>,
        radius: f32,
        roi_weight: f32,
    },

    /// Split by percentages and pick one bucket (0-based)
    #[serde(rename = "partition")]
    Partition { percentages: Vec<u8>, pick: usize },
}

/// Let ROI centers be provided as either [x,y,z] or {x=..,y=..,z=..}
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum RoiCenter {
    Arr([f32; 3]),
    Obj { x: f32, y: f32, z: f32 },
}
impl RoiCenter {
    pub fn to_xyz(&self) -> [f32; 3] {
        match *self {
            RoiCenter::Arr(a) => a,
            RoiCenter::Obj { x, y, z } => [x, y, z],
        }
    }
}
