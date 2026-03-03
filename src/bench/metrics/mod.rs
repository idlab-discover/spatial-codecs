//! Public exports for benchmark metrics.
//!
//! The bench harness pulls geometry and colour evaluators, along with the reference cache
//! helpers, from this module. Keeping the re-exports in one place makes it easier to
//! document and extend the metric suite.

mod color;
mod geometry;
mod nn;
mod normals;
mod ref_cache;

pub use color::evaluate_color;
pub use geometry::{evaluate_geometry, Aggregates, MetricSummary, MetricsScratch};
pub use ref_cache::{RefCache, RefCacheMap};
