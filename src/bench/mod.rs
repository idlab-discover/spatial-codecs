//! Benchmark harness modules.
//!
//! Each submodule focuses on a single concern:
//! - [`config`] - serde-deserialisable run configuration.
//! - [`io`] - dataset discovery and loading.
//! - [`sampling`] - dataset resampling helpers.
//! - [`metrics`] - geometry/colour metrics with reusable scratch buffers.
//! - [`runner`] - orchestrates end-to-end benchmark execution.
//! - [`report`] - human-friendly summaries.
//! - [`serialize`] - CSV/JSONL writers.
//! - [`progress`] / [`timing`] - utilities used by the runner.

pub mod config;
pub mod io;
pub mod metrics;
pub mod progress;
pub mod report;
pub mod runner;
pub mod serialize;
pub mod timing;
