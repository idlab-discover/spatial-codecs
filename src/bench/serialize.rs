//! Serialisation utilities for benchmark outputs.
//!
//! Benchmarks typically want both machine-readable (JSONL/CSV) and human-readable artefacts.
//! The helpers here produce flat records summarising each `(dataset, sweep)` combination.

use serde::Serialize;
use std::{fs::File, io::Write, path::PathBuf};

/// Row written by the benchmark reporters.
#[derive(Debug, Serialize)]
pub struct BenchOutcome {
    pub dataset_id: String,
    pub dataset_path: String,
    pub sweep_name: String,
    pub params_json: String,
    pub num_points_ref: usize,
    pub num_points_dec: usize,
    pub bytes: usize,
    pub bpp: f64,
    pub encode_ns: u128,
    pub decode_ns: u128,
    pub encode_ms: f64,
    pub decode_ms: f64,
    pub d1_rmse: f64,
    pub d1_median: f64,
    pub d1_p95: f64,
    pub d2_rmse: Option<f64>,
    pub d2_median: Option<f64>,
    pub d2_p95: Option<f64>,
    pub n_ang_mean_deg: Option<f64>,
    pub n_ang_p95_deg: Option<f64>,
    pub outlier_ratio: f64,
    pub y_psnr: Option<f64>,
    pub cb_psnr: Option<f64>,
    pub cr_psnr: Option<f64>,
}

/// Write outcomes as CSV with a fixed column order.
pub fn write_csv(path: &PathBuf, rows: &[BenchOutcome]) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = File::create(path)?;
    writeln!(
        f,
        "dataset_id,dataset_path,sweep_name,params_json,num_points_ref,num_points_dec,bytes,bpp,encode_ns,decode_ns,encode_ms,decode_ms,d1_rmse,d1_median,d1_p95,d2_rmse,d2_median,d2_p95,n_ang_mean_deg,n_ang_p95_deg,outlier_ratio,y_psnr,cb_psnr,cr_psnr"
    )?;
    for r in rows {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{:.6},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{:.6},{},{},{}",
            esc(&r.dataset_id),
            esc(&r.dataset_path),
            esc(&r.sweep_name),
            esc(&r.params_json),
            r.num_points_ref,
            r.num_points_dec,
            r.bytes,
            r.bpp,
            r.encode_ns,
            r.decode_ns,
            r.encode_ms,
            r.decode_ms,
            r.d1_rmse,
            r.d1_median,
            r.d1_p95,
            optf(r.d2_rmse),
            optf(r.d2_median),
            optf(r.d2_p95),
            optf(r.n_ang_mean_deg),
            optf(r.n_ang_p95_deg),
            r.outlier_ratio,
            optf(r.y_psnr),
            optf(r.cb_psnr),
            optf(r.cr_psnr)
        )?;
    }
    Ok(())
}

/// Write outcomes as newline-delimited JSON (`.jsonl`).
pub fn write_jsonl(
    path: &PathBuf,
    rows: &[BenchOutcome],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = File::create(path)?;
    for r in rows {
        writeln!(f, "{}", serde_json::to_string(r)?)?;
    }
    Ok(())
}

/// Escape CSV fields when they contain commas, quotes, or line breaks.
fn esc(s: &str) -> String {
    if s.contains(&[',', '"', '\n'][..]) {
        let mut v = String::from("\"");
        for ch in s.chars() {
            if ch == '"' {
                v.push_str("\"\"");
            } else {
                v.push(ch);
            }
        }
        v.push('"');
        v
    } else {
        s.to_string()
    }
}
/// Format optional floats, emitting an empty field when `None`.
fn optf(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.6}")).unwrap_or_default()
}
