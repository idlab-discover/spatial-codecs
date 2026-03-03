//! Benchmark reporting helpers.
//!
//! The `report` module provides lightweight human-readable summaries for benchmark runs.
//! The heavy lifting (serialisation, CSV/JSON) happens elsewhere - this file is about quick
//! console inspection.

use crate::bench::config::BenchConfig;
use crate::bench::runner::RunOptions;
use crate::bench::serialize::BenchOutcome;

/// Print a friendly header describing the upcoming run.
pub fn print_run_header(cfg: &BenchConfig, opts: &RunOptions) {
    println!("== spatial_codecs benchmark ==");
    println!(
        "Datasets: {}",
        cfg.datasets
            .roots
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("Sweeps  : {}", cfg.sweeps.len());
    println!(
        "Mode    : {}",
        if opts.mode_throughput {
            "Throughput (parallel)"
        } else {
            "Accuracy (sequential)"
        }
    );
    if opts.mode_throughput {
        println!(
            "Jobs    : {}",
            if opts.jobs == 0 {
                "auto".into()
            } else {
                opts.jobs.to_string()
            }
        );
    }
    println!("Warm-up : {}", if opts.warmup { "on" } else { "off" });
    println!(
        "Progress: {}",
        if opts.show_progress { "on" } else { "off" }
    );
    println!();
}

/// Summarise the list of outcomes by sweep name.
///
/// We report medians rather than means because encode/decode timings and RMSE can be skewed
/// by occasional spikes (e.g. OS scheduling, I/O hiccups). Medians are more robust for the
/// quick-look scenario this summary targets.
pub fn print_outcomes_summary(rows: &[BenchOutcome]) {
    use std::collections::BTreeMap;
    if rows.is_empty() {
        println!("No results.");
        return;
    }

    // aggregate by sweep name
    let mut by_name: BTreeMap<&str, Vec<&BenchOutcome>> = BTreeMap::new();
    for r in rows {
        by_name.entry(&r.sweep_name).or_default().push(r);
    }

    println!("\n== Summary by sweep ==");
    println!(
        "{:>40}  {:>8}  {:>10}  {:>10}  {:>9}  {:>9}",
        "Sweep", "Runs", "bpp~med", "D1_rmse~med", "enc(ms)", "dec(ms)"
    );
    for (name, v) in by_name {
        if v.is_empty() {
            continue;
        }
        let med_bpp = median(v.iter().map(|r| r.bpp));
        let med_d1 = median(v.iter().map(|r| r.d1_rmse));
        let med_enc_ns = median_u128(v.iter().map(|r| r.encode_ns));
        let med_dec_ns = median_u128(v.iter().map(|r| r.decode_ns));
        let med_enc_ms = (med_enc_ns as f64) / 1_000_000.0;
        let med_dec_ms = (med_dec_ns as f64) / 1_000_000.0;
        println!(
            "{name:>20}  {:>8}  {:>10.4}  {:>10.4}  {:>9.3}  {:>9.3}",
            v.len(),
            med_bpp,
            med_d1,
            med_enc_ms,
            med_dec_ms
        );
    }
    println!();
}

fn median<I: Iterator<Item = f64>>(iter: I) -> f64 {
    let mut v: Vec<f64> = iter.collect();
    if v.is_empty() {
        return 0.0;
    }
    // Sorting is acceptable here because the result set is typically small (datasets × sweeps).
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}
fn median_u128<I: Iterator<Item = u128>>(iter: I) -> u128 {
    let mut v: Vec<u128> = iter.collect();
    if v.is_empty() {
        return 0;
    }
    v.sort();
    v[v.len() / 2]
}
