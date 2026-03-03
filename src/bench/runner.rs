//! Benchmark harness entry points.
//!
//! This module coordinates dataset discovery, codec execution, metrics, and reporting.
//! The design goals are:
//! - Reuse allocations wherever possible between sweeps.
//! - Keep timing measurements honest by supporting optional warm-up runs.
//! - Make it easy to add new codecs or metrics without modifying the harness core.

use crate::bench::config::{BenchConfig, ResampleSpec};
use crate::bench::io::{discover_pc_files, load_ref_points};
use crate::bench::metrics::{evaluate_color, evaluate_geometry, MetricsScratch, RefCacheMap};
use crate::bench::progress::make_bar;
use crate::bench::serialize::BenchOutcome;
use crate::bench::timing::time_nanos;
use serde_json;

use crate::bench::config::MetricsConfig;
use crate::decoder::decode_to_points_vec;
use crate::encoder::encode_from_points_with_params;

/// Tuning knobs for [`run_bench`].
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// When `true`, the runner focuses on throughput and enables optional parallelism.
    pub mode_throughput: bool,
    /// Number of worker threads to use when throughput mode is active (`0` = auto-detect).
    pub jobs: usize,
    /// Enable/disable the progress bar.
    pub show_progress: bool,
    /// Print per-sweep stats to stdout as they are produced.
    pub verbose: bool,
    /// Perform a warm-up iteration before each measurement to amortise one-off allocations.
    pub warmup: bool,
}

/// Run all configured sweeps across the discovered datasets.
///
/// Returns a flat vector of [`BenchOutcome`] (one per `(dataset, sweep)` pair).
/// The harness is deterministic as long as resampling specifications are
/// deterministic - the dataset order and sweep order are fixed.
pub fn run_bench(
    cfg: &BenchConfig,
    opts: &RunOptions,
) -> Result<Vec<BenchOutcome>, Box<dyn std::error::Error>> {
    let files = discover_pc_files(&cfg.datasets.roots);
    println!("Discovered {} point cloud files.", files.len());

    let pb = make_bar(files.len() as u64, opts.show_progress);

    // Optional pool (throughput mode). In accuracy mode, stay sequential.
    #[cfg(feature = "bench_parallel")]
    let pool = if opts.mode_throughput {
        let n = if opts.jobs == 0 {
            num_cpus::get()
        } else {
            opts.jobs
        };
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap(),
        )
    } else {
        None
    };

    let mut outcomes = Vec::with_capacity(files.len() * cfg.sweeps.len());

    // Closure to process a single file (sequential inside)
    let process_file = |path: &std::path::Path| -> Vec<BenchOutcome> {
        let ds_id = path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let mut pts_ref = match load_ref_points(path) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Failed to load {}: {e}", path.display());
                return Vec::new();
            }
        };

        // If there are no reference points, skip
        if pts_ref.is_empty() {
            eprintln!("Skipping {}: no reference points.", path.display());
            return Vec::new();
        }

        // Optional dataset-level resample
        // let original_n = pts_ref.len();
        if let Some(rs) = &cfg.datasets.resample {
            pts_ref = apply_resample(&pts_ref, rs);
            if pts_ref.is_empty() {
                eprintln!("Skipping {}: resample produced 0 points.", path.display());
                return Vec::new();
            }
        }

        // Pre-slice views for metrics. We store everything as f64 because the KD-tree
        // implementation expects double precision, and conversion happens only once
        // per dataset rather than per sweep.
        let ref_pos: Vec<[f64; 3]> = pts_ref
            .iter()
            .map(|p| [p.x as f64, p.y as f64, p.z as f64])
            .collect();
        let ref_rgb: Vec<[u8; 3]> = pts_ref.iter().map(|p| [p.r, p.g, p.b]).collect();
        let mut ref_cache_map = RefCacheMap::new();
        // Geometry scratch buffers are sized to the reference cloud once and then reused
        // by each sweep. This keeps allocations O(datasets) instead of O(datasets×sweeps).
        let mut geom_scratch = MetricsScratch::with_capacity(ref_pos.len());

        let mut rows = Vec::with_capacity(cfg.sweeps.len());

        for sw in &cfg.sweeps {
            let name = sw.name.clone().unwrap_or_else(|| "sweep".into());
            let params_json = serde_json::to_string(&sw.params).unwrap_or_else(|_| "{}".into());

            // Encode (optionally warm up)
            let (enc_ns, enc_bytes) = {
                let f =
                    || encode_from_points_with_params(pts_ref.clone(), &sw.params).expect("encode");
                // `time_nanos` optionally runs `f` twice (warm-up + measurement) to smooth out
                // allocator cold-starts and instruction cache misses.
                let (t, out) = time_nanos(f, opts.warmup);
                (t, out.unwrap())
            };

            // Size & bpp
            let n_pts = pts_ref.len();
            let bytes = enc_bytes.len();
            let bpp = 8.0 * (bytes as f64) / (n_pts.max(1) as f64);

            // Decode (optionally warm up)
            let (dec_ns, dec_pts) = {
                let dec = |b: Vec<u8>| move || decode_to_points_vec(&b).expect("decode");
                // We re-clone the encoded buffer per sweep so the decoder receives the same data
                // whether or not warm-up is enabled.
                let (t, out) = time_nanos(dec(enc_bytes), opts.warmup);
                (t, out.unwrap())
            };
            let enc_ms = enc_ns as f64 / 1_000_000.0;
            let dec_ms = dec_ns as f64 / 1_000_000.0;

            // Convert decoded output into f64 / u8 slices so metrics can reuse the same
            // code path as the reference preparation step above.
            let test_pos: Vec<[f64; 3]> = dec_pts
                .iter()
                .map(|p| [p.x as f64, p.y as f64, p.z as f64])
                .collect();
            let test_rgb: Vec<[u8; 3]> = dec_pts.iter().map(|p| [p.r, p.g, p.b]).collect();

            // Metrics
            let mcfg: MetricsConfig = cfg.metrics.clone();
            let ref_cache = ref_cache_map.get_or_build(&ref_pos, &ref_rgb, &mcfg);
            let geom = evaluate_geometry(ref_cache, &mut geom_scratch, &test_pos, None, &mcfg);
            let color = if mcfg.color_psnr {
                let c = evaluate_color(ref_cache, &test_pos, &test_rgb);
                (c.y, c.cb, c.cr)
            } else {
                (None, None, None)
            };

            if opts.verbose {
                println!(
                    "[{ds_id}] {name} | pts={n_pts} | size={bytes} B (bpp={bpp:.4}) | encode={enc_ms:.3}ms decode={dec_ms:.3}ms | D1_RMSE={:.4} D1_P95={:.4}{}", geom.d1.rmse, geom.d1.p95,
                    color.0.map(|v| format!(" | PSNR-Y={v:.2}dB")).unwrap_or_default()
                );
            }

            rows.push(BenchOutcome {
                dataset_id: ds_id.clone(),
                dataset_path: path.display().to_string(),
                sweep_name: name,
                params_json,
                num_points_ref: n_pts,
                num_points_dec: dec_pts.len(),
                bytes,
                bpp,
                encode_ns: enc_ns,
                decode_ns: dec_ns,
                encode_ms: enc_ms,
                decode_ms: dec_ms,
                d1_rmse: geom.d1.rmse,
                d1_median: geom.d1.median,
                d1_p95: geom.d1.p95,
                d2_rmse: geom.d2.as_ref().map(|a| a.rmse),
                d2_median: geom.d2.as_ref().map(|a| a.median),
                d2_p95: geom.d2.as_ref().map(|a| a.p95),
                n_ang_mean_deg: geom.n_ang_mean_deg,
                n_ang_p95_deg: geom.n_ang_p95_deg,
                outlier_ratio: geom.outlier_ratio,
                y_psnr: color.0,
                cb_psnr: color.1,
                cr_psnr: color.2,
            });
        }

        rows
    };

    // Iterate files
    #[cfg(feature = "bench_parallel")]
    if let Some(pool) = &pool {
        pool.install(|| {
            use rayon::prelude::*;
            let rows: Vec<Vec<BenchOutcome>> = files.par_iter().map(|p| process_file(p)).collect();
            for chunk in rows {
                if let Some(pb) = &pb {
                    pb.inc(1);
                }
                outcomes.extend(chunk);
            }
        });
    } else {
        for p in &files {
            let rows = process_file(p);
            if let Some(pb) = &pb {
                pb.inc(1);
            }
            outcomes.extend(rows);
        }
    }

    #[cfg(not(feature = "bench_parallel"))]
    {
        for p in &files {
            let rows = process_file(p);
            if let Some(pb) = &pb {
                pb.inc(1);
            }
            outcomes.extend(rows);
        }
    }

    if let Some(pb) = &pb {
        pb.finish_with_message("done");
    }
    Ok(outcomes)
}

/// Apply dataset-level resampling prior to encoding.
///
/// Resampling allows sweeps to operate on comparable point counts, which helps when
/// interpreting compression ratios and PSNR across wildly different frames.
pub fn apply_resample(
    pts: &[crate::bench::io::Pt],
    spec: &ResampleSpec,
) -> Vec<crate::bench::io::Pt> {
    use spatial_utils::{point::Point3D, sampling};
    match spec {
        ResampleSpec::Rate { rate } => {
            let mut out = sampling::bernoulli::random_sampling(pts, *rate);
            if out.is_empty() && !pts.is_empty() && *rate > 0.0 {
                // Avoid producing an empty cloud when the target rate is >0 but the RNG
                // happened to drop every sample (possible with tiny clouds).
                out.push(pts[0]);
            }
            out
        }
        ResampleSpec::Count { count } => {
            let want = (*count).min(pts.len());
            sampling::exact_random::exact_random_sampling(pts, want)
        }
        ResampleSpec::Biased {
            count,
            roi_centers,
            radius,
            roi_weight,
        } => {
            let rois: Vec<Point3D> = roi_centers
                .iter()
                .map(|c| {
                    let [x, y, z] = c.to_xyz();
                    Point3D {
                        x,
                        y,
                        z,
                        r: 0,
                        g: 0,
                        b: 0,
                    }
                })
                .collect();
            let want = (*count).min(pts.len());
            sampling::exact_random::biased_exact_random_sampling(
                pts,
                want,
                &rois,
                *radius,
                *roi_weight,
            )
        }
        ResampleSpec::Partition { percentages, pick } => {
            let buckets =
                sampling::chunker::percentage_chunks::partition_by_percentages(pts, percentages)
                    .unwrap_or_default();
            if buckets.is_empty() {
                return Vec::new();
            }
            let i = (*pick).min(buckets.len().saturating_sub(1));
            // Convert Vec<&Pt> into Vec<Pt> by cloning the referenced points
            buckets
                .into_iter()
                .nth(i)
                .unwrap_or_default()
                .into_iter()
                .cloned()
                .collect()
        }
    }
}
