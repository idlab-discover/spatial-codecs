use clap::{Parser, ValueEnum};
use spatial_codecs::bench::config::BenchConfig;
use spatial_codecs::bench::report::{print_outcomes_summary, print_run_header};
use spatial_codecs::bench::runner::{run_bench, RunOptions};
use std::{fs::File, io::Write, path::PathBuf};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Mode {
    Accuracy = 0,
    Throughput = 1,
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mode::Accuracy => write!(f, "accuracy"),
            Mode::Throughput => write!(f, "throughput"),
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "pc_bench", version)]
struct Cli {
    /// Path to bench config (TOML)
    #[arg(long)]
    config: PathBuf,
    /// Output path (CSV or JSONL by extension)
    #[arg(long)]
    out: PathBuf,
    /// Mode: Accuracy (sequential, stable timing) or Throughput (parallel)
    #[arg(long, default_value_t=Mode::Accuracy)]
    mode: Mode,
    /// Max parallel jobs (only used in Throughput mode). 0 = num_cpus
    #[arg(long, default_value_t = 0)]
    jobs: usize,
    /// Print per-run lines to console
    #[arg(long, default_value_t = true)]
    verbose: bool,
    /// Disable progress bars (useful for CI logs)
    #[arg(long, default_value_t = false)]
    no_progress: bool,
    /// Warm-up encode/decode once before timing
    #[arg(long, default_value_t = true)]
    warmup: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let cfg_str = std::fs::read_to_string(&cli.config)?;
    let cfg: BenchConfig = toml::from_str(&cfg_str)?;

    let opts = RunOptions {
        mode_throughput: matches!(cli.mode, Mode::Throughput),
        jobs: cli.jobs,
        show_progress: !cli.no_progress,
        verbose: cli.verbose,
        warmup: cli.warmup,
    };

    print_run_header(&cfg, &opts);

    let outcomes = run_bench(&cfg, &opts)?;

    // Write output
    match cli.out.extension().and_then(|s| s.to_str()) {
        Some("jsonl") => spatial_codecs::bench::serialize::write_jsonl(&cli.out, &outcomes)?,
        _ => spatial_codecs::bench::serialize::write_csv(&cli.out, &outcomes)?,
    }

    // Small hint file
    let mut note = File::create(cli.out.with_extension("txt"))?;
    writeln!(note, "Wrote results to {}", cli.out.display())?;

    // Console summary (nice readable table)
    print_outcomes_summary(&outcomes);

    Ok(())
}
