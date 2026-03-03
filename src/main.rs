use std::{
    fmt, fs, path::{Path, PathBuf}, time::Instant
};

use clap::{Parser, ValueEnum};
use spatial_utils::{
    point::{Point3RgbF32, Point3RgbF64, Point3RgbaF32, Point3RgbaF64},
    sampling::{
        chunker::{
            grid_chunks::{chunk_by_grid_counts, chunk_by_tile_size},
            percentage_chunks::build_index_buckets,
            roi_chunks::{chunk_by_rois, Roi},
        },
        exact_random::exact_random_sampling,
    },
    splat::GaussianSplatF32,
    traits::{SpatialSink, SpatialOwnedFull},
    utils::point_scalar::PointScalar,
};
use spatial_codecs::{
    decoder,
    encoder::{self, EncodingFormat, EncodingParams},
};

type DynResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PointType {
    PointsRgbF32,
    PointsRgbaF32,
    PointsRgbF64,
    PointsRgbaF64,
    SplatsRgbaF32,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CodecChoice {
    Ply,
    Draco,
    Gsplat16,
    Tmf,
    Bitcode,
    Quantize,
    Sogp,
    Gzip,
    Zstd,
    Lz4,
    Snappy,
    Openzl,
}

impl From<CodecChoice> for EncodingFormat {
    fn from(c: CodecChoice) -> Self {
        match c {
            CodecChoice::Ply => EncodingFormat::Ply,
            CodecChoice::Draco => EncodingFormat::Draco,
            CodecChoice::Gsplat16 => EncodingFormat::Gsplat16,
            CodecChoice::Tmf => EncodingFormat::Tmf,
            CodecChoice::Bitcode => EncodingFormat::Bitcode,
            CodecChoice::Quantize => EncodingFormat::Quantize,
            CodecChoice::Sogp => EncodingFormat::Sogp,
            CodecChoice::Gzip => EncodingFormat::Gzip,
            CodecChoice::Zstd => EncodingFormat::Zstd,
            CodecChoice::Lz4 => EncodingFormat::Lz4,
            CodecChoice::Snappy => EncodingFormat::Snappy,
            CodecChoice::Openzl => EncodingFormat::Openzl,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "pc_tool",
    about = "Decode → transform → encode point clouds",
    version
)]
struct Cli {
    /// Input file or directory containing encoded point clouds.
    #[arg(short, long)]
    input: PathBuf,

    /// Output file or directory. Directories are required for multi-file or chunked output.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Use default parameters for this codec (mutually exclusive with --params).
    #[arg(long, conflicts_with = "params")]
    codec: Option<CodecChoice>,

    /// TOML snippet describing EncodingParams (same shape as configs/bench.toml).
    /// Example: 'Zstd = { level = 3, inner = { Bitcode = {} } }'
    #[arg(long, value_name = "TOML", conflicts_with = "codec")]
    params: Option<String>,

    /// Point representation to decode into and re-encode from.
    #[arg(long, value_enum, default_value_t = PointType::PointsRgbF32)]
    point_type: PointType,

    /// Uniform exact random downsampling to this many points before chunking.
    #[arg(long)]
    sample_count: Option<usize>,

    /// Split into percentage buckets (e.g., "70,20,10") after sampling.
    #[arg(long, value_delimiter = ',', num_args = 1.., value_parser = clap::value_parser!(u8))]
    chunks: Option<Vec<u8>>,

    /// Split into a uniform 3D grid with the given divisions along X,Y,Z (e.g., 4,4,4).
    #[arg(long, value_delimiter = ',', num_args = 3, value_parser = clap::value_parser!(u32))]
    grid_divisions: Option<Vec<u32>>,

    /// Split into cubic tiles of this edge length (same units as input coordinates).
    #[arg(long)]
    tile_size: Option<f64>,

    /// Split by semicolon-delimited ROIs; each ROI is "xmin,ymin,zmin,xmax,ymax,zmax".
    /// Example: --rois "0,0,0,1,1,1;1,1,1,2,2,2"
    #[arg(long, value_delimiter = ';')]
    rois: Option<Vec<String>>,

    /// When using ROIs, also emit a remainder bucket for points outside all ROIs.
    #[arg(long, default_value_t = true)]
    roi_keep_unmatched: bool,

    /// Recurse into subdirectories when input is a directory.
    #[arg(long, default_value_t = false)]
    recursive: bool,

    /// Allow overwriting existing output files.
    #[arg(long, default_value_t = false)]
    overwrite: bool,
}

fn main() -> DynResult<()> {
    let cli = Cli::parse();

    let params = resolve_params(&cli)?;
    let inputs = collect_inputs(&cli.input, cli.recursive)?;
    if inputs.is_empty() {
        return Err("No input files found".into());
    }

    let chunk_count = if let Some(rois) = &cli.rois {
        rois.len() + cli.roi_keep_unmatched as usize
    } else if let Some(divs) = &cli.grid_divisions {
        if divs.len() == 3 {
            divs[0]
                .saturating_mul(divs[1])
                .saturating_mul(divs[2]) as usize
        } else {
            1
        }
    } else if cli.tile_size.is_some() {
        // Unknown until AABB is known; assume >1 to pick dir when needed.
        2
    } else if let Some(v) = &cli.chunks {
        v.len()
    } else {
        1
    };
    let input_is_dir = cli.input.is_dir();
    let must_use_dir = input_is_dir || inputs.len() > 1 || chunk_count > 1;
    let output_is_dir = match &cli.output {
        Some(out) => must_use_dir || out.is_dir(),
        None => false,
    };

    if let Some(out) = &cli.output {
        if must_use_dir && out.exists() && !out.is_dir() {
            return Err(format!(
                "Output '{}' must be a directory when processing multiple clouds or chunks",
                out.display()
            )
            .into());
        }
        if output_is_dir {
            fs::create_dir_all(out)?;
        }
    }

    let do_write = cli.output.is_some();

    let chunk_flags =
        cli.chunks.is_some() as u8
            + cli.grid_divisions.is_some() as u8
            + cli.tile_size.is_some() as u8
            + cli.rois.is_some() as u8;
    if chunk_flags > 1 {
        return Err("Choose only one of --chunks, --grid-divisions, --tile-size, or --rois".into());
    }

    match cli.point_type {
        PointType::PointsRgbF32 => {
            run_pipeline::<Point3RgbF32, f32>(&cli, &params, &inputs, output_is_dir, do_write)
        }
        PointType::PointsRgbaF32 => {
            run_pipeline::<Point3RgbaF32, f32>(&cli, &params, &inputs, output_is_dir, do_write)
        }
        PointType::PointsRgbF64 => {
            run_pipeline::<Point3RgbF64, f64>(&cli, &params, &inputs, output_is_dir, do_write)
        }
        PointType::PointsRgbaF64 => {
            run_pipeline::<Point3RgbaF64, f64>(&cli, &params, &inputs, output_is_dir, do_write)
        }
        PointType::SplatsRgbaF32 => {
            run_pipeline::<GaussianSplatF32, f32>(&cli, &params, &inputs, output_is_dir, do_write)
        }
    }
}

fn run_pipeline<P, S>(
    cli: &Cli,
    params: &EncodingParams,
    inputs: &[PathBuf],
    output_is_dir: bool,
    do_write: bool,
) -> DynResult<()>
where
    P: SpatialOwnedFull<S> + SpatialSink<Scalar = S> + 'static,
    S: PointScalar,
{
    let mut decoded: Vec<P> = Vec::new();
    let mut encoded: Vec<u8> = Vec::new();
    let codec_label = codec_chain_label(params);

    let mut encode_us_per_chunk: Vec<u128> = Vec::new();
    let mut write_us_per_chunk: Vec<u128> = Vec::new();

    for input in inputs {
        decoded.clear();
        let data = fs::read(input)?;

        // Decode
        let t0 = Instant::now();
        decoder::decode_into::<P>(&data, &mut decoded)?;
        let decode_us = t0.elapsed().as_micros();

        // Move decoded data out to avoid clones when no sampling.
        let mut working = std::mem::take(&mut decoded);
        let original_len = working.len();

        // Sampling timing
        let t_sample = Instant::now();
        if let Some(target) = cli.sample_count {
            if target < working.len() {
                working = exact_random_sampling(&working, target);
            }
        }
        let sample_us = t_sample.elapsed().as_micros();
        let after_sample_len = working.len();

        // Chunking timing
        let t_chunk = Instant::now();
        let mut chunked: Vec<Vec<P>>;
        let mut reuse_decode_buf: Vec<P> = Vec::new();

        if let Some(percentages) = &cli.chunks {
            let plan = build_index_buckets(working.len(), percentages)?;
            let buckets = plan.bucket_count();
            chunked = Vec::with_capacity(buckets);

            for b in 0..buckets {
                let idxs = plan.bucket_indices(b).ok_or("Invalid index plan")?;
                let mut chunk = Vec::with_capacity(idxs.len());
                for &i in idxs {
                    // Safe by construction: plan built for this length
                    chunk.push(working[i]);
                }
                chunked.push(chunk);
            }

            reuse_decode_buf = working; // keep ownership to reuse allocation
        } else if let Some(divs) = &cli.grid_divisions {
            let arr = divs
                .as_slice()
                .try_into()
                .map_err(|_| "Expected exactly 3 values for --grid-divisions")?;
            let buckets = chunk_by_grid_counts(&working, arr)?;
            chunked = buckets_to_owned(buckets);
            reuse_decode_buf = working;
        } else if let Some(tile) = cli.tile_size {
            let tile_s = S::from_f64(tile);
            if !tile_s.is_finite() || tile_s <= S::ZERO {
                return Err("tile_size must be positive and finite".into());
            }
            let buckets = chunk_by_tile_size(&working, tile_s)?;
            chunked = buckets_to_owned(buckets);
            reuse_decode_buf = working;
        } else if let Some(raw_rois) = &cli.rois {
            let rois = parse_rois::<S>(raw_rois)?;
            let buckets = chunk_by_rois(&working, &rois, cli.roi_keep_unmatched)?;
            chunked = buckets_to_owned(buckets);
            reuse_decode_buf = working;
        } else {
            chunked = vec![working];
        }
        let chunk_us = t_chunk.elapsed().as_micros();

        // Encode timings (per chunk)
        encode_us_per_chunk.clear();
        write_us_per_chunk.clear();
        if do_write {
            encode_us_per_chunk.reserve(chunked.len());
            write_us_per_chunk.reserve(chunked.len());

            for (idx, chunk) in chunked.iter().enumerate() {
                encoded.clear();

                let t_enc = Instant::now();
                encoder::encode_into(chunk, params, &mut encoded)?;
                encode_us_per_chunk.push(t_enc.elapsed().as_micros());

                let out_path = make_output_path(
                    cli.output.as_ref().unwrap(),
                    input,
                    idx,
                    &codec_label,
                    output_is_dir,
                    chunked.len(),
                );

                // Optional: write timing (often dominates, nice to see separately)
                let t_write = Instant::now();
                write_output(&out_path, &encoded, cli.overwrite)?;
                write_us_per_chunk.push(t_write.elapsed().as_micros());
            }
        }

        // Verbose timing output
        println!("{}", input.display());
        println!("  decoded:  {} pts in {} ms", original_len, Ms3(decode_us));
        println!("  sampled:  {} pts in {} ms", after_sample_len, Ms3(sample_us));
        println!(
            "  chunked:  {} chunks in {} ms",
            chunked.len(),
            Ms3(chunk_us)
        );

        if chunked.len() > 1 {
            for (idx, chunk) in chunked.iter().enumerate() {
                println!("    chunk {}: {} pts", idx, chunk.len());
            }
        }

        if do_write {
            let enc_list = format_us_csv_ms3(&encode_us_per_chunk);
            let enc_total: u128 = encode_us_per_chunk.iter().sum();
            println!("  encoded:  {} ms total (per chunk: {})", Ms3(enc_total), enc_list);

            let wr_list = format_us_csv_ms3(&write_us_per_chunk);
            let wr_total: u128 = write_us_per_chunk.iter().sum();
            println!("  wrote:    {} ms total (per chunk: {})", Ms3(wr_total), wr_list);
        } else {
            println!("  encoded:  skipped (no --output)");
        }

        // Reuse the largest available buffer for the next decode step.
        decoded = if !reuse_decode_buf.is_empty() {
            reuse_decode_buf
        } else {
            chunked.into_iter().next().unwrap_or_default()
        };
    }

    Ok(())
}

fn resolve_params(cli: &Cli) -> DynResult<EncodingParams> {
    match (&cli.codec, &cli.params) {
        (Some(c), None) => Ok(encoder::get_default_params((*c).into())),
        (None, Some(raw)) => load_params_from_str(raw),
        (Some(_), Some(_)) => Err("Specify either --codec or --params, not both".into()),
        (None, None) => {
            Err("You must provide --codec for defaults or --params for custom settings".into())
        }
    }
}

fn load_params_from_str(raw: &str) -> DynResult<EncodingParams> {
    #[derive(serde::Deserialize)]
    struct Wrapper {
        params: EncodingParams,
    }

    if let Ok(w) = toml::from_str::<Wrapper>(raw.trim()) {
        return Ok(w.params);
    }
    if let Ok(p) = toml::from_str::<EncodingParams>(raw.trim()) {
        return Ok(p);
    }

    // Fallback: allow files that only contain the inner map without "params =".
    let wrapped = format!("params = {{\n{raw}\n}}\n");
    if let Ok(w) = toml::from_str::<Wrapper>(&wrapped) {
        return Ok(w.params);
    }

    Err("Failed to parse params. Expected either 'params = {...}' or a top-level EncodingParams map.".to_string()
    .into())
}

fn collect_inputs(root: &Path, recursive: bool) -> DynResult<Vec<PathBuf>> {
    if root.is_file() {
        return Ok(vec![root.to_path_buf()]);
    }
    if !root.is_dir() {
        return Err(format!("Input path '{}' does not exist", root.display()).into());
    }

    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                files.push(path);
            } else if recursive && path.is_dir() {
                stack.push(path);
            }
        }
    }

    files.sort();
    Ok(files)
}

fn codec_chain_label(params: &EncodingParams) -> String {
    fn push_chain<'a>(p: &'a EncodingParams, out: &mut Vec<&'a str>) {
        match p {
            EncodingParams::Ply(_) => out.push("ply"),
            EncodingParams::Draco(_) => out.push("draco"),
            EncodingParams::Gsplat16(_) => out.push("gsplat16"),
            EncodingParams::Tmf(_) => out.push("tmf"),
            EncodingParams::Bitcode(_) => out.push("bitcode"),
            EncodingParams::Quantize(_) => out.push("quantize"),
            EncodingParams::Sogp(_) => out.push("sogp"),
            EncodingParams::LASzip => out.push("laszip"),
            EncodingParams::Gzip { inner, .. } => {
                out.push("gzip");
                push_chain(inner.as_ref(), out);
            }
            EncodingParams::Zstd { inner, .. } => {
                out.push("zstd");
                push_chain(inner.as_ref(), out);
            }
            EncodingParams::Lz4 { inner } => {
                out.push("lz4");
                push_chain(inner.as_ref(), out);
            }
            EncodingParams::Snappy { inner } => {
                out.push("snappy");
                push_chain(inner.as_ref(), out);
            }
            #[cfg(feature = "openzl")]
            EncodingParams::Openzl(p) => {
                out.push("openzl");
                if let spatial_codecs::codecs::openzl::encoder::OpenzlParams::Serial {
                    inner,
                    ..
                } = p
                {
                    push_chain(inner.as_ref(), out);
                }
            }
            #[cfg(not(feature = "openzl"))]
            EncodingParams::Openzl(_) => {
                out.push("openzl");
            }
        }
    }

    let mut parts = Vec::new();
    push_chain(params, &mut parts);
    if parts.is_empty() {
        "pc".to_string()
    } else {
        parts.join("+")
    }
}

fn make_output_path(
    output: &Path,
    input: &Path,
    chunk_idx: usize,
    codec_label: &str,
    output_is_dir: bool,
    chunk_count: usize,
) -> PathBuf {
    if !output_is_dir {
        return output.to_path_buf();
    }

    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("cloud");

    let mut filename = String::from(stem);
    if chunk_count > 1 {
        filename.push_str(&format!("_chunk{chunk_idx}"));
    }

    filename.push('.');
    filename.push_str(codec_label);

    output.join(filename)
}

fn write_output(path: &Path, data: &[u8], overwrite: bool) -> DynResult<()> {
    if path.exists() && !overwrite {
        return Err(format!(
            "Refusing to overwrite existing file '{}'; pass --overwrite to replace",
            path.display()
        )
        .into());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, data)?;
    Ok(())
}

fn buckets_to_owned<P: Clone>(buckets: Vec<Vec<&P>>) -> Vec<Vec<P>> {
    let mut out = Vec::with_capacity(buckets.len());
    for bucket in buckets {
        let mut owned = Vec::with_capacity(bucket.len());
        owned.extend(bucket.into_iter().cloned());
        out.push(owned);
    }
    out
}

fn parse_rois<S: PointScalar>(raw: &[String]) -> DynResult<Vec<Roi<S>>> {
    let mut out = Vec::with_capacity(raw.len());
    for r in raw {
        let parts: Vec<_> = r.split(',').collect();
        if parts.len() != 6 {
            return Err(format!("ROI must have 6 comma-separated numbers: '{r}'").into());
        }
        let mut vals = [0f64; 6];
        for (i, p) in parts.iter().enumerate() {
            vals[i] = p.trim().parse::<f64>().map_err(|_| {
                format!("Failed to parse ROI value '{p}' in '{r}' as number")
            })?;
            if !vals[i].is_finite() {
                return Err(format!("ROI value '{p}' in '{r}' is not finite").into());
            }
        }
        if vals[0] > vals[3] || vals[1] > vals[4] || vals[2] > vals[5] {
            return Err(format!("ROI mins must be <= maxs: '{r}'").into());
        }
        out.push(Roi::new(
            [
                S::from_f64(vals[0]),
                S::from_f64(vals[1]),
                S::from_f64(vals[2]),
            ],
            [
                S::from_f64(vals[3]),
                S::from_f64(vals[4]),
                S::from_f64(vals[5]),
            ],
        ));
    }
    Ok(out)
}

/// Formats microseconds as milliseconds with 3 decimals (x.xxx ms), deterministically.
struct Ms3(u128);

impl fmt::Display for Ms3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let us = self.0;
        let ms = us / 1_000;
        let frac = us % 1_000; // thousandths of a ms
        write!(f, "{ms}.{frac:03}")
    }
}

fn format_us_csv_ms3(values: &[u128]) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    for (i, v) in values.iter().enumerate() {
        if i != 0 {
            s.push_str(" ms , ");
        }
        let _ = write!(&mut s, "{}", Ms3(*v));
    }
    s.push_str(" ms");
    s
}
