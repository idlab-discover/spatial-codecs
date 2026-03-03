# spatial_codecs

`spatial_codecs` is a standalone Rust crate that implements a suite of point‐cloud encoders, decoders, and benchmarking tools used throughout Broadcast for XR. Every codec exposes the same API surface, letting higher layers pick an encoding at runtime without wiring bespoke glue.

This document explains the crate layout, the supported codecs, and dives into the custom **Quantize** codec, including its bitstream layout and optional tools (palettes, deltas, fixed-width packing).

---

## Crate layout

```
Libraries/spatial_codecs
├── Cargo.toml
├── README.md                # this file
├── configs                  # benchmark configs (datasets, sweeps, metrics)
└── src
    ├── bench                # benchmarking harness and metric implementations
    ├── main.rs              # CLI: decode → optional sampling/chunking → encode (or stats-only)
    ├── codecs
    │   ├── bitcode          # bit-packed integer codec
    │   ├── draco            # Google Draco wrapper
    │   ├── gzip / zstd /
    │   │   lz4 / snappy     # wrapper codecs (compress another stream)
    │   ├── openzl           # OpenZL bridge
    │   ├── ply              # PLY encoders/decoders (ASCII + binary)
    │   ├── quantize         # custom codec (see below)
    │   └── tmf              # MPEG TMF codec support
    ├── encoder.rs           # format-agnostic `EncodingParams` API
    └── decoder.rs           # symmetric decoding entry point
```

The **crate API** centres on:

- `EncodingParams` - an enum with one variant per codec; wrapper codecs (`Gzip`, `Zstd`, `Openzl`, …) themselves hold an `inner: Box<EncodingParams>`.
- `encode_from_payload_into(points: &[Point3D], params: EncodingParams, out: &mut Vec<u8>)` - stream encoder.
- `decode_into(data: &[u8], out: &mut Vec<Point3D>)` - stream decoder.
- `bench` module - dataset loading, metric computation (geometry error, colour PSNR, etc.) and reporting utilities used by `configs/bench.toml`.

All codecs share the `Point3D` type from `spatial_utils::`x/y/z: S`, `r/g/b: u8`.

---

## CLI pipeline (decode → transform → encode)

`src/main.rs` builds the package’s default binary (`spatial_codecs`). It lets you read an encoded cloud (file or directory), optionally downsample and/or split into percentage buckets, then re-encode with any supported codec—or run in stats-only mode.

### Quick starts
- Default params for a codec:  
  `cargo run -p spatial_codecs -- --input data/frame.bc1 --output out.bc1 --codec bitcode`
- Custom params via TOML snippet (no file needed):  
  `cargo run -p spatial_codecs -- --input data/frame.bc1 --output out.zst --params 'Zstd = { level = 3, inner = { Bitcode = {} } }'`
- Info only (no output written): omit `--output` to print decode time and point counts:  
  `cargo run -p spatial_codecs -- --input data/frame.bc1 --codec bitcode`
- Sample then chunk a directory recursively:  
  `cargo run -p spatial_codecs -- --input Datasets --recursive --sample-count 200000 --chunks 15,25,60 --output out_dir --codec zstd`

### Arguments (common)
- `--input <path>`: encoded file or directory (use `--recursive` to descend).
- `--output <file|dir>`: where to write results. If omitted → stats-only (no encode).
- `--codec <name>`: use built-in defaults (`ply|draco|tmf|bitcode|quantize|sogp|gzip|zstd|lz4|snappy|openzl`).
- `--params "<toml>"`: custom `EncodingParams` TOML snippet (mutually exclusive with `--codec`). Example `Zstd = { level = 3, inner = { Bitcode = {} } }`.
- `--point-type <variant>`: one of `PointsRgbF32` (default), `PointsRgbaF32`, `PointsRgbF64`, `PointsRgbaF64`, `SplatsRgbaF32`.
- `--sample-count N`: exact random sampling to N points before chunking.
- `--chunks "p1,p2,...\"`: split into percentage buckets (sum ≤ 100); outputs `_chunk{idx}` files.
- `--recursive`: when input is a directory, walk subdirs.
- `--overwrite`: allow replacing existing output files.

Behavior notes:
- If multiple inputs or chunks are produced, `--output` must be a directory; filenames include the input stem, optional `_chunk{idx}`, and a codec-chain suffix (e.g., `bitcode+zstd`).
- In stats-only mode the tool prints decode duration (ms) and point counts per file/chunk, then exits without encoding.
- Parameter parsing accepts either `params = { ... }` or a bare map; errors report if TOML is malformed.

---

## Available codecs

| Codec      | Module                   | Notes                                                                              |
|------------|--------------------------|------------------------------------------------------------------------------------|
| Quantize   | `codecs::quantize`       | Custom integer codec with optional palettes, delta coding, and bit packing.        |
| Bitcode    | `codecs::bitcode`        | Efficient bit-packed representation.                                               |
| Draco      | `codecs::draco`          | Wraps Google Draco (Kd-tree and sequential modes).                                 |
| TMF        | `codecs::tmf`            | A quick and efficient codec for point clouds.                                      |
| PLY        | `codecs::ply`            | ASCII and binary PLY encoder/decoder for interoperability.                         |
| OpenZL     | `codecs::openzl`         | Columnar / serial layout wrappers (often used downstream of Quantize).             |
| Gzip/Zstd/Lz4/Snappy | `codecs::{gzip,zstd,lz4,snappy}` | Compress a child codec via the relevant compression library.     |

Bench configurations (`configs/bench.toml`) describe *sweeps*: combinations of datasets, codecs, optional wrappers, and quantisation settings that the harness executes.

---

## Quantize codec overview

`codecs::quantize` is the in-house codec designed for low-latency packing of large point clouds. Version 2 of the format adds feature flags so that additional transforms remain backwards compatible, while the header still fits into a few dozen bytes.

### Bitstream layout

```
magic[3] = "QNT"
version  = 0x02
position_bits (u8)
color_bits   (u8)  // upper bits carry feature flags
[flags (u8)]       // present when COLOR_BITFLAG_HAS_FLAGS is set
point_count (u32 LE)
position mins (3 × f32 LE)
position maxs (3 × f32 LE)
color mins    (3 × u8)
color maxs    (3 × u8)
[palette_len (u16 LE) + palette_len × 3 bytes]  // if COLOR_BITFLAG_PALETTE
body bitstream
```

Feature bits in the “color bits” field:

- `0x80` - palette enabled.
- `0x40` - extra flag byte present.
- `0x3F` - stored colour (or palette index) bit depth.

Extra flag byte (`FLAG_*` constants):

- `0x01` (`delta_positions`) - encode XYZ as first-order deltas.
- `0x02` (`delta_colors`) - encode colour channels / palette index as deltas.
- `0x04` (`pack_positions` = false) - positions stored at fixed width (8/16/32 bits instead of tight packing).
- `0x08` (`pack_colors` = false) - same for colour channels / palette indices.

The body bitstream contains one record per point in the input order.

### Encoding steps

1. **Bounds collection** - Single pass to record min/max for XYZ and RGB. Empty streams degenerate to zero ranges.
2. **Palette construction (optional)** - When `max_palette_colors > 0`, colours are bucketed on a 3D grid (5 or 6 bits per channel chosen adaptively) via `palette::build_palette_indices_grid`. The densest buckets become palette entries; others fall back to nearest neighbour search. The encoder stores palette indices instead of raw colours.
3. **Header emission** - `header::write_header_v2` writes the shared metadata, palette flag, optional flag byte, and the palette table itself.
4. **Quantisation** - Positions are quantised to `position_bits` precision over the observed range. Colours are quantised to `color_bits` (unless a palette is active in which case the raw palette index is used). Fast paths exist for 8-bit RGB full-range data to avoid lookup tables.
5. **Delta coding (optional)** - With `delta_positions` / `delta_colors`, values become differences to the previous sample modulo the maximum representable value. The first sample remains absolute. This shrinks entropy and improves wraparound locality.
6. **Bit packing** - `pack_positions = true` writes exactly `position_bits`; otherwise values are rounded up to 8/16/32 bits. Colour packing obeys `pack_colors` and applies to either the per-channel quantised values or palette indices.
7. **Body write** - A `TurboBitWriter` streams the LSB-first bits into a pre-sized buffer to avoid repeated reallocations.

### Decoding steps

1. **Header parse** - `header::parse_header` validates the magic/version, extracts flags, bounds, palette table (if any), and returns a `ParsedHeader` struct.
2. **Reader setup** - `FastBitReader` prepares to consume the body bitstream using the effective (packed or rounded) bit widths.
3. **Sample reconstruction** - For each point:
   - Read quantised XYZ, apply delta if flagged, dequantise via `min + q * range / max`.
   - Read palette index or colour channels, apply delta if flagged.
   - If palette: lookup RGB from the table; otherwise dequantise or use raw bytes.
4. **Output** - Append reconstructed `Point3D` to the destination, or fill flattened XYZ / RGB vectors depending on API call.

### Quantize parameters

Defined in `codecs/quantize/types.rs`:

```rust
pub struct QuantizeParams {
    pub position_bits: u8,        // 1..=32 (default 12)
    pub color_bits: u8,           // 1..=32 (default 8)
    pub max_palette_colors: u16,  // 0 disables palette
    pub delta_positions: bool,    // default false
    pub delta_colors: bool,       // default false
    pub pack_positions: bool,     // default true (tight packing)
    pub pack_colors: bool,        // default true
}
```

#### Choosing values

- **Precision** - 11-13 bits for XYZ usually keep quantisation error at or below sub-millimetre for metre-scale captures; 8 bits for colours replicates sRGB values.
- **Palette** - Try `max_palette_colors = 256` or `512` for scenes with a limited colour gamut. Benchmark both palette and non-palette modes; the helper `palette::palette_wins_estimate` offers a rough guess.
- **Delta** - Enable on smooth motion/geometry where neighbours are correlated. Avoid on random point shuffles as wraparound can increase entropy.
- **Packing** - Keep `pack_* = true` unless a downstream wrapper expects byte/word-aligned attributes (for example when feeding into a fixed-struct columnar store).

---

## Benchmark harness

The `bench` directory provides:

- `configs/bench.toml` - dataset roots, resampling strategies, sweeps (codec parameter grids), metrics to evaluate (symmetric Chamfer distance, colour PSNR, etc.).
- `bench/runner.rs` - orchestrates dataset loading, codec invocation and metric calculation.
- `bench/metrics/*` - individual metric implementations with caching helpers for reference clouds.
- `bench/report.rs` - summary tables (CSV/JSON) for aggregated results.

To add a new benchmark sweep, append an entry to `configs/bench.toml`:

```toml
[[sweeps]]
name = "Quantize@p12c8+delta"
params = { Quantize = { position_bits = 12, color_bits = 8, delta_positions = true, delta_colors = true } }
```

Run the harness (typically via workspace scripts) to compare sizes, encode/decode throughput, and reconstruction error across sweeps.

---

## Extending the crate

- **New codec** - add a module under `src/codecs/<name>`, expose `encode_from_payload_into` and `decode_into` functions, and register the variant in `src/encoder.rs` / `src/decoder.rs` by extending `EncodingParams`.
- **Wrapper codec** - follow the existing Gzip/Zstd/Lz4/Snappy pattern: take an `inner: Box<EncodingParams>`, run the inner codec into a temporary buffer, then wrap it.
- **Quantize tweaks** - update the constants/header helpers in `codecs::quantize::types/header` and touch both encoder/decoder. Keep the flag byte stable to preserve compatibility.
- **Bench metrics** - implement `Metric` trait in `bench/metrics`, register in `bench/metrics/mod.rs`, and configure via `configs/bench.toml`.

Unit tests live inside each module (e.g. `codecs::quantize::tests`) and the bench harness; run via `cargo test -p spatial_codecs`.

---

## Useful entry points

- `src/encoder.rs` / `src/decoder.rs` - unify codec dispatch.
- `src/codecs/quantize/encoder.rs` / `decoder.rs` - full Quantize implementation.
- `src/codecs/quantize/header.rs` / `bitio.rs` / `palette.rs` - shared utilities.
- `configs/bench.toml` - edit sweeps to exercise new features.

Refer back to this README when tuning codec parameters, adding new transforms, or integrating the crate into additional pipelines.

---

## Benchmark metrics in detail

The `bench/metrics` module contains several metrics that quantify geometric fidelity, colour reconstruction, and statistical properties of decoded clouds. Each metric can be toggled in `configs/bench.toml` under the `[metrics]` table.

To run a benchmark, first build the crate with the build command from the root README, then execute the benchmark runner with the desired configuration file using:

```bash
./target/x86_64-unknown-linux-gnu/release/pc_bench --config ./Libraries/spatial_codecs/configs/bench.toml --out codecs_benchmark.csv
```

### Geometry

Implemented in `bench/metrics/geometry.rs`, the geometry metric computes a *symmetric k-nearest neighbour distance* (akin to a bounded Chamfer distance):

1. **Nearest neighbours** - For every point in the reconstructed cloud, find the closest `k_normals` points in the reference cloud (default `k = 16`). Repeat inversely (reference → reconstructed). Both passes use KD-tree acceleration.
2. **Distance aggregation** - Square Euclidean distances accumulate into histograms (`d2` when enabled) and compute average/min/max/statistics. Because the measure is symmetric, the reported error reflects both missed points and spurious additions.
3. **Normal similarity** (optional) - When `normal_angle = true`, stored normals are compared for each neighbour pair and the angular deviation is averaged in degrees. This helps detect orientation errors even when positional error is low.
4. **Outlier rejection** - Controlled via `outlier_threshold` (default `{ kind = "mad3" }`). Distances above three median absolute deviations (MAD) are considered outliers and reported separately in the summary so that catastrophic spikes do not drown the “typical” error metrics.

### Colour

Located in `bench/metrics/color.rs`, colour quality is reported via PSNR (Peak Signal-to-Noise Ratio) over 8-bit RGB channels:

1. **Pairing** - Uses the geometric pairing step (same nearest neighbours) to map reconstructed points to reference colour samples.
2. **Error computation** - Mean squared error (MSE) is computed per channel; PSNR = 10·log₁₀(255² / MSE). Infinite PSNR indicates an exact colour match.
3. **Palette-awareness** - Because the decoder returns 8-bit RGB, this metric is agnostic to the encoding method (palette, direct quantisation, etc.), ensuring fair comparison between codecs.

### Reference cache

`bench/metrics/ref_cache.rs` builds and caches KD-trees (`RefCache`) per dataset so repeated metric runs avoid rebuilding spatial indices. The cache layer also holds per-point normals when available.

### Reporting

`bench/report.rs` consumes metric outputs and produces:

- Aggregate stats (mean, median, standard deviation) for geometry distances and normals.
- Colour PSNR per channel and combined luminance-like value.
- Outlier counts and thresholds used during the analysis.

When you enable/disable metrics in `bench.toml`:

```toml
[metrics]
symmetric = true       # enable symmetric geometry distance
k_normals = 16         # neighbourhood size
outlier_threshold = { kind = "mad3" }
normal_angle = true
d2 = true              # emit squared distance stats
color_psnr = true      # compute PSNR
```

- Setting `symmetric = false` switches to a single-direction distance (reconstructed→reference).
- Increase `k_normals` for denser clouds; larger neighbourhoods smooth out local noise at the cost of runtime.
- Alternative `outlier_threshold` strategies (e.g., `{ kind = "percentile", value = 0.99 }`) are supported if you need quantile-based filtering.
- Disabling `color_psnr` skips colour processing entirely, useful for geometry-only experiments.

All metrics share the same dataset resampling policy (configured in `[datasets.resample]`) so comparisons remain consistent across sweeps.
