# spatial_codecs

High-throughput, **format-agnostic** encoding/decoding for spatial data (point clouds, Gaussian splats, …), built around a simple idea:

> Every encoded payload starts with a **3-byte magic tag**. The decoder reads those first bytes and automatically selects the right codec.

This makes the pipeline much easier to evolve: producers can switch codecs without forcing all consumers to be reconfigured.

`spatial_codecs` is designed for **throughput**:

- encoding/decoding APIs that write **into caller-provided buffers**
- minimal allocations and copy-avoidance where possible
- an FFI layer that favors **flat arrays** for interop and speed

## What data types does it support?

We support spatial primitives from the `spatial_utils` crate (e.g. points and splats). Not all codecs support all spatial types.

When a direct path is missing, the library will:

1. perform a fast conversion when possible, or
2. return a **clear, typed error** when it cannot.

Adding the missing case is intentionally straightforward—see `docs/design.md`.

## Crate structure

- `src/decoder.rs` - **format-agnostic decoding** (dispatch by 3-byte magic)
- `src/encoder.rs` - **format-agnostic encoding** (explicit `EncodingParams`)
- `src/codecs/*` - per-codec adapters
- `src/ffi.rs` - C ABI (and binding generation support)

More details:

- `docs/design.md` - architecture, dispatch, conversions, performance notes
- `docs/ffi.md` - how to build and use the C / C# bindings

## Getting started (Rust)

Add the crate:

```toml
[dependencies]
spatial_codecs = "0.1.0"
spatial_utils = "0.1.0"
```

Tip: check crate versions on [crates.io/crates/spatial_codecs](https://crates.io/crates/spatial_codecs) and [crates.io/crates/spatial_utils](https://crates.io/crates/spatial_utils) for the latest release.

Encode into a reusable `Vec<u8>`:

```rust
use spatial_codecs::encoder::{encode_into_generic, EncodingParams};
use spatial_utils::point::Point3RgbF32;

let points: Vec<Point3RgbF32> = vec![
    Point3RgbF32::new(0.0, 0.0, 0.0, 255, 0, 0),
    Point3RgbF32::new(1.0, 2.0, 3.0, 0, 255, 0),
];

let params = EncodingParams::default();
let mut out = Vec::with_capacity(1024);

encode_into_generic::<Point3RgbF32, f32>(&points, &params, &mut out)?;
```

Decode without knowing the codec up front:

```rust
use spatial_codecs::decoder::decode_into;
use spatial_utils::point::Point3RgbF32;

let mut decoded: Vec<Point3RgbF32> = Vec::new();
decode_into(&out, &mut decoded)?;
```

## Optional codecs

Some codecs are behind Cargo features to keep the default build lean.

Example: enable the Draco bridge via:

```toml
spatial_codecs = { version = "0.1.0", default-features = false, features = ["draco"] }
```

## Building for FFI

Build optimized artifacts:

```bash
cargo build --release
```

You’ll typically load the **shared library** from C# (P/Invoke) or from C/C++:

- Linux: `libspatial_codecs.so`
- macOS: `libspatial_codecs.dylib`
- Windows: `spatial_codecs.dll`

Static libraries may also be produced (`.a` / `.lib`) depending on your toolchain.

Generate C/C++ and C# bindings:

```bash
cargo run --release --bin pc_generate_bindings
```

This writes the generated bindings under `bindings/`.

## Examples

See the `examples/` directory for end-to-end samples. They are intentionally written to be **allocation-light** and suitable for high-throughput pipelines.

---

If you’re extending codec/type coverage, start with `docs/design.md`.
