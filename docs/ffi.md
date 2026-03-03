# FFI

`spatial_codecs` ships a C ABI intended for **high-throughput** interop from:

- C / C++
- C# (P/Invoke)

Bindings are generated into:

- `bindings/c/spatial_codecs_interops.h`
- `bindings/csharp/SpatialCodecsInterop.cs`

## Build artifacts

Build optimized binaries:

```bash
cargo build --release
```

Look in `target/release/` for:

- Linux: `libspatial_codecs.so`
- macOS: `libspatial_codecs.dylib`
- Windows: `spatial_codecs.dll`

Depending on your platform/toolchain, you may also see a static library:

- Linux/macOS: `libspatial_codecs.a`
- Windows: `spatial_codecs.lib`

### Which one should I use?

- **C#**: load the shared library (`.so` / `.dylib` / `.dll`).
- **C/C++**:
  - shared library for dynamic linking, or
  - static library if you want a fully static binary.

## Generating bindings

The repo includes a small helper binary:

```bash
cargo run --release --bin pc_generate_bindings
```

This regenerates the C header and the C# interop file under `bindings/`.

## Handles, ownership, and lifetime rules

The ABI uses **opaque handles** (pointers to internal Rust structs) for:

- encoders/decoders
- encoding parameter objects

This allows the library to cache buffers and reduce allocations.

### Output buffers

Many “encode”/“decode” functions write into buffers owned by the handle and return a pointer + length view.

- The returned pointer is valid **until the next call that reuses the same handle**, or until the handle is freed.
- If you need to keep the bytes longer, **copy** them into your own storage.

### Threading

For best throughput:

- create **one handle per thread**
- reuse it for many calls

Do not call into a single handle from multiple threads concurrently.

## Encoding parameters

Encoding is configured through an `EncodingParams` handle.

Common ways to obtain one:

- default params: `spatial_codecs_encoding_params_default()`
- from TOML: `spatial_codecs_encoding_params_from_toml(...)`

The TOML route is useful when you want to tune parameters without recompiling.

## Data layouts: flattened arrays

FFI APIs favor **flattened arrays**:

- positions: `[x0, y0, z0, x1, y1, z1, ...]`
- colors: `[r0, g0, b0, r1, g1, b1, ...]`

This layout maps cleanly to C arrays and avoids per-point marshaling overhead in C#.

The Rust library also includes helpers to convert between flattened vectors and “compact” `[T; 3]` representations. Use those helpers in Rust code; in C/C#/C++ you typically keep flattened arrays.

## What can I encode/decode?

The ABI provides dedicated entry points for the spatial primitives supported by the library (e.g. points and Gaussian splats).

Not every codec supports every spatial type. When a requested combination isn’t supported, the call returns a failure code and a readable error message.

## Examples

See `examples/` for:

- C and C++ examples using the generated header
- C# example using `SpatialCodecsInterop.cs`

These examples are written to be **allocation-light** and to demonstrate some best practices.
