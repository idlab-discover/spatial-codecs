# Design

This document describes how `spatial_codecs` is structured, how decoding selects a codec automatically, and where conversions and performance-critical decisions live.

## Goals

- **Format-agnostic decoding**: consumers don’t need to know which codec was used.
- **High throughput**: avoid per-call allocations, avoid unnecessary copies.
- **Interop-friendly**: offer a C ABI that works well from C/C++ and C#.
- **Extensible**: adding a codec or adding a missing spatial type mapping should be local and obvious.

## High-level idea: 3-byte magic dispatch

Every encoded payload starts with a **3-byte magic tag** (the first three bytes of the byte stream). Decoding works like this:

1. read `data[0..3]`
2. map that tag to a codec implementation
3. call that codec’s decoder

This makes the payload **self-describing** and keeps your application logic free from “which codec is this?” configuration.

The dispatch logic lives in:

- `src/decoder.rs` (format-agnostic decoder front door)

Codec-specific logic lives under:

- `src/codecs/<codec>/decoder.rs`

## Encoding: explicit `EncodingParams`

Encoding is explicit: you choose a format/config via `EncodingParams` and pass it to the encoder.

- `EncodingParams` is **`serde`-serializable** and is intended to be used as a config surface.
- The library can also wrap inner formats (e.g. “encode with X, then compress with Y”) via the wrapper variants.

The entry point is:

- `src/encoder.rs` (`encode_into_generic`, `encode_into_any`, …)

### Why explicit encoding but implicit decoding?

- **Decoding** benefits from being automatic: a single entry point can accept payloads from many producers.
- **Encoding** usually needs deliberate choices (format, speed/quality trade-offs, wrapper selection). Those choices belong in a config.

## Spatial types and conversions

`spatial_codecs` targets spatial primitives from the `spatial_utils` crate (points, splats, …).

Not all codecs support all types. When you request an encode/decode into a type that a codec cannot produce directly, the library:

- performs a conversion when there is a cheap, well-defined mapping, or
- returns a clear error describing what is missing.

The “happy path” for throughput is:

- decode directly into the desired output type, and
- avoid intermediate representations.
- When buffers are needed, try to reuse them across calls. Be mindfull of threading and ownership rules.

### Adding missing type support

Most “missing type” work is local:

1. add the conversion or mapping in the codec bridge
2. wire it into the codec’s encode/decode implementation
3. add a small round-trip test in that codec module

## Buffering and allocation strategy

### Rust API

The Rust APIs prefer **caller-provided buffers**:

- encoding writes into `&mut Vec<u8>`
- decoding writes into `&mut Vec<T>` (and also into flattened `Vec`s)

This lets you reuse allocations across frames.

Recommended pattern in hot loops:

- keep `Vec<u8>` and output vectors around
- `clear()` them per iteration (do not reallocate)
- reserve once up front when you have typical sizes

### Flattened vs “compact” coordinates

Internally, some codecs naturally operate on flattened arrays (`[x0,y0,z0,x1,y1,z1,…]`), while others work on `[[f32; 3]]`.

The crate provides helper conversions to move between:

- `Vec<[T; N]>` (compact)
- `Vec<T>` (flattened)

These helpers exist to keep FFI fast and to avoid duplicating conversion code in every codec.

Of course, converting in your application code is also allowed. For this, you need to know that both the compacted and flattend forms use the same underlying memory layout, so you can safely `transmute` between them if you want to avoid the copy. The crate’s helpers do this under the hood.

## FFI overview

The FFI layer (`src/ffi.rs`) is built around a few principles:

- **opaque handles** for encoders/decoders and parameter objects
- output buffers owned by the handle to avoid repeated heap traffic
- “flat array” APIs to keep marshaling overhead low in C# and C

Details are in `docs/ffi.md`.

## Threading model

The core codecs are intended to be used in multi-threaded pipelines.

- Prefer **one encoder/decoder handle per thread** for runtime optimization to be effective.
- Treat handles as not thread-safe unless explicitly documented otherwise.
- `EncodingParams` values are cheap to clone and can be shared/configured per thread. But don't create one per frame for the same reason as above: avoid per-call allocations.

## Where to look next

- `src/decoder.rs` - magic dispatch
- `src/encoder.rs` - `EncodingParams` and format selection
- `src/codecs/*` - actual codec bridges
- `docs/ffi.md` - ABI, bindings, and ownership rules
