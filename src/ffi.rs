//! FFI for high-throughput Gaussian Splat decoding into caller-provided buffers.
//!
//! Goals:
//! - Reuse `crate::decoder::decode_into::<GaussianSplatF32>` (no codec reimplementation).
//! - Separate timings:
//!     - decode: bytes -> Vec<GaussianSplatF32>
//!     - convert: Vec<GaussianSplatF32> -> caller-provided AoS buffers (Unity-like arrays)
//! - Allow array reuse across runs (caller owns buffers).
//!
//! Threading model:
//! - A decoder handle is NOT safe for concurrent use.
//! - Use one decoder per thread for maximum throughput.

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use interoptopus::{
    ffi_function, ffi_type,
    patterns::slice::FFISlice,
    function, Inventory, InventoryBuilder,
};
use crate::encoder::EncodingParams;
use spatial_utils::splat::{GaussianSplatF32, SH_C0};
use std::{
    ffi::{c_void, CStr, CString},
    panic::{catch_unwind, AssertUnwindSafe},
};

/// Opaque, heap-allocated encoding parameters.
///
/// This keeps the C ABI stable while still allowing callers to configure codecs.
/// Create an instance once (e.g. per thread) and reuse it to avoid per-call parsing.
///
/// - Construct with [`pc_encoding_params_default_points`], [`pc_encoding_params_default_gsplat`]
///   or [`pc_encoding_params_from_toml`].
/// - Use with `pc_point_encode_with_params` / `pc_gsplat_encode_with_params`.
/// - Free with [`pc_encoding_params_free`].
///
/// The handle is immutable and can be shared across threads.
#[ffi_type(opaque)]
pub struct PcEncodingParamsOpaque {
    _private: [u8; 0],
}

/// Internal representation behind [`PcEncodingParamsOpaque`].
struct PcEncodingParamsInner {
    params: EncodingParams,
}

/// Status codes for FFI calls.
#[ffi_type]
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PcStatus {
    Ok = 0,
    NullPointer = 1,
    InvalidSlice = 2,
    DecodeFailed = 3,
    SizeOverflow = 4,
    Panic = 5,
}

/// 3-float vector (Unity `Vector3` compatible layout).
#[ffi_type]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct PcVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// 4-float vector (Unity `Vector4` compatible layout).
#[ffi_type]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct PcVec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

/// Metadata (count always valid after decode; bounds valid after write_to_buffers).
#[ffi_type]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct PcGsplatMeta {
    pub splat_count: u32,
    pub sh_bands: u8,
    pub _reserved0: u8,
    pub _reserved1: u16,
    pub bounds_center: PcVec3,
    pub bounds_extents: PcVec3,
}

/// Mutable slice for `PcVec3` output.
#[ffi_type]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct PcMutSliceVec3 {
    pub data: *mut PcVec3,
    pub len: u64,
}

/// Mutable slice for `PcVec4` output.
#[ffi_type]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct PcMutSliceVec4 {
    pub data: *mut PcVec4,
    pub len: u64,
}

/// Caller-provided output buffers (AoS arrays).
#[ffi_type]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct PcGsplatOutBuffers {
    pub positions: PcMutSliceVec3,
    pub colors: PcMutSliceVec4,    // (dc_r, dc_g, dc_b, opacity01)
    pub scales: PcMutSliceVec3,
    pub rotations: PcMutSliceVec4, // stored as (w,x,y,z) in (x,y,z,w)
}

/// Immutable slice for `f32` input.
#[ffi_type]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct PcSliceF32 {
    pub data: *const f32,
    pub len: u64,
}

/// Immutable slice for `u8` input.
#[ffi_type]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct PcSliceU8 {
    pub data: *const u8,
    pub len: u64,
}

/// Mutable slice for `f32` output.
#[ffi_type]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct PcMutSliceF32 {
    pub data: *mut f32,
    pub len: u64,
}

/// Mutable slice for `u8` output.
#[ffi_type]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct PcMutSliceU8 {
    pub data: *mut u8,
    pub len: u64,
}

/// Flattened point-cloud IO buffers.
///
/// Positions are `XYZXYZ...` and colors are `RGBRGB...`.
#[ffi_type]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct PcPointInBuffers {
    pub positions: PcSliceF32,
    pub colors: PcSliceU8,
}

/// Flattened point-cloud output buffers.
#[ffi_type]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct PcPointOutBuffers {
    pub positions: PcMutSliceF32,
    pub colors: PcMutSliceU8,
}


/// Metadata for point-cloud decoding.
///
/// `point_count` is valid after a successful decode.
#[ffi_type]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct PcPointMeta {
    pub point_count: u32,
    pub _reserved0: u32,
}

/// Returned view into an encoder-owned byte buffer.
#[ffi_type]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct PcByteSlice {
    pub data: *const u8,
    pub len: u64,
}

#[allow(dead_code)]
struct GsplatDecoderOwned {
    decoded: Vec<GaussianSplatF32>,
    dc_lut: [f32; 256],
    inv_255: f32,
}

struct GsplatEncoderOwned {
    splats: Vec<GaussianSplatF32>,
    bytes: Vec<u8>,
}

struct PointEncoderOwned {
    points: Vec<spatial_utils::point::Point3RgbF32>,
    bytes: Vec<u8>,
}

struct PointDecoderOwned {
    positions: Vec<f32>,
    colors: Vec<u8>,
}

thread_local! {
    static TLS_LAST_ERROR: std::cell::RefCell<Option<CString>> = const { std::cell::RefCell::new(None) };
}

#[inline(always)]
fn set_last_error(msg: impl AsRef<str>) {
    let sanitized = msg.as_ref().replace('\0', "�");
    let c = CString::new(sanitized).unwrap_or_else(|_| CString::new("ffi: error").unwrap());
    TLS_LAST_ERROR.with(|cell| *cell.borrow_mut() = Some(c));
}

#[inline(always)]
fn clear_last_error() {
    TLS_LAST_ERROR.with(|cell| *cell.borrow_mut() = None);
}

/// Return the last error message for the calling thread (null-terminated).
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_last_error_message() -> *const std::os::raw::c_char {
    TLS_LAST_ERROR.with(|cell| {
        if let Some(s) = cell.borrow().as_ref() {
            s.as_ptr()
        } else {
            static EMPTY: &[u8] = b"\0";
            unsafe { CStr::from_bytes_with_nul_unchecked(EMPTY) }.as_ptr()
        }
    })
}

/// Clear last error for the calling thread.
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_clear_last_error() {
    clear_last_error();
}

#[inline(always)]
fn make_dc_lut() -> [f32; 256] {
    let inv_255 = 1.0f64 / 255.0f64;
    let mut lut = [0.0f32; 256];
    for (i, v) in lut.iter_mut().enumerate() {
        // dc = ((c/255) - 0.5) / SH_C0
        let c = i as f64;
        *v = (((c * inv_255) - 0.5) / SH_C0) as f32;
    }
    lut
}

#[inline(always)]
unsafe fn decoder_mut<'a>(h: *mut c_void) -> Result<&'a mut GsplatDecoderOwned, PcStatus> {
    if h.is_null() {
        return Err(PcStatus::NullPointer);
    }
    Ok(&mut *(h as *mut GsplatDecoderOwned))
}

#[inline(always)]
fn u64_to_usize(v: u64) -> Result<usize, PcStatus> {
    usize::try_from(v).map_err(|_| PcStatus::SizeOverflow)
}

#[inline(always)]
fn normalize_quat_wxyz(mut q: [f32; 4]) -> [f32; 4] {
    let n2 = (q[0] as f64) * (q[0] as f64)
        + (q[1] as f64) * (q[1] as f64)
        + (q[2] as f64) * (q[2] as f64)
        + (q[3] as f64) * (q[3] as f64);

    if n2.is_finite() && n2 > 0.0 {
        let inv = (1.0 / n2.sqrt()) as f32;
        q[0] *= inv;
        q[1] *= inv;
        q[2] *= inv;
        q[3] *= inv;
        q
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

/// Create a decoder handle (owning an internal `Vec<GaussianSplatF32>` reused across runs).
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_gsplat_decoder_create(out_handle: *mut *mut c_void) -> PcStatus {
    clear_last_error();
    if out_handle.is_null() {
        set_last_error("ffi: out_handle is null");
        return PcStatus::NullPointer;
    }

    let res = catch_unwind(AssertUnwindSafe(|| {
        let d = GsplatDecoderOwned {
            decoded: Vec::new(),
            dc_lut: make_dc_lut(),
            inv_255: 1.0f32 / 255.0f32,
        };
        let p = Box::into_raw(Box::new(d)) as *mut c_void;
        unsafe { *out_handle = p; }
        PcStatus::Ok
    }));

    match res {
        Ok(s) => s,
        Err(_) => {
            set_last_error("ffi: panic in pc_gsplat_decoder_create");
            PcStatus::Panic
        }
    }
}

/// Free a decoder handle (safe to call with null).
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_gsplat_decoder_free(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(handle as *mut GsplatDecoderOwned));
    }
}

/// Decode bytes into the decoder’s internal `Vec<GaussianSplatF32>`.
/// Returns meta with count; bounds are produced in `pc_gsplat_decoder_write_to_buffers`.
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_gsplat_decoder_decode_from_bytes(
    handle: *mut c_void,
    data: FFISlice<u8>,
    out_meta: *mut PcGsplatMeta,
) -> PcStatus {
    clear_last_error();
    if out_meta.is_null() {
        set_last_error("ffi: out_meta is null");
        return PcStatus::NullPointer;
    }

    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        let d = decoder_mut(handle)?;
        let bytes = data.as_slice();

        d.decoded.clear();
        if let Err(e) = crate::decoder::decode_into::<GaussianSplatF32>(bytes, &mut d.decoded) {
            set_last_error(format!("decode failed: {e}"));
            return Err(PcStatus::DecodeFailed);
        }

        let n: u32 = d.decoded.len().try_into().map_err(|_| PcStatus::SizeOverflow)?;

        *out_meta = PcGsplatMeta {
            splat_count: n,
            sh_bands: 0,
            _reserved0: 0,
            _reserved1: 0,
            bounds_center: PcVec3::default(),
            bounds_extents: PcVec3::default(),
        };

        Ok::<(), PcStatus>(())
    }));

    match res {
        Ok(Ok(())) => PcStatus::Ok,
        Ok(Err(s)) => s,
        Err(_) => {
            set_last_error("ffi: panic in pc_gsplat_decoder_decode_from_bytes");
            PcStatus::Panic
        }
    }
}

/// Convert decoded splats into caller-provided output buffers.
/// Writes the first N elements (N = last decoded `splat_count`).
/// Also computes bounds and returns them in `out_meta`.
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_gsplat_decoder_write_to_buffers(
    handle: *mut c_void,
    out_buffers: PcGsplatOutBuffers,
    out_meta: *mut PcGsplatMeta,
) -> PcStatus {
    clear_last_error();
    if out_meta.is_null() {
        set_last_error("ffi: out_meta is null");
        return PcStatus::NullPointer;
    }

    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        let d = decoder_mut(handle)?;
        let n_usize = d.decoded.len();
        let n_u32: u32 = n_usize.try_into().map_err(|_| PcStatus::SizeOverflow)?;

        // Validate buffers (len >= n, pointer non-null if n>0).
        let pos_len = u64_to_usize(out_buffers.positions.len)?;
        let col_len = u64_to_usize(out_buffers.colors.len)?;
        let scl_len = u64_to_usize(out_buffers.scales.len)?;
        let rot_len = u64_to_usize(out_buffers.rotations.len)?;

        if n_usize > 0
            && (out_buffers.positions.data.is_null()
                || out_buffers.colors.data.is_null()
                || out_buffers.scales.data.is_null()
                || out_buffers.rotations.data.is_null())
            {
                set_last_error("ffi: one or more output buffer pointers are null");
                return Err(PcStatus::NullPointer);
            }

        if pos_len < n_usize || col_len < n_usize || scl_len < n_usize || rot_len < n_usize {
            set_last_error("ffi: one or more output buffers smaller than splat_count");
            return Err(PcStatus::InvalidSlice);
        }

        let pos_ptr = out_buffers.positions.data;
        let col_ptr = out_buffers.colors.data;
        let scl_ptr = out_buffers.scales.data;
        let rot_ptr = out_buffers.rotations.data;

        // Bounds
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];

        //let lut = &d.dc_lut;
        let inv_255 = d.inv_255;

        // Hot loop: write AoS directly.
        for i in 0..n_usize {
            let s = d.decoded.get_unchecked(i);

            let p = s.mean;
            min[0] = min[0].min(p[0]); max[0] = max[0].max(p[0]);
            min[1] = min[1].min(p[1]); max[1] = max[1].max(p[1]);
            min[2] = min[2].min(p[2]); max[2] = max[2].max(p[2]);

            *pos_ptr.add(i) = PcVec3 { x: p[0], y: p[1], z: p[2] };

            let c = s.rgba;
            *col_ptr.add(i) = PcVec4 {
                x: (c.r as f32) * inv_255,
                y: (c.g as f32) * inv_255,
                z: (c.b as f32) * inv_255,
                w: (c.a as f32) * inv_255,
            };

            let sc = s.scale;
            *scl_ptr.add(i) = PcVec3 { x: sc[0], y: sc[1], z: sc[2] };

            // Stored as (w,x,y,z) but written into PcVec4 (x,y,z,w) memory slots.
            let rq = normalize_quat_wxyz(s.rotation);
            *rot_ptr.add(i) = PcVec4 { x: rq[0], y: rq[1], z: rq[2], w: rq[3] };
        }

        let center = PcVec3 {
            x: 0.5 * (min[0] + max[0]),
            y: 0.5 * (min[1] + max[1]),
            z: 0.5 * (min[2] + max[2]),
        };
        let extents = PcVec3 {
            x: 0.5 * (max[0] - min[0]),
            y: 0.5 * (max[1] - min[1]),
            z: 0.5 * (max[2] - min[2]),
        };

        *out_meta = PcGsplatMeta {
            splat_count: n_u32,
            sh_bands: 0,
            _reserved0: 0,
            _reserved1: 0,
            bounds_center: center,
            bounds_extents: extents,
        };

        Ok::<(), PcStatus>(())
    }));

    match res {
        Ok(Ok(())) => PcStatus::Ok,
        Ok(Err(s)) => s,
        Err(_) => {
            set_last_error("ffi: panic in pc_gsplat_decoder_write_to_buffers");
            PcStatus::Panic
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Point encode/decode (flattened buffers)
// -------------------------------------------------------------------------------------------------

/// Create a point encoder handle.
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_point_encoder_create() -> *mut c_void {
    let owned = PointEncoderOwned {
        points: Vec::new(),
        bytes: Vec::new(),
    };
    Box::into_raw(Box::new(owned)) as *mut c_void
}

/// Free a point encoder handle created by [`pc_point_encoder_create`].
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_point_encoder_free(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(handle as *mut PointEncoderOwned));
    }
}

/// Encode a flattened point cloud (`XYZXYZ...` + `RGBRGB...`) into `spatial_codecs` bytes.
///
/// Returns a view into an internal buffer owned by `handle`. The returned pointer stays valid
/// until the next call to this encoder handle or until the handle is freed.
///
/// This uses default encoding parameters. For custom parameters, use
/// [`pc_point_encode_with_params`].
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_point_encode(
    handle: *mut c_void,
    input: PcPointInBuffers,
    out_bytes: *mut PcByteSlice,
) -> bool {
    pc_point_encode_with_params(handle, input, std::ptr::null(), out_bytes)
}

/// Encode a flattened point cloud (`XYZXYZ...` + `RGBRGB...`) into `spatial_codecs` bytes,
/// using explicit encoding parameters.
///
/// If `prms` is null, defaults are used.
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_point_encode_with_params(
    handle: *mut c_void,
    input: PcPointInBuffers,
    prms: *const PcEncodingParamsOpaque,
    out_bytes: *mut PcByteSlice,
) -> bool {
    if out_bytes.is_null() {
        set_last_error("ffi: out_bytes is null");
        return false;
    }
    unsafe { *out_bytes = PcByteSlice::default() };

    if handle.is_null() {
        set_last_error("ffi: handle is null");
        return false;
    }

    if input.positions.data.is_null() {
        set_last_error("ffi: input.positions.data is null");
        return false;
    }
    if input.colors.data.is_null() {
        set_last_error("ffi: input.colors.data is null");
        return false;
    }

    let pos_ptr = input.positions.data;
    let col_ptr = input.colors.data;

    let pos_len = input.positions.len as usize;
    let col_len = input.colors.len as usize;
    if !pos_len.is_multiple_of(3) {
        set_last_error("ffi: positions.len must be a multiple of 3");
        return false;
    }
    if !col_len.is_multiple_of(3) {
        set_last_error("ffi: colors.len must be a multiple of 3");
        return false;
    }
    if pos_len != col_len {
        set_last_error("ffi: positions.len must equal colors.len");
        return false;
    }

    let num_points = pos_len / 3;
    let positions = unsafe { std::slice::from_raw_parts(pos_ptr, pos_len) };
    let colors = unsafe { std::slice::from_raw_parts(col_ptr, col_len) };

    let owned = unsafe { &mut *(handle as *mut PointEncoderOwned) };
    owned.points.clear();
    owned.points.reserve(num_points);

    for i in 0..num_points {
        let b = i * 3;
        owned.points.push(spatial_utils::point::Point3RgbF32::new(
            positions[b],
            positions[b + 1],
            positions[b + 2],
            colors[b],
            colors[b + 1],
            colors[b + 2],
        ));
    }

    owned.bytes.clear();
    let params = if prms.is_null() {
        EncodingParams::default()
    } else {
        // SAFETY: must originate from `pc_encoding_params_*`.
        let inner = unsafe { &*(prms as *const PcEncodingParamsInner) };
        inner.params.clone()
    };

    if let Err(e) = crate::encoder::encode_into_generic::<spatial_utils::point::Point3RgbF32, f32>(
        &owned.points,
        &params,
        &mut owned.bytes,
    ) {
        set_last_error(format!("encode failed: {e}"));
        return false;
    }

    unsafe {
        *out_bytes = PcByteSlice {
            data: owned.bytes.as_ptr(),
            len: owned.bytes.len() as u64,
        };
    }
    true
}

/// Create a point decoder handle.
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_point_decoder_create() -> *mut c_void {
    let owned = PointDecoderOwned {
        positions: Vec::new(),
        colors: Vec::new(),
    };
    Box::into_raw(Box::new(owned)) as *mut c_void
}

/// Free a point decoder handle.
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_point_decoder_free(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(handle as *mut PointDecoderOwned));
    }
}


/// Decode bytes into the decoder’s internal buffers.
///
/// This is the first step of a two-step decode:
/// 1) [`pc_point_decoder_decode_from_bytes`]: parse bytes and determine `point_count`.
/// 2) [`pc_point_decoder_write_to_buffers`]: copy into caller-provided buffers.
///
/// This enables dynamic buffer sizing, similar to the gsplat decode API.
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_point_decoder_decode_from_bytes(
    handle: *mut c_void,
    data: FFISlice<u8>,
    out_meta: *mut PcPointMeta,
) -> PcStatus {
    pc_clear_last_error();

    if out_meta.is_null() {
        set_last_error("ffi: out_meta is null");
        return PcStatus::NullPointer;
    }
    unsafe { *out_meta = PcPointMeta::default() };

    if handle.is_null() {
        set_last_error("ffi: handle is null");
        return PcStatus::NullPointer;
    }

    let res = catch_unwind(AssertUnwindSafe(|| {
        let owned = unsafe { &mut *(handle as *mut PointDecoderOwned) };
        owned.positions.clear();
        owned.colors.clear();

        if let Err(e) = crate::decoder::decode_into_flattened_vecs(&data, &mut owned.positions, &mut owned.colors) {
            set_last_error(format!("decode failed: {e}"));
            return PcStatus::DecodeFailed;
        }

        if owned.positions.len() != owned.colors.len() || (owned.positions.len() % 3) != 0 {
            set_last_error("decode produced inconsistent output lengths");
            return PcStatus::DecodeFailed;
        }

        let point_count = (owned.positions.len() / 3) as u64;
        if point_count > (u32::MAX as u64) {
            set_last_error("ffi: point_count overflow (exceeds u32::MAX)");
            return PcStatus::SizeOverflow;
        }

        unsafe {
            (*out_meta).point_count = point_count as u32;
        }

        PcStatus::Ok
    }));

    match res {
        Ok(st) => st,
        Err(_) => {
            set_last_error("ffi: panic");
            PcStatus::Panic
        }
    }
}

/// Copy the last decoded point cloud into caller-provided buffers.
///
/// `out_meta->point_count` will be updated to the decoded count (and can be used by callers
/// to validate they allocated the expected size).
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_point_decoder_write_to_buffers(
    handle: *mut c_void,
    output: PcPointOutBuffers,
    out_meta: *mut PcPointMeta,
) -> PcStatus {
    pc_clear_last_error();

    if out_meta.is_null() {
        set_last_error("ffi: out_meta is null");
        return PcStatus::NullPointer;
    }
    unsafe { *out_meta = PcPointMeta::default() };

    if handle.is_null() {
        set_last_error("ffi: handle is null");
        return PcStatus::NullPointer;
    }

    let res = catch_unwind(AssertUnwindSafe(|| {
        if output.positions.data.is_null() || output.colors.data.is_null() {
            set_last_error("ffi: output pointers are null");
            return PcStatus::NullPointer;
        }

        let owned = unsafe { &mut *(handle as *mut PointDecoderOwned) };

        // Allow empty point clouds (0 points).
        if owned.positions.len() != owned.colors.len() || (owned.positions.len() % 3) != 0 {
            set_last_error("ffi: internal decoded buffers are inconsistent (call decode_from_bytes first)");
            return PcStatus::DecodeFailed;
        }

        let n = owned.positions.len();
        let point_count = (n / 3) as u64;
        if point_count > (u32::MAX as u64) {
            set_last_error("ffi: point_count overflow (exceeds u32::MAX)");
            return PcStatus::SizeOverflow;
        }

        if (output.positions.len as usize) < n || (output.colors.len as usize) < n {
            set_last_error("ffi: output buffers too small");
            return PcStatus::InvalidSlice;
        }

        unsafe {
            std::ptr::copy_nonoverlapping(owned.positions.as_ptr(), output.positions.data, n);
            std::ptr::copy_nonoverlapping(owned.colors.as_ptr(), output.colors.data, n);

            (*out_meta).point_count = point_count as u32;
        }

        PcStatus::Ok
    }));

    match res {
        Ok(st) => st,
        Err(_) => {
            set_last_error("ffi: panic");
            PcStatus::Panic
        }
    }
}


/// Decode `data` into flattened output buffers.
///
/// The output buffers must have capacity for `num_points * 3` values each.
/// The number of points decoded is returned via `output_points`.
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_point_decode_into(
    handle: *mut c_void,
    data: FFISlice<u8>,
    output: PcPointOutBuffers,
    output_points: *mut u64,
) -> bool {
    if output_points.is_null() {
        set_last_error("ffi: output_points is null");
        return false;
    }
    unsafe { *output_points = 0 };

    if handle.is_null() {
        set_last_error("ffi: handle is null");
        return false;
    }

    if output.positions.data.is_null() || output.colors.data.is_null() {
        set_last_error("ffi: output pointers are null");
        return false;
    }

    let owned = unsafe { &mut *(handle as *mut PointDecoderOwned) };
    owned.positions.clear();
    owned.colors.clear();

    if let Err(e) = crate::decoder::decode_into_flattened_vecs(&data, &mut owned.positions, &mut owned.colors) {
        set_last_error(format!("decode failed: {e}"));
        return false;
    }

    if owned.positions.len() != owned.colors.len() || owned.positions.len() % 3 != 0 {
        set_last_error("decode produced inconsistent output lengths");
        return false;
    }

    let n = owned.positions.len();
    if (output.positions.len as usize) < n || (output.colors.len as usize) < n {
        set_last_error("ffi: output buffers too small");
        return false;
    }

    unsafe {
        std::ptr::copy_nonoverlapping(owned.positions.as_ptr(), output.positions.data, n);
        std::ptr::copy_nonoverlapping(owned.colors.as_ptr(), output.colors.data, n);
        *output_points = (n / 3) as u64;
    }
    true
}

// -------------------------------------------------------------------------------------------------
// Gsplat encode (AoS via SoA buffers)
// -------------------------------------------------------------------------------------------------

/// Create a gsplat encoder handle.
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_gsplat_encoder_create() -> *mut c_void {
    let owned = GsplatEncoderOwned {
        splats: Vec::new(),
        bytes: Vec::new(),
    };
    Box::into_raw(Box::new(owned)) as *mut c_void
}

/// Free a gsplat encoder handle.
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_gsplat_encoder_free(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(handle as *mut GsplatEncoderOwned));
    }
}

/// Encode gsplats from SoA buffers.
///
/// Returns a view into an internal buffer owned by `handle`.
///
/// This uses default encoding parameters. For custom parameters, use
/// [`pc_gsplat_encode_with_params`].
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_gsplat_encode(
    handle: *mut c_void,
    input: PcGsplatOutBuffers,
    out_bytes: *mut PcByteSlice,
) -> bool {
    pc_gsplat_encode_with_params(handle, input, std::ptr::null(), out_bytes)
}

/// Encode gsplats from SoA buffers using explicit encoding parameters.
///
/// If `prms` is null, defaults are used.
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_gsplat_encode_with_params(
    handle: *mut c_void,
    input: PcGsplatOutBuffers,
    prms: *const PcEncodingParamsOpaque,
    out_bytes: *mut PcByteSlice,
) -> bool {
    if out_bytes.is_null() {
        set_last_error("ffi: out_bytes is null");
        return false;
    }
    unsafe { *out_bytes = PcByteSlice::default() };

    if handle.is_null() {
        set_last_error("ffi: handle is null");
        return false;
    }

    // Validate pointers.
    if input.positions.data.is_null()
        || input.colors.data.is_null()
        || input.scales.data.is_null()
        || input.rotations.data.is_null()
    {
        set_last_error("ffi: one or more input pointers are null");
        return false;
    }

    let n = input.positions.len as usize;
    if input.colors.len as usize != n || input.scales.len as usize != n || input.rotations.len as usize != n {
        set_last_error("ffi: gsplat input buffers must have equal lengths");
        return false;
    }

    let positions = unsafe { std::slice::from_raw_parts(input.positions.data, n) };
    let colors = unsafe { std::slice::from_raw_parts(input.colors.data, n) };
    let scales = unsafe { std::slice::from_raw_parts(input.scales.data, n) };
    let rotations = unsafe { std::slice::from_raw_parts(input.rotations.data, n) };

    let owned = unsafe { &mut *(handle as *mut GsplatEncoderOwned) };
    owned.splats.clear();
    owned.splats.reserve(n);

    #[inline]
    fn f01_to_u8(x: f32) -> u8 {
        if !x.is_finite() { return 0; }
        let clamped = x.clamp(0.0, 1.0);
        (clamped * 255.0 + 0.5).floor() as u8
    }

    for i in 0..n {
        let p = positions[i];
        let c = colors[i];
        let s = scales[i];
        let r = rotations[i];
        let rgba = spatial_utils::color::Rgba8::new(
            f01_to_u8(c.x),
            f01_to_u8(c.y),
            f01_to_u8(c.z),
            f01_to_u8(c.w),
        );
        owned.splats.push(GaussianSplatF32::new(
            [p.x, p.y, p.z],
            rgba,
            [s.x, s.y, s.z],
            [r.x, r.y, r.z, r.w],
        ));
    }

    owned.bytes.clear();
    let params = if prms.is_null() {
        EncodingParams::default()
    } else {
        // SAFETY: must originate from `pc_encoding_params_*`.
        let inner = unsafe { &*(prms as *const PcEncodingParamsInner) };
        inner.params.clone()
    };
    if let Err(e) = crate::encoder::encode_into_generic::<GaussianSplatF32, f32>(
        &owned.splats,
        &params,
        &mut owned.bytes,
    ) {
        set_last_error(format!("encode failed: {e}"));
        return false;
    }

    unsafe {
        *out_bytes = PcByteSlice {
            data: owned.bytes.as_ptr(),
            len: owned.bytes.len() as u64,
        };
    }
    true
}

/// Create default encoding parameters for point encoding.
///
/// The returned handle must be freed with [`pc_encoding_params_free`].
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_encoding_params_default_points(
    output: *mut *mut PcEncodingParamsOpaque,
) -> bool {
    if output.is_null() {
        set_last_error("ffi: output is null");
        return false;
    }
    let boxed: Box<PcEncodingParamsInner> = Box::new(PcEncodingParamsInner {
        params: EncodingParams::default(),
    });
    unsafe {
        *output = Box::into_raw(boxed) as *mut PcEncodingParamsOpaque;
    }
    true
}

/// Create default encoding parameters for Gaussian splat encoding.
///
/// The returned handle must be freed with [`pc_encoding_params_free`].
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_encoding_params_default_gsplat(
    output: *mut *mut PcEncodingParamsOpaque,
) -> bool {
    if output.is_null() {
        set_last_error("ffi: output is null");
        return false;
    }
    let boxed: Box<PcEncodingParamsInner> = Box::new(PcEncodingParamsInner {
        params: EncodingParams::default(),
    });
    unsafe {
        *output = Box::into_raw(boxed) as *mut PcEncodingParamsOpaque;
    }
    true
}

/// Parse encoding parameters from TOML.
///
/// Intended for configuration-time use; for high-throughput encoding, parse once and reuse.
///
/// The returned handle must be freed with [`pc_encoding_params_free`].
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_encoding_params_from_toml(
    toml_ptr: *const u8,
    toml_len: u64,
    output: *mut *mut PcEncodingParamsOpaque,
) -> bool {
    if output.is_null() {
        set_last_error("ffi: output is null");
        return false;
    }
    if toml_ptr.is_null() {
        set_last_error("ffi: toml_ptr is null");
        return false;
    }
    let toml_len: usize = match usize::try_from(toml_len) {
        Ok(v) => v,
        Err(_) => {
            set_last_error("ffi: toml_len too large");
            return false;
        }
    };
    let bytes = unsafe { std::slice::from_raw_parts(toml_ptr, toml_len) };
    let s = match std::str::from_utf8(bytes) {
        Ok(v) => v,
        Err(_) => {
            set_last_error("ffi: TOML must be valid UTF-8");
            return false;
        }
    };
    let params: EncodingParams = match toml::from_str(s) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(format!("ffi: Failed to parse TOML: {e}"));
            return false;
        }
    };
    let boxed: Box<PcEncodingParamsInner> = Box::new(PcEncodingParamsInner { params });
    unsafe {
        *output = Box::into_raw(boxed) as *mut PcEncodingParamsOpaque;
    }
    true
}

/// Free an encoding parameters handle created by `pc_encoding_params_*`.
#[ffi_function]
#[no_mangle]
pub extern "C" fn pc_encoding_params_free(ptr: *mut PcEncodingParamsOpaque) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(ptr as *mut PcEncodingParamsInner));
    }
}

/// Interoptopus inventory.
pub fn build_binding_inventory() -> Inventory {
    InventoryBuilder::new()
        .register(function!(pc_last_error_message))
        .register(function!(pc_clear_last_error))
        .register(function!(pc_gsplat_decoder_create))
        .register(function!(pc_gsplat_decoder_free))
        .register(function!(pc_gsplat_decoder_decode_from_bytes))
        .register(function!(pc_gsplat_decoder_write_to_buffers))
        .register(function!(pc_gsplat_encoder_create))
        .register(function!(pc_gsplat_encoder_free))
        .register(function!(pc_gsplat_encode))
        .register(function!(pc_gsplat_encode_with_params))
        .register(function!(pc_point_encoder_create))
        .register(function!(pc_point_encoder_free))
        .register(function!(pc_point_encode))
        .register(function!(pc_point_encode_with_params))
        .register(function!(pc_encoding_params_default_points))
        .register(function!(pc_encoding_params_default_gsplat))
        .register(function!(pc_encoding_params_from_toml))
        .register(function!(pc_encoding_params_free))
        .register(function!(pc_point_decoder_create))
        .register(function!(pc_point_decoder_free))
        .register(function!(pc_point_decoder_decode_from_bytes))
        .register(function!(pc_point_decoder_write_to_buffers))
        .register(function!(pc_point_decode_into))
        .inventory()
}
