//! GSP16: fixed-width (16 bytes/record) packed Points/Splats container.
//!
//! Record layout (little-endian u32 words, 16 bytes total):
//!   word0: RGBA8 packed as 0xAABBGGRR
//!   word1: f16 X (low16), f16 Y (high16)
//!   word2: f16 Z (low16), octU (bits16..23), octV (bits24..31)
//!   word3: scaleX (bits0..7), scaleY (bits8..15), scaleZ (bits16..23), angleU8 (bits24..31)
//!
//! Header layout (16 bytes):
//!   magic[3] = "GSP"
//!   version  = 1
//!   flags    = bit0: 1=splats, 0=points; bit1: 1=alpha meaningful
//!   rsv0     = 0
//!   rsv1     = u16(0)
//!   count    = u32 LE (number of records)
//!   rsv2     = u32(0)
//! 
//! Derived from https://github.com/wuyize25/gsplat-unity/pull/12/changes

pub mod decoder;
pub mod encoder;
pub mod types;


#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod f16c {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    // Rust stdarch only accepts the 3-bit rounding mode immediate (0..=7).
    // _MM_FROUND_TO_NEAREST_INT is 0 (round-to-nearest-even).
    const ROUND_NEAREST: i32 = _MM_FROUND_TO_NEAREST_INT;

    /// Convert 4x f32 to 4x f16 *bit patterns* (IEEE-754 binary16) using F16C hardware.
    ///
    /// Lanes low->high correspond to `(a, b, c, d)`.
    ///
    /// # Safety
    ///
    /// This function requires the CPU feature `f16c` to be available. It must only be called
    /// when the `f16c` feature is detected at runtime or the target guarantees its availability.
    #[target_feature(enable = "f16c")]
    pub unsafe fn f32x4_to_f16x4_bits(a: f32, b: f32, c: f32, d: f32) -> (u16, u16, u16, u16) {
        // lanes low->high: a, b, c, d
        let v = _mm_set_ps(d, c, b, a);

        // IMPORTANT: do NOT OR _MM_FROUND_NO_EXC here; stdarch rejects bit3.
        let h = _mm_cvtps_ph(v, ROUND_NEAREST);

        // low 4 u16 lanes are in the low 64 bits
        let lo = _mm_cvtsi128_si64(h) as u64;
        (
            (lo & 0xFFFF) as u16,
            ((lo >> 16) & 0xFFFF) as u16,
            ((lo >> 32) & 0xFFFF) as u16,
            ((lo >> 48) & 0xFFFF) as u16,
        )
    }

    /// Convert 4x f16 *bit patterns* (IEEE-754 binary16) to 4x f32 using F16C hardware.
    ///
    /// Lanes low->high correspond to `(a, b, c, d)`.
    ///
    /// # Safety
    ///
    /// This function requires the CPU feature `f16c` to be available. It must only be called
    /// when the `f16c` feature is detected at runtime or the target guarantees its availability.
    #[target_feature(enable = "f16c")]
    pub unsafe fn f16x4_bits_to_f32x4(a: u16, b: u16, c: u16, d: u16) -> (f32, f32, f32, f32) {
        let packed: u64 = (a as u64)
            | ((b as u64) << 16)
            | ((c as u64) << 32)
            | ((d as u64) << 48);

        let h = _mm_cvtsi64_si128(packed as i64);
        let v = _mm_cvtph_ps(h);

        let mut out = [0f32; 4];
        _mm_storeu_ps(out.as_mut_ptr(), v);
        (out[0], out[1], out[2], out[3])
    }
}


/// AArch64 FP16 conversion helpers (runtime-selected fast path).
///
/// These functions require the CPU feature `fp16` and are guarded by:
/// - compile-time `cfg(target_arch="aarch64")`
/// - runtime detection via `std::arch::is_aarch64_feature_detected!("fp16")`
///
/// They are safe to call only when the feature is detected, because they are
/// annotated with `#[target_feature(enable="fp16")]`.
#[cfg(target_arch = "aarch64")]
pub mod fp16 {
    use core::arch::aarch64::*;

    /// Convert 4x f32 to 4x f16 *bit patterns* (IEEE-754 binary16) using FP16 hardware.
    ///
    /// Lanes low->high correspond to `(a, b, c, d)`.
    #[inline(always)]
    #[target_feature(enable = "fp16")]
    pub unsafe fn f32x4_to_f16x4_bits(a: f32, b: f32, c: f32, d: f32) -> (u16, u16, u16, u16) {
        // Build float32x4_t without extra memory traffic.
        let mut v = vdupq_n_f32(0.0);
        v = vsetq_lane_f32(a, v, 0);
        v = vsetq_lane_f32(b, v, 1);
        v = vsetq_lane_f32(c, v, 2);
        v = vsetq_lane_f32(d, v, 3);

        // Convert to half and reinterpret to u16 bits.
        let h: float16x4_t = vcvt_f16_f32(v);
        let hu: uint16x4_t = vreinterpret_u16_f16(h);

        (
            vget_lane_u16(hu, 0),
            vget_lane_u16(hu, 1),
            vget_lane_u16(hu, 2),
            vget_lane_u16(hu, 3),
        )
    }

    /// Convert 4x f16 *bit patterns* (IEEE-754 binary16) to 4x f32 using FP16 hardware.
    ///
    /// Lanes low->high correspond to `(a, b, c, d)`.
    #[inline(always)]
    #[target_feature(enable = "fp16")]
    pub unsafe fn f16x4_bits_to_f32x4(a: u16, b: u16, c: u16, d: u16) -> (f32, f32, f32, f32) {
        // Load as u16 lanes then reinterpret as f16 lanes.
        let lanes = [a, b, c, d];
        let hu: uint16x4_t = vld1_u16(lanes.as_ptr());
        let h: float16x4_t = vreinterpret_f16_u16(hu);

        // Convert to f32.
        let v: float32x4_t = vcvt_f32_f16(h);

        // Extract.
        let mut out = [0f32; 4];
        vst1q_f32(out.as_mut_ptr(), v);
        (out[0], out[1], out[2], out[3])
    }
}

#[cfg(test)]
mod tests {
    use super::{decoder, encoder, types::Gsplat16Params};
    use spatial_utils::{point::Point3RgbF32, traits::{HasPosition3, HasRgb8u}};

    #[test]
    fn gsplat16_roundtrip_smoke_points() {
        let input: Vec<Point3RgbF32> = vec![
            Point3RgbF32::new(0.0, 0.0, 0.0, 255, 0, 0),
            Point3RgbF32::new(1.25, 2.5, 3.75, 0, 255, 0),
            Point3RgbF32::new(-4.0, 5.0, -6.0, 0, 0, 255),
            Point3RgbF32::new(10.0, -2.5, 0.125, 123, 45, 67),
        ];

        let params = Gsplat16Params::default();

        let mut encoded = Vec::new();
        encoder::encode_from_payload_into::<Point3RgbF32, f32>(&input, &params, &mut encoded)
            .expect("encode must succeed");
        assert!(!encoded.is_empty());

        let mut out: Vec<Point3RgbF32> = Vec::new();
        decoder::decode_into(&encoded, &mut out).expect("decode must succeed");
        assert_eq!(out.len(), input.len());

        // Positions are stored in f16 -> allow small numeric error.
        let eps = 1.0e-2_f32;
        for (a, b) in input.iter().zip(out.iter()) {
            assert!((a.x() - b.x()).abs() <= eps, "x mismatch: {a:?} vs {b:?}");
            assert!((a.y() - b.y()).abs() <= eps, "y mismatch: {a:?} vs {b:?}");
            assert!((a.z() - b.z()).abs() <= eps, "z mismatch: {a:?} vs {b:?}");

            // Colors should roundtrip exactly.
            assert_eq!(a.r_u8(), b.r_u8());
            assert_eq!(a.g_u8(), b.g_u8());
            assert_eq!(a.b_u8(), b.b_u8());
        }
    }
}