//! Codec-agnostic encoder entry points.
//!
//! The `spatial_codecs` crate exposes many codecs behind a single enum (`EncodingParams`).
//! Higher layers interact with this module rather than calling codec-specific functions.
//! This keeps benchmark and runtime code simple while letting codec modules remain focused
//! on their own transforms.

use spatial_utils::{
    traits::{ColorKind, SpatialKind, SpatialMeta, SpatialOwnedFull},
    utils::point_scalar::PointScalar,
};
use serde::{Deserialize, Serialize};

use crate::CodecError;

/// Runtime-selectable encoding format.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Copy, Clone)]
pub enum EncodingFormat {
    Ply = 0,
    Draco,
    Gsplat16,
    LASzip,
    Tmf,
    Bitcode,
    Gzip,
    Zstd,
    Lz4,
    Snappy,
    Sogp,
    Quantize,
    Openzl,
}

impl EncodingFormat {
    // Returns a string representation of the encoding format.
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ply => "PLY",
            Self::Draco => "Draco",
            Self::Gsplat16 => "GSP16",
            Self::LASzip => "LASzip",
            Self::Tmf => "TMF",
            Self::Bitcode => "Bitcode",
            Self::Gzip => "Gzip",
            Self::Zstd => "Zstd",
            Self::Lz4 => "Lz4",
            Self::Snappy => "Snappy",
            Self::Sogp => "SOGP",
            Self::Quantize => "Quantize",
            Self::Openzl => "OpenZL",
        }
    }
    // Convert from &str to EncodingFormat (returns Option)
    #[inline]
    pub fn from_str_opt(format: &str) -> Option<Self> {
        match format.to_lowercase().as_str() {
            "ply" => Some(Self::Ply),
            "draco" => Some(Self::Draco),
            "gsplat16" | "gsp16" | "gsp" => Some(Self::Gsplat16),
            "laszip" => Some(Self::LASzip),
            "tmf" => Some(Self::Tmf),
            "bitcode" => Some(Self::Bitcode),
            "gzip" => Some(Self::Gzip),
            "zstd" => Some(Self::Zstd),
            "lz4" => Some(Self::Lz4),
            "snappy" => Some(Self::Snappy),
            "sogp" => Some(Self::Sogp),
            "quantize" => Some(Self::Quantize),
            "openzl" => Some(Self::Openzl),
            _ => None,
        }
    }
}

#[inline(always)]
fn supports_spatial_kind(params: &EncodingParams, kind: SpatialKind) -> bool {
    match kind {
        SpatialKind::Points => true,
        SpatialKind::Splats => match params {
            EncodingParams::Bitcode(_) => true,
            EncodingParams::Ply(_) => true,
            EncodingParams::Gsplat16(_) => true,

            // Wrapper codecs: support splats iff inner supports splats.
            EncodingParams::Gzip { inner, .. } => supports_spatial_kind(inner.as_ref(), kind),
            EncodingParams::Zstd { inner, .. } => supports_spatial_kind(inner.as_ref(), kind),
            EncodingParams::Lz4 { inner } => supports_spatial_kind(inner.as_ref(), kind),
            EncodingParams::Snappy { inner } => supports_spatial_kind(inner.as_ref(), kind),

            // OpenZL Serial is a wrapper over inner bytes -> propagate.
            #[cfg(feature = "openzl")]
            EncodingParams::Openzl(crate::codecs::openzl::encoder::OpenzlParams::Serial {
                inner,
                ..
            }) => supports_spatial_kind(inner.as_ref(), kind),
            #[cfg(not(feature = "openzl"))]
            EncodingParams::Openzl(_) => {
                // When OpenZL is disabled, we conservatively assume it could support splats.
                true
            }

            // Everything else cannot represent splat fields yet.
            _ => false,
        },
    }
}

fn push_chain(params: &EncodingParams, out: &mut Vec<EncodingFormat>) {
    match params {
        EncodingParams::Ply(_) => out.push(EncodingFormat::Ply),
        EncodingParams::Draco(_) => out.push(EncodingFormat::Draco),
        EncodingParams::Gsplat16(_) => out.push(EncodingFormat::Gsplat16),
        EncodingParams::Tmf(_) => out.push(EncodingFormat::Tmf),
        EncodingParams::Bitcode(_) => out.push(EncodingFormat::Bitcode),
        EncodingParams::Quantize(_) => out.push(EncodingFormat::Quantize),
        EncodingParams::Sogp(_) => out.push(EncodingFormat::Sogp),
        EncodingParams::LASzip => out.push(EncodingFormat::LASzip),

        EncodingParams::Gzip { inner, .. } => {
            out.push(EncodingFormat::Gzip);
            push_chain(inner.as_ref(), out);
        }
        EncodingParams::Zstd { inner, .. } => {
            out.push(EncodingFormat::Zstd);
            push_chain(inner.as_ref(), out);
        }
        EncodingParams::Lz4 { inner } => {
            out.push(EncodingFormat::Lz4);
            push_chain(inner.as_ref(), out);
        }
        EncodingParams::Snappy { inner } => {
            out.push(EncodingFormat::Snappy);
            push_chain(inner.as_ref(), out);
        }
        #[cfg(feature = "openzl")]
        EncodingParams::Openzl(p) => match p {
            crate::codecs::openzl::encoder::OpenzlParams::Serial { inner, .. } => {
                out.push(EncodingFormat::Openzl);
                push_chain(inner.as_ref(), out);
            }
            crate::codecs::openzl::encoder::OpenzlParams::Columnar { .. } => {
                out.push(EncodingFormat::Openzl);
            }
        },
        #[cfg(not(feature = "openzl"))]
        EncodingParams::Openzl(_) => {
            out.push(EncodingFormat::Openzl);
        }
    }
}

fn validate_spatial_kind<P, S>(params: &EncodingParams) -> Result<(), Box<dyn std::error::Error>>
where
    P: SpatialOwnedFull<S>,
    S: PointScalar,
{
    let spatial_kind = <P as SpatialMeta>::KIND;
    let color_kind = <P as SpatialMeta>::COLOR_KIND;

    fn get_error(
        spatial_kind: SpatialKind,
        color_kind: ColorKind,
        params: &EncodingParams,
    ) -> CodecError {
        let mut codec_chain = Vec::new();
        push_chain(params, &mut codec_chain);
        CodecError::UnsupportedSpatialKind {
            spatial_kind,
            color_kind,
            codec_chain,
        }
    }

    // A small shortcut: assuming that Points + RGB8 is always supported.
    if spatial_kind == SpatialKind::Points && color_kind == ColorKind::Rgb8 {
        return Ok(());
    }
    if !supports_spatial_kind(params, spatial_kind) {
        return Err(Box::new(get_error(spatial_kind, color_kind, params)));
    }

    // TODO: add color kind checks

    Ok(())
}

/// Parameters for each encoder. Variants wrap the codec’s native parameter structure
/// (for example `QuantizeParams`) so we do not lose any settings when serialising/deserialising.
#[repr(u8)]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum EncodingParams {
    Ply(crate::codecs::ply::encoder::PlyParams) = 0,
    #[cfg(feature = "draco")]
    Draco(crate::codecs::draco::encoder::DracoParams) = 1,
    #[cfg(not(feature = "draco"))]
    Draco(()) = 1, // Placeholder when Draco is disabled
    Gsplat16(crate::codecs::gsplat16::types::Gsplat16Params) = 2,
    Tmf(crate::codecs::tmf::encoder::TmfParams) = 3,
    Bitcode(crate::codecs::bitcode::encoder::BitcodeParams) = 4,
    Quantize(crate::codecs::quantize::types::QuantizeParams) = 5,
    LASzip = 6,
    // Reserve variants for external codecs:
    // Gpcc(GpccParams), Vpcc(VpccParams),
    Gzip {
        inner: Box<EncodingParams>,
        level: Option<u32>, // 0..9 (None = flate2 default)
    } = 7,
    Zstd {
        inner: Box<EncodingParams>,
        level: Option<i32>,
    } = 8,
    Lz4 {
        inner: Box<EncodingParams>,
    } = 9,
    Snappy {
        inner: Box<EncodingParams>,
    } = 10,
    Sogp(crate::codecs::sogp::types::SogpParams) = 11,
    #[cfg(feature = "openzl")]
    Openzl(crate::codecs::openzl::encoder::OpenzlParams) = 12,
    #[cfg(not(feature = "openzl"))]
    Openzl(()) = 12, // Placeholder when OpenZL is disabled
}

impl EncodingParams {
    #[inline]
    /// Return `true` when the variant wraps another codec (e.g. Gzip/Zstd).
    #[cfg(feature = "openzl")]
    pub fn is_wrapper(&self) -> bool {
        matches!(
            self,
            Self::Gzip { .. }
                | Self::Zstd { .. }
                | Self::Lz4 { .. }
                | Self::Snappy { .. }
                | Self::Openzl(crate::codecs::openzl::encoder::OpenzlParams::Serial { .. })
        )
    }
    #[cfg(not(feature = "openzl"))]
    pub fn is_wrapper(&self) -> bool {
        matches!(
            self,
            Self::Gzip { .. }
                | Self::Zstd { .. }
                | Self::Lz4 { .. }
                | Self::Snappy { .. }
        )
    }
}

impl Default for EncodingParams {
    fn default() -> Self {
        get_default_params(EncodingFormat::Bitcode)
    }
}

/// Convenience helper to obtain default parameters per format.
pub fn get_default_params(format: EncodingFormat) -> EncodingParams {
    match format {
        // The standalone codecs
        EncodingFormat::Ply => {
            EncodingParams::Ply(crate::codecs::ply::encoder::PlyParams::default())
        }
        EncodingFormat::Draco => {
            #[cfg(feature = "draco")]
            return EncodingParams::Draco(crate::codecs::draco::encoder::DracoParams::default());
            #[cfg(not(feature = "draco"))]
            return EncodingParams::Draco(());
        }
        EncodingFormat::Gsplat16 => {
            EncodingParams::Gsplat16(crate::codecs::gsplat16::types::Gsplat16Params::default())
        }
        EncodingFormat::Tmf => {
            EncodingParams::Tmf(crate::codecs::tmf::encoder::TmfParams::default())
        }
        EncodingFormat::Quantize => {
            EncodingParams::Quantize(crate::codecs::quantize::types::QuantizeParams::default())
        }
        EncodingFormat::Sogp => {
            EncodingParams::Sogp(crate::codecs::sogp::types::SogpParams::default())
        }
        EncodingFormat::Bitcode => {
            EncodingParams::Bitcode(crate::codecs::bitcode::encoder::BitcodeParams::default())
        }
        // Codecs that can be both standalone and a wrapper
            #[cfg(feature = "openzl")]
        EncodingFormat::Openzl => {
            EncodingParams::Openzl(crate::codecs::openzl::encoder::OpenzlParams::columnar_default())
        }
        #[cfg(not(feature = "openzl"))]
        EncodingFormat::Openzl => {
            EncodingParams::Openzl(())
        }
        // The wrapper codecs
        EncodingFormat::Gzip => EncodingParams::Gzip {
            inner: Box::new(get_default_params(EncodingFormat::Bitcode)),
            level: None,
        },
        EncodingFormat::Zstd => EncodingParams::Zstd {
            inner: Box::new(get_default_params(EncodingFormat::Bitcode)),
            level: None,
        },
        EncodingFormat::Lz4 => EncodingParams::Lz4 {
            inner: Box::new(get_default_params(EncodingFormat::Bitcode)),
        },
        EncodingFormat::Snappy => EncodingParams::Snappy {
            inner: Box::new(get_default_params(EncodingFormat::Bitcode)),
        },
        _ => panic!("Unsupported encoding format for default parameters"),
    }
}

/// Streaming-friendly: encode points into caller-owned `out`.
/// TODO: investigate whether a scratch buffer can help here.
pub(crate) fn encode_into_unchecked<P, S>(
    payload: &[P],
    params: &EncodingParams,
    out: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>>
where
    P: SpatialOwnedFull<S>,
    S: PointScalar,
{
    match params {
        // The standalone codecs
        EncodingParams::Ply(p) => {
            crate::codecs::ply::encoder::encode_from_payload_into::<P, S>(payload, p, out)
        }
        #[cfg(feature = "draco")]
        EncodingParams::Draco(p) => {
            crate::codecs::draco::encoder::encode_from_payload_into::<P, S>(payload, p, out)
        }
        #[cfg(not(feature = "draco"))]
        EncodingParams::Draco(_) => {
            Err("Draco encoding not available (feature disabled)".into())
        }
        EncodingParams::Gsplat16(p) => {
            crate::codecs::gsplat16::encoder::encode_from_payload_into::<P, S>(payload, p, out)
        }
        EncodingParams::Tmf(p) => {
            crate::codecs::tmf::encoder::encode_from_payload_into::<P, S>(payload, p, out)
        }
        EncodingParams::Bitcode(p) => {
            crate::codecs::bitcode::encoder::encode_from_payload_into::<P, S>(payload, p, out)
        }
        EncodingParams::Quantize(p) => {
            crate::codecs::quantize::encoder::encode_from_payload_into::<P, S>(payload, p, out)
        }
        EncodingParams::Sogp(p) => {
            crate::codecs::sogp::encoder::encode_from_payload_into::<P, S>(payload, p, out)
        }
        // Codecs that can be both standalone and a wrapper
        #[cfg(feature = "openzl")]
        EncodingParams::Openzl(p) => {
            crate::codecs::openzl::encoder::encode_from_payload_into(payload, p, out)
        }
        #[cfg(not(feature = "openzl"))]
        EncodingParams::Openzl(_) => {
            Err("OpenZL encoding not available (feature disabled)".into())
        }
        // The wrapper codecs
        EncodingParams::Gzip { inner, level } => crate::codecs::gzip::encoder::wrap_gzip_into(
            |wbuf: &mut Vec<u8>| encode_into::<P, S>(payload, inner, wbuf),
            *level,
            out,
        ),
        EncodingParams::Zstd { inner, level } => crate::codecs::zstd::encoder::wrap_zstd_into(
            |wbuf: &mut Vec<u8>| encode_into::<P, S>(payload, inner, wbuf),
            *level,
            out,
        ),
        EncodingParams::Lz4 { inner } => crate::codecs::lz4::encoder::wrap_lz4_into(
            |wbuf: &mut Vec<u8>| encode_into::<P, S>(payload, inner, wbuf),
            out,
        ),
        EncodingParams::Snappy { inner } => crate::codecs::snappy::encoder::wrap_snappy_into(
            |wbuf: &mut Vec<u8>| encode_into::<P, S>(payload, inner, wbuf),
            out,
        ),
        // Unsupported codecs
        EncodingParams::LASzip => Err("LASzip not implemented".into()),
    }
}

pub fn encode_into_generic<P, S>(
    payload: &[P],
    params: &EncodingParams,
    out: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>>
where
    P: SpatialOwnedFull<S>,
    S: PointScalar,
{
    validate_spatial_kind::<P, S>(params)?;
    encode_into_unchecked::<P, S>(payload, params, out)
}

pub fn encode_into<P, S>(
    payload: &[P],
    params: &EncodingParams,
    out: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>>
where
    P: SpatialOwnedFull<S>,
    S: PointScalar,
{
    encode_into_generic(payload, params, out)
}

pub fn encode_from_points_with_params<P, S>(
    points: Vec<P>,
    params: &EncodingParams,
) -> Result<Vec<u8>, Box<dyn std::error::Error>>
where
    P: SpatialOwnedFull<S>,
    S: PointScalar,
{
    let mut out = Vec::new();
    encode_into(&points, params, &mut out)?;
    Ok(out)
}

pub fn encode_from_points<P, S>(
    points: Vec<P>,
    encoding: EncodingFormat,
) -> Result<Vec<u8>, Box<dyn std::error::Error>>
where
    P: SpatialOwnedFull<S>,
    S: PointScalar,
{
    let params = get_default_params(encoding);
    encode_from_points_with_params(points, &params)
}

#[cfg(test)]
mod spatial_kind_capability_tests {
    use super::*;
    use spatial_utils::color::Rgba8;
    use spatial_utils::point::Point3Rgb;
    use spatial_utils::splat::GaussianSplatF32;

    fn gzip_over_bitcode_params() -> EncodingParams {
        EncodingParams::Gzip {
            inner: Box::new(EncodingParams::Bitcode(
                crate::codecs::bitcode::encoder::BitcodeParams::default(),
            )),
            level: None,
        }
    }

    #[test]
    fn points_still_encode_via_quantize() {
        let pts: Vec<Point3Rgb<f32>> = vec![
            Point3Rgb::new(1.0, 2.0, 3.0, 10, 20, 30),
            Point3Rgb::new(4.0, 5.0, 6.0, 40, 50, 60),
        ];

        let params =
            EncodingParams::Quantize(crate::codecs::quantize::types::QuantizeParams::default());
        let mut out = Vec::new();

        encode_into_generic::<Point3Rgb<f32>, f32>(&pts, &params, &mut out).unwrap();
        assert!(out.starts_with(b"QNT"));
    }

    #[test]
    fn splats_encode_via_bitcode() {
        let splats = vec![GaussianSplatF32::new(
            [1.0, 2.0, 3.0],
            Rgba8::new(1, 2, 3, 4),
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        )];

        let params =
            EncodingParams::Bitcode(crate::codecs::bitcode::encoder::BitcodeParams::default());
        let mut out = Vec::new();

        encode_into_generic::<GaussianSplatF32, f32>(&splats, &params, &mut out).unwrap();
        assert!(out.starts_with(b"BC1"));
    }

    #[test]
    fn splats_rejected_by_quantize_with_typed_error() {
        let splats = vec![GaussianSplatF32::new(
            [1.0, 2.0, 3.0],
            Rgba8::new(1, 2, 3, 4),
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        )];

        let params =
            EncodingParams::Quantize(crate::codecs::quantize::types::QuantizeParams::default());
        let mut out = Vec::new();

        let err =
            encode_into_generic::<GaussianSplatF32, f32>(&splats, &params, &mut out).unwrap_err();
        let ce = err
            .downcast::<crate::error::CodecError>()
            .expect("expected CodecError");

        match *ce {
            crate::error::CodecError::UnsupportedSpatialKind {
                spatial_kind,
                color_kind,
                ref codec_chain,
            } => {
                assert_eq!(spatial_kind, spatial_utils::traits::SpatialKind::Splats);
                assert_eq!(color_kind, spatial_utils::traits::ColorKind::Rgba8);
                assert_eq!(codec_chain.as_slice(), &[EncodingFormat::Quantize]);
            }
        }
    }

    #[test]
    fn splats_allowed_through_wrapper_chain_over_bitcode() {
        let splats = vec![GaussianSplatF32::new(
            [1.0, 2.0, 3.0],
            Rgba8::new(1, 2, 3, 4),
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        )];

        let params = EncodingParams::Gzip {
            inner: Box::new(EncodingParams::Bitcode(
                crate::codecs::bitcode::encoder::BitcodeParams::default(),
            )),
            level: None,
        };

        let mut out = Vec::new();
        encode_into_generic::<GaussianSplatF32, f32>(&splats, &params, &mut out).unwrap();
        assert!(out.starts_with(b"GZP"));
    }

    #[test]
    fn splats_allowed_through_snappy_over_gzip_over_bitcode() {
        let splats = vec![GaussianSplatF32::new(
            [1.0, 2.0, 3.0],
            Rgba8::new(1, 2, 3, 4),
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        )];

        let params = EncodingParams::Snappy {
            inner: Box::new(gzip_over_bitcode_params()),
        };

        let mut out = Vec::new();
        encode_into_generic::<GaussianSplatF32, f32>(&splats, &params, &mut out).unwrap();
        assert!(out.starts_with(b"SNP"));
    }

    #[cfg(feature = "openzl")]
    #[test]
    fn splats_rejected_by_openzl_columnar() {
        let splats = vec![GaussianSplatF32::new(
            [0.0, 0.0, 0.0],
            Rgba8::new(1, 1, 1, 1),
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        )];

        let params = EncodingParams::Openzl(
            crate::codecs::openzl::encoder::OpenzlParams::columnar_default(),
        );
        let mut out = Vec::new();
        let err =
            encode_into_generic::<GaussianSplatF32, f32>(&splats, &params, &mut out).unwrap_err();
        let ce = err
            .downcast::<crate::error::CodecError>()
            .expect("expected CodecError");
        match *ce {
            crate::error::CodecError::UnsupportedSpatialKind {
                spatial_kind,
                color_kind,
                ref codec_chain,
            } => {
                assert_eq!(spatial_kind, SpatialKind::Splats);
                assert_eq!(color_kind, ColorKind::Rgba8);
                assert_eq!(codec_chain.as_slice(), &[EncodingFormat::Openzl]);
            }
        }
    }

    #[test]
    fn points_allowed_even_when_wrapped() {
        let pts: Vec<Point3Rgb<f32>> = vec![Point3Rgb::new(0.0, 0.0, 0.0, 1, 2, 3)];
        let params = gzip_over_bitcode_params();
        let mut out = Vec::new();
        encode_into_generic::<Point3Rgb<f32>, f32>(&pts, &params, &mut out).unwrap();
        assert!(out.starts_with(b"GZP"));
    }
}

#[cfg(test)]
mod format_and_validation_tests {
    use super::*;
    use spatial_utils::point::Point3RgbF32;

    #[test]
    fn encoding_format_roundtrips_strings_case_insensitive() {
        assert_eq!(EncodingFormat::Ply.as_str(), "PLY");
        assert_eq!(EncodingFormat::Draco.as_str(), "Draco");
        assert_eq!(
            EncodingFormat::from_str_opt("ply"),
            Some(EncodingFormat::Ply)
        );
        assert_eq!(
            EncodingFormat::from_str_opt("DrAcO"),
            Some(EncodingFormat::Draco)
        );
        assert_eq!(EncodingFormat::from_str_opt("unknown"), None);
    }

    #[test]
    fn supports_spatial_kind_matches_wrapper_chains() {
        // Splats supported when inner eventually is Bitcode
        let params = EncodingParams::Gzip {
            inner: Box::new(EncodingParams::Zstd {
                inner: Box::new(EncodingParams::Bitcode(
                    crate::codecs::bitcode::encoder::BitcodeParams::default(),
                )),
                level: None,
            }),
            level: None,
        };
        assert!(super::supports_spatial_kind(&params, SpatialKind::Splats));

        // Splats rejected when inner chain never supports splats
        let params = EncodingParams::Snappy {
            inner: Box::new(EncodingParams::Quantize(
                crate::codecs::quantize::types::QuantizeParams::default(),
            )),
        };
        assert!(!super::supports_spatial_kind(&params, SpatialKind::Splats));
    }

    #[test]
    fn validate_spatial_kind_shortcut_accepts_rgb_points() {
        let params = EncodingParams::Ply(crate::codecs::ply::encoder::PlyParams::default());
        let result = super::validate_spatial_kind::<Point3RgbF32, f32>(&params);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_spatial_kind_accepts_splats_for_ply() {
        let params = EncodingParams::Ply(crate::codecs::ply::encoder::PlyParams::default());
        let mut out = Vec::new();
        let splats = vec![spatial_utils::splat::GaussianSplatF32::new(
            [0.0, 0.0, 0.0],
            spatial_utils::color::Rgba8::new(0, 0, 0, 0),
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        )];

        super::encode_into_generic::<spatial_utils::splat::GaussianSplatF32, f32>(&splats, &params, &mut out)
            .expect("PLY should accept splats");
    }
}

#[cfg(test)]
mod gsp16_tests {
    use crate::{decoder, encoder::{self, EncodingParams}};
    use spatial_utils::{
        color::Rgba8,
        point::Point3RgbF32,
        splat::GaussianSplatF32,
        traits::{SpatialView, HasPosition3, HasRgb8u, HasRgba8u}
    };

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn approx_rel(a: f32, b: f32, rel_eps: f32, abs_eps: f32) -> bool {
        let diff = (a - b).abs();
        diff <= abs_eps.max(rel_eps * b.abs())
    }

    #[test]
    fn gsp16_points_roundtrip() {
        let pts = vec![
            Point3RgbF32::new(1.25, -2.5, 3.75, 10, 20, 30),
            Point3RgbF32::new(-4.0, 5.0, -6.0, 40, 50, 60),
        ];

        let params = EncodingParams::Gsplat16(crate::codecs::gsplat16::types::Gsplat16Params::default());
        let mut bytes = Vec::new();
        encoder::encode_into(&pts, &params, &mut bytes).unwrap();
        assert!(bytes.starts_with(b"GSP"));

        let mut out: Vec<Point3RgbF32> = Vec::new();
        decoder::decode_into(&bytes, &mut out).unwrap();
        assert_eq!(out.len(), pts.len());

        for (a, b) in out.iter().zip(pts.iter()) {
            // f16 quantization => allow small epsilon
            assert!(approx(a.x(), b.x(), 0.01));
            assert!(approx(a.y(), b.y(), 0.01));
            assert!(approx(a.z(), b.z(), 0.01));
            assert_eq!(a.r_u8(), b.r_u8());
            assert_eq!(a.g_u8(), b.g_u8());
            assert_eq!(a.b_u8(), b.b_u8());
        }
    }

    #[test]
    fn gsp16_splats_roundtrip() {
        let splats = vec![
            GaussianSplatF32::new(
                [1.0, 2.0, 3.0],
                Rgba8::new(128, 64, 32, 200),
                [0.5, 1.25, 2.0],
                [0.70710677, 0.0, 0.70710677, 0.0], // w,x,y,z
            ),
        ];

        let params = EncodingParams::Gsplat16(crate::codecs::gsplat16::types::Gsplat16Params::default());
        let mut bytes = Vec::new();
        encoder::encode_into(&splats, &params, &mut bytes).unwrap();
        assert!(bytes.starts_with(b"GSP"));

        let mut out: Vec<GaussianSplatF32> = Vec::new();
        decoder::decode_into(&bytes, &mut out).unwrap();
        assert_eq!(out.len(), splats.len());

        let a = &out[0];
        let b = &splats[0];

        // mean: f16 epsilon
        assert!(approx(a.x(), b.x(), 0.01));
        assert!(approx(a.y(), b.y(), 0.01));
        assert!(approx(a.z(), b.z(), 0.01));

        // color: exact
        assert_eq!(a.r_u8(), b.r_u8());
        assert_eq!(a.g_u8(), b.g_u8());
        assert_eq!(a.b_u8(), b.b_u8());
        assert_eq!(a.a_u8(), b.a_u8());

        // scale: log-quantized => allow a bit more drift
        assert!(approx_rel(a.scale_x(), b.scale_x(), 0.05, 0.02));
        assert!(approx_rel(a.scale_y(), b.scale_y(), 0.05, 0.02));
        assert!(approx_rel(a.scale_z(), b.scale_z(), 0.05, 0.02));

        // rotation: compare absolute dot (q and -q represent same rotation)
        let dot =
            a.rot_w() * b.rot_w() +
            a.rot_x() * b.rot_x() +
            a.rot_y() * b.rot_y() +
            a.rot_z() * b.rot_z();
        assert!(dot.abs() > 0.98);
    }


}
