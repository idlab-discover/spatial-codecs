use core::fmt;
use spatial_utils::traits::{ColorKind, SpatialKind};

use crate::encoder::EncodingFormat;

/// Errors produced by `spatial_codecs` entry points.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CodecError {
    /// The selected codec chain cannot represent the requested spatial kind.
    UnsupportedSpatialKind {
        spatial_kind: SpatialKind,
        color_kind: ColorKind,
        /// Codec chain from outermost wrapper to innermost codec.
        codec_chain: Vec<EncodingFormat>,
    },
}

impl CodecError {
    #[inline]
    pub fn spatial_kind(&self) -> SpatialKind {
        match self {
            Self::UnsupportedSpatialKind { spatial_kind, .. } => *spatial_kind,
        }
    }

    #[inline]
    pub fn codec_chain(&self) -> &[EncodingFormat] {
        match self {
            Self::UnsupportedSpatialKind { codec_chain, .. } => codec_chain.as_slice(),
        }
    }
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSpatialKind {
                spatial_kind,
                color_kind,
                codec_chain,
            } => {
                let spatial_kind_str = match spatial_kind {
                    SpatialKind::Points => "Points",
                    SpatialKind::Splats => "Splats",
                };
                let color_kind_str = match color_kind {
                    ColorKind::Rgb8 => "RGB8",
                    ColorKind::Rgba8 => "RGBA8",
                };

                write!(f, "unsupported spatial kind {spatial_kind_str} with color kind {color_kind_str} for codec chain ")?;
                for (i, step) in codec_chain.iter().enumerate() {
                    if i != 0 {
                        write!(f, " -> ")?;
                    }
                    write!(f, "{}", step.as_str())?;
                }

                if *spatial_kind == SpatialKind::Splats {
                    write!(
                        f,
                        " (Splats encoding is supported by Bitcode, PLY, GSP16, and wrappers over those)"
                    )?;
                }

                Ok(())
            }
        }
    }
}

impl std::error::Error for CodecError {}
