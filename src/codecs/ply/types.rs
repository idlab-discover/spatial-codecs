//! Core PLY types (encoding + header model) and small utilities.
use std::fmt;

use serde::{Deserialize, Serialize};

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum PlyEncoding {
    Ascii = 0,
    BinaryLittleEndian,
    BinaryBigEndian,
}

impl PlyEncoding {
    pub fn from_format_line(s: &str) -> Option<(Self, String)> {
        // Accept lines like: "format ascii 1.0" (extra spaces allowed)
        let mut it = s.split_whitespace();
        if it.next()? != "format" {
            return None;
        }
        let enc = match it.next()? {
            "ascii" => PlyEncoding::Ascii,
            "binary_little_endian" => PlyEncoding::BinaryLittleEndian,
            "binary_big_endian" => PlyEncoding::BinaryBigEndian,
            _ => return None,
        };
        let ver = it.next().unwrap_or("1.0").to_string();
        Some((enc, ver))
    }

    pub fn as_header_str(&self) -> &'static str {
        match self {
            PlyEncoding::Ascii => "ascii",
            PlyEncoding::BinaryLittleEndian => "binary_little_endian",
            PlyEncoding::BinaryBigEndian => "binary_big_endian",
        }
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ScalarType {
    Char = 0,
    UChar,
    Short,
    UShort,
    Int,
    UInt,
    Float,
    Double,
}

impl ScalarType {
    #[inline]
    pub fn size_of(self) -> usize {
        use ScalarType::*;
        match self {
            Char | UChar => 1,
            Short | UShort => 2,
            Int | UInt | Float => 4,
            Double => 8,
        }
    }
    #[inline]
    pub fn from_name(name: &str) -> Option<Self> {
        use ScalarType::*;
        match name.to_ascii_lowercase().as_str() {
            "char" | "int8" => Some(Char),
            "uchar" | "uint8" => Some(UChar),
            "short" | "int16" => Some(Short),
            "ushort" | "uint16" => Some(UShort),
            "int" | "int32" => Some(Int),
            "uint" | "uint32" => Some(UInt),
            "float" | "float32" => Some(Float),
            "double" | "float64" => Some(Double),
            _ => None,
        }
    }
    #[inline]
    pub fn display_name(self) -> &'static str {
        use ScalarType::*;
        match self {
            Char => "char",
            UChar => "uchar",
            Short => "short",
            UShort => "ushort",
            Int => "int",
            UInt => "uint",
            Float => "float",
            Double => "double",
        }
    }
}

#[repr(u8)]
// add near PlyEncoding / ScalarType
#[derive(Copy, Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AsciiFloatMode {
    /// Fixed digits after the decimal, e.g. 6
    Fixed(usize) = 0,
    /// Shortest round-trip (minimal bytes, fast)
    Shortest = 1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PropertyType {
    Scalar(ScalarType),
    List { count: ScalarType, item: ScalarType },
}

#[derive(Clone, Debug)]
pub struct PropertyDef {
    pub name: String,
    pub ty: PropertyType,
}

#[derive(Clone, Debug)]
pub struct ElementDef {
    pub name: String,
    pub count: usize,
    pub properties: Vec<PropertyDef>,
}

#[derive(Clone, Debug)]
pub struct Header {
    pub encoding: PlyEncoding,
    pub version: String,
    pub comments: Vec<String>,
    pub obj_info: Vec<String>,
    pub elements: Vec<ElementDef>,
}

impl Header {
    pub fn find_element(&self, name: &str) -> Option<&ElementDef> {
        self.elements.iter().find(|e| e.name == name)
    }
}

/// Helper describing how to pick vertex indices quickly.
#[derive(Copy, Clone, Debug, Default)]
pub struct VertexPropIndex {
    pub ix: Option<usize>,
    pub iy: Option<usize>,
    pub iz: Option<usize>,
    pub ir: Option<usize>,
    pub ig: Option<usize>,
    pub ib: Option<usize>,
    pub ia: Option<usize>,
}

impl VertexPropIndex {
    pub fn from_element(e: &ElementDef) -> Self {
        let mut out = VertexPropIndex::default();
        for (i, p) in e.properties.iter().enumerate() {
            let name = p.name.as_str();
            // Accept common aliases (r/g/b). Add more if you see them in the wild.
            match name {
                "x" => out.ix = Some(i),
                "y" => out.iy = Some(i),
                "z" => out.iz = Some(i),
                "red" | "r" => out.ir = Some(i),
                "green" | "g" => out.ig = Some(i),
                "blue" | "b" => out.ib = Some(i),
                "alpha" | "a" | "opacity" => out.ia = Some(i),
                _ => {}
            }
        }
        out
    }
    pub fn has_xyz(&self) -> bool {
        self.ix.is_some() && self.iy.is_some() && self.iz.is_some()
    }

    pub fn has_rgb(&self) -> bool {
        self.ir.is_some() && self.ig.is_some() && self.ib.is_some()
    }

    pub fn has_alpha(&self) -> bool {
        self.ia.is_some()
    }
}

/// Helper describing how to pick Gaussian splat vertex indices quickly.
///
/// This targets the common 3D Gaussian Splatting PLY schemas, e.g.:
/// - x y z f_dc_0 f_dc_1 f_dc_2 opacity rot_0..3 scale_0..2
/// - x y z nx ny nz f_dc_0..2 opacity scale_0..2 rot_0..3
///   plus optional trailing `f_rest_*` (ignored by the decoder for now).
#[derive(Copy, Clone, Debug, Default)]
pub struct SplatPropIndex {
    pub ix: Option<usize>,
    pub iy: Option<usize>,
    pub iz: Option<usize>,
    pub if_dc0: Option<usize>,
    pub if_dc1: Option<usize>,
    pub if_dc2: Option<usize>,
    pub iopacity: Option<usize>,
    pub irot0: Option<usize>,
    pub irot1: Option<usize>,
    pub irot2: Option<usize>,
    pub irot3: Option<usize>,
    pub iscale0: Option<usize>,
    pub iscale1: Option<usize>,
    pub iscale2: Option<usize>,
}

impl SplatPropIndex {
    /// Build an index map by property name (single pass; no allocations).
    #[inline]
    pub fn from_element(e: &ElementDef) -> Self {
        let mut out = SplatPropIndex::default();
        for (i, p) in e.properties.iter().enumerate() {
            match p.name.as_str() {
                "x" => out.ix = Some(i),
                "y" => out.iy = Some(i),
                "z" => out.iz = Some(i),
                "f_dc_0" => out.if_dc0 = Some(i),
                "f_dc_1" => out.if_dc1 = Some(i),
                "f_dc_2" => out.if_dc2 = Some(i),
                "opacity" => out.iopacity = Some(i),
                "rot_0" => out.irot0 = Some(i),
                "rot_1" => out.irot1 = Some(i),
                "rot_2" => out.irot2 = Some(i),
                "rot_3" => out.irot3 = Some(i),
                "scale_0" => out.iscale0 = Some(i),
                "scale_1" => out.iscale1 = Some(i),
                "scale_2" => out.iscale2 = Some(i),
                _ => {}
            }
        }
        out
    }

    /// True if the element contains the minimum set needed to decode splats.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.ix.is_some()
            && self.iy.is_some()
            && self.iz.is_some()
            && self.if_dc0.is_some()
            && self.if_dc1.is_some()
            && self.if_dc2.is_some()
            && self.iopacity.is_some()
            && self.irot0.is_some()
            && self.irot1.is_some()
            && self.irot2.is_some()
            && self.irot3.is_some()
            && self.iscale0.is_some()
            && self.iscale1.is_some()
            && self.iscale2.is_some()
    }

    /// Largest property index that must be parsed to obtain all required fields.
    /// Used to early-cutoff loops when there are trailing `f_rest_*` props.
    #[inline]
    pub fn max_required_prop_index(&self) -> Option<usize> {
        if !self.is_complete() {
            return None;
        }
        let mut m = 0usize;
        for v in [
            self.ix,
            self.iy,
            self.iz,
            self.if_dc0,
            self.if_dc1,
            self.if_dc2,
            self.iopacity,
            self.irot0,
            self.irot1,
            self.irot2,
            self.irot3,
            self.iscale0,
            self.iscale1,
            self.iscale2,
        ] {
            m = m.max(v.unwrap());
        }
        Some(m)
    }
}

impl fmt::Display for PropertyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PropertyType::Scalar(s) => write!(f, "{}", s.display_name()),
            PropertyType::List { count, item } => {
                write!(f, "list {} {}", count.display_name(), item.display_name())
            }
        }
    }
}
