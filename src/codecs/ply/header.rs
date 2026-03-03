//! ASCII header parser & writer. Liberal in what it accepts; strict in what it writes.
use crate::codecs::ply::types::*;
use std::io::{self, BufRead, Write};

pub fn read_header<R: BufRead>(mut r: R) -> io::Result<(Header, R)> {
    // PLY header is ASCII lines terminated by `end_header`.
    // We return the reader back so the caller can continue after the header.
    let mut first = String::new();
    r.read_line(&mut first)?;
    if first.trim() != "ply" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "missing magic 'ply'",
        ));
    }

    let mut encoding = PlyEncoding::Ascii; // default until proven otherwise
    let mut version = "1.0".to_string();
    let mut comments = Vec::new();
    let mut obj_info = Vec::new();
    let mut elements: Vec<ElementDef> = Vec::new();
    let mut cur_elem: Option<ElementDef> = None;

    let mut line = String::new();
    loop {
        line.clear();
        if r.read_line(&mut line)? == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "header EOF"));
        }
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if t == "end_header" {
            if let Some(e) = cur_elem.take() {
                elements.push(e)
            };
            break;
        }

        if let Some((enc, ver)) = PlyEncoding::from_format_line(t) {
            encoding = enc;
            version = ver;
            continue;
        }
        if let Some(rest) = t.strip_prefix("comment ") {
            comments.push(rest.to_string());
            continue;
        }
        if let Some(rest) = t.strip_prefix("obj_info ") {
            obj_info.push(rest.to_string());
            continue;
        }

        if let Some(rest) = t.strip_prefix("element ") {
            if let Some(e) = cur_elem.take() {
                elements.push(e);
            }
            let mut it = rest.split_whitespace();
            let name = it
                .next()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad element"))?
                .to_string();
            let count: usize = it
                .next()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad element count"))?
                .parse()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad element count"))?;
            cur_elem = Some(ElementDef {
                name,
                count,
                properties: Vec::new(),
            });
            continue;
        }
        if let Some(rest) = t.strip_prefix("property ") {
            let e = cur_elem.as_mut().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "property without element")
            })?;
            let mut it = rest.split_whitespace();
            let head = it
                .next()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad property"))?;
            let ty = if head == "list" {
                let ct = ScalarType::from_name(it.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "bad list count type")
                })?)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "unknown list count type")
                })?;
                let itp = ScalarType::from_name(it.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "bad list item type")
                })?)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "unknown list item type")
                })?;
                PropertyType::List {
                    count: ct,
                    item: itp,
                }
            } else {
                PropertyType::Scalar(ScalarType::from_name(head).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "unknown scalar type")
                })?)
            };
            let name = it
                .next()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing property name"))?
                .to_string();
            e.properties.push(PropertyDef { name, ty });
            continue;
        }
        // Unknown header line -> ignore (robustness)
    }

    let hdr = Header {
        encoding,
        version,
        comments,
        obj_info,
        elements,
    };
    Ok((hdr, r))
}

pub struct WriteHeaderOpts<'a> {
    pub encoding: PlyEncoding,
    pub version: &'a str,
    pub comments: &'a [String],
    pub vertex_properties: &'a [PropertyDef],
    pub vertex_count: usize,
}

pub fn write_minimal_header<W: Write>(mut w: W, opts: WriteHeaderOpts<'_>) -> io::Result<()> {
    writeln!(w, "ply")?;
    writeln!(
        w,
        "format {} {}",
        opts.encoding.as_header_str(),
        opts.version
    )?;
    for c in opts.comments {
        writeln!(w, "comment {c}")?;
    }
    writeln!(w, "element vertex {}", opts.vertex_count)?;
    for p in opts.vertex_properties {
        writeln!(
            w,
            "property {} {}",
            match &p.ty {
                PropertyType::Scalar(s) => s.display_name().to_string(),
                PropertyType::List { count, item } =>
                    format!("list {} {}", count.display_name(), item.display_name()),
            },
            p.name
        )?;
    }
    // We purposely omit other elements; many consumers expect vertex‑only point clouds.
    writeln!(w, "end_header")
}

#[cfg(test)]
mod tests {

    use super::*;
    use std::io::Cursor;

    #[test]
    fn read_header_rejects_missing_magic() {
        let input = Cursor::new(b"notply\nformat ascii 1.0\nend_header\n".to_vec());
        assert!(read_header(input).is_err());
    }

    #[test]
    fn read_header_parses_vertex_element_and_properties() {
        let hdr_txt = b"ply
format ascii 1.0
comment hello
element vertex 2
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
";
        let input = Cursor::new(hdr_txt.to_vec());
        let (hdr, _r) = read_header(input).unwrap();
        assert_eq!(hdr.encoding, PlyEncoding::Ascii);
        assert_eq!(hdr.elements.len(), 1);
        let v = &hdr.elements[0];
        assert_eq!(v.name, "vertex");
        assert_eq!(v.count, 2);
        assert_eq!(v.properties.len(), 6);
        assert_eq!(hdr.comments, vec!["hello".to_string()]);
    }

    #[test]
    fn write_minimal_header_emits_expected_lines() {
        let props = vec![
            PropertyDef {
                name: "x".to_string(),
                ty: PropertyType::Scalar(ScalarType::Float),
            },
            PropertyDef {
                name: "y".to_string(),
                ty: PropertyType::Scalar(ScalarType::Float),
            },
            PropertyDef {
                name: "z".to_string(),
                ty: PropertyType::Scalar(ScalarType::Float),
            },
        ];
        let mut out = Vec::new();
        write_minimal_header(
            &mut out,
            WriteHeaderOpts {
                encoding: PlyEncoding::Ascii,
                version: "1.0",
                comments: &[],
                vertex_properties: &props,
                vertex_count: 42,
            },
        )
        .unwrap();

        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("ply\n"));
        assert!(s.contains("format ascii 1.0\n"));
        assert!(s.contains("element vertex 42\n"));
        assert!(s.contains("property float x\n"));
        assert!(s.ends_with("end_header\n"));
    }
}
