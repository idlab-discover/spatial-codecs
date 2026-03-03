//! High‑level decode entry points expected by your central dispatcher.
use crate::codecs::ply::{ascii as asc, binary as bin, header, types::*};
use spatial_utils::{traits::SpatialSink, utils::point_scalar::PointScalar};
use crate::BasicResult;
use std::io::{self, BufReader, Cursor};

fn decode_into_flattened_vecs_impl<S>(
    data: &[u8],
    pos_out: &mut Vec<S>,
    color_out: &mut Vec<u8>,
) -> io::Result<()>
where
    S: PointScalar,
{
    let cur = Cursor::new(data);
    let br = BufReader::new(cur);
    let (hdr, mut br) = header::read_header(br)?; // returns the same BufRead back after header

    // Process elements in declared order to stay spec‑compliant.
    for elem in &hdr.elements {
        if elem.name == "vertex" {
            match hdr.encoding {
                PlyEncoding::Ascii => {
                    asc::read_vertices_into_flat(&mut br, elem, pos_out, color_out)?
                }
                PlyEncoding::BinaryLittleEndian => {
                    bin::read_vertices_into_flat(&mut br, elem, false, pos_out, color_out)?
                }
                PlyEncoding::BinaryBigEndian => {
                    bin::read_vertices_into_flat(&mut br, elem, true, pos_out, color_out)?
                }
            }
        } else {
            // Skip unknown elements efficiently while keeping stream alignment.
            match hdr.encoding {
                PlyEncoding::Ascii => asc::skip_element_lines(&mut br, elem)?,
                PlyEncoding::BinaryLittleEndian => {
                    bin::skip_element(&mut br, elem, false)?;
                }
                PlyEncoding::BinaryBigEndian => {
                    bin::skip_element(&mut br, elem, true)?;
                }
            }
        }
    }
    Ok(())
}

pub fn decode_into<P>(data: &[u8], out: &mut Vec<P>) -> BasicResult
where
    P: SpatialSink,
{
    let cur = Cursor::new(data);
    let br = BufReader::new(cur);
    let (hdr, mut br) = header::read_header(br)?;

    for elem in &hdr.elements {
        if elem.name == "vertex" {
            match hdr.encoding {
                PlyEncoding::Ascii => asc::read_vertices_into_target(&mut br, elem, out)?,
                PlyEncoding::BinaryLittleEndian => {
                    bin::read_vertices_into_target(&mut br, elem, false, out)?
                }
                PlyEncoding::BinaryBigEndian => {
                    bin::read_vertices_into_target(&mut br, elem, true, out)?
                }
            }
        } else {
            match hdr.encoding {
                PlyEncoding::Ascii => asc::skip_element_lines(&mut br, elem)?,
                PlyEncoding::BinaryLittleEndian => bin::skip_element(&mut br, elem, false)?,
                PlyEncoding::BinaryBigEndian => bin::skip_element(&mut br, elem, true)?,
            }
        }
    }
    Ok(())
}

pub fn decode_into_flattened_vecs<S>(
    data: &[u8],
    pos_out: &mut Vec<S>,
    color_out: &mut Vec<u8>,
) -> BasicResult
where
    S: PointScalar,
{
    decode_into_flattened_vecs_impl(data, pos_out, color_out)?;
    Ok(())
}
