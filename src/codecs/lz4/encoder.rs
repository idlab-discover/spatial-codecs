//! LZ4 frame wrapper codec.

use lz4_flex::frame::FrameEncoder;
use std::io::Write;

/// Compress the result of `write_inner` using LZ4-frame and append it to `out`.
pub fn wrap_lz4_into<F>(
    mut write_inner: F,
    out: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut(&mut Vec<u8>) -> Result<(), Box<dyn std::error::Error>>,
{
    out.extend_from_slice(b"LZ4");
    let mut inner = Vec::new();
    write_inner(&mut inner)?;
    let mut enc = FrameEncoder::new(out);
    enc.write_all(&inner)?;
    enc.finish()?;
    Ok(())
}
