//! Zstd wrapper encoder.

use std::io::Write;

/// Compress the result of `write_inner` using Zstandard and append it to `out`.
pub fn wrap_zstd_into<F>(
    mut write_inner: F, // writes inner payload into provided Vec
    level: Option<i32>,
    out: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut(&mut Vec<u8>) -> Result<(), Box<dyn std::error::Error>>,
{
    // 1) write 3-byte magic
    out.extend_from_slice(b"ZST");

    // 2) compress inner into `out` directly
    //    (stage inner to scratch.tmp to avoid re-reading from `out`)
    let mut inner = Vec::new();
    write_inner(&mut inner)?;
    let mut enc = zstd::stream::write::Encoder::new(out, level.unwrap_or(1))?;
    enc.write_all(&inner)?;
    enc.finish()?; // flushes into `out` (which includes the header we wrote)
    Ok(())
}
