//! Gzip wrapper codec.
//!
//! Emits a payload framed as `GZP | gzip(data…)`, where `data` is produced by an inner
//! codec. The wrapper is kept extremely small: no streaming state beyond a temporary
//! buffer for the inner payload.

use flate2::{write::GzEncoder, Compression};
use std::io::Write;

/// Compress the result of `write_inner` with gzip and append it to `out`.
pub fn wrap_gzip_into<F>(
    mut write_inner: F,
    level: Option<u32>,
    out: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut(&mut Vec<u8>) -> Result<(), Box<dyn std::error::Error>>,
{
    out.extend_from_slice(b"GZP");
    let mut inner = Vec::new();
    write_inner(&mut inner)?;
    let mut enc = GzEncoder::new(out, Compression::new(level.unwrap_or(6)));
    enc.write_all(&inner)?;
    enc.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {

    use super::*;
    use flate2::read::GzDecoder;
    use std::io::Read;

    #[test]
    fn gzip_wrapper_frames_and_decompresses_inner_bytes() {
        let mut out = Vec::new();
        wrap_gzip_into(
            |buf| {
                buf.extend_from_slice(b"hello");
                Ok(())
            },
            Some(6),
            &mut out,
        )
        .unwrap();
        assert_eq!(&out[..3], b"GZP");

        let mut dec = GzDecoder::new(&out[3..]);
        let mut inner = Vec::new();
        dec.read_to_end(&mut inner).unwrap();
        assert_eq!(inner, b"hello");
    }
}
