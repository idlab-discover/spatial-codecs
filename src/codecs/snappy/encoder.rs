//! Snappy-frame wrapper codec.

use std::io::Write;

/// Compress the result of `write_inner` using Snappy framing and append it to `out`.
pub fn wrap_snappy_into<F>(
    mut write_inner: F,
    out: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut(&mut Vec<u8>) -> Result<(), Box<dyn std::error::Error>>,
{
    out.extend_from_slice(b"SNP");
    let mut inner = Vec::new();
    write_inner(&mut inner)?;
    let mut enc = snap::write::FrameEncoder::new(out);
    enc.write_all(&inner)?;
    enc.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {

    use super::*;
    use std::io::Read;

    #[test]
    fn snappy_wrapper_frames_and_decompresses_inner_bytes() {
        let mut out = Vec::new();
        wrap_snappy_into(
            |buf| {
                buf.extend_from_slice(b"hello");
                Ok(())
            },
            &mut out,
        )
        .unwrap();
        assert_eq!(&out[..3], b"SNP");

        let mut dec = snap::read::FrameDecoder::new(&out[3..]);
        let mut inner = Vec::new();
        dec.read_to_end(&mut inner).unwrap();
        assert_eq!(inner, b"hello");
    }
}
