//! Small, defensive byte cursor for parsing binary formats.
//!
//! Many codecs need the same "read u8/u32/f32/take" helpers. Keeping them here avoids
//! copy-paste across codecs and reduces parsing bugs.

use core::mem;

use spatial_utils::utils::point_scalar::PointScalar;

/// Cursor over a byte slice with bounds-checked reads.
#[derive(Clone, Copy)]
pub(crate) struct ByteCursor<'a> {
    cur: &'a [u8],
}

impl<'a> ByteCursor<'a> {
    #[inline]
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { cur: bytes }
    }

    #[inline]
    pub(crate) fn remaining(&self) -> usize {
        self.cur.len()
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn peek(&self, n: usize) -> Option<&'a [u8]> {
        if self.cur.len() < n {
            None
        } else {
            Some(&self.cur[..n])
        }
    }

    #[inline]
    pub(crate) fn read_u8(&mut self) -> Result<u8, &'static str> {
        if self.cur.is_empty() {
            return Err("truncated (u8)");
        }
        let v = self.cur[0];
        self.cur = &self.cur[1..];
        Ok(v)
    }

    #[inline]
    pub(crate) fn read_u32_le(&mut self) -> Result<u32, &'static str> {
        if self.cur.len() < 4 {
            return Err("truncated (u32)");
        }
        let v = u32::from_le_bytes(self.cur[..4].try_into().unwrap());
        self.cur = &self.cur[4..];
        Ok(v)
    }

    #[inline]
    pub(crate) fn read_f32_le(&mut self) -> Result<f32, &'static str> {
        if self.cur.len() < mem::size_of::<f32>() {
            return Err("truncated (f32)");
        }
        let v = f32::from_le_bytes(self.cur[..4].try_into().unwrap());
        self.cur = &self.cur[4..];
        Ok(v)
    }

    #[inline]
    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8], &'static str> {
        if self.cur.len() < n {
            return Err("truncated (take)");
        }
        let (h, t) = self.cur.split_at(n);
        self.cur = t;
        Ok(h)
    }
}

#[inline]
pub(crate) fn write_u8(out: &mut Vec<u8>, v: u8) {
    out.push(v);
}

#[inline]
pub(crate) fn write_u32_le(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
#[allow(dead_code)]
pub(crate) fn write_f32_le(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[inline]
pub(crate) fn write_scalar_le<S>(out: &mut Vec<u8>, v: S)
where
    S: PointScalar,
{
    out.extend_from_slice(&v.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_reads_and_advances() {
        let bytes = [0x01u8, 0x02, 0x03, 0x04, 0xAA, 0xBB, 0xCC, 0xDD];
        let mut c = ByteCursor::new(&bytes);

        assert_eq!(c.remaining(), 8);
        assert_eq!(c.peek(2).unwrap(), &[0x01, 0x02]);

        assert_eq!(c.read_u8().unwrap(), 0x01);
        assert_eq!(c.read_u32_le().unwrap(), 0xAA040302u32);
        // After reading u8 + u32: consumed 5 bytes.
        assert_eq!(c.remaining(), 3);

        let tail = c.take(3).unwrap();
        assert_eq!(tail, &[0xBB, 0xCC, 0xDD]);
        assert_eq!(c.remaining(), 0);
    }

    #[test]
    fn cursor_detects_truncation() {
        let bytes = [0x01u8, 0x02, 0x03];
        let mut c = ByteCursor::new(&bytes);

        assert!(c.read_u32_le().is_err());
        assert_eq!(c.read_u8().unwrap(), 0x01);
        assert!(c.read_f32_le().is_err());
        assert!(c.take(10).is_err());
    }

    #[test]
    fn writers_append_expected_le_bytes() {
        let mut out = Vec::new();
        write_u8(&mut out, 0x7F);
        write_u32_le(&mut out, 0x11223344);
        assert_eq!(&out[..], &[0x7F, 0x44, 0x33, 0x22, 0x11]);
    }
}
