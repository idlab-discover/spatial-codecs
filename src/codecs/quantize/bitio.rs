#[allow(dead_code)]
pub struct TurboBitWriter<'a> {
    ptr: *mut u8,
    len: usize,
    pos: usize,
    acc: u128,     // LSB-first accumulator
    acc_bits: u32, // valid bits
    _keep: core::marker::PhantomData<&'a mut Vec<u8>>,
}

impl<'a> TurboBitWriter<'a> {
    /// # Safety
    /// `start` must be a valid range inside `out`.
    #[inline(always)]
    pub unsafe fn new_at(out: &'a mut Vec<u8>, start: usize, len: usize) -> Self {
        let ptr = out.as_mut_ptr().add(start);
        Self {
            ptr,
            len,
            pos: 0,
            acc: 0,
            acc_bits: 0,
            _keep: core::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub fn write_masked(&mut self, value: u32, bits: u8, mask: u64) {
        if bits == 0 {
            return;
        }
        debug_assert!(bits <= 32);
        let v = (value as u128) & (mask as u128);
        self.acc |= v << self.acc_bits;
        self.acc_bits += bits as u32;

        while self.acc_bits >= 64 {
            debug_assert!(self.pos + 8 <= self.len);
            unsafe {
                core::ptr::write_unaligned(
                    self.ptr.add(self.pos) as *mut u64,
                    (self.acc as u64).to_le(),
                );
            }
            self.pos += 8;
            self.acc >>= 64;
            self.acc_bits -= 64;
        }
        while self.acc_bits >= 32 {
            debug_assert!(self.pos + 4 <= self.len);
            unsafe {
                core::ptr::write_unaligned(
                    self.ptr.add(self.pos) as *mut u32,
                    (self.acc as u32).to_le(),
                );
            }
            self.pos += 4;
            self.acc >>= 32;
            self.acc_bits -= 32;
        }
        while self.acc_bits >= 8 {
            debug_assert!(self.pos < self.len);
            unsafe {
                *self.ptr.add(self.pos) = self.acc as u8;
            }
            self.pos += 1;
            self.acc >>= 8;
            self.acc_bits -= 8;
        }
    }

    #[inline(always)]
    pub fn finish(mut self) {
        if self.acc_bits > 0 {
            debug_assert!(self.pos < self.len);
            unsafe {
                *self.ptr.add(self.pos) = self.acc as u8;
            }
            self.pos += 1;
        }
        debug_assert!(
            self.pos == self.len,
            "body size mismatch: wrote {} / {}",
            self.pos,
            self.len
        );
    }
}

#[derive(Clone)]
pub struct FastBitReader<'a> {
    ptr: *const u8,
    end: *const u8,
    acc: u128,
    acc_bits: u32,
    _keep: core::marker::PhantomData<&'a [u8]>,
}

impl<'a> FastBitReader<'a> {
    #[inline(always)]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            ptr: bytes.as_ptr(),
            end: unsafe { bytes.as_ptr().add(bytes.len()) },
            acc: 0,
            acc_bits: 0,
            _keep: core::marker::PhantomData,
        }
    }

    #[inline(always)]
    fn refill(&mut self, need: u32) {
        while self.acc_bits < need && self.ptr < self.end {
            let rem = (self.end as usize) - (self.ptr as usize);
            unsafe {
                if rem >= 8 {
                    let chunk = core::ptr::read_unaligned(self.ptr as *const u64).to_le();
                    self.ptr = self.ptr.add(8);
                    self.acc |= (chunk as u128) << self.acc_bits;
                    self.acc_bits += 64;
                } else {
                    let mut tmp = 0u64;
                    for i in 0..rem {
                        tmp |= (*self.ptr.add(i) as u64) << (8 * i);
                    }
                    self.ptr = self.ptr.add(rem);
                    self.acc |= (tmp as u128) << self.acc_bits;
                    self.acc_bits += (rem as u32) * 8;
                }
            }
        }
    }

    #[inline(always)]
    pub fn read(&mut self, bits: u8) -> Option<u32> {
        debug_assert!(bits <= 32);
        self.refill(bits as u32);
        if self.acc_bits < bits as u32 {
            return None;
        }
        let mask = if bits == 32 {
            u32::MAX
        } else {
            (1u32 << bits) - 1
        };
        let v = (self.acc as u32) & mask;
        self.acc >>= bits;
        self.acc_bits -= bits as u32;
        Some(v)
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn read_u8(&mut self) -> Option<u8> {
        self.refill(8);
        if self.acc_bits < 8 {
            return None;
        }
        let v = self.acc as u8;
        self.acc >>= 8;
        self.acc_bits -= 8;
        Some(v)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn bit_writer_reader_roundtrip_varied_bit_widths() {
        // Deterministic "random" pattern without RNG.
        let values: [u32; 8] = [
            0x0,
            0x1,
            0x3,
            0x7,
            0xFF,
            0x1234_5678,
            0xDEAD_BEEF,
            0xFFFF_FFFF,
        ];
        let widths: [u8; 7] = [1, 2, 3, 7, 8, 13, 32];

        for &bits in &widths {
            let mask: u64 = if bits == 32 {
                u32::MAX as u64
            } else {
                (1u64 << bits) - 1
            };
            let per_val_bits = bits as usize;
            let total_bits = values.len() * per_val_bits;
            let body_len = total_bits.div_ceil(8);

            let mut out = vec![0u8; body_len];
            unsafe {
                let mut w = TurboBitWriter::new_at(&mut out, 0, body_len);
                for &v in &values {
                    w.write_masked(v, bits, mask);
                }
                w.finish();
            }

            let mut r = FastBitReader::new(&out);
            for &v in &values {
                let got = r.read(bits).expect("should have bits");
                assert_eq!(got, (v & (mask as u32)));
            }
            assert_eq!(r.read(bits), None);
        }
    }

    #[test]
    fn read_u8_matches_byte_stream() {
        let bytes = [0x12u8, 0x34, 0x56];
        let mut r = FastBitReader::new(&bytes);
        assert_eq!(r.read_u8(), Some(0x12));
        assert_eq!(r.read_u8(), Some(0x34));
        assert_eq!(r.read_u8(), Some(0x56));
        assert_eq!(r.read_u8(), None);
    }
}
