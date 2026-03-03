#[inline(always)]
pub fn encode_delta(value: u32, prev: &mut u32, mask: u32) -> u32 {
    let delta = value.wrapping_sub(*prev) & mask;
    *prev = value & mask;
    delta
}

#[inline(always)]
pub fn decode_delta(delta: u32, prev: &mut u32, mask: u32) -> u32 {
    let value = prev.wrapping_add(delta) & mask;
    *prev = value;
    value
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn delta_roundtrip_basic() {
        let mask = 0xFFu32;
        let mut prev_enc = 0u32;
        let mut prev_dec = 0u32;

        for &v in &[0u32, 1, 2, 10, 255, 0, 5] {
            let d = encode_delta(v, &mut prev_enc, mask);
            let r = decode_delta(d, &mut prev_dec, mask);
            assert_eq!(r, v & mask);
        }
    }

    #[test]
    fn delta_wraps_with_mask() {
        let mask = 0x0Fu32; // 4 bits
        let mut prev_enc = 0u32;
        let mut prev_dec = 0u32;

        let seq = [0u32, 15u32, 0u32, 1u32, 14u32];
        for &v in &seq {
            let d = encode_delta(v, &mut prev_enc, mask);
            let r = decode_delta(d, &mut prev_dec, mask);
            assert_eq!(r, v & mask);
        }
    }
}
