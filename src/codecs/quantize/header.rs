use spatial_utils::utils::point_scalar::PointScalar;

use super::types::*;

pub struct ParsedHeader<S> {
    pub position_bits: u8,
    pub color_bits: u8, // if palette, this holds palette index bits
    pub point_count: usize,
    pub mins: [S; 3],
    pub maxs: [S; 3],
    pub color_mins: [u8; 3],
    pub color_maxs: [u8; 3],
    pub palette: Vec<[u8; 3]>,
    pub has_palette: bool,
    pub flags: u8,
    pub offset: usize,
}

pub fn write_header_v2<S>(
    out: &mut Vec<u8>,
    position_bits: u8,
    color_bits_header: u8, // palette + flags indicator already applied by caller
    flags: Option<u8>,
    point_count: u32,
    mins: [S; 3],
    maxs: [S; 3],
    color_mins: [u8; 3],
    color_maxs: [u8; 3],
    palette: Option<&[[u8; 3]]>,
) where
    S: PointScalar,
{
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.push(position_bits);
    out.push(color_bits_header);
    if let Some(f) = flags {
        out.push(f);
    }
    out.extend_from_slice(&point_count.to_le_bytes());
    for &v in &mins {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &maxs {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&color_mins);
    out.extend_from_slice(&color_maxs);
    if let Some(pal) = palette {
        let len = pal.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        for rgb in pal {
            out.extend_from_slice(rgb);
        }
    }
}

pub fn parse_header<S>(data: &[u8]) -> Result<ParsedHeader<S>, Box<dyn std::error::Error>>
where
    S: PointScalar,
{
    if data.len() < HEADER_FIXED_SIZE {
        return Err("Quantize decoder: buffer too small for header".into());
    }
    if &data[0..3] != MAGIC {
        return Err("Quantize decoder: invalid magic".into());
    }

    let mut off = MAGIC.len();
    let ver = data[off];
    off += 1;

    let mut palette_flag = false;
    let mut has_extra_flags = false;
    let (pos_bits, col_bits) = match ver {
        1 => {
            let p = data[off];
            off += 1;
            (p, p)
        }
        2 => {
            if data.len() < HEADER_EXTENDED_SIZE {
                return Err("Quantize decoder: buffer too small for v2 header".into());
            }
            let p = data[off];
            let raw = data[off + 1];
            off += 2;
            palette_flag = (raw & super::types::COLOR_BITFLAG_PALETTE) != 0;
            has_extra_flags = (raw & super::types::COLOR_BITFLAG_HAS_FLAGS) != 0;
            (p, raw & super::types::COLOR_BITS_MASK)
        }
        _ => return Err("Quantize decoder: unsupported version".into()),
    };

    if pos_bits == 0 || pos_bits > 32 {
        return Err("Quantize decoder: invalid position bits".into());
    }
    if !palette_flag && (col_bits == 0 || col_bits > 32) {
        return Err("Quantize decoder: invalid color bits".into());
    }
    if palette_flag && col_bits > 32 {
        return Err("Quantize decoder: invalid color bits".into());
    }

    let mut flags_byte = 0u8;
    if has_extra_flags {
        if data.len() <= off {
            return Err("Quantize decoder: missing flags byte".into());
        }
        flags_byte = data[off];
        off += 1;
    }

    if data.len() < off + 4 {
        return Err("Quantize decoder: truncated point count".into());
    }
    let mut u4 = [0u8; 4];
    u4.copy_from_slice(&data[off..off + 4]);
    let point_count = u32::from_le_bytes(u4) as usize;
    off += 4;

    let mut mins = [S::ZERO; 3];
    let mut maxs = [S::ZERO; 3];
    for i in 0..3 {
        // This from le bytes uses strictly 4 bytes, never 8, even when S = f64
        let mut b = [0u8; 4];
        b.copy_from_slice(&data[off..off + 4]);
        mins[i] = S::from_le_bytes(b);
        off += 4;
    }
    for i in 0..3 {
        // This from le bytes uses strictly 4 bytes, never 8, even when S = f64
        let mut b = [0u8; 4];
        b.copy_from_slice(&data[off..off + 4]);
        maxs[i] = S::from_le_bytes(b);
        off += 4;
    }

    if data.len() < off + 6 {
        return Err("Quantize decoder: truncated color mins/maxs".into());
    }
    let mut cmins = [0u8; 3];
    cmins.copy_from_slice(&data[off..off + 3]);
    off += 3;
    let mut cmaxs = [0u8; 3];
    cmaxs.copy_from_slice(&data[off..off + 3]);
    off += 3;

    let mut palette = Vec::new();
    if palette_flag {
        if data.len() < off + PALETTE_LEN_FIELD_SIZE {
            return Err("Quantize decoder: truncated palette length".into());
        }
        let mut l2 = [0u8; 2];
        l2.copy_from_slice(&data[off..off + 2]);
        off += 2;
        let plen = u16::from_le_bytes(l2) as usize;
        if plen == 0 {
            return Err("Quantize decoder: palette length cannot be zero".into());
        }
        let bytes = plen
            .checked_mul(3)
            .ok_or("Quantize decoder: palette length overflow")?;
        if data.len() < off + bytes {
            return Err("Quantize decoder: truncated palette entries".into());
        }
        palette.reserve(plen);
        for i in 0..plen {
            let base = off + i * 3;
            palette.push([data[base], data[base + 1], data[base + 2]]);
        }
        off += bytes;
    }

    Ok(ParsedHeader {
        position_bits: pos_bits,
        color_bits: col_bits,
        point_count,
        mins,
        maxs,
        color_mins: cmins,
        color_maxs: cmaxs,
        palette,
        has_palette: palette_flag,
        flags: flags_byte,
        offset: off,
    })
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn parse_rejects_invalid_magic() {
        let data = b"BAD\x02\x10\x10\0\0\0\0";
        let res = parse_header::<f32>(data);
        assert!(res.is_err());
    }

    #[test]
    fn v2_header_with_flags_and_palette_parses() {
        let mins = [0.0f32, 1.0, 2.0];
        let maxs = [3.0f32, 4.0, 5.0];
        let palette: [[u8; 3]; 2] = [[1, 2, 3], [4, 5, 6]];
        let mut out = Vec::new();

        let color_bits_header =
            (8u8 & COLOR_BITS_MASK) | COLOR_BITFLAG_PALETTE | COLOR_BITFLAG_HAS_FLAGS;
        write_header_v2(
            &mut out,
            12,
            color_bits_header,
            Some(0b1010_0001),
            123,
            mins,
            maxs,
            [0, 0, 0],
            [255, 255, 255],
            Some(&palette),
        );

        let ph = parse_header::<f32>(&out).expect("parse header");
        assert_eq!(ph.position_bits, 12);
        assert_eq!(ph.color_bits, 8);
        assert_eq!(ph.point_count, 123);
        assert!(ph.has_palette);
        assert_eq!(ph.flags, 0b1010_0001);
        assert_eq!(ph.palette, palette.to_vec());
        assert_eq!(ph.mins, mins);
        assert_eq!(ph.maxs, maxs);
        assert!(ph.offset <= out.len());
    }

    #[test]
    fn v2_palette_length_zero_rejected() {
        // Build a minimal v2 header by hand with palette flag + zero length.
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.push(2); // version
        out.push(10); // pos bits
        out.push((8u8 & COLOR_BITS_MASK) | COLOR_BITFLAG_PALETTE); // palette flag
        out.extend_from_slice(&0u32.to_le_bytes()); // point_count
        out.extend_from_slice(&0.0f32.to_le_bytes()); // mins x
        out.extend_from_slice(&0.0f32.to_le_bytes()); // mins y
        out.extend_from_slice(&0.0f32.to_le_bytes()); // mins z
        out.extend_from_slice(&0.0f32.to_le_bytes()); // maxs x
        out.extend_from_slice(&0.0f32.to_le_bytes()); // maxs y
        out.extend_from_slice(&0.0f32.to_le_bytes()); // maxs z
        out.extend_from_slice(&[0u8; 3]); // cmins
        out.extend_from_slice(&[255u8; 3]); // cmaxs
        out.extend_from_slice(&0u16.to_le_bytes()); // palette length = 0

        assert!(parse_header::<f32>(&out).is_err());
    }
}
