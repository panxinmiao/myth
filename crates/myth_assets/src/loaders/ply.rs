//! PLY loader for 3D Gaussian Splatting point clouds.
//!
//! Parses the binary PLY format exported by standard 3DGS training pipelines
//! (e.g. the original "3D Gaussian Splatting" by Kerbl et al.) and produces
//! a [`GaussianCloud`] asset ready for GPU upload.
//!
//! # Supported Properties
//!
//! | PLY property      | Interpretation |
//! |--------------------|---------------|
//! | `x`, `y`, `z`     | Position      |
//! | `nx`, `ny`, `nz`  | Normals (skipped) |
//! | `f_dc_0..2`       | SH degree-0 (base colour) |
//! | `f_rest_*`         | Higher-order SH coefficients |
//! | `opacity`          | Raw opacity (sigmoid-activated on load) |
//! | `scale_0..2`       | Log-space scale (exponentiated on load) |
//! | `rot_0..3`         | Quaternion rotation |

use std::io::{BufRead, BufReader, Read, Seek};

use glam::Vec3;
use half::f16;
use myth_core::{AssetError, Error, Result};
use myth_resources::gaussian_splat::{GaussianCloud, GaussianSHCoefficients, GaussianSplat};
use myth_resources::image::ColorSpace;

// ─── PLY Header Parsing ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyFormat {
    BinaryLittleEndian,
    BinaryBigEndian,
}

struct PlyHeader {
    format: PlyFormat,
    vertex_count: usize,
    /// Number of SH coefficients per channel (including DC).
    num_sh_coeffs: usize,
    /// Byte size of a single vertex element in the binary stream.
    vertex_byte_size: usize,
    property_offsets: PropertyOffsets,
}

/// Byte offsets of known properties within a single vertex element.
struct PropertyOffsets {
    x: usize,
    opacity: usize,
    scale_0: usize,
    rot_0: usize,
    f_dc_0: usize,
    f_rest_start: usize,
}

fn parse_header<R: BufRead>(reader: &mut R) -> Result<PlyHeader> {
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| Error::Asset(AssetError::Io(e)))?;
    if !line.trim().starts_with("ply") {
        return Err(Error::Asset(AssetError::Format("not a PLY file".into())));
    }

    let mut format = None;
    let mut vertex_count = 0usize;
    let mut in_vertex_element = false;
    let mut properties: Vec<(String, usize)> = Vec::new(); // (name, byte_size)
    let mut current_offset = 0usize;

    loop {
        line.clear();
        reader
            .read_line(&mut line)
            .map_err(|e| Error::Asset(AssetError::Io(e)))?;
        let trimmed = line.trim();

        if trimmed == "end_header" {
            break;
        }

        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "format" => {
                format = match parts.get(1) {
                    Some(&"binary_little_endian") => Some(PlyFormat::BinaryLittleEndian),
                    Some(&"binary_big_endian") => Some(PlyFormat::BinaryBigEndian),
                    _ => {
                        return Err(Error::Asset(AssetError::Format(
                            "only binary PLY formats are supported".into(),
                        )));
                    }
                };
            }
            "element" if parts.get(1) == Some(&"vertex") => {
                vertex_count = parts
                    .get(2)
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| Error::Asset(AssetError::Format("bad vertex count".into())))?;
                in_vertex_element = true;
                current_offset = 0;
                properties.clear();
            }
            "element" => {
                in_vertex_element = false;
            }
            "property" if in_vertex_element => {
                let type_name = parts.get(1).unwrap_or(&"");
                let prop_name = parts.get(2).unwrap_or(&"").to_string();
                let byte_size = match *type_name {
                    "float" | "float32" => 4,
                    "double" | "float64" => 8,
                    "uchar" | "uint8" => 1,
                    "short" | "int16" => 2,
                    "ushort" | "uint16" => 2,
                    "int" | "int32" => 4,
                    "uint" | "uint32" => 4,
                    _ => 4, // default assumption
                };
                properties.push((prop_name, current_offset));
                current_offset += byte_size;
            }
            _ => {}
        }
    }

    let format =
        format.ok_or_else(|| Error::Asset(AssetError::Format("missing format line".into())))?;

    let vertex_byte_size = current_offset;

    // Locate known properties
    let find = |name: &str| -> Result<usize> {
        properties
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, off)| *off)
            .ok_or_else(|| Error::Asset(AssetError::Format(format!("missing property: {name}"))))
    };

    let x = find("x")?;
    let opacity = find("opacity")?;
    let scale_0 = find("scale_0")?;
    let rot_0 = find("rot_0")?;
    let f_dc_0 = find("f_dc_0")?;

    // Count SH rest coefficients
    let num_rest = properties
        .iter()
        .filter(|(n, _)| n.starts_with("f_rest_"))
        .count();
    let num_sh_per_channel = 1 + num_rest / 3;

    let f_rest_start = if num_rest > 0 {
        properties
            .iter()
            .find(|(n, _)| n == "f_rest_0")
            .map(|(_, off)| *off)
            .unwrap_or(f_dc_0 + 12) // 3 floats after f_dc
    } else {
        f_dc_0 + 12
    };

    Ok(PlyHeader {
        format,
        vertex_count,
        num_sh_coeffs: num_sh_per_channel,
        vertex_byte_size,
        property_offsets: PropertyOffsets {
            x,
            opacity,
            scale_0,
            rot_0,
            f_dc_0,
            f_rest_start,
        },
    })
}

// ─── Math Utilities ────────────────────────────────────────────────────────

/// Numerically stable sigmoid activation.
#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Builds the upper triangle of the 3×3 covariance matrix from a normalised
/// quaternion `q` and log-space scale `s`.
///
/// Returns `[c00, c01, c02, c11, c12, c22]`.
pub(crate) fn build_covariance(q: [f32; 4], s: [f32; 3]) -> [f32; 6] {
    // Rotation matrix from quaternion (w, x, y, z)
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let r00 = 1.0 - 2.0 * (y * y + z * z);
    let r01 = 2.0 * (x * y - w * z);
    let r02 = 2.0 * (x * z + w * y);
    let r10 = 2.0 * (x * y + w * z);
    let r11 = 1.0 - 2.0 * (x * x + z * z);
    let r12 = 2.0 * (y * z - w * x);
    let r20 = 2.0 * (x * z - w * y);
    let r21 = 2.0 * (y * z + w * x);
    let r22 = 1.0 - 2.0 * (x * x + y * y);

    // L = R * S  (S is diagonal)
    let l00 = r00 * s[0];
    let l01 = r01 * s[1];
    let l02 = r02 * s[2];
    let l10 = r10 * s[0];
    let l11 = r11 * s[1];
    let l12 = r12 * s[2];
    let l20 = r20 * s[0];
    let l21 = r21 * s[1];
    let l22 = r22 * s[2];

    // M = L * L^T  (symmetric)
    let m00 = l00 * l00 + l01 * l01 + l02 * l02;
    let m01 = l00 * l10 + l01 * l11 + l02 * l12;
    let m02 = l00 * l20 + l01 * l21 + l02 * l22;
    let m11 = l10 * l10 + l11 * l11 + l12 * l12;
    let m12 = l10 * l20 + l11 * l21 + l12 * l22;
    let m22 = l20 * l20 + l21 * l21 + l22 * l22;

    [m00, m01, m02, m11, m12, m22]
}

/// Packs two `f32` values into a single `u32` via `f16` conversion.
#[inline]
pub(crate) fn pack2x16float(a: f32, b: f32) -> u32 {
    let lo = f16::from_f32(a).to_bits();
    let hi = f16::from_f32(b).to_bits();
    (lo as u32) | ((hi as u32) << 16)
}

// ─── PLY Loader ────────────────────────────────────────────────────────────

/// Loads a 3DGS `.ply` file into a [`GaussianCloud`].
///
/// # Errors
///
/// Returns an error if the file is not a valid binary PLY, or if required
/// vertex properties are missing.
pub fn load_gaussian_ply<R: Read + Seek>(reader: R) -> Result<GaussianCloud> {
    let mut buf = BufReader::new(reader);
    let header = parse_header(&mut buf)?;

    if header.format == PlyFormat::BinaryBigEndian {
        return Err(Error::Asset(AssetError::Format(
            "big-endian PLY is not supported yet".into(),
        )));
    }

    let n = header.vertex_count;
    let offsets = &header.property_offsets;
    let stride = header.vertex_byte_size;

    let mut gaussians = Vec::with_capacity(n);
    let mut sh_coefficients = Vec::with_capacity(n);

    let mut aabb_min = Vec3::splat(f32::INFINITY);
    let mut aabb_max = Vec3::splat(f32::NEG_INFINITY);

    let num_rest = (header.num_sh_coeffs - 1) * 3;
    let num_sh_coeffs = header.num_sh_coeffs;

    // Read all vertex data into a flat buffer for efficient parsing
    let total_bytes = n * stride;
    let mut raw = vec![0u8; total_bytes];
    buf.read_exact(&mut raw)
        .map_err(|e| Error::Asset(AssetError::Io(e)))?;

    for i in 0..n {
        let base = i * stride;
        let vertex = &raw[base..base + stride];

        // Position
        let x = f32::from_le_bytes(vertex[offsets.x..offsets.x + 4].try_into().unwrap());
        let y = f32::from_le_bytes(vertex[offsets.x + 4..offsets.x + 8].try_into().unwrap());
        let z = f32::from_le_bytes(vertex[offsets.x + 8..offsets.x + 12].try_into().unwrap());

        let pos = Vec3::new(x, y, z);
        aabb_min = aabb_min.min(pos);
        aabb_max = aabb_max.max(pos);

        // Opacity (sigmoid activation)
        let raw_opacity = f32::from_le_bytes(
            vertex[offsets.opacity..offsets.opacity + 4]
                .try_into()
                .unwrap(),
        );
        let opacity = sigmoid(raw_opacity);

        // Scale (exponentiate from log-space)
        let s0 = f32::from_le_bytes(
            vertex[offsets.scale_0..offsets.scale_0 + 4]
                .try_into()
                .unwrap(),
        )
        .exp();
        let s1 = f32::from_le_bytes(
            vertex[offsets.scale_0 + 4..offsets.scale_0 + 8]
                .try_into()
                .unwrap(),
        )
        .exp();
        let s2 = f32::from_le_bytes(
            vertex[offsets.scale_0 + 8..offsets.scale_0 + 12]
                .try_into()
                .unwrap(),
        )
        .exp();

        // Rotation quaternion (w, x, y, z) — normalise
        let r0 = f32::from_le_bytes(vertex[offsets.rot_0..offsets.rot_0 + 4].try_into().unwrap());
        let r1 = f32::from_le_bytes(
            vertex[offsets.rot_0 + 4..offsets.rot_0 + 8]
                .try_into()
                .unwrap(),
        );
        let r2 = f32::from_le_bytes(
            vertex[offsets.rot_0 + 8..offsets.rot_0 + 12]
                .try_into()
                .unwrap(),
        );
        let r3 = f32::from_le_bytes(
            vertex[offsets.rot_0 + 12..offsets.rot_0 + 16]
                .try_into()
                .unwrap(),
        );
        let len = (r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3).sqrt();
        let inv_len = if len > 1e-12 { 1.0 / len } else { 0.0 };
        let q = [r0 * inv_len, r1 * inv_len, r2 * inv_len, r3 * inv_len];

        // Build covariance and pack to f16
        let cov = build_covariance(q, [s0, s1, s2]);
        let packed_cov = [
            pack2x16float(cov[0], cov[1]),
            pack2x16float(cov[2], cov[3]),
            pack2x16float(cov[4], cov[5]),
        ];

        gaussians.push(GaussianSplat {
            x,
            y,
            z,
            opacity: pack2x16float(opacity, 0.0),
            sh_idx: i as u32,
            cov: packed_cov,
        });

        // SH coefficients: DC (f_dc_0..2) + rest (interleaved channel-first)
        let mut sh_data = [0u32; 24];

        // DC component
        let dc0 = f32::from_le_bytes(
            vertex[offsets.f_dc_0..offsets.f_dc_0 + 4]
                .try_into()
                .unwrap(),
        );
        let dc1 = f32::from_le_bytes(
            vertex[offsets.f_dc_0 + 4..offsets.f_dc_0 + 8]
                .try_into()
                .unwrap(),
        );
        let dc2 = f32::from_le_bytes(
            vertex[offsets.f_dc_0 + 8..offsets.f_dc_0 + 12]
                .try_into()
                .unwrap(),
        );

        // Pack SH as [[f16;3];16] into [u32;24]
        // Each coefficient c has 3 channel values (R, G, B).
        // Two f16 values are packed per u32.
        let mut sh_flat = [[0.0f32; 3]; 16];
        sh_flat[0] = [dc0, dc1, dc2];

        // Read rest coefficients — stored channel-first in PLY:
        //   f_rest_0..N-1 for channel 0, then channel 1, then channel 2
        if num_rest > 0 {
            let rest_base = offsets.f_rest_start;
            let coeffs_per_channel = num_sh_coeffs - 1;
            for c_idx in 0..coeffs_per_channel.min(15) {
                for ch in 0..3 {
                    let off = rest_base + (ch * coeffs_per_channel + c_idx) * 4;
                    if off + 4 <= vertex.len() {
                        sh_flat[c_idx + 1][ch] =
                            f32::from_le_bytes(vertex[off..off + 4].try_into().unwrap());
                    }
                }
            }
        }

        // Pack into u32 array: 16 coefficients × 3 channels = 48 f16 = 24 u32
        for (c_idx, coeff) in sh_flat.iter().enumerate() {
            // channel_idx -> which u32 and which half
            // half_idx = c_idx * 3 + ch  (0..48)
            // u32_idx = half_idx / 2, sub_idx = half_idx % 2
            for ch in 0..3 {
                let half_idx = c_idx * 3 + ch;
                let u32_idx = half_idx / 2;
                let sub_idx = half_idx % 2;
                let h = f16::from_f32(coeff[ch]).to_bits();
                sh_data[u32_idx] |= (h as u32) << (sub_idx * 16);
            }
        }

        sh_coefficients.push(GaussianSHCoefficients { data: sh_data });
    }

    let center = (aabb_min + aabb_max) * 0.5;

    Ok(GaussianCloud {
        gaussians,
        sh_coefficients,
        sh_degree: sh_degree_from_coeffs(num_sh_coeffs as u32),
        num_points: n,
        aabb_min,
        aabb_max,
        center,
        mip_splatting: false,
        kernel_size: 0.3,
        color_space: ColorSpace::Srgb,
        opacity_compensation: 1.0,
    })
}

/// Derives the SH degree from the total number of coefficients per channel.
fn sh_degree_from_coeffs(n: u32) -> u32 {
    let sqrt = (n as f32).sqrt();
    if sqrt.fract() == 0.0 {
        (sqrt as u32).saturating_sub(1)
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(100.0) > 0.999);
        assert!(sigmoid(-100.0) < 0.001);
    }

    #[test]
    fn test_sh_degree() {
        assert_eq!(sh_degree_from_coeffs(1), 0);
        assert_eq!(sh_degree_from_coeffs(4), 1);
        assert_eq!(sh_degree_from_coeffs(9), 2);
        assert_eq!(sh_degree_from_coeffs(16), 3);
    }

    #[test]
    fn test_pack2x16float() {
        let packed = pack2x16float(1.0, -1.0);
        let lo = f16::from_bits((packed & 0xFFFF) as u16);
        let hi = f16::from_bits(((packed >> 16) & 0xFFFF) as u16);
        assert!((lo.to_f32() - 1.0).abs() < 0.01);
        assert!((hi.to_f32() + 1.0).abs() < 0.01);
    }
}
