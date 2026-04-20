//! NPZ loader for 3D Gaussian Splatting point clouds.
//!
//! Parses the compressed NPZ format produced by 3DGS training pipelines
//! (e.g. compact-3dgs, LightGaussian) and produces a [`GaussianCloud`]
//! asset ready for GPU upload.
//!
//! The NPZ file is a ZIP archive of NumPy `.npy` arrays containing
//! quantized Gaussian attributes. Key arrays:
//!
//! | Array               | Dtype  | Shape           | Description |
//! |---------------------|--------|-----------------|-------------|
//! | `xyz`               | f16    | `(N, 3)`        | Position    |
//! | `opacity`           | i8     | `(N, 1)`        | Quantized opacity |
//! | `scaling`           | i8     | `(N, 3)` or `(M, 3)` | Quantized log-scale |
//! | `rotation`          | i8     | `(N, 4)` or `(M, 4)` | Quantized quaternion |
//! | `features_dc`       | i8     | `(K, 3)`        | Quantized SH DC |
//! | `features_rest`     | i8     | `(K, C)`        | Quantized SH rest |
//! | `*_scale` / `*_zero_point` | scalar | `(1,)` | Dequantization parameters |
//!
//! Dequantization: `value = (raw - zero_point) * scale`

use std::io::{Read, Seek};

use glam::Vec3;
use half::f16;
use myth_core::{AssetError, Error, Result};
use myth_resources::gaussian_splat::{GaussianCloud, GaussianSHCoefficients, GaussianSplat};

use super::ply::{build_covariance, pack2x16float, sigmoid};

// ─── Minimal NumPy Array Parser ────────────────────────────────────────────

/// Parsed header of a `.npy` array inside the NPZ archive.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct NpyHeader {
    /// Element data type.
    dtype: NpyDtype,
    /// Shape of the array (row-major / C-order).
    shape: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NpyDtype {
    F16,
    F32,
    F64,
    I8,
    I32,
    U32,
    Bool,
}

#[allow(dead_code)]
impl NpyDtype {
    fn byte_size(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 => 8,
            Self::I8 | Self::Bool => 1,
        }
    }
}

/// Parses the `.npy` binary header from a byte slice.
fn parse_npy_header(data: &[u8]) -> Result<(NpyHeader, usize)> {
    // Magic: \x93NUMPY
    if data.len() < 10 || &data[..6] != b"\x93NUMPY" {
        return Err(Error::Asset(AssetError::Format(
            "invalid .npy magic bytes".into(),
        )));
    }
    let major = data[6];
    let header_len = if major >= 2 {
        u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize
    } else {
        u16::from_le_bytes(data[8..10].try_into().unwrap()) as usize
    };
    let header_offset = if major >= 2 { 12 } else { 10 };
    let header_str = std::str::from_utf8(&data[header_offset..header_offset + header_len])
        .map_err(|_| Error::Asset(AssetError::Format("invalid npy header encoding".into())))?;

    let dtype = parse_dtype_from_header(header_str)?;
    let shape = parse_shape_from_header(header_str)?;

    Ok((NpyHeader { dtype, shape }, header_offset + header_len))
}

fn parse_dtype_from_header(header: &str) -> Result<NpyDtype> {
    // Look for 'descr': '<f2', '<f4', '|i1', '<i4', '<u4', '|b1', etc.
    let descr = extract_field(header, "descr")
        .ok_or_else(|| Error::Asset(AssetError::Format("missing descr in npy header".into())))?;
    match descr
        .trim_matches('\'')
        .trim_matches(|c| c == '<' || c == '>' || c == '|' || c == '=')
    {
        "f2" | "e" => Ok(NpyDtype::F16),
        "f4" | "f" => Ok(NpyDtype::F32),
        "f8" | "d" => Ok(NpyDtype::F64),
        "i1" | "b" => Ok(NpyDtype::I8),
        "i4" | "i" => Ok(NpyDtype::I32),
        "u4" => Ok(NpyDtype::U32),
        "b1" => Ok(NpyDtype::Bool),
        other => Err(Error::Asset(AssetError::Format(format!(
            "unsupported npy dtype: {other}"
        )))),
    }
}

fn parse_shape_from_header(header: &str) -> Result<Vec<usize>> {
    let shape_str = extract_field(header, "shape")
        .ok_or_else(|| Error::Asset(AssetError::Format("missing shape in npy header".into())))?;
    // shape_str looks like "(1234,)" or "(1234, 3)" or "()"
    let inner = shape_str.trim_matches(|c: char| c == '(' || c == ')' || c.is_whitespace());
    if inner.is_empty() {
        return Ok(vec![]);
    }
    inner
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|_| Error::Asset(AssetError::Format(format!("bad shape element: {s}"))))
        })
        .collect()
}

/// Extracts a Python dict field value from the numpy header string.
fn extract_field<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let pattern = format!("'{key}':");
    let start = header.find(&pattern)?;
    let after = &header[start + pattern.len()..];
    let after = after.trim_start();

    // Handle tuple/parenthesized values
    if after.starts_with('(') {
        let end = after.find(')')?;
        Some(&after[..=end])
    } else if after.starts_with('\'') {
        // String value
        let rest = &after[1..];
        let end = rest.find('\'')?;
        Some(&after[..end + 2])
    } else {
        // Plain value (bool, number)
        let end = after.find(',')?;
        Some(after[..end].trim())
    }
}

// ─── NPZ Archive Reader ───────────────────────────────────────────────────

/// Reads a single named array from the NPZ zip archive.
fn read_npy_array(
    archive: &mut zip::ZipArchive<impl Read + Seek>,
    name: &str,
) -> Result<Option<(NpyHeader, Vec<u8>)>> {
    let file_name = if name.ends_with(".npy") {
        name.to_string()
    } else {
        format!("{name}.npy")
    };
    let mut entry = match archive.by_name(&file_name) {
        Ok(e) => e,
        Err(zip::result::ZipError::FileNotFound) => return Ok(None),
        Err(e) => {
            return Err(Error::Asset(AssetError::Format(format!(
                "zip error reading '{file_name}': {e}"
            ))));
        }
    };
    let mut buf = Vec::new();
    entry
        .read_to_end(&mut buf)
        .map_err(|e| Error::Asset(AssetError::Io(e)))?;

    let (header, data_offset) = parse_npy_header(&buf)?;
    let data = buf[data_offset..].to_vec();
    Ok(Some((header, data)))
}

/// Reads a scalar value from a named NPZ array (e.g. `opacity_scale`).
fn read_scalar<T: NpyScalar>(
    archive: &mut zip::ZipArchive<impl Read + Seek>,
    name: &str,
) -> Result<Option<T>> {
    let Some((header, data)) = read_npy_array(archive, name)? else {
        return Ok(None);
    };
    T::from_npy(&header, &data)
}

/// Reads a 1D or 2D array of i8 values.
fn read_i8_array(
    archive: &mut zip::ZipArchive<impl Read + Seek>,
    name: &str,
) -> Result<Option<Vec<i8>>> {
    let Some((_header, data)) = read_npy_array(archive, name)? else {
        return Ok(None);
    };
    Ok(Some(data.iter().map(|&b| b as i8).collect()))
}

/// Reads a 1D or 2D array of f16 values.
fn read_f16_array(
    archive: &mut zip::ZipArchive<impl Read + Seek>,
    name: &str,
) -> Result<Option<Vec<f16>>> {
    let Some((header, data)) = read_npy_array(archive, name)? else {
        return Ok(None);
    };
    if header.dtype != NpyDtype::F16 {
        return Err(Error::Asset(AssetError::Format(format!(
            "expected f16 for '{name}', got {:?}",
            header.dtype
        ))));
    }
    let values: Vec<f16> = data
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]))
        .collect();
    Ok(Some(values))
}

trait NpyScalar: Sized {
    fn from_npy(header: &NpyHeader, data: &[u8]) -> Result<Option<Self>>;
}

impl NpyScalar for f32 {
    fn from_npy(header: &NpyHeader, data: &[u8]) -> Result<Option<Self>> {
        match header.dtype {
            NpyDtype::F32 if data.len() >= 4 => {
                Ok(Some(f32::from_le_bytes(data[..4].try_into().unwrap())))
            }
            NpyDtype::F64 if data.len() >= 8 => Ok(Some(f64::from_le_bytes(
                data[..8].try_into().unwrap(),
            ) as f32)),
            _ => Ok(None),
        }
    }
}

impl NpyScalar for i32 {
    fn from_npy(header: &NpyHeader, data: &[u8]) -> Result<Option<Self>> {
        match header.dtype {
            NpyDtype::I32 if data.len() >= 4 => {
                Ok(Some(i32::from_le_bytes(data[..4].try_into().unwrap())))
            }
            NpyDtype::I8 if !data.is_empty() => Ok(Some(data[0] as i8 as i32)),
            _ => Ok(None),
        }
    }
}

impl NpyScalar for bool {
    fn from_npy(header: &NpyHeader, data: &[u8]) -> Result<Option<Self>> {
        match header.dtype {
            NpyDtype::Bool if !data.is_empty() => Ok(Some(data[0] != 0)),
            NpyDtype::I8 if !data.is_empty() => Ok(Some(data[0] != 0)),
            _ => Ok(None),
        }
    }
}

// ─── NPZ Loader ────────────────────────────────────────────────────────────

/// Loads a compressed 3DGS `.npz` file into a [`GaussianCloud`].
///
/// Dequantizes integer-quantized attributes using per-attribute scale and
/// zero-point values stored in the archive. The resulting cloud uses the
/// same GPU-ready format as the PLY loader.
///
/// # Errors
///
/// Returns an error if the archive is malformed, required arrays are
/// missing, or shapes are inconsistent.
pub fn load_gaussian_npz<R: Read + Seek>(reader: R) -> Result<GaussianCloud> {
    let mut archive = zip::ZipArchive::new(reader).map_err(|e| {
        Error::Asset(AssetError::Format(format!(
            "failed to open NPZ archive: {e}"
        )))
    })?;

    // ─── Read dequantization parameters ────────────────────────────
    let opacity_scale: f32 = read_scalar(&mut archive, "opacity_scale")?.unwrap_or(1.0);
    let opacity_zero_point: i32 = read_scalar(&mut archive, "opacity_zero_point")?.unwrap_or(0);

    let scaling_scale: f32 = read_scalar(&mut archive, "scaling_scale")?.unwrap_or(1.0);
    let scaling_zero_point: i32 = read_scalar(&mut archive, "scaling_zero_point")?.unwrap_or(0);

    let rotation_scale: f32 = read_scalar(&mut archive, "rotation_scale")?.unwrap_or(1.0);
    let rotation_zero_point: i32 = read_scalar(&mut archive, "rotation_zero_point")?.unwrap_or(0);

    let features_dc_scale: f32 = read_scalar(&mut archive, "features_dc_scale")?.unwrap_or(1.0);
    let features_dc_zero_point: i32 =
        read_scalar(&mut archive, "features_dc_zero_point")?.unwrap_or(0);

    let features_rest_scale: f32 = read_scalar(&mut archive, "features_rest_scale")?.unwrap_or(1.0);
    let features_rest_zero_point: i32 =
        read_scalar(&mut archive, "features_rest_zero_point")?.unwrap_or(0);

    let scaling_factor_scale: f32 =
        read_scalar(&mut archive, "scaling_factor_scale")?.unwrap_or(1.0);
    let scaling_factor_zero_point: i32 =
        read_scalar(&mut archive, "scaling_factor_zero_point")?.unwrap_or(0);

    let mip_splatting_flag: bool = read_scalar(&mut archive, "mip_splatting")?.unwrap_or(false);
    let kernel_size_val: f32 = read_scalar(&mut archive, "kernel_size")?.unwrap_or(0.3);

    // ─── Read data arrays ──────────────────────────────────────────
    let xyz_f16 = read_f16_array(&mut archive, "xyz")?
        .ok_or_else(|| Error::Asset(AssetError::Format("missing 'xyz' array in NPZ".into())))?;
    let num_points = xyz_f16.len() / 3;

    let opacity_raw = read_i8_array(&mut archive, "opacity")?
        .ok_or_else(|| Error::Asset(AssetError::Format("missing 'opacity' array".into())))?;

    let scaling_raw = read_i8_array(&mut archive, "scaling")?
        .ok_or_else(|| Error::Asset(AssetError::Format("missing 'scaling' array".into())))?;

    let rotation_raw = read_i8_array(&mut archive, "rotation")?
        .ok_or_else(|| Error::Asset(AssetError::Format("missing 'rotation' array".into())))?;

    let features_dc_raw = read_i8_array(&mut archive, "features_dc")?
        .ok_or_else(|| Error::Asset(AssetError::Format("missing 'features_dc' array".into())))?;

    let features_rest_raw = read_i8_array(&mut archive, "features_rest")?;

    let scaling_factor_raw = read_i8_array(&mut archive, "scaling_factor")?;

    let feature_indices: Option<Vec<u32>> = {
        if let Some((_header, data)) = read_npy_array(&mut archive, "feature_indices")? {
            Some(
                data.chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32)
                    .collect(),
            )
        } else {
            None
        }
    };

    let gaussian_indices: Option<Vec<u32>> = {
        if let Some((_header, data)) = read_npy_array(&mut archive, "gaussian_indices")? {
            Some(
                data.chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32)
                    .collect(),
            )
        } else {
            None
        }
    };

    // ─── Determine SH degree ───────────────────────────────────────
    let num_dc_entries = features_dc_raw.len() / 3;
    let rest_channels = if let Some(ref rest) = features_rest_raw {
        if num_dc_entries > 0 {
            rest.len() / num_dc_entries
        } else {
            0
        }
    } else {
        0
    };
    let num_sh_coeffs_per_channel = 1 + rest_channels / 3;
    let sh_degree = sh_degree_from_coeffs(num_sh_coeffs_per_channel as u32);

    let has_scaling_factor = scaling_factor_raw.is_some();

    // ─── Decode and pack ───────────────────────────────────────────
    let num_geom = scaling_raw.len() / 3;
    let mut gaussians = Vec::with_capacity(num_points);
    let mut aabb_min = Vec3::splat(f32::INFINITY);
    let mut aabb_max = Vec3::splat(f32::NEG_INFINITY);

    for i in 0..num_points {
        // ── Position ───────────────────────────────────────────────
        let x = xyz_f16[i * 3].to_f32();
        let y = xyz_f16[i * 3 + 1].to_f32();
        let z = xyz_f16[i * 3 + 2].to_f32();
        let pos = Vec3::new(x, y, z);
        aabb_min = aabb_min.min(pos);
        aabb_max = aabb_max.max(pos);

        // ── Opacity ────────────────────────────────────────────────
        let raw_op = (opacity_raw[i] as f32 - opacity_zero_point as f32) * opacity_scale;
        let opacity = sigmoid(raw_op);

        // ── Geometry index (for shared rotation/scale) ─────────────
        let geom_idx = match gaussian_indices {
            Some(ref gi) => gi[i] as usize,
            None => i,
        };
        let geom_idx = geom_idx.min(num_geom - 1);

        // ── Scale ──────────────────────────────────────────────────
        let s0_raw = (scaling_raw[geom_idx * 3] as f32 - scaling_zero_point as f32) * scaling_scale;
        let s1_raw =
            (scaling_raw[geom_idx * 3 + 1] as f32 - scaling_zero_point as f32) * scaling_scale;
        let s2_raw =
            (scaling_raw[geom_idx * 3 + 2] as f32 - scaling_zero_point as f32) * scaling_scale;

        let (s0, s1, s2) = if has_scaling_factor {
            // Normalised scaling direction + per-Gaussian scaling factor
            let len = (s0_raw * s0_raw + s1_raw * s1_raw + s2_raw * s2_raw)
                .sqrt()
                .max(1e-12);
            let dir = [s0_raw / len, s1_raw / len, s2_raw / len];
            let sf = scaling_factor_raw.as_ref().unwrap();
            let factor =
                ((sf[i] as f32 - scaling_factor_zero_point as f32) * scaling_factor_scale).exp();
            (dir[0] * factor, dir[1] * factor, dir[2] * factor)
        } else {
            (s0_raw.exp(), s1_raw.exp(), s2_raw.exp())
        };

        // ── Rotation ───────────────────────────────────────────────
        let r0 = (rotation_raw[geom_idx * 4] as f32 - rotation_zero_point as f32) * rotation_scale;
        let r1 =
            (rotation_raw[geom_idx * 4 + 1] as f32 - rotation_zero_point as f32) * rotation_scale;
        let r2 =
            (rotation_raw[geom_idx * 4 + 2] as f32 - rotation_zero_point as f32) * rotation_scale;
        let r3 =
            (rotation_raw[geom_idx * 4 + 3] as f32 - rotation_zero_point as f32) * rotation_scale;
        let len = (r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3).sqrt();
        let inv_len = if len > 1e-12 { 1.0 / len } else { 0.0 };
        let q = [r0 * inv_len, r1 * inv_len, r2 * inv_len, r3 * inv_len];

        // ── Covariance ─────────────────────────────────────────────
        let cov = build_covariance(q, [s0, s1, s2]);
        let packed_cov = [
            pack2x16float(cov[0], cov[1]),
            pack2x16float(cov[2], cov[3]),
            pack2x16float(cov[4], cov[5]),
        ];

        let sh_idx = match feature_indices {
            Some(ref fi) => fi[i] as usize,
            None => i,
        };
        let sh_idx = sh_idx.min(num_dc_entries.saturating_sub(1));

        gaussians.push(GaussianSplat {
            x,
            y,
            z,
            opacity: pack2x16float(opacity, 0.0),
            sh_idx: sh_idx as u32,
            cov: packed_cov,
        });
    }

    let mut sh_coefficients = Vec::with_capacity(num_dc_entries);
    for sh_idx in 0..num_dc_entries {
        let mut sh_flat = [[0.0f32; 3]; 16];

        let dc0 = (features_dc_raw[sh_idx * 3] as f32 - features_dc_zero_point as f32)
            * features_dc_scale;
        let dc1 = (features_dc_raw[sh_idx * 3 + 1] as f32 - features_dc_zero_point as f32)
            * features_dc_scale;
        let dc2 = (features_dc_raw[sh_idx * 3 + 2] as f32 - features_dc_zero_point as f32)
            * features_dc_scale;
        sh_flat[0] = [dc0, dc1, dc2];

        if let Some(ref rest) = features_rest_raw {
            let rest_per_entry = rest_channels;
            let base = sh_idx * rest_per_entry;
            let coeffs_per_channel = num_sh_coeffs_per_channel - 1;
            for c_idx in 0..coeffs_per_channel.min(15) {
                let off = base + c_idx * 3;
                for ch in 0..3 {
                    if off + ch < rest.len() {
                        sh_flat[c_idx + 1][ch] = (rest[off + ch] as f32
                            - features_rest_zero_point as f32)
                            * features_rest_scale;
                    }
                }
            }
        }

        let mut sh_data = [0u32; 24];
        for (c_idx, coeff) in sh_flat.iter().enumerate() {
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
        sh_degree,
        num_points,
        aabb_min,
        aabb_max,
        center,
        mip_splatting: mip_splatting_flag,
        kernel_size: kernel_size_val,
    })
}

/// Derives the SH degree from the number of coefficients per channel.
fn sh_degree_from_coeffs(n: u32) -> u32 {
    let sqrt = (n as f32).sqrt();
    if sqrt.fract() == 0.0 {
        (sqrt as u32).saturating_sub(1)
    } else {
        0
    }
}
