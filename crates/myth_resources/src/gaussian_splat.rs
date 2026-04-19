//! 3D Gaussian Splatting data types.
//!
//! Defines GPU-aligned structures for representing 3D Gaussian primitives,
//! as well as the high-level [`GaussianCloud`] asset that bundles raw splat
//! data with metadata and bounding information.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

// ─── GPU Structures ────────────────────────────────────────────────────────

/// A single 3D Gaussian primitive in GPU-ready layout.
///
/// Position (`x`, `y`, `z`) and opacity are stored as `f32`.
/// Covariance is stored as the upper triangle of a symmetric 3×3 matrix
/// (6 `f16` values packed into 3 `u32` slots) to match the format used
/// in the preprocessing compute shader.
///
/// The spherical-harmonic DC component and per-Gaussian metadata are
/// stored in the separate [`GaussianSHCoefficients`] buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GaussianSplat {
    /// World-space x coordinate.
    pub x: f32,
    /// World-space y coordinate.
    pub y: f32,
    /// World-space z coordinate.
    pub z: f32,
    /// Pre-activated opacity stored as a packed `f16` pair
    /// (only the lower half-float is used; the upper half is padding).
    pub opacity: u32,
    /// Upper triangle of the 3×3 covariance matrix, packed as
    /// `[pack2x16float(c00, c01), pack2x16float(c02, c11), pack2x16float(c12, c22)]`.
    pub cov: [u32; 3],
}

/// Per-Gaussian spherical harmonic coefficients (up to degree 3).
///
/// 16 RGB coefficients × 3 channels = 48 half-floats packed into 24 `u32`
/// slots via `pack2x16float`.  Degree-0 is the base colour; higher degrees
/// capture view-dependent colour.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GaussianSHCoefficients {
    /// 24 packed `u32` values, each containing two `f16` channel values.
    pub data: [u32; 24],
}

/// Intermediate 2D screen-space splat produced by the preprocessing
/// compute shader.  Consumed by the rendering vertex shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Splat2D {
    /// Screen-space centre
    pub pos: [f32; 2],
    /// Scaled eigenvector 0 (2 × f16 packed).
    pub v0: u32,
    /// Scaled eigenvector 1 (2 × f16 packed).
    pub v1: u32,
    /// Reverse-Z NDC depth at the splat centre.
    pub depth: f32,
    /// RGBA colour channels 0–1 (2 × f16 packed).
    pub color_rg: u32,
    /// RGBA colour channels 2–3 (2 × f16 packed).
    pub color_ba: u32,

    pub _pad: u32,
}

// ─── High-Level Asset ──────────────────────────────────────────────────────

/// Maximum supported spherical-harmonic degree.
pub const MAX_SH_DEGREE: u32 = 3;

/// A point cloud of 3D Gaussians ready for GPU upload.
///
/// This is the CPU-side asset produced by the PLY loader and consumed by the
/// renderer to allocate GPU storage buffers.
pub struct GaussianCloud {
    /// Raw Gaussian primitive data.
    pub gaussians: Vec<GaussianSplat>,
    /// Per-Gaussian SH coefficients (same length as `gaussians`).
    pub sh_coefficients: Vec<GaussianSHCoefficients>,
    /// Spherical-harmonic degree (0–3).
    pub sh_degree: u32,
    /// Number of Gaussian primitives.
    pub num_points: usize,
    /// Axis-aligned bounding box minimum.
    pub aabb_min: Vec3,
    /// Axis-aligned bounding box maximum.
    pub aabb_max: Vec3,
    /// Centroid of the point cloud.
    pub center: Vec3,
    /// Whether mip-splatting filtering is enabled.
    pub mip_splatting: bool,
    /// Kernel size for mip-splatting anti-aliasing.
    pub kernel_size: f32,
}

impl GaussianCloud {
    /// Scene extent (max half-diagonal of the bounding box).
    #[inline]
    #[must_use]
    pub fn scene_extent(&self) -> f32 {
        (self.aabb_max - self.aabb_min).length() * 0.5
    }
}
