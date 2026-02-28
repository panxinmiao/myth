//! SSAO (Screen Space Ambient Occlusion) Configuration
//!
//! This module defines SSAO settings as pure data structures, following the same
//! pattern as [`BloomSettings`](super::bloom::BloomSettings) and
//! [`ToneMappingSettings`](super::tone_mapping::ToneMappingSettings).
//!
//! # GPU Uniform Struct
//!
//! - [`SsaoUniforms`]: Contains the sample kernel, radius, bias, intensity, and
//!   noise scale parameters used by the SSAO shader.
//!
//! This struct is defined here (rather than in the render pass) so that
//! `SsaoSettings` can own the `CpuBuffer<SsaoUniforms>`. User-facing setters
//! write directly into the buffer via `CpuBuffer::write()`, which automatically
//! tracks the data version. The render pass only calls `ensure_buffer_id()` and
//! never writes to the buffers itself; GPU sync happens only when the version
//! has actually changed.
//!
//! # Algorithm
//!
//! The SSAO implementation uses:
//! 1. A hemisphere sample kernel (up to 64 samples) with importance-weighted
//!    distribution concentrated near the origin
//! 2. A 4×4 tiled rotation noise texture that randomizes the kernel orientation
//!    per-pixel, breaking banding artifacts while keeping sample count low
//! 3. Range-checked occlusion with smooth distance falloff
//! 4. A cross-bilateral blur pass (depth-aware + normal-aware) that smooths
//!    the noisy raw AO while preserving geometric edges

use glam::Vec4;

use crate::define_gpu_data_struct;
use crate::resources::WgslType;
use crate::resources::buffer::CpuBuffer;
use crate::resources::uniforms::UniformArray;

// ============================================================================
// GPU Uniform Struct
// ============================================================================

define_gpu_data_struct!(
    struct SsaoUniforms {
        pub samples: UniformArray<Vec4, 64>,
        pub radius: f32,
        pub bias: f32,
        pub intensity: f32,
        pub sample_count: u32,
        pub(crate) noise_scale: glam::Vec2,
        pub(crate) __pad: UniformArray<u32, 2>,
    }
);

// ============================================================================
// SsaoSettings
// ============================================================================

/// SSAO post-processing configuration (pure data + automatic version control).
///
/// This struct holds all parameters for the screen-space ambient occlusion pass.
/// Dynamic GPU uniform data lives in a `CpuBuffer<SsaoUniforms>` field; the
/// internal version is automatically bumped when setter methods modify values
/// via `CpuBuffer::write()`. The render pass calls `ensure_buffer_id()` which
/// only performs a GPU upload when the version has changed.
///
/// # Usage
///
/// ```rust,ignore
/// // Access via scene
/// let ssao = &mut scene.ssao;
///
/// // Enable/disable
/// ssao.set_enabled(true);
///
/// // Tune parameters
/// ssao.set_radius(0.5);
/// ssao.set_intensity(1.5);
/// ssao.set_sample_count(32);
/// ```
#[derive(Debug, Clone)]
pub struct SsaoSettings {
    /// Whether SSAO is enabled.
    pub enabled: bool,

    /// GPU uniform buffer containing sample kernel and parameters.
    /// Updated via setter methods — version tracking is automatic.
    pub(crate) uniforms: CpuBuffer<SsaoUniforms>,
}

impl Default for SsaoSettings {
    fn default() -> Self {
        let kernel = generate_ssao_kernel(32);

        let mut samples = UniformArray::<Vec4, 64>::default();
        for (i, s) in kernel.iter().enumerate() {
            samples[i] = *s;
        }

        let uniforms = SsaoUniforms {
            samples,
            radius: 0.5,
            bias: 0.025,
            intensity: 1.0,
            sample_count: 32,
            noise_scale: glam::Vec2::new(1.0, 1.0), // Will be updated at runtime from screen size
            ..Default::default()
        };

        Self {
            enabled: false,
            uniforms: CpuBuffer::new(
                uniforms,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("SSAO Uniforms"),
            ),
        }
    }
}

impl SsaoSettings {
    /// Creates new SSAO settings with default values (disabled).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets whether SSAO is enabled.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Sets the sampling radius in view-space units (meters).
    ///
    /// Larger values detect occlusion from more distant geometry;
    /// typical range is 0.1–2.0.
    pub fn set_radius(&mut self, radius: f32) {
        self.uniforms.write().radius = radius.max(0.01);
    }

    /// Returns the current sampling radius.
    #[inline]
    #[must_use]
    pub fn radius(&self) -> f32 {
        self.uniforms.read().radius
    }

    /// Sets the depth bias to prevent self-occlusion artifacts.
    ///
    /// Typical range is 0.01–0.05.
    pub fn set_bias(&mut self, bias: f32) {
        self.uniforms.write().bias = bias.max(0.0);
    }

    /// Returns the current depth bias.
    #[inline]
    #[must_use]
    pub fn bias(&self) -> f32 {
        self.uniforms.read().bias
    }

    /// Sets the AO intensity (exponent applied to the final occlusion value).
    ///
    /// Higher values produce stronger, darker shadows. Typical range is 1.0–3.0.
    pub fn set_intensity(&mut self, intensity: f32) {
        self.uniforms.write().intensity = intensity.max(0.0);
    }

    /// Returns the current AO intensity.
    #[inline]
    #[must_use]
    pub fn intensity(&self) -> f32 {
        self.uniforms.read().intensity
    }

    /// Sets the number of hemisphere samples.
    ///
    /// More samples produce smoother results at higher GPU cost.
    /// Clamped to 1..64. The kernel is regenerated when the count changes.
    pub fn set_sample_count(&mut self, count: u32) {
        let count = count.clamp(1, 64);
        let current = self.uniforms.read().sample_count;
        if current != count {
            let kernel = generate_ssao_kernel(count);
            let mut guard = self.uniforms.write();
            guard.sample_count = count;
            for (i, s) in kernel.iter().enumerate() {
                guard.samples[i] = *s;
            }
        }
    }

    /// Returns the current sample count.
    #[inline]
    #[must_use]
    pub fn sample_count(&self) -> u32 {
        self.uniforms.read().sample_count
    }

    /// Updates the noise tiling scale based on screen resolution.
    ///
    /// Called by the render pass during `prepare()` when the screen
    /// size changes. `noise_scale = (width / 4.0, height / 4.0)`.
    pub fn update_noise_scale(&mut self, width: u32, height: u32) {
        let scale = glam::Vec2::new(width as f32 / 4.0, height as f32 / 4.0);
        let current = self.uniforms.read().noise_scale;
        if (current - scale).length_squared() > f32::EPSILON {
            self.uniforms.write().noise_scale = scale;
        }
    }
}

// ============================================================================
// Kernel & Noise Generation
// ============================================================================

use glam::Vec3;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Generates a hemisphere sample kernel with importance-weighted distribution.
///
/// Uses a fixed seed for deterministic results across frames and sessions.
/// Samples are concentrated near the origin via a quadratic fall-off curve,
/// producing better occlusion sampling efficiency.
#[must_use]
pub fn generate_ssao_kernel(samples: u32) -> Vec<Vec4> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut kernel = Vec::with_capacity(samples as usize);

    for i in 0..samples {
        // Random direction in the upper hemisphere (Z > 0)
        let mut sample = Vec3::new(
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(0.01..1.0),
        )
        .normalize();

        // Random magnitude within the hemisphere volume
        sample *= rng.random_range(0.0..1.0f32);

        // Quadratic scale: concentrate samples near the origin
        // This follows the principle that nearby occlusion is more important
        let scale = i as f32 / samples as f32;
        let scale = lerp(0.1, 1.0, scale * scale);
        sample *= scale;

        kernel.push(Vec4::new(sample.x, sample.y, sample.z, 0.0));
    }
    kernel
}

/// Generates a 4×4 rotation noise texture (16 RGBA8 pixels).
///
/// Each pixel encodes a random 2D rotation vector in XY (Z = 0).
/// Used to rotate the sample kernel per-pixel, breaking regular
/// banding patterns while keeping the sample count low.
///
/// The noise texture should use `Repeat` addressing and `Nearest` filtering.
#[must_use]
pub fn generate_ssao_noise() -> Vec<[u8; 4]> {
    let mut rng = StdRng::seed_from_u64(12345);
    let mut noise = Vec::with_capacity(16);
    for _ in 0..16 {
        let xy = Vec3::new(
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            0.0,
        )
        .normalize();

        noise.push([
            ((xy.x * 0.5 + 0.5) * 255.0) as u8,
            ((xy.y * 0.5 + 0.5) * 255.0) as u8,
            0,
            255,
        ]);
    }
    noise
}

fn lerp(a: f32, b: f32, f: f32) -> f32 {
    a + f * (b - a)
}
