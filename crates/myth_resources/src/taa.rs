//! TAA (Temporal Anti-Aliasing) Configuration
//!
//! Pure data structure for TAA tuning parameters, following the same pattern
//! as [`FxaaSettings`](super::fxaa::FxaaSettings) and
//! [`BloomSettings`](super::bloom::BloomSettings).
//!
//! TAA resolves all aliasing categories (geometric, specular, sub-pixel) by
//! accumulating multiple jittered frames over time.  The trade-off is slight
//! temporal softness and potential ghosting on fast-moving objects — both
//! controlled by the `feedback_weight` parameter.
//!
//! # Usage
//!
//! ```rust,ignore
//! scene.taa.feedback_weight = 0.90;
//! ```

/// TAA settings exposed to the user.
#[derive(Debug, Clone, Copy)]
pub struct TaaSettings {
    /// History frame blend weight (`0.0..=1.0`).
    ///
    /// Higher values produce smoother results but increase ghosting on
    /// fast-moving objects.  Typical range: `0.85..=0.95`.
    pub feedback_weight: f32,

    /// Contrast Adaptive Sharpening intensity (`0.0..=1.0`).
    ///
    /// Applied after TAA resolve to recover fine detail lost to temporal
    /// filtering.  `0.0` disables sharpening; `1.0` is maximum.
    /// Typical range: `0.4..=0.8`.
    pub sharpen_intensity: f32,
}

impl Default for TaaSettings {
    fn default() -> Self {
        Self {
            feedback_weight: 0.90,
            sharpen_intensity: 0.5,
        }
    }
}
