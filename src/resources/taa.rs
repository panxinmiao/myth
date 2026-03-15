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
}

impl Default for TaaSettings {
    fn default() -> Self {
        Self {
            feedback_weight: 0.90,
        }
    }
}
