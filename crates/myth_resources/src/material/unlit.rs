use glam::Vec4;
use myth_macros::myth_material;

use crate::uniforms::Mat3Uniform;

#[myth_material(shader = "entry/main/unlit", crate_path = "crate")]
pub struct UnlitMaterial {
    /// Base color.
    #[uniform(default = "Vec4::ONE")]
    pub color: Vec4,

    /// Opacity value.
    #[uniform(default = "1.0")]
    pub opacity: f32,

    /// Alpha test threshold.
    #[uniform]
    pub alpha_test: f32,

    /// Dynamic extrusion thickness (0.0 to disable).
    #[uniform(default = "0.0")]
    pub extrusion: f32,

    /// The color map.
    #[texture]
    pub map: TextureSlot,
}

impl UnlitMaterial {
    /// Creates a new unlit material with the given base color.
    #[must_use]
    pub fn new(color: Vec4) -> Self {
        Self::from_uniforms(UnlitUniforms {
            color,
            ..Default::default()
        })
    }
}
