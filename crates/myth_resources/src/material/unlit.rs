use glam::Vec4;
use myth_macros::myth_material;

use crate::TextureHandle;
use crate::material::{AlphaMode, Side};
use crate::uniforms::UnlitUniforms;

#[myth_material(shader = "templates/unlit", crate_path = "crate", uniforms = UnlitUniforms)]
pub struct UnlitMaterial {
    /// Base color.
    #[uniform]
    pub color: Vec4,

    /// Opacity value.
    #[uniform]
    pub opacity: f32,

    /// Alpha test threshold.
    #[uniform]
    pub alpha_test: f32,

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

    /// Sets the base color (builder).
    #[must_use]
    pub fn with_color(self, color: Vec4) -> Self {
        self.uniforms.write().color = color;
        self
    }

    /// Sets the opacity (builder).
    #[must_use]
    pub fn with_opacity(self, opacity: f32) -> Self {
        self.uniforms.write().opacity = opacity;
        self
    }

    /// Sets the color map texture (builder).
    #[must_use]
    pub fn with_map(self, handle: TextureHandle) -> Self {
        self.set_map(Some(handle));
        self
    }

    /// Sets the face culling side (builder).
    #[must_use]
    pub fn with_side(self, side: Side) -> Self {
        self.set_side(side);
        self
    }

    /// Sets the alpha mode (builder).
    #[must_use]
    pub fn with_alpha_mode(self, mode: AlphaMode) -> Self {
        self.set_alpha_mode(mode);
        self
    }

    /// Sets depth write (builder).
    #[must_use]
    pub fn with_depth_write(self, enabled: bool) -> Self {
        self.set_depth_write(enabled);
        self
    }
}

impl Default for UnlitMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}
