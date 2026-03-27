use glam::{Vec2, Vec3, Vec4};
use myth_macros::myth_material;

use crate::TextureHandle;
use crate::material::{AlphaMode, Side};
use crate::uniforms::Mat3Uniform;

#[myth_material(shader = "templates/phong", crate_path = "crate")]
pub struct PhongMaterial {
    /// Diffuse color.
    #[uniform(default = "Vec4::ONE")]
    pub color: Vec4,

    /// Specular color.
    #[uniform(default = "Vec3::splat(0.06667)")]
    pub specular: Vec3,

    /// Opacity value.
    #[uniform(default = "1.0")]
    pub opacity: f32,

    /// Emissive color.
    #[uniform]
    pub emissive: Vec3,

    /// Emissive intensity.
    #[uniform(default = "1.0")]
    pub emissive_intensity: f32,

    /// Normal map scale.
    #[uniform(default = "Vec2::ONE")]
    pub normal_scale: Vec2,

    /// Shininess factor.
    #[uniform(default = "30.0")]
    pub shininess: f32,

    /// Alpha test threshold.
    #[uniform]
    pub alpha_test: f32,

    /// The color map.
    #[texture]
    pub map: TextureSlot,

    /// The normal map.
    #[texture]
    pub normal_map: TextureSlot,

    /// The specular map.
    #[texture]
    pub specular_map: TextureSlot,

    /// The emissive map.
    #[texture]
    pub emissive_map: TextureSlot,
}

impl PhongMaterial {
    /// Creates a new Phong material with the given diffuse color.
    #[must_use]
    pub fn new(color: Vec4) -> Self {
        Self::from_uniforms(PhongUniforms {
            color,
            ..Default::default()
        })
    }

    /// Sets the diffuse color (builder).
    #[must_use]
    pub fn with_color(self, color: Vec4) -> Self {
        self.uniforms.write().color = color;
        self
    }

    /// Sets the shininess factor (builder).
    #[must_use]
    pub fn with_shininess(self, s: f32) -> Self {
        self.uniforms.write().shininess = s;
        self
    }

    /// Sets the specular color (builder).
    #[must_use]
    pub fn with_specular(self, specular: Vec3) -> Self {
        self.uniforms.write().specular = specular;
        self
    }

    /// Sets the emissive color and intensity (builder).
    #[must_use]
    pub fn with_emissive(self, color: Vec3, intensity: f32) -> Self {
        {
            let mut u = self.uniforms.write();
            u.emissive = color;
            u.emissive_intensity = intensity;
        }
        self
    }

    /// Sets the normal map scale (builder).
    #[must_use]
    pub fn with_normal_scale(self, scale: Vec2) -> Self {
        self.uniforms.write().normal_scale = scale;
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

    /// Sets the normal map texture (builder).
    #[must_use]
    pub fn with_normal_map(self, handle: TextureHandle) -> Self {
        self.set_normal_map(Some(handle));
        self
    }

    /// Sets the specular map texture (builder).
    #[must_use]
    pub fn with_specular_map(self, handle: TextureHandle) -> Self {
        self.set_specular_map(Some(handle));
        self
    }

    /// Sets the emissive map texture (builder).
    #[must_use]
    pub fn with_emissive_map(self, handle: TextureHandle) -> Self {
        self.set_emissive_map(Some(handle));
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

impl Default for PhongMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}
