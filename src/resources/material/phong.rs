use std::sync::atomic::AtomicU64;

use glam::{Vec2, Vec3, Vec4};
use parking_lot::RwLock;

use crate::assets::TextureHandle;
use crate::resources::buffer::CpuBuffer;
use crate::resources::material::{AlphaMode, MaterialSettings, Side, TextureSlot};
use crate::resources::texture::SamplerSource;
use crate::resources::uniforms::MeshPhongUniforms;
use crate::{impl_material_api, impl_material_trait};

#[derive(Clone, Default, Debug)]
pub struct MeshPhongTextureSet {
    pub map: TextureSlot,
    pub normal_map: TextureSlot,
    pub specular_map: TextureSlot,
    pub emissive_map: TextureSlot,

    pub map_sampler: Option<SamplerSource>,
    pub normal_map_sampler: Option<SamplerSource>,
    pub specular_map_sampler: Option<SamplerSource>,
    pub emissive_map_sampler: Option<SamplerSource>,
}

#[derive(Debug)]
pub struct MeshPhongMaterial {
    pub(crate) uniforms: CpuBuffer<MeshPhongUniforms>,
    pub(crate) settings: RwLock<MaterialSettings>,
    pub(crate) version: AtomicU64,

    pub(crate) textures: RwLock<MeshPhongTextureSet>,

    pub auto_sync_texture_to_uniforms: bool,
}

impl MeshPhongMaterial {
    #[must_use]
    pub fn new(color: Vec4) -> Self {
        let uniform_data = MeshPhongUniforms {
            color,
            ..Default::default()
        };

        Self {
            uniforms: CpuBuffer::new(
                uniform_data,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("MeshPhongUniforms"),
            ),
            settings: RwLock::new(MaterialSettings::default()),
            version: AtomicU64::new(0),
            textures: RwLock::new(MeshPhongTextureSet::default()),
            auto_sync_texture_to_uniforms: false,
        }
    }

    // -- Builder pattern (chainable at construction time) --

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

impl_material_api!(
    MeshPhongMaterial,
    MeshPhongUniforms,
    uniforms: [
        (color,              Vec4, "Diffuse color."),
        (alpha_test,         f32,  "Alpha test threshold."),
        (specular,           Vec3, "Specular color."),
        (opacity,            f32,  "Opacity value."),
        (emissive,           Vec3, "Emissive color."),
        (emissive_intensity, f32,  "Emissive intensity."),
        (normal_scale,       Vec2, "Normal map scale."),
        (shininess,          f32,  "Shininess factor."),
    ],
    textures: [
        (map,          "The color map."),
        (normal_map,   "The normal map."),
        (specular_map, "The specular map."),
        (emissive_map, "The emissive map."),
    ]
);

impl_material_trait!(
    MeshPhongMaterial,
    "templates/mesh_phong",
    MeshPhongUniforms,
    textures: [
        map,
        normal_map,
        specular_map,
        emissive_map,
    ]
);

impl Default for MeshPhongMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}
