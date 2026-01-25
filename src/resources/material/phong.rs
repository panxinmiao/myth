use glam::{Vec2, Vec3, Vec4};

use crate::resources::buffer::CpuBuffer;
use crate::resources::material::{MaterialSettings, SettingsGuard, TextureSlot};
use crate::resources::texture::SamplerSource;
use crate::resources::uniforms::MeshPhongUniforms;
use crate::{impl_material_api, impl_material_trait};

#[derive(Debug)]
pub struct MeshPhongMaterial {
    pub(crate) uniforms: CpuBuffer<MeshPhongUniforms>,
    pub(crate) settings: MaterialSettings,
    pub(crate) version: u64,

    pub map: TextureSlot,
    pub map_sampler: Option<SamplerSource>,
    pub normal_map: TextureSlot,
    pub normal_map_sampler: Option<SamplerSource>,
    pub specular_map: TextureSlot,
    pub specular_map_sampler: Option<SamplerSource>,
    pub emissive_map: TextureSlot,
    pub emissive_map_sampler: Option<SamplerSource>,
}

impl MeshPhongMaterial {
    pub fn new(color: Vec4) -> Self {
        let uniform_data = MeshPhongUniforms { color, ..Default::default() };
        
        Self {
            uniforms: CpuBuffer::new(
                uniform_data,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("MeshPhongUniforms")
            ),
            settings: MaterialSettings::default(),
            version: 0,

            map: TextureSlot::default(),
            map_sampler: None,
            normal_map: TextureSlot::default(),
            normal_map_sampler: None,
            specular_map: TextureSlot::default(),
            specular_map_sampler: None,
            emissive_map: TextureSlot::default(),
            emissive_map_sampler: None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn uniforms_mut(&mut self) -> crate::resources::buffer::BufferGuard<'_, MeshPhongUniforms> {
        self.uniforms.write()
    }

    #[allow(dead_code)]
    pub(crate) fn settings_mut(&mut self) -> SettingsGuard<'_> {
        SettingsGuard {
            initial_settings: self.settings.clone(),
            settings: &mut self.settings,
            version: &mut self.version,
        }
    }
}

impl_material_api!(
    MeshPhongMaterial, 
    MeshPhongUniforms,
    uniforms: [
        (color,              Vec4, "Diffuse color."),
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
    "mesh_phong",
    MeshPhongUniforms,
    default_defines: [],
    textures: [
        (map,          "use_map"),
        (normal_map,   "use_normal_map"),
        (specular_map, "use_specular_map"),
        (emissive_map, "use_emissive_map"),
    ]
);

impl Default for MeshPhongMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}

