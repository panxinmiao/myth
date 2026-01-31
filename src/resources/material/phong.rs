use std::sync::atomic::AtomicU64;

use glam::{Vec2, Vec3, Vec4};
use parking_lot::RwLock;

use crate::resources::buffer::CpuBuffer;
use crate::resources::material::{MaterialSettings, TextureSlot};
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

    // pub(crate) map: TextureSlot,
    // pub map_sampler: Option<SamplerSource>,
    // pub(crate) normal_map: TextureSlot,
    // pub normal_map_sampler: Option<SamplerSource>,
    // pub(crate) specular_map: TextureSlot,
    // pub specular_map_sampler: Option<SamplerSource>,
    // pub(crate) emissive_map: TextureSlot,
    // pub emissive_map_sampler: Option<SamplerSource>,

    pub auto_sync_texture_to_uniforms: bool,
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
            settings: RwLock::new(MaterialSettings::default()),
            version: AtomicU64::new(0),

            textures: RwLock::new(MeshPhongTextureSet::default()),

            // map: TextureSlot::default(),
            // map_sampler: None,
            // normal_map: TextureSlot::default(),
            // normal_map_sampler: None,
            // specular_map: TextureSlot::default(),
            // specular_map_sampler: None,
            // emissive_map: TextureSlot::default(),
            // emissive_map_sampler: None,

            auto_sync_texture_to_uniforms: false,
        }
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
    "mesh_phong",
    MeshPhongUniforms,
    default_defines: [],
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

