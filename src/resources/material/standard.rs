use std::sync::atomic::AtomicU64;

use glam::{Vec2, Vec3, Vec4};
use parking_lot::RwLock;

use crate::resources::buffer::CpuBuffer;
use crate::resources::material::{MaterialSettings, TextureSlot};
use crate::resources::texture::SamplerSource;
use crate::resources::uniforms::MeshStandardUniforms;
use crate::{impl_material_api, impl_material_trait};

#[derive(Clone, Default, Debug)]
pub struct MeshStandardTextureSet {
    pub map: TextureSlot,
    pub normal_map: TextureSlot,
    pub roughness_map: TextureSlot,
    pub metalness_map: TextureSlot,
    pub ao_map: TextureSlot,
    pub emissive_map: TextureSlot,
    pub specular_map: TextureSlot,



    pub map_sampler: Option<SamplerSource>,
    pub normal_map_sampler: Option<SamplerSource>,
    pub roughness_map_sampler: Option<SamplerSource>,
    pub metalness_map_sampler: Option<SamplerSource>,
    pub ao_map_sampler: Option<SamplerSource>,
    pub emissive_map_sampler: Option<SamplerSource>,
    pub specular_map_sampler: Option<SamplerSource>,


}

#[derive(Debug)]
pub struct MeshStandardMaterial {
    pub(crate) uniforms: CpuBuffer<MeshStandardUniforms>,
    pub(crate) settings: RwLock<MaterialSettings>,
    pub(crate) version: AtomicU64,

    pub(crate) textures: RwLock<MeshStandardTextureSet>,

    // pub(crate) map: TextureSlot,
    // pub map_sampler: Option<SamplerSource>,
    // pub(crate) normal_map: TextureSlot,
    // pub normal_map_sampler: Option<SamplerSource>,
    // pub(crate) roughness_map: TextureSlot,
    // pub roughness_map_sampler: Option<SamplerSource>,
    // pub(crate) metalness_map: TextureSlot,
    // pub metalness_map_sampler: Option<SamplerSource>,
    // pub(crate) ao_map: TextureSlot,
    // pub ao_map_sampler: Option<SamplerSource>,
    // pub(crate) emissive_map: TextureSlot,
    // pub emissive_map_sampler: Option<SamplerSource>,
    // pub(crate) specular_map: TextureSlot,
    // pub specular_map_sampler: Option<SamplerSource>,

    pub auto_sync_texture_to_uniforms: bool,
}

impl MeshStandardMaterial {
    pub fn new(color: Vec4) -> Self {
        let uniform_data = MeshStandardUniforms { color, ..Default::default() };
        
        Self {
            uniforms: CpuBuffer::new(
                uniform_data,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("MeshStandardUniforms")
            ),
            settings: RwLock::new(MaterialSettings::default()),
            version: AtomicU64::new(0),

            textures: RwLock::new(MeshStandardTextureSet::default()),

            // map: TextureSlot::default(),
            // map_sampler: None,
            // normal_map: TextureSlot::default(),
            // normal_map_sampler: None,
            // roughness_map: TextureSlot::default(),
            // roughness_map_sampler: None,
            // metalness_map: TextureSlot::default(),
            // metalness_map_sampler: None,
            // ao_map: TextureSlot::default(),
            // ao_map_sampler: None,
            // emissive_map: TextureSlot::default(),
            // emissive_map_sampler: None,
            // specular_map: TextureSlot::default(),
            // specular_map_sampler: None,

            auto_sync_texture_to_uniforms: false,
        }
    }

}

impl_material_api!(
    MeshStandardMaterial, 
    MeshStandardUniforms,
    uniforms: [
        (color,              Vec4, "Base color."),
        (alpha_test,         f32,  "Alpha test threshold."),
        (roughness,          f32,  "Roughness factor."),
        (metalness,          f32,  "Metalness factor."),
        (opacity,            f32,  "Opacity value."),
        (emissive,           Vec3, "Emissive color."),
        (emissive_intensity, f32,  "Emissive intensity."),
        (normal_scale,       Vec2, "Normal map scale."),
        (ao_map_intensity,   f32,  "AO map intensity."),
        (specular,           Vec3, "Specular color."),
        (specular_intensity, f32,  "Specular intensity."),
    ],
    textures: [
        (map,           "The color map."),
        (normal_map,    "The normal map."),
        (roughness_map, "The roughness map."),
        (metalness_map, "The metalness map."),
        (ao_map,        "The AO map."),
        (emissive_map,  "The emissive map."),
        (specular_map,  "The specular map."),
    ]
);

impl_material_trait!(
    MeshStandardMaterial,
    "mesh_standard",
    MeshStandardUniforms,
    default_defines: [("USE_IBL", "1")],
    textures: [
        map,
        normal_map,
        roughness_map,
        metalness_map,
        ao_map,
        emissive_map,
        specular_map,
    ]
);

impl Default for MeshStandardMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}
