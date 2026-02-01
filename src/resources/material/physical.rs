use std::sync::atomic::AtomicU64;

use glam::{Vec2, Vec3, Vec4};
use parking_lot::RwLock;

use crate::resources::buffer::CpuBuffer;
use crate::resources::material::{MaterialSettings, TextureSlot};
use crate::resources::texture::SamplerSource;
use crate::resources::uniforms::MeshPhysicalUniforms;
use crate::{impl_material_api, impl_material_trait};


#[derive(Clone, Default, Debug)]
pub struct MeshPhysicalTextureSet {
    pub map: TextureSlot,
    pub normal_map: TextureSlot,
    pub roughness_map: TextureSlot,
    pub metalness_map: TextureSlot,
    pub ao_map: TextureSlot,
    pub emissive_map: TextureSlot,
    pub specular_map: TextureSlot,
    pub specular_intensity_map: TextureSlot,
    pub clearcoat_map: TextureSlot,
    pub clearcoat_roughness_map: TextureSlot,
    pub clearcoat_normal_map: TextureSlot,
    pub sheen_color_map: TextureSlot,
    pub sheen_roughness_map: TextureSlot,

    pub map_sampler: Option<SamplerSource>,
    pub normal_map_sampler: Option<SamplerSource>,
    pub roughness_map_sampler: Option<SamplerSource>,
    pub metalness_map_sampler: Option<SamplerSource>,
    pub ao_map_sampler: Option<SamplerSource>,
    pub emissive_map_sampler: Option<SamplerSource>,
    pub specular_map_sampler: Option<SamplerSource>,
    pub specular_intensity_map_sampler: Option<SamplerSource>,
    pub clearcoat_map_sampler: Option<SamplerSource>,
    pub clearcoat_roughness_map_sampler: Option<SamplerSource>,
    pub clearcoat_normal_map_sampler: Option<SamplerSource>,
    pub sheen_color_map_sampler: Option<SamplerSource>,
    pub sheen_roughness_map_sampler: Option<SamplerSource>,

}


#[derive(Debug)]
pub struct MeshPhysicalMaterial {
    pub(crate) uniforms: CpuBuffer<MeshPhysicalUniforms>,
    pub(crate) settings: RwLock<MaterialSettings>,
    pub(crate) version: AtomicU64,

    pub(crate) textures: RwLock<MeshPhysicalTextureSet>,

    pub auto_sync_texture_to_uniforms: bool,
}

impl MeshPhysicalMaterial {
    pub fn new(color: Vec4) -> Self {
        let uniform_data = MeshPhysicalUniforms { color, ..Default::default() };
        
        Self {
            uniforms: CpuBuffer::new(
                uniform_data,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("MeshPhysicalUniforms")
            ),
            settings: RwLock::new(MaterialSettings::default()),
            version: AtomicU64::new(0),

            textures: RwLock::new(MeshPhysicalTextureSet::default()),

            auto_sync_texture_to_uniforms: false,
        }
    }
}

impl_material_api!(
    MeshPhysicalMaterial, 
    MeshPhysicalUniforms,
    uniforms: [
        (color,               Vec4, "Base color."),
        (alpha_test,          f32,  "Alpha test threshold."),
        (roughness,           f32,  "Roughness factor."),
        (metalness,           f32,  "Metalness factor."),
        (opacity,             f32,  "Opacity value."),
        (emissive,            Vec3, "Emissive color."),
        (emissive_intensity,  f32,  "Emissive intensity."),
        (normal_scale,        Vec2, "Normal map scale."),
        (ao_map_intensity,    f32,  "AO map intensity."),
        (specular_color,      Vec3, "Specular color."),
        (specular_intensity,  f32,  "Specular intensity."),
        (clearcoat,           f32,  "Clearcoat factor."),
        (clearcoat_roughness, f32, "Clearcoat roughness factor."),
        (ior,                 f32,  "Index of Refraction."),

        (sheen_color,         Vec3,  "The sheen tint. Default is (0, 0, 0), black."),
        (sheen_roughness,     f32,   "The sheen roughness. Default is 1.0."),
    ],
    textures: [
        (map,                    "The color map."),
        (normal_map,             "The normal map."),
        (roughness_map,          "The roughness map."),
        (metalness_map,          "The metalness map."),
        (ao_map,                 "The AO map."),
        (emissive_map,           "The emissive map."),
        (specular_map,           "The specular map."),
        (specular_intensity_map, "The specular intensity map."),
        (clearcoat_map,          "The clearcoat map."),
        (clearcoat_roughness_map, "The clearcoat roughness map."),
        (clearcoat_normal_map,   "The clearcoat normal map."),
        (sheen_color_map,        "The sheen color map."),
        (sheen_roughness_map,    "The sheen roughness map."),
    ]
);

impl_material_trait!(
    MeshPhysicalMaterial,
    "mesh_physical",
    MeshPhysicalUniforms,
    default_defines: [
        ("USE_IBL", "1"),
        ("USE_SPECULAR", "1"),
        ("USE_IOR", "1"),
        ("USE_CLEARCOAT", "1"),
        // ("USE_SHEEN", "1"),
    ],
    textures: [
        map,
        normal_map,
        roughness_map,
        metalness_map,
        ao_map,
        emissive_map,
        specular_map,
        specular_intensity_map,
        clearcoat_map,
        clearcoat_roughness_map,
        clearcoat_normal_map,
        sheen_color_map,
        sheen_roughness_map,
    ]
);

impl Default for MeshPhysicalMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}
