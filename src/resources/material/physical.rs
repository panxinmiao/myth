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

}


#[derive(Debug)]
pub struct MeshPhysicalMaterial {
    pub(crate) uniforms: CpuBuffer<MeshPhysicalUniforms>,
    pub(crate) settings: RwLock<MaterialSettings>,
    pub(crate) version: AtomicU64,

    pub(crate) textures: RwLock<MeshPhysicalTextureSet>,

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
    // pub(crate) specular_intensity_map: TextureSlot,
    // pub specular_intensity_map_sampler: Option<SamplerSource>,
    // pub(crate) clearcoat_map: TextureSlot,
    // pub clearcoat_map_sampler: Option<SamplerSource>,
    // pub(crate) clearcoat_roughness_map: TextureSlot,
    // pub clearcoat_roughness_map_sampler: Option<SamplerSource>,
    // pub(crate) clearcoat_normal_map: TextureSlot,
    // pub clearcoat_normal_map_sampler: Option<SamplerSource>,

    pub auto_sync_texture_to_uniforms: bool,
}

// impl Clone for MeshPhysicalMaterial {
//     fn clone(&self) -> Self {
//         use std::sync::atomic::Ordering;
//         Self {
//             // 1. Uniforms: CpuBuffer 已经实现了 Clone (拷贝数据+新ID)
//             uniforms: self.uniforms.clone(),

//             // 2. Settings: 读锁 -> 拷贝数据 -> 新锁
//             settings: parking_lot::RwLock::new(self.settings.read().clone()),

//             // 3. Textures: 读锁 -> 拷贝数据 -> 新锁
//             textures: parking_lot::RwLock::new(self.textures.read().clone()),

//             // 4. Version: 原子读取 -> 新原子变量
//             // 克隆出来的新材质通常从当前版本号开始，或者重置为 0 也可以
//             version: std::sync::atomic::AtomicU64::new(
//                 self.version.load(Ordering::Relaxed)
//             ),

//             auto_sync_texture_to_uniforms: self.auto_sync_texture_to_uniforms,
//         }
//     }
// }

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
            // specular_intensity_map: TextureSlot::default(),
            // specular_intensity_map_sampler: None,
            // clearcoat_map: TextureSlot::default(),
            // clearcoat_map_sampler: None,
            // clearcoat_roughness_map: TextureSlot::default(),
            // clearcoat_roughness_map_sampler: None,
            // clearcoat_normal_map: TextureSlot::default(),
            // clearcoat_normal_map_sampler: None,

            auto_sync_texture_to_uniforms: false,
        }
    }


}

impl_material_api!(
    MeshPhysicalMaterial, 
    MeshPhysicalUniforms,
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
        (specular_color,     Vec3, "Specular color."),
        (specular_intensity, f32,  "Specular intensity."),
        (clearcoat,          f32,  "Clearcoat factor."),
        (clearcoat_roughness, f32, "Clearcoat roughness factor."),
        (ior,                f32,  "Index of Refraction."),
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
    ]
);

impl Default for MeshPhysicalMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}
