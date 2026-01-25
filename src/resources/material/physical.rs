use glam::{Vec2, Vec3, Vec4};

use crate::resources::buffer::CpuBuffer;
use crate::resources::material::{MaterialSettings, SettingsGuard, TextureSlot};
use crate::resources::texture::SamplerSource;
use crate::resources::uniforms::MeshPhysicalUniforms;
use crate::{impl_material_api, impl_material_trait};

#[derive(Debug)]
pub struct MeshPhysicalMaterial {
    pub(crate) uniforms: CpuBuffer<MeshPhysicalUniforms>,
    // #[allow(deprecated)]
    // pub(crate) bindings: MaterialBindings,
    pub(crate) settings: MaterialSettings,
    pub(crate) version: u64,

    pub map: TextureSlot,
    pub map_sampler: Option<SamplerSource>,
    pub normal_map: TextureSlot,
    pub normal_map_sampler: Option<SamplerSource>,
    pub roughness_map: TextureSlot,
    pub roughness_map_sampler: Option<SamplerSource>,
    pub metalness_map: TextureSlot,
    pub metalness_map_sampler: Option<SamplerSource>,
    pub ao_map: TextureSlot,
    pub ao_map_sampler: Option<SamplerSource>,
    pub emissive_map: TextureSlot,
    pub emissive_map_sampler: Option<SamplerSource>,
    pub specular_map: TextureSlot,
    pub specular_map_sampler: Option<SamplerSource>,
    pub specular_intensity_map: TextureSlot,
    pub specular_intensity_map_sampler: Option<SamplerSource>,
    pub clearcoat_map: TextureSlot,
    pub clearcoat_map_sampler: Option<SamplerSource>,
    pub clearcoat_roughness_map: TextureSlot,
    pub clearcoat_roughness_map_sampler: Option<SamplerSource>,
    pub clearcoat_normal_map: TextureSlot,
    pub clearcoat_normal_map_sampler: Option<SamplerSource>,
}

impl MeshPhysicalMaterial {
    // #[allow(deprecated)]
    pub fn new(color: Vec4) -> Self {
        let uniform_data = MeshPhysicalUniforms { color, ..Default::default() };
        
        Self {
            uniforms: CpuBuffer::new(
                uniform_data,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("MeshPhysicalUniforms")
            ),
            // bindings: MaterialBindings::default(),
            settings: MaterialSettings::default(),
            version: 0,

            map: TextureSlot::default(),
            map_sampler: None,
            normal_map: TextureSlot::default(),
            normal_map_sampler: None,
            roughness_map: TextureSlot::default(),
            roughness_map_sampler: None,
            metalness_map: TextureSlot::default(),
            metalness_map_sampler: None,
            ao_map: TextureSlot::default(),
            ao_map_sampler: None,
            emissive_map: TextureSlot::default(),
            emissive_map_sampler: None,
            specular_map: TextureSlot::default(),
            specular_map_sampler: None,
            specular_intensity_map: TextureSlot::default(),
            specular_intensity_map_sampler: None,
            clearcoat_map: TextureSlot::default(),
            clearcoat_map_sampler: None,
            clearcoat_roughness_map: TextureSlot::default(),
            clearcoat_roughness_map_sampler: None,
            clearcoat_normal_map: TextureSlot::default(),
            clearcoat_normal_map_sampler: None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn uniforms_mut(&mut self) -> crate::resources::buffer::BufferGuard<'_, MeshPhysicalUniforms> {
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
    MeshPhysicalMaterial, 
    MeshPhysicalUniforms,
    uniforms: [
        (color,              Vec4, "Base color."),
        (roughness,          f32,  "Roughness factor."),
        (metalness,          f32,  "Metalness factor."),
        (opacity,            f32,  "Opacity value."),
        (emissive,           Vec3, "Emissive color."),
        (emissive_intensity, f32,  "Emissive intensity."),
        (normal_scale,       Vec2, "Normal map scale."),
        (occlusion_strength, f32,  "Occlusion strength."),
        (ao_map_intensity,   f32,  "AO map intensity."),
        (specular_color,     Vec3, "Specular color."),
        (specular_intensity, f32,  "Specular intensity."),
        (ior,                f32,  "Index of Refraction."),
    ],
    textures: [
        (map,                    map_transform,                    "The color map."),
        (normal_map,             normal_map_transform,             "The normal map."),
        (roughness_map,          roughness_map_transform,          "The roughness map."),
        (metalness_map,          metalness_map_transform,          "The metalness map."),
        (ao_map,                 ao_map_transform,                 "The AO map."),
        (emissive_map,           emissive_map_transform,           "The emissive map."),
        (specular_map,           specular_map_transform,           "The specular map."),
        (specular_intensity_map, specular_intensity_map_transform, "The specular intensity map."),
        (clearcoat_map,          clearcoat_map_transform,          "The clearcoat map."),
        (clearcoat_roughness_map, clearcoat_roughness_map_transform, "The clearcoat roughness map."),
        (clearcoat_normal_map,   clearcoat_normal_map_transform,   "The clearcoat normal map."),
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
        (map,           "use_map"),
        (normal_map,    "use_normal_map"),
        (roughness_map, "use_roughness_map"),
        (metalness_map, "use_metalness_map"),
        (ao_map,        "use_ao_map"),
        (emissive_map,  "use_emissive_map"),
        (specular_map,  "use_specular_map"),
        (specular_intensity_map, "use_specular_intensity_map"),
        (clearcoat_map, "use_clearcoat_map"),
        (clearcoat_roughness_map, "use_clearcoat_roughness_map"),
        (clearcoat_normal_map, "use_clearcoat_normal_map"),
    ]
);

impl Default for MeshPhysicalMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}
