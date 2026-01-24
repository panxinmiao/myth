use glam::{Vec2, Vec3, Vec4};

use crate::resources::buffer::CpuBuffer;
use crate::resources::material::{MaterialBindings, MaterialFeatures, MaterialSettings, SettingsGuard};
use crate::resources::uniforms::MeshPhysicalUniforms;
use crate::{impl_material_api, impl_material_trait};

#[derive(Debug)]
pub struct MeshPhysicalMaterial {
    pub(crate) uniforms: CpuBuffer<MeshPhysicalUniforms>,
    pub(crate) bindings: MaterialBindings,
    pub(crate) settings: MaterialSettings,
    pub(crate) version: u64,
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
            bindings: MaterialBindings::default(),
            settings: MaterialSettings::default(),
            version: 0,
        }
    }
    
    #[allow(dead_code)]
    pub(crate) fn bindings_mut(&mut self) -> &mut MaterialBindings {
        &mut self.bindings
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
        (map,           "The color map."),
        (normal_map,    "The normal map."),
        (roughness_map, "The roughness map."),
        (metalness_map, "The metalness map."),
        (ao_map,        "The AO map."),
        (emissive_map,  "The emissive map."),
        (specular_map,  "The specular map."),
        (specular_intensity_map, "The specular intensity map."),
        (clearcoat_map, "The clearcoat map."),
        (clearcoat_roughness_map, "The clearcoat roughness map."),
        (clearcoat_normal_map, "The clearcoat normal map."),
    ]
);

impl_material_trait!(
    MeshPhysicalMaterial,
    "mesh_physical",
    MeshPhysicalUniforms,
    default_features: MaterialFeatures::USE_IBL | MaterialFeatures::USE_SPECULAR | MaterialFeatures::USE_IOR | MaterialFeatures::USE_CLEARCOAT,
    textures: [
        (map,           USE_MAP),
        (normal_map,    USE_NORMAL_MAP),
        (roughness_map, USE_ROUGHNESS_MAP),
        (metalness_map, USE_METALNESS_MAP),
        (ao_map,        USE_AO_MAP),
        (emissive_map,  USE_EMISSIVE_MAP),
        (specular_map,  USE_SPECULAR_MAP),
        (specular_intensity_map, USE_SPECULAR_INTENSITY_MAP),
        (clearcoat_map, USE_CLEARCOAT_MAP),
        (clearcoat_roughness_map, USE_CLEARCOAT_ROUGHNESS_MAP),
        (clearcoat_normal_map, USE_CLEARCOAT_NORMAL_MAP),
    ]
);

impl Default for MeshPhysicalMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}
