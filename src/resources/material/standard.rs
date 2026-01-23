use glam::{Vec2, Vec3, Vec4};

use crate::resources::buffer::CpuBuffer;
use crate::resources::material::{MaterialBindings, MaterialFeatures, MaterialSettings, SettingsGuard};
use crate::resources::uniforms::MeshStandardUniforms;
use crate::{impl_material_api, impl_material_trait};

#[derive(Debug)]
pub struct MeshStandardMaterial {
    pub(crate) uniforms: CpuBuffer<MeshStandardUniforms>,
    pub(crate) bindings: MaterialBindings,
    pub(crate) settings: MaterialSettings,
    pub(crate) version: u64,
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
            bindings: MaterialBindings::default(),
            settings: MaterialSettings::default(),
            version: 0,
        }
    }
    
    pub(crate) fn bindings_mut(&mut self) -> &mut MaterialBindings {
        &mut self.bindings
    }

    pub(crate) fn uniforms_mut(&mut self) -> crate::resources::buffer::BufferGuard<'_, MeshStandardUniforms> {
        self.uniforms.write()
    }

    pub(crate) fn settings_mut(&mut self) -> SettingsGuard<'_> {
        SettingsGuard {
            initial_settings: self.settings.clone(),
            settings: &mut self.settings,
            version: &mut self.version,
        }
    }
}

impl_material_api!(
    MeshStandardMaterial, 
    MeshStandardUniforms,
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
    default_features: MaterialFeatures::USE_IBL,
    textures: [
        (map,           USE_MAP),
        (normal_map,    USE_NORMAL_MAP),
        (roughness_map, USE_ROUGHNESS_MAP),
        (metalness_map, USE_METALNESS_MAP),
        (ao_map,        USE_AO_MAP),
        (emissive_map,  USE_EMISSIVE_MAP),
        (specular_map,  USE_SPECULAR_MAP),
    ]
);

impl Default for MeshStandardMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}
