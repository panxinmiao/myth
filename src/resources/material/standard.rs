use glam::{Vec2, Vec3, Vec4};

use crate::resources::buffer::CpuBuffer;
use crate::resources::material::{MaterialBindings, MaterialSettings, SettingsGuard};
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
    
    #[allow(dead_code)]
    pub(crate) fn bindings_mut(&mut self) -> &mut MaterialBindings {
        &mut self.bindings
    }

    #[allow(dead_code)]
    pub(crate) fn uniforms_mut(&mut self) -> crate::resources::buffer::BufferGuard<'_, MeshStandardUniforms> {
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
    default_defines: [("USE_IBL", "1")],
    textures: [
        (map,           "use_map"),
        (normal_map,    "use_normal_map"),
        (roughness_map, "use_roughness_map"),
        (metalness_map, "use_metalness_map"),
        (ao_map,        "use_ao_map"),
        (emissive_map,  "use_emissive_map"),
        (specular_map,  "use_specular_map"),
    ]
);

impl Default for MeshStandardMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}
