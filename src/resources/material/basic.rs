use glam::Vec4;

use crate::resources::buffer::CpuBuffer;
use crate::resources::material::{MaterialSettings, SettingsGuard, TextureSlot};
use crate::resources::texture::SamplerSource;
use crate::resources::uniforms::MeshBasicUniforms;
use crate::{impl_material_api, impl_material_trait};

#[derive(Debug)]
pub struct MeshBasicMaterial {
    pub(crate) uniforms: CpuBuffer<MeshBasicUniforms>,
    // #[allow(deprecated)]
    // pub(crate) bindings: MaterialBindings,
    pub(crate) settings: MaterialSettings,
    pub(crate) version: u64,

    pub map: TextureSlot,
    pub map_sampler: Option<SamplerSource>,
}

impl MeshBasicMaterial {
    // #[allow(deprecated)]
    pub fn new(color: Vec4) -> Self {
        let uniform_data = MeshBasicUniforms { color, ..Default::default() };
        
        Self {
            uniforms: CpuBuffer::new(
                uniform_data, 
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("MeshBasicUniforms")
            ),
            // bindings: MaterialBindings::default(),
            settings: MaterialSettings::default(),
            version: 0,

            map: TextureSlot::default(),
            map_sampler: None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn uniforms_mut(&mut self) -> crate::resources::buffer::BufferGuard<'_, MeshBasicUniforms> {
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
    MeshBasicMaterial, 
    MeshBasicUniforms,
    uniforms: [
        (color,   Vec4, "Base color."),
        (opacity, f32,  "Opacity value."),
    ],
    textures: [
        (map, map_transform, "The color map."),
    ]
);

impl_material_trait!(
    MeshBasicMaterial,
    "mesh_basic",
    MeshBasicUniforms,
    default_defines: [],
    textures: [
        (map, "use_map"),
    ]
);

impl Default for MeshBasicMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}
