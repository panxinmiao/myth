use glam::Vec4;

use crate::resources::buffer::CpuBuffer;
use crate::resources::material::{MaterialBindings, MaterialFeatures, MaterialSettings, SettingsGuard};
use crate::resources::uniforms::MeshBasicUniforms;
use crate::{impl_material_api, impl_material_trait};

#[derive(Debug)]
pub struct MeshBasicMaterial {
    pub(crate) uniforms: CpuBuffer<MeshBasicUniforms>,
    pub(crate) bindings: MaterialBindings,
    pub(crate) settings: MaterialSettings,
    pub(crate) version: u64,
}

impl MeshBasicMaterial {
    pub fn new(color: Vec4) -> Self {
        let uniform_data = MeshBasicUniforms { color, ..Default::default() };
        
        Self {
            uniforms: CpuBuffer::new(
                uniform_data, 
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("MeshBasicUniforms")
            ),
            bindings: MaterialBindings::default(),
            settings: MaterialSettings::default(),
            version: 0,
        }
    }
    
    pub(crate) fn bindings_mut(&mut self) -> &mut MaterialBindings {
        &mut self.bindings
    }

    pub(crate) fn uniforms_mut(&mut self) -> crate::resources::buffer::BufferGuard<'_, MeshBasicUniforms> {
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
    MeshBasicMaterial, 
    MeshBasicUniforms,
    uniforms: [
        (color,   Vec4, "Base color."),
        (opacity, f32,  "Opacity value."),
    ],
    textures: [
        (map, "The color map."),
    ]
);

impl_material_trait!(
    MeshBasicMaterial,
    "mesh_basic",
    MeshBasicUniforms,
    default_features: MaterialFeatures::empty(),
    textures: [
        (map, USE_MAP),
    ]
);

impl Default for MeshBasicMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}
