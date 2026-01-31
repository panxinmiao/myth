use std::sync::atomic::AtomicU64;

use glam::Vec4;
use parking_lot::RwLock;

use crate::resources::buffer::CpuBuffer;
use crate::resources::material::{MaterialSettings, TextureSlot};
use crate::resources::texture::SamplerSource;
use crate::resources::uniforms::MeshBasicUniforms;
use crate::{impl_material_api, impl_material_trait};

#[derive(Clone, Default, Debug)]
pub struct MeshBasicTextureSet {
    pub map: TextureSlot,

    pub map_sampler: Option<SamplerSource>,
}

#[derive(Debug)]
pub struct MeshBasicMaterial {
    pub(crate) uniforms: CpuBuffer<MeshBasicUniforms>,
    pub(crate) settings: RwLock<MaterialSettings>,
    pub(crate) version: AtomicU64,

    pub(crate) textures : RwLock<MeshBasicTextureSet>,

    pub auto_sync_texture_to_uniforms: bool,
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
            settings: RwLock::new(MaterialSettings::default()),
            version: AtomicU64::new(0),

            textures: RwLock::new(MeshBasicTextureSet::default()),

            // map: TextureSlot::default(),
            // map_sampler: None,

            auto_sync_texture_to_uniforms: false,
        }
    }

}

impl_material_api!(
    MeshBasicMaterial, 
    MeshBasicUniforms,
    uniforms: [
        (color,   Vec4, "Base color."),
        (opacity, f32,  "Opacity value."),
        (alpha_test, f32, "Alpha test threshold."),
    ],
    textures: [
        (map, "The color map."),
    ]
);

impl_material_trait!(
    MeshBasicMaterial,
    "mesh_basic",
    MeshBasicUniforms,
    default_defines: [],
    textures: [
        map,
    ]
);

impl Default for MeshBasicMaterial {
    fn default() -> Self {
        Self::new(Vec4::ONE)
    }
}
