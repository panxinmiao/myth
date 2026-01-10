use std::sync::{Arc, RwLock};
use crate::core::texture::Texture;
use crate::core::buffer::BufferRef; 
use crate::core::material::{Material, MaterialData};
use crate::core::geometry::Geometry;
use crate::core::world::WorldEnvironment;
use crate::core::uniforms::*;
use crate::renderer::resource_builder::ResourceBuilder;

/// 实际的绑定资源数据 (用于生成 BindGroup)
/// 层只持有 ID 或 数据引用，不持有 GPU 句柄
#[derive(Debug, Clone)]
pub enum BindingResource {

    /// 持有 CPU Buffer 的引用 (统一了 Vertex/Index/Uniform/Storage)
    Buffer {
        buffer: BufferRef,
        offset: u64,        // 偏移量 (默认为 0)
        size: Option<u64>,  // 绑定窗口大小 (None 表示整个 Buffer)
    },

    Texture(Option<Arc<RwLock<Texture>>>),
    Sampler(Option<Arc<RwLock<Texture>>>),

}

// pub fn define_texture_binding(builder: &mut ResourceBuilder, name: &'static str, texture_id: Option<Uuid>) {
//     builder.add_texture(
//         name,
//         texture_id,
//         wgpu::TextureSampleType::Float { filterable: true },
//         wgpu::TextureViewDimension::D2,
//         ShaderStages::FRAGMENT,
//     );
//     builder.add_sampler(
//         name,
//         texture_id,
//         wgpu::SamplerBindingType::Filtering,
//         ShaderStages::FRAGMENT,
//     );
// }

pub trait Bindings {
    fn define_bindings(&self, builder: &mut ResourceBuilder);
}


impl Bindings for MaterialData {
    // ... shader_name, flush_uniforms ...

    fn define_bindings(&self, builder: &mut ResourceBuilder) {
        match self {
            Self::Basic(m) => {
                // 1. Uniforms
                builder.add_uniform::<MeshBasicUniforms>(
                    "material", 
                    &m.uniform_buffer, 
                    wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX
                );
                
                // 2. Maps (Texture + Sampler)
                // 注意：这里直接把 Arc 传给 builder
                builder.add_texture(
                    "map", 
                    m.map.clone(), 
                    wgpu::TextureSampleType::Float { filterable: true }, 
                    wgpu::TextureViewDimension::D2, 
                    wgpu::ShaderStages::FRAGMENT
                );
                
                builder.add_sampler(
                    "map", 
                    m.map.clone(), 
                    wgpu::SamplerBindingType::Filtering, 
                    wgpu::ShaderStages::FRAGMENT
                );
            },
            Self::Standard(m) => {
                builder.add_uniform::<MeshStandardUniforms>(
                    "material", 
                    &m.uniform_buffer, 
                    wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX
                );
                
                // 批量添加 Standard 材质的贴图
                // Map
                builder.add_texture("map", m.map.clone(), wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                builder.add_sampler("map", m.map.clone(), wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);

                // Normal Map
                builder.add_texture("normalMap", m.normal_map.clone(), wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                builder.add_sampler("normalMap", m.normal_map.clone(), wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
                
                // ... 其他贴图 (Roughness, Metalness ...)
            }
        }
    }
}



impl Bindings for Material {
    // ...
    fn define_bindings(&self, builder: &mut ResourceBuilder) {
        self.data.define_bindings(builder);
    }
}



impl Bindings for Geometry {
    fn define_bindings(&self, _builder: &mut ResourceBuilder) {
        // 需要根据 morph_attributes 来自动生成 morph texture 资源（或 storage buffer）
        // 然后绑定到 Group 2 中
       
    }
}

impl Bindings for WorldEnvironment {
    fn define_bindings(&self, builder: &mut ResourceBuilder) {
        // Binding 0: Frame Uniforms (Vertex + Fragment)
        builder.add_uniform::<GlobalFrameUniforms>(
            "global", 
            &self.frame_uniforms, // 传入 BufferRef
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT
        );

        // Binding 1: Light Uniforms (Fragment only)
        builder.add_uniform::<GlobalLightUniforms>(
            "lights",
            &self.light_uniforms,
            wgpu::ShaderStages::FRAGMENT
        );
    }

    // 未来可以在这里添加：
    // builder.add_texture("env_map", self.env_map_id, ...)
}

