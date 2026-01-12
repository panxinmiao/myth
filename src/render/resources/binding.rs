use crate::resources::buffer::BufferRef; 
use crate::assets::TextureHandle;
use crate::resources::material::{Material, MaterialData};
use crate::resources::geometry::Geometry;
use crate::scene::environment::Environment;
use crate::resources::uniforms::*;
use crate::render::resources::builder::ResourceBuilder;
use crate::render::RenderState;

/// 实际的绑定资源数据 (用于生成 BindGroup)
/// 层只持有 ID 或 数据引用，不持有 GPU 句柄
#[derive(Debug, Clone)]
pub enum BindingResource {

    Buffer {
        buffer: BufferRef,
        offset: u64,        // 偏移量 (默认为 0)
        size: Option<u64>,  // 绑定窗口大小 (None 表示整个 Buffer)
    },

    Texture(Option<TextureHandle>),
    Sampler(Option<TextureHandle>),

}

pub trait Bindings {
    fn define_bindings(&self, builder: &mut ResourceBuilder);
}


impl Bindings for MaterialData {
    // ... shader_name, flush_uniforms ...

    fn define_bindings(&self, builder: &mut ResourceBuilder) {
        match self {
            Self::Basic(m) => {
                // 1. Uniforms
                builder.add_uniform_with_struct::<MeshBasicUniforms>(
                    "material", 
                    &m.uniform_buffer, 
                    wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX
                );
                
                // 2. Maps (Texture + Sampler)
                // 注意：这里直接把 Arc 传给 builder
                if let Some(map) = &m.map {
                    // Texture
                    builder.add_texture(
                        "map", 
                        map.clone(), 
                        wgpu::TextureSampleType::Float { filterable: true }, 
                        wgpu::TextureViewDimension::D2, 
                        wgpu::ShaderStages::FRAGMENT
                    );
                    
                    // Sampler
                    builder.add_sampler(
                        "map", 
                        map.clone(), 
                        wgpu::SamplerBindingType::Filtering, 
                        wgpu::ShaderStages::FRAGMENT
                    );
                    
                }

            },
            Self::Phong(m) => {
                builder.add_uniform_with_struct::<MeshPhongUniforms>(
                    "material", 
                    &m.uniform_buffer, 
                    wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX
                );

                // 批量添加 Phong 材质的贴图
                // Map
                if let Some(map) = &m.map {
                    builder.add_texture("map", *map, wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                    builder.add_sampler("map", *map, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
                }

                if let Some(map) = &m.normal_map {
                    builder.add_texture("normal_map", *map, wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                    builder.add_sampler("normal_map", *map, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
                }

                if let Some(map) = &m.specular_map {
                    builder.add_texture("specular_map", *map, wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                    builder.add_sampler("specular_map", *map, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
                    
                }

                if let Some(map) = &m.emissive_map {
                    builder.add_texture("emissive_map", *map, wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                    builder.add_sampler("emissive_map", *map, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
                }
            },


            Self::Standard(m) => {
                builder.add_uniform_with_struct::<MeshStandardUniforms>(
                    "material", 
                    &m.uniform_buffer, 
                    wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX
                );
                
                // 批量添加 Standard 材质的贴图
                // Map
                if let Some(map) = &m.map {
                    builder.add_texture("map", *map, wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                    builder.add_sampler("map", *map, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);

                }

                if let Some(map) = &m.ao_map {
                    builder.add_texture("ao_map", *map, wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                    builder.add_sampler("ao_map", *map, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
                }

                // Normal Map
                if let Some(map) = &m.normal_map {
                    builder.add_texture("normal_map", *map, wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                    builder.add_sampler("normal_map", *map, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
                }
                
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

impl Bindings for Environment {
    fn define_bindings(&self, builder: &mut ResourceBuilder) {
        builder.add_uniform_with_struct::<EnvironmentUniforms>(
            "environment", 
            &self.uniform_buffer,
            wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX
        );
        builder.add_storage_with_struct::<GpuLightStorage>("lights", &self.light_storage_buffer, true, wgpu::ShaderStages::FRAGMENT);
    }

    // 未来可以在这里添加：
    // builder.add_texture("env_map", self.env_map_id, ...)
}

impl Bindings for RenderState {
    fn define_bindings(&self, builder: &mut ResourceBuilder) {
        builder.add_uniform_with_struct::<RenderStateUniforms>(
            "render_state", 
            &self.uniform_buffer,
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT
        );
    }
}
