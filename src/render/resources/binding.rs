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
pub enum BindingResource<'a> {
    /// UniformSlot（推荐用于小型Uniform）
    UniformSlot {
        slot_id: u64,
        data: &'a [u8],    // 直接持有数据引用
        label: &'a str,    // 标签引用（用于创建GPU Buffer）
    },

    /// BufferRef（用于Storage/Vertex/Dynamic Uniform）
    Buffer {
        buffer: BufferRef,
        offset: u64,        // 偏移量 (默认为 0)
        size: Option<u64>,  // 绑定窗口大小 (None 表示整个 Buffer)
    },

    Texture(Option<TextureHandle>),
    Sampler(Option<TextureHandle>),
    
    // 使用 PhantomData 来标记生命周期（对于不使用引用的 variant）
    _Phantom(std::marker::PhantomData<&'a ()>),
}

pub trait Bindings {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>);
}


impl Bindings for MaterialData {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        match self {
            Self::Basic(m) => {
                // 使用新的 add_uniform_slot 方法
                builder.add_uniform_slot::<MeshBasicUniforms>(
                    "material", 
                    &m.uniforms, 
                    wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX
                );
                
                // 纹理绑定保持不变
                if let Some(map) = &m.map {
                    builder.add_texture(
                        "map", 
                        map.clone(), 
                        wgpu::TextureSampleType::Float { filterable: true }, 
                        wgpu::TextureViewDimension::D2, 
                        wgpu::ShaderStages::FRAGMENT
                    );
                    
                    builder.add_sampler(
                        "map", 
                        map.clone(), 
                        wgpu::SamplerBindingType::Filtering, 
                        wgpu::ShaderStages::FRAGMENT
                    );
                }
            },
            Self::Phong(m) => {
                builder.add_uniform_slot::<MeshPhongUniforms>(
                    "material", 
                    &m.uniforms, 
                    wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX
                );

                // 纹理绑定
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
                builder.add_uniform_slot::<MeshStandardUniforms>(
                    "material", 
                    &m.uniforms, 
                    wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX
                );
                
                // 纹理绑定
                if let Some(map) = &m.map {
                    builder.add_texture("map", *map, wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                    builder.add_sampler("map", *map, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
                }

                if let Some(map) = &m.ao_map {
                    builder.add_texture("ao_map", *map, wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                    builder.add_sampler("ao_map", *map, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
                }

                if let Some(map) = &m.normal_map {
                    builder.add_texture("normal_map", *map, wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                    builder.add_sampler("normal_map", *map, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
                }
            }
        }
    }
}



impl Bindings for Material {
    // ...
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        self.data.define_bindings(builder);
    }
}



impl Bindings for Geometry {
    fn define_bindings<'a>(&'a self, _builder: &mut ResourceBuilder<'a>) {
        // 需要根据 morph_attributes 来自动生成 morph texture 资源（或 storage buffer）
        // 然后绑定到 Group 2 中
       
    }
}

impl Bindings for Environment {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        // Uniform 使用 UniformSlot
        builder.add_uniform_slot::<EnvironmentUniforms>(
            "environment", 
            &self.uniforms,
            wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX
        );
        // Storage 继续使用 BufferRef
        builder.add_storage_with_struct::<GpuLightStorage>(
            "lights", 
            &self.light_storage_buffer, 
            true, 
            wgpu::ShaderStages::FRAGMENT
        );
    }
}

impl Bindings for RenderState {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        builder.add_uniform_slot::<RenderStateUniforms>(
            "render_state", 
            &self.uniforms,
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT
        );
    }
}
