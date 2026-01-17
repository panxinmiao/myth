//! GPU 绑定资源
//!
//! 定义 BindGroup 的资源类型和绑定 Trait

use crate::Mesh;
use crate::resources::buffer::BufferRef;
use crate::assets::TextureHandle;
use crate::resources::material::{Material, MaterialData};
use crate::resources::geometry::Geometry;
use crate::scene::environment::Environment;
use crate::resources::uniforms::*;
use crate::renderer::core::builder::{ResourceBuilder};
use crate::renderer::graph::RenderState;

/// 实际的绑定资源数据 (用于生成 BindGroup)
#[derive(Debug, Clone)]
pub enum BindingResource<'a> {
    Buffer {
        buffer: BufferRef,
        offset: u64,
        size: Option<u64>,
        data: Option<&'a [u8]>,
    },
    Texture(Option<TextureHandle>),
    Sampler(Option<TextureHandle>),
    _Phantom(std::marker::PhantomData<&'a ()>),
}

/// 绑定资源 Trait
pub trait Bindings {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>);
}

impl Bindings for MaterialData {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        match self {
            Self::Basic(m) => {
                builder.add_uniform::<MeshBasicUniforms>(
                    "material",
                    &m.uniforms,
                    wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX
                );
                
                if let Some(map) = &m.bindings().map {
                    builder.add_texture("map", *map, wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                    builder.add_sampler("map", *map, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
                }
            },
            Self::Phong(m) => {
                builder.add_uniform::<MeshPhongUniforms>(
                    "material",
                    &m.uniforms,
                    wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX
                );

                if let Some(map) = &m.bindings().map {
                    builder.add_texture("map", *map, wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                    builder.add_sampler("map", *map, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
                }

                if let Some(map) = &m.bindings().normal_map {
                    builder.add_texture("normal_map", *map, wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                    builder.add_sampler("normal_map", *map, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
                }

                if let Some(map) = &m.bindings().specular_map {
                    builder.add_texture("specular_map", *map, wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                    builder.add_sampler("specular_map", *map, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
                }

                if let Some(map) = &m.bindings().emissive_map {
                    builder.add_texture("emissive_map", *map, wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                    builder.add_sampler("emissive_map", *map, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
                }
            },

            Self::Standard(m) => {
                builder.add_uniform::<MeshStandardUniforms>(
                    "material",
                    &m.uniforms,
                    wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX
                );
                
                if let Some(map) = &m.bindings().map {
                    builder.add_texture("map", *map, wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                    builder.add_sampler("map", *map, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
                }

                if let Some(map) = &m.bindings().normal_map {
                    builder.add_texture("normal_map", *map, wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                    builder.add_sampler("normal_map", *map, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
                }

                if let Some(map) = &m.bindings().roughness_map {
                    builder.add_texture("roughness_map", *map, wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                    builder.add_sampler("roughness_map", *map, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
                }

                if let Some(map) = &m.bindings().metalness_map {
                    builder.add_texture("metalness_map", *map, wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                    builder.add_sampler("metalness_map", *map, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
                }

                if let Some(map) = &m.bindings().ao_map {
                    builder.add_texture("ao_map", *map, wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                    builder.add_sampler("ao_map", *map, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
                }

                if let Some(map) = &m.bindings().emissive_map {
                    builder.add_texture("emissive_map", *map, wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D2, wgpu::ShaderStages::FRAGMENT);
                    builder.add_sampler("emissive_map", *map, wgpu::SamplerBindingType::Filtering, wgpu::ShaderStages::FRAGMENT);
                }
            }
        }
    }
}

impl Bindings for Material {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        self.data.define_bindings(builder);
    }
}

impl Bindings for Geometry {
    fn define_bindings<'a>(&'a self, _builder: &mut ResourceBuilder<'a>) {
        // for (morph_name, data) in &self.morph_attributes {

        // }
    }
}

impl Bindings for Mesh {
    fn define_bindings<'a>(&'a self, _builder: &mut ResourceBuilder<'a>) {
        // 
    }
}

impl Bindings for Environment {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        builder.add_uniform::<EnvironmentUniforms>(
            "environment",
            self.uniforms(),
            wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX
        );
        
        if !self.light_storage.read().is_empty() {
            builder.add_storage::<GpuLightStorage>(
                "lights",
                self.light_storage.handle(),
                self.light_storage.as_bytes(),
                true,
                wgpu::ShaderStages::FRAGMENT
            );
        }

        if let Some(env_map) = &self.bindings().env_map {
            builder.add_texture(
                "env_map", 
                *env_map, 
                wgpu::TextureSampleType::Float { filterable: true }, 
                wgpu::TextureViewDimension::Cube, 
                wgpu::ShaderStages::FRAGMENT
            );
            builder.add_sampler(
                "env_map", 
                *env_map, 
                wgpu::SamplerBindingType::Filtering, 
                wgpu::ShaderStages::FRAGMENT
            );
        }
    }
}

impl Bindings for RenderState {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        builder.add_uniform::<RenderStateUniforms>(
            "render_state",
            self.uniforms(),
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT
        );
    }
}
