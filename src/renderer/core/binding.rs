//! GPU 绑定资源
//!
//! 定义 BindGroup 的资源类型和绑定 Trait

use crate::assets::TextureHandle;
use crate::resources::texture::{SamplerSource, TextureSource};
use crate::{Mesh, Scene};
use crate::resources::buffer::BufferRef;
use crate::resources::material::Material;
use crate::resources::geometry::Geometry;
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
    Texture(Option<TextureSource>),
    Sampler(Option<SamplerSource>),
    _Phantom(std::marker::PhantomData<&'a ()>),
}

/// 绑定资源 Trait
pub trait Bindings {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>);
}

impl Bindings for Material {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        self.data.define_bindings(builder);
    }
}

impl Bindings for Geometry {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        // Morph Target Storage Buffers
        if self.has_morph_targets() {
            // Position morph storage
            if let (Some(buffer), Some(data)) = (&self.morph_position_buffer, self.morph_position_bytes()) {
                builder.add_storage_buffer(
                    "morph_positions",
                    buffer,
                    Some(data),
                    true,
                    wgpu::ShaderStages::VERTEX,
                    Some(crate::renderer::core::builder::WgslStructName::Name("f32".into()))
                );
            }
            
            // Normal morph storage (optional)
            if let (Some(buffer), Some(data)) = (&self.morph_normal_buffer, self.morph_normal_bytes()) {
                builder.add_storage_buffer(
                    "morph_normals",
                    buffer,
                    Some(data),
                    true,
                    wgpu::ShaderStages::VERTEX,
                    Some(crate::renderer::core::builder::WgslStructName::Name("f32".into()))
                );
            }
            
            // Tangent morph storage (optional)
            if let (Some(buffer), Some(data)) = (&self.morph_tangent_buffer, self.morph_tangent_bytes()) {
                builder.add_storage_buffer(
                    "morph_tangents",
                    buffer,
                    Some(data),
                    true,
                    wgpu::ShaderStages::VERTEX,
                    Some(crate::renderer::core::builder::WgslStructName::Name("f32".into()))
                );
            }
        }
    }
}

impl Bindings for Mesh {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        // todo: 是否需要检查 geometry features 包含 USE_MORPHING？
        builder.add_uniform::<MorphUniforms>(
            "morph_targets",
            &self.morph_uniforms,
            wgpu::ShaderStages::VERTEX
        );
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

impl Bindings for Scene {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        // Binding 1: Environment Uniforms
        builder.add_uniform_buffer(
            "environment",
            self.uniforms_buffer.handle(),
            Some(self.uniforms_buffer.as_bytes()),
            wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
            false,
            None,
            Some(crate::renderer::core::builder::WgslStructName::Generator(
                crate::resources::uniforms::EnvironmentUniforms::wgsl_struct_def
            ))
        );
        
        // Binding 2: Light Storage Buffer
        builder.add_storage_buffer(
            "lights",
            self.light_storage_buffer.handle(),
            Some(self.light_storage_buffer.as_bytes()),
            true,
            wgpu::ShaderStages::FRAGMENT,
            Some(crate::renderer::core::builder::WgslStructName::Generator(
                crate::resources::uniforms::GpuLightStorage::wgsl_struct_def
            ))
        );

        // Binding 3-4: Environment Map (Cube) and Sampler
        let env_map_handle = self.environment.pmrem_map.unwrap_or(TextureHandle::dummy_env_map().into());
        builder.add_texture(
            "env_map",
            Some(env_map_handle),
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::Cube,
            wgpu::ShaderStages::FRAGMENT
        );
        builder.add_sampler(
            "env_map",
            Some(SamplerSource::Default),
            wgpu::SamplerBindingType::Filtering,
            wgpu::ShaderStages::FRAGMENT
        );

        builder.add_texture(
            "pmrem_map",
            self.environment.pmrem_map,
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::Cube,
            wgpu::ShaderStages::FRAGMENT
        );
        builder.add_sampler(
            "pmrem_map",
            Some(SamplerSource::Default),
            wgpu::SamplerBindingType::Filtering,
            wgpu::ShaderStages::FRAGMENT
        );

        builder.add_texture(
            "brdf_lut",
            self.environment.brdf_lut,
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::D2,
            wgpu::ShaderStages::FRAGMENT
        );
        builder.add_sampler(
            "brdf_lut",
            Some(SamplerSource::Default),
            wgpu::SamplerBindingType::Filtering,
            wgpu::ShaderStages::FRAGMENT
        );
            
    }
}
