//! GPU 绑定资源
//!
//! 定义 BindGroup 的资源类型和绑定 Trait

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::assets::TextureHandle;
use crate::resources::texture::{SamplerSource, TextureSource};
use crate::{Mesh, Scene};
use crate::resources::buffer::BufferRef;
use crate::resources::material::{Material, RenderableMaterialTrait};
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
            &self.uniforms_buffer.handle(),
            None,
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
            &self.light_storage_buffer.handle(),
            None,
            true,
            wgpu::ShaderStages::FRAGMENT,
            Some(crate::renderer::core::builder::WgslStructName::Generator(
                crate::resources::uniforms::GpuLightStorage::wgsl_struct_def
            ))
        );

        // Binding 3-4: Environment Map (Cube) and Sampler - use processed_env_map for Skybox
        let env_map_handle = self.environment.get_processed_env_map()
            .cloned()
            .unwrap_or(TextureHandle::dummy_env_map().into());
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



pub struct GlobalBindGroupCache {
    cache: FxHashMap<BindGroupKey, wgpu::BindGroup>,
}

impl GlobalBindGroupCache {
    pub fn new() -> Self {
        Self {
            cache: FxHashMap::default(),
        }
    }

    pub fn get(&self, key: &BindGroupKey) -> Option<&wgpu::BindGroup> {
        self.cache.get(key)
    }

    pub fn insert(&mut self, key: BindGroupKey, bind_group: wgpu::BindGroup) {
        self.cache.insert(key, bind_group);
    }

    pub fn get_or_create(
        &mut self,
        key: BindGroupKey,
        factory: impl FnOnce() -> wgpu::BindGroup,
    ) -> &wgpu::BindGroup {
        self.cache.entry(key).or_insert_with(factory)
    }

    /// 在 Resize 时调用，彻底清空
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}


/// 全局 BindGroup 缓存键
/// 包含 Layout 的唯一标识 + 所有绑定资源的唯一标识
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BindGroupKey {
    layout_id: u64,      
    resources: SmallVec<[u64; 8]>,
}

impl BindGroupKey {
    pub fn new(layout_id: u64) -> Self {
        Self {
            layout_id,
            resources: SmallVec::with_capacity(8), // 预估常见大小
        }
    }

    pub fn with_resource(mut self, id: u64) -> Self {
        self.resources.push(id);
        self
    }
}