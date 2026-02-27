//! GPU binding resources
//!
//! Defines `BindGroup` resource types and the binding Trait

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::renderer::core::builder::ResourceBuilder;
use crate::renderer::graph::RenderState;
use crate::resources::buffer::BufferRef;
use crate::resources::geometry::Geometry;
use crate::resources::material::{Material, RenderableMaterialTrait};
use crate::resources::texture::{SamplerSource, TextureSource};
use crate::resources::uniforms::{MorphUniforms, RenderStateUniforms};
use crate::{Mesh, Scene};

/// Actual binding resource data (used for generating `BindGroup`)
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

/// Binding resource Trait
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
            if let (Some(buffer), Some(data)) =
                (&self.morph_position_buffer, self.morph_position_bytes())
            {
                builder.add_storage_buffer(
                    "morph_positions",
                    buffer,
                    Some(data),
                    true,
                    wgpu::ShaderStages::VERTEX,
                    Some(crate::renderer::core::builder::WgslStructName::Name(
                        "f32".into(),
                    )),
                );
            }

            // Normal morph storage (optional)
            if let (Some(buffer), Some(data)) =
                (&self.morph_normal_buffer, self.morph_normal_bytes())
            {
                builder.add_storage_buffer(
                    "morph_normals",
                    buffer,
                    Some(data),
                    true,
                    wgpu::ShaderStages::VERTEX,
                    Some(crate::renderer::core::builder::WgslStructName::Name(
                        "f32".into(),
                    )),
                );
            }

            // Tangent morph storage (optional)
            if let (Some(buffer), Some(data)) =
                (&self.morph_tangent_buffer, self.morph_tangent_bytes())
            {
                builder.add_storage_buffer(
                    "morph_tangents",
                    buffer,
                    Some(data),
                    true,
                    wgpu::ShaderStages::VERTEX,
                    Some(crate::renderer::core::builder::WgslStructName::Name(
                        "f32".into(),
                    )),
                );
            }
        }
    }
}

impl Bindings for Mesh {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        // todo: should we check if geometry features contain USE_MORPHING?
        builder.add_uniform::<MorphUniforms>(
            "morph_targets",
            &self.morph_uniforms,
            wgpu::ShaderStages::VERTEX,
        );
    }
}

impl Bindings for RenderState {
    fn define_bindings<'a>(&'a self, builder: &mut ResourceBuilder<'a>) {
        builder.add_uniform::<RenderStateUniforms>(
            "render_state",
            self.uniforms(),
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        );
    }
}

impl Bindings for Scene {
    fn define_bindings<'a>(&'a self, _builder: &mut ResourceBuilder<'a>) {
        // Scene-level global bindings are now built by
        // `ResourceManager::define_global_scene_bindings` which resolves
        // environment textures from the GPU cache instead of from Environment.
        // This impl is kept empty for trait coherence; the actual bindings
        // are constructed in `ResourceManager::create_global_state`.
    }
}

pub struct GlobalBindGroupCache {
    cache: FxHashMap<BindGroupKey, wgpu::BindGroup>,
}

impl Default for GlobalBindGroupCache {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalBindGroupCache {
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: FxHashMap::default(),
        }
    }

    #[must_use]
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

    /// Called on resize to completely clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

/// Global `BindGroup` cache key
/// Contains the unique identifier of the Layout + unique identifiers of all binding resources
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BindGroupKey {
    layout_id: u64,
    resources: SmallVec<[u64; 8]>,
}

impl BindGroupKey {
    #[must_use]
    pub fn new(layout_id: u64) -> Self {
        Self {
            layout_id,
            resources: SmallVec::with_capacity(8), // Estimated common size
        }
    }

    #[must_use]
    pub fn with_resource(mut self, id: u64) -> Self {
        self.resources.push(id);
        self
    }
}
