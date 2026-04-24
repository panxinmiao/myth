use std::marker::PhantomData;

use crate::core::ResourceManager;
use crate::core::WgpuContext;
use crate::core::binding::{BindGroupKey, GlobalBindGroupCache};
use crate::core::gpu::{CommonSampler, MipmapGenerator, SamplerRegistry, SystemTextures, Tracked};
use crate::graph::frame::{BakedRenderLists, RenderLists};
use crate::graph::{ExtractedScene, RenderState};
use crate::pipeline::{PipelineCache, ShaderManager};
use myth_assets::AssetServer;
use myth_scene::RenderCamera;
use wgpu::{Device, Queue, TextureView};

use super::allocator::{SubViewKey, TransientPool};
use super::types::{
    BufferNodeId, GraphResourceType, RenderTargetOps, ResourceKind, ResourceNodeId, ResourceRecord,
    TextureNodeId,
};

// ─── Extract Context (Feature Pre-RDG Phase) ────────────────────────────────

/// Rich context available during the **Feature extract-and-prepare** phase.
///
/// Each `Feature::extract_and_prepare(&mut self, ctx: &mut ExtractContext)`
/// is called **before** the render graph is built.  The context provides full
/// access to GPU infrastructure, scene data, and the asset server so that
/// features can:
///
/// - Create / cache `wgpu::BindGroupLayout`s
/// - Compile pipelines via [`PipelineCache`]
/// - Upload non-transient GPU data (uniform buffers, noise textures, etc.)
/// - Assemble cached persistent bind groups through
///   [`ExtractContext::build_bind_group`]
///
/// After this phase, Features hold only lightweight pipeline IDs.  The
/// per-frame ephemeral `PassNode` created by `Feature::add_to_graph()`
/// carries those IDs into the graph.
pub struct ExtractContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub pipeline_cache: &'a mut PipelineCache,
    pub shader_manager: &'a mut ShaderManager,
    // pub sampler_registry: &'a mut SamplerRegistry,
    pub global_bind_group_cache: &'a mut GlobalBindGroupCache,
    pub resource_manager: &'a mut ResourceManager,
    pub wgpu_ctx: &'a WgpuContext,
    pub render_lists: &'a mut RenderLists,
    pub extracted_scene: &'a ExtractedScene,
    pub render_state: &'a RenderState,
    pub assets: &'a AssetServer,
    pub render_camera: &'a RenderCamera,
}

// ─── Prepare Context (Transient-Only) ─────────────────────────────────────────

/// Minimal context available during the RDG **prepare** phase.
///
/// After the render graph has been compiled and transient resources allocated,
/// each pass's [`PassNode::prepare`] receives this context to assemble
/// `wgpu::BindGroup`s that reference RDG-managed textures and buffers.
///
/// This context is deliberately kept **pure**: it provides only the GPU
/// device, transient view resolver, sampler registry, and the global bind
/// group cache.  Passes then build bind groups through
/// [`PrepareContext::build_bind_group`], which centralizes cache-key
/// construction and safe logical-size truncation for RDG buffers.
pub struct PrepareContext<'a> {
    /// Transient resource view resolver.
    pub views: ViewResolver<'a>,
    /// GPU device for creating bind groups and sub-views.
    pub device: &'a wgpu::Device,
    /// GPU queue for immediate buffer uploads (rare in prepare).
    pub queue: &'a wgpu::Queue,
    /// Shared sampler registry (persistent, immutable during prepare).
    pub sampler_registry: &'a SamplerRegistry,
    /// Mutable cache for transient bind groups with TTL eviction.
    pub global_bind_group_cache: &'a mut GlobalBindGroupCache,
    /// System fallback textures and Group 3 bind-group infrastructure.
    pub system_textures: &'a SystemTextures,
}

pub struct ViewResolver<'a> {
    pub resources: &'a [ResourceRecord],
    pub pool: &'a mut TransientPool,
}

#[inline]
fn common_sampler_resource_id(sampler: CommonSampler) -> u64 {
    0x434f_4d4d_4f4e_0000 | sampler as u64
}

/// Explicit binding wrapper for raw buffers that are not RDG-managed.
///
/// Callers must provide a stable `resource_id` for cache-key construction.
/// Use this path sparingly: `BufferNodeId` remains the preferred API because
/// it applies logical-size truncation automatically for pooled transient
/// buffers.
#[derive(Clone, Copy)]
pub struct RawBufferBinding<'a> {
    buffer: &'a wgpu::Buffer,
    resource_id: u64,
    size: Option<wgpu::BufferSize>,
}

impl<'a> RawBufferBinding<'a> {
    #[must_use]
    pub fn new(buffer: &'a wgpu::Buffer, resource_id: u64, size: Option<wgpu::BufferSize>) -> Self {
        Self {
            buffer,
            resource_id,
            size,
        }
    }

    #[must_use]
    pub fn whole(buffer: &'a wgpu::Buffer, resource_id: u64) -> Self {
        Self::new(buffer, resource_id, None)
    }
}

/// Explicit binding wrapper for raw texture views that are not stored as
/// `Tracked<wgpu::TextureView>`.
#[derive(Clone, Copy)]
pub struct RawTextureViewBinding<'a> {
    view: &'a wgpu::TextureView,
    resource_id: u64,
}

impl<'a> RawTextureViewBinding<'a> {
    #[must_use]
    pub const fn new(view: &'a wgpu::TextureView, resource_id: u64) -> Self {
        Self { view, resource_id }
    }
}

/// Explicit binding wrapper for raw samplers that do not come from the
/// shared sampler registry.
#[derive(Clone, Copy)]
pub struct RawSamplerBinding<'a> {
    sampler: &'a wgpu::Sampler,
    resource_id: u64,
}

impl<'a> RawSamplerBinding<'a> {
    #[must_use]
    pub const fn new(sampler: &'a wgpu::Sampler, resource_id: u64) -> Self {
        Self {
            sampler,
            resource_id,
        }
    }
}

/// Unified binding descriptor used by [`BindGroupBuilder`] to hide the
/// differences between RDG-managed and raw WGPU resources.
#[derive(Clone, Copy)]
pub enum GraphBinding<'a> {
    Buffer(BufferNodeId),
    Texture(TextureNodeId),
    TrackedBuffer(&'a Tracked<wgpu::Buffer>),
    TrackedTextureView(&'a Tracked<wgpu::TextureView>),
    TrackedSampler(&'a Tracked<wgpu::Sampler>),
    RawBuffer(RawBufferBinding<'a>),
    RawTextureView(RawTextureViewBinding<'a>),
    Sampler(RawSamplerBinding<'a>),
    CommonSampler(CommonSampler),
}

/// Fluent bind-group builder shared by feature extract code and RDG prepare.
///
/// The critical safety property lives in [`Self::bind_buffer`]: binding a
/// `BufferNodeId` always resolves through [`ViewResolver::get_buffer_binding`],
/// which truncates the physical pooled allocation to the resource's logical
/// size. This prevents shaders from reading stale bytes past the logical end
/// of a transient power-of-two buffer allocation.
pub struct BindGroupBuilder<'ctx, 'frame> {
    views: Option<&'ctx ViewResolver<'frame>>,
    cache: &'ctx mut GlobalBindGroupCache,
    device: &'ctx wgpu::Device,
    sampler_registry: &'ctx SamplerRegistry,
    layout: &'ctx wgpu::BindGroupLayout,
    label: Option<&'static str>,
    key: BindGroupKey,
    entries: Vec<wgpu::BindGroupEntry<'ctx>>,
    _marker: PhantomData<&'frame wgpu::BindGroup>,
}

impl<'ctx, 'frame> BindGroupBuilder<'ctx, 'frame> {
    fn new(
        views: Option<&'ctx ViewResolver<'frame>>,
        cache: &'ctx mut GlobalBindGroupCache,
        device: &'ctx wgpu::Device,
        sampler_registry: &'ctx SamplerRegistry,
        layout: &'ctx Tracked<wgpu::BindGroupLayout>,
        label: Option<&'static str>,
    ) -> Self {
        Self {
            views,
            cache,
            device,
            sampler_registry,
            layout,
            label,
            key: BindGroupKey::new(layout.id()),
            entries: Vec::with_capacity(8),
            _marker: PhantomData,
        }
    }

    fn push_entry(
        mut self,
        binding: u32,
        resource_id: u64,
        resource: wgpu::BindingResource<'ctx>,
    ) -> Self {
        self.key = self.key.with_resource(resource_id);
        self.entries
            .push(wgpu::BindGroupEntry { binding, resource });
        self
    }

    #[must_use]
    pub fn bind_graph_binding(self, binding: u32, resource: GraphBinding<'ctx>) -> Self {
        match resource {
            GraphBinding::Buffer(id) => self.bind_buffer(binding, id),
            GraphBinding::Texture(id) => self.bind_texture(binding, id),
            GraphBinding::TrackedBuffer(buffer) => self.bind_tracked_buffer(binding, buffer),
            GraphBinding::TrackedTextureView(view) => self.bind_tracked_texture_view(binding, view),
            GraphBinding::TrackedSampler(sampler) => self.bind_tracked_sampler(binding, sampler),
            GraphBinding::RawBuffer(buffer) => self.bind_raw_buffer(binding, buffer),
            GraphBinding::RawTextureView(view) => self.bind_raw_texture_view(binding, view),
            GraphBinding::Sampler(sampler) => self.bind_raw_sampler(binding, sampler),
            GraphBinding::CommonSampler(sampler) => self.bind_common_sampler(binding, sampler),
        }
    }

    #[must_use]
    pub fn bind_resource<R>(self, binding: u32, resource: R) -> Self
    where
        R: BindableResource<'ctx, 'frame>,
    {
        resource.add_to_builder(binding, self)
    }

    /// Binds an RDG buffer with automatic logical-size truncation.
    #[must_use]
    pub fn bind_buffer(self, binding: u32, id: BufferNodeId) -> Self {
        let views = self
            .views
            .expect("BufferNodeId binding requires PrepareContext-backed RDG views");
        let resource_id = views.get_physical_buffer_uid(id);
        let resource = wgpu::BindingResource::Buffer(views.get_buffer_binding(id));
        self.push_entry(binding, resource_id, resource)
    }

    #[must_use]
    pub fn bind_texture(self, binding: u32, id: TextureNodeId) -> Self {
        let views = self
            .views
            .expect("TextureNodeId binding requires PrepareContext-backed RDG views");
        let view = views.get_texture_view(id);
        self.push_entry(binding, view.id(), wgpu::BindingResource::TextureView(view))
    }

    #[must_use]
    pub fn bind_tracked_buffer_with_size(
        self,
        binding: u32,
        buffer: &'ctx Tracked<wgpu::Buffer>,
        size: Option<wgpu::BufferSize>,
    ) -> Self {
        self.push_entry(
            binding,
            buffer.id(),
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size,
            }),
        )
    }

    #[must_use]
    pub fn bind_tracked_buffer(self, binding: u32, buffer: &'ctx Tracked<wgpu::Buffer>) -> Self {
        self.bind_tracked_buffer_with_size(binding, buffer, None)
    }

    #[must_use]
    pub fn bind_tracked_texture_view(
        self,
        binding: u32,
        view: &'ctx Tracked<wgpu::TextureView>,
    ) -> Self {
        self.push_entry(binding, view.id(), wgpu::BindingResource::TextureView(view))
    }

    #[must_use]
    pub fn bind_tracked_sampler(self, binding: u32, sampler: &'ctx Tracked<wgpu::Sampler>) -> Self {
        self.push_entry(
            binding,
            sampler.id(),
            wgpu::BindingResource::Sampler(sampler),
        )
    }

    #[must_use]
    pub fn bind_raw_buffer(self, binding: u32, raw: RawBufferBinding<'ctx>) -> Self {
        self.push_entry(
            binding,
            raw.resource_id,
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: raw.buffer,
                offset: 0,
                size: raw.size,
            }),
        )
    }

    #[must_use]
    pub fn bind_raw_texture_view(self, binding: u32, raw: RawTextureViewBinding<'ctx>) -> Self {
        self.push_entry(
            binding,
            raw.resource_id,
            wgpu::BindingResource::TextureView(raw.view),
        )
    }

    #[must_use]
    pub fn bind_texture_view_with_id(
        self,
        binding: u32,
        view: &'ctx wgpu::TextureView,
        resource_id: u64,
    ) -> Self {
        self.bind_raw_texture_view(binding, RawTextureViewBinding::new(view, resource_id))
    }

    #[must_use]
    pub fn bind_raw_sampler(self, binding: u32, raw: RawSamplerBinding<'ctx>) -> Self {
        self.push_entry(
            binding,
            raw.resource_id,
            wgpu::BindingResource::Sampler(raw.sampler),
        )
    }

    #[must_use]
    pub fn bind_sampler_with_id(
        self,
        binding: u32,
        sampler: &'ctx wgpu::Sampler,
        resource_id: u64,
    ) -> Self {
        self.bind_raw_sampler(binding, RawSamplerBinding::new(sampler, resource_id))
    }

    #[must_use]
    pub fn bind_sampler(self, binding: u32, sampler: &'ctx wgpu::Sampler) -> Self {
        self.bind_sampler_with_id(
            binding,
            sampler,
            (std::ptr::from_ref(sampler) as *const wgpu::Sampler as usize) as u64,
        )
    }

    #[must_use]
    pub fn bind_common_sampler(self, binding: u32, sampler: CommonSampler) -> Self {
        let resource_id = common_sampler_resource_id(sampler);
        let resource = wgpu::BindingResource::Sampler(self.sampler_registry.get_common(sampler));
        self.push_entry(binding, resource_id, resource)
    }

    /// Builds or reuses a cached bind group and returns a frame-stable
    /// reference suitable for storing on a `PassNode<'frame>`.
    #[must_use]
    pub fn build(self) -> &'frame wgpu::BindGroup {
        let Self {
            cache,
            device,
            layout,
            label,
            key,
            entries,
            ..
        } = self;

        cache.get_or_create_bg(key, || {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label,
                layout,
                entries: &entries,
            })
        })
    }
}

/// Trait implemented by resources that can append themselves to a
/// [`BindGroupBuilder`].
pub trait BindableResource<'ctx, 'frame>: Sized {
    fn add_to_builder(
        self,
        binding: u32,
        builder: BindGroupBuilder<'ctx, 'frame>,
    ) -> BindGroupBuilder<'ctx, 'frame>;
}

impl<'ctx, 'frame> BindableResource<'ctx, 'frame> for GraphBinding<'ctx> {
    fn add_to_builder(
        self,
        binding: u32,
        builder: BindGroupBuilder<'ctx, 'frame>,
    ) -> BindGroupBuilder<'ctx, 'frame> {
        builder.bind_graph_binding(binding, self)
    }
}

impl<'ctx, 'frame> BindableResource<'ctx, 'frame> for BufferNodeId {
    fn add_to_builder(
        self,
        binding: u32,
        builder: BindGroupBuilder<'ctx, 'frame>,
    ) -> BindGroupBuilder<'ctx, 'frame> {
        builder.bind_buffer(binding, self)
    }
}

impl<'ctx, 'frame> BindableResource<'ctx, 'frame> for TextureNodeId {
    fn add_to_builder(
        self,
        binding: u32,
        builder: BindGroupBuilder<'ctx, 'frame>,
    ) -> BindGroupBuilder<'ctx, 'frame> {
        builder.bind_texture(binding, self)
    }
}

impl<'ctx, 'frame> BindableResource<'ctx, 'frame> for &'ctx Tracked<wgpu::Buffer> {
    fn add_to_builder(
        self,
        binding: u32,
        builder: BindGroupBuilder<'ctx, 'frame>,
    ) -> BindGroupBuilder<'ctx, 'frame> {
        builder.bind_tracked_buffer(binding, self)
    }
}

impl<'ctx, 'frame> BindableResource<'ctx, 'frame> for &'ctx Tracked<wgpu::TextureView> {
    fn add_to_builder(
        self,
        binding: u32,
        builder: BindGroupBuilder<'ctx, 'frame>,
    ) -> BindGroupBuilder<'ctx, 'frame> {
        builder.bind_tracked_texture_view(binding, self)
    }
}

impl<'ctx, 'frame> BindableResource<'ctx, 'frame> for &'ctx Tracked<wgpu::Sampler> {
    fn add_to_builder(
        self,
        binding: u32,
        builder: BindGroupBuilder<'ctx, 'frame>,
    ) -> BindGroupBuilder<'ctx, 'frame> {
        builder.bind_tracked_sampler(binding, self)
    }
}

impl<'ctx, 'frame> BindableResource<'ctx, 'frame> for RawBufferBinding<'ctx> {
    fn add_to_builder(
        self,
        binding: u32,
        builder: BindGroupBuilder<'ctx, 'frame>,
    ) -> BindGroupBuilder<'ctx, 'frame> {
        builder.bind_raw_buffer(binding, self)
    }
}

impl<'ctx, 'frame> BindableResource<'ctx, 'frame> for RawTextureViewBinding<'ctx> {
    fn add_to_builder(
        self,
        binding: u32,
        builder: BindGroupBuilder<'ctx, 'frame>,
    ) -> BindGroupBuilder<'ctx, 'frame> {
        builder.bind_raw_texture_view(binding, self)
    }
}

impl<'ctx, 'frame> BindableResource<'ctx, 'frame> for RawSamplerBinding<'ctx> {
    fn add_to_builder(
        self,
        binding: u32,
        builder: BindGroupBuilder<'ctx, 'frame>,
    ) -> BindGroupBuilder<'ctx, 'frame> {
        builder.bind_raw_sampler(binding, self)
    }
}

impl<'ctx, 'frame> BindableResource<'ctx, 'frame> for CommonSampler {
    fn add_to_builder(
        self,
        binding: u32,
        builder: BindGroupBuilder<'ctx, 'frame>,
    ) -> BindGroupBuilder<'ctx, 'frame> {
        builder.bind_common_sampler(binding, self)
    }
}

#[macro_export]
macro_rules! myth_bind_group {
    ($ctx:expr, $layout:expr, $label:expr, [ $( $binding:expr => $res:expr ),* $(,)? ]) => {{
        let builder = $ctx.build_bind_group($layout, $label);
        $(
            let builder = builder.bind_resource($binding, $res);
        )*
        builder.build()
    }};
}

impl<'frame> PrepareContext<'frame> {
    /// Starts a cached bind-group build using RDG-safe resource resolution.
    #[must_use]
    pub fn build_bind_group<'ctx>(
        &'ctx mut self,
        layout: &'ctx Tracked<wgpu::BindGroupLayout>,
        label: Option<&'static str>,
    ) -> BindGroupBuilder<'ctx, 'frame> {
        BindGroupBuilder::new(
            Some(&self.views),
            self.global_bind_group_cache,
            self.device,
            self.sampler_registry,
            layout,
            label,
        )
    }
}

impl ExtractContext<'_> {
    /// Starts a cached bind-group build for persistent resources during
    /// feature extract-and-prepare.
    #[must_use]
    pub fn build_bind_group<'ctx>(
        &'ctx mut self,
        layout: &'ctx Tracked<wgpu::BindGroupLayout>,
        label: Option<&'static str>,
    ) -> BindGroupBuilder<'ctx, 'ctx> {
        BindGroupBuilder::new(
            None,
            self.global_bind_group_cache,
            self.device,
            &self.resource_manager.sampler_registry,
            layout,
            label,
        )
    }
}

#[must_use]
pub fn resolve_root_id<T: GraphResourceType>(
    resources: &[ResourceRecord],
    mut id: ResourceNodeId<T>,
) -> ResourceNodeId<T> {
    while let Some(parent) = resources[id.index() as usize].alias_of {
        id = ResourceNodeId::from_erased(parent);
    }
    id
}

#[inline]
fn resolve_texture_resource(
    resources: &[ResourceRecord],
    id: TextureNodeId,
) -> (TextureNodeId, &ResourceRecord) {
    let root_id = resolve_root_id(resources, id);
    let res = &resources[root_id.index() as usize];
    debug_assert!(matches!(res.kind, ResourceKind::Texture { .. }));
    (root_id, res)
}

#[inline]
fn resolve_buffer_resource(
    resources: &[ResourceRecord],
    id: BufferNodeId,
) -> (BufferNodeId, &ResourceRecord) {
    let root_id = resolve_root_id(resources, id);
    let res = &resources[root_id.index() as usize];
    debug_assert!(matches!(res.kind, ResourceKind::Buffer { .. }));
    (root_id, res)
}

impl ViewResolver<'_> {
    /// Resolve a virtual [`TextureNodeId`] to its physical [`Tracked<TextureView>`].
    ///
    /// For external resources, the view is looked up in `external_resources`.
    /// For transient resources, the **default** view is obtained from the pool.
    #[must_use]
    pub fn get_texture_view(&self, id: TextureNodeId) -> &Tracked<wgpu::TextureView> {
        let (_, res) = resolve_texture_resource(self.resources, id);

        if res.is_external {
            let ptr = res
                .external_texture_ptr()
                .expect("External texture missing view pointer!");
            unsafe { &*ptr }
        } else {
            let physical_index = res.physical_index.expect("No physical memory!");
            self.pool.get_tracked_view(physical_index)
        }
    }

    /// Returns the physical-texture allocation UID for the given node.
    ///
    /// Useful for dirty-checking: if the UID hasn't changed between frames,
    /// the physical texture is the same and derived state can be reused.
    #[must_use]
    pub fn get_physical_texture_uid(&self, id: TextureNodeId) -> u64 {
        let (_, res) = resolve_texture_resource(self.resources, id);
        let physical_index = res.physical_index.expect("No physical memory!");
        self.pool.get_uid(physical_index)
    }

    /// Returns the raw `wgpu::Texture` handle for the given node.
    ///
    /// Useful for passes that need to create custom views (e.g. Bloom mip chain).
    #[must_use]
    pub fn get_texture(&self, id: TextureNodeId) -> &wgpu::Texture {
        let (_, res) = resolve_texture_resource(self.resources, id);
        let physical_index = res.physical_index.expect("No physical memory!");
        self.pool.get_texture(physical_index)
    }

    /// Lazily creates and caches a sub-view for a transient resource.
    ///
    /// Typical use: obtaining a `DepthOnly` aspect view from the combined
    /// depth-stencil texture for bind-group sampling.
    pub fn get_or_create_sub_view(
        &mut self,
        id: TextureNodeId,
        key: &SubViewKey,
    ) -> &Tracked<wgpu::TextureView> {
        let (_, res) = resolve_texture_resource(self.resources, id);
        let physical_index = res.physical_index.expect("No physical memory!");
        self.pool.get_or_create_sub_view(physical_index, key)
    }

    #[must_use]
    pub fn get_sub_view(
        &self,
        id: TextureNodeId,
        key: &SubViewKey,
    ) -> Option<&Tracked<wgpu::TextureView>> {
        let (_, res) = resolve_texture_resource(self.resources, id);
        let physical_index = res.physical_index.expect("No physical memory!");
        self.pool.get_sub_view(physical_index, key)
    }

    /// Resolve a virtual [`BufferNodeId`] to its tracked physical buffer.
    #[must_use]
    pub fn get_tracked_buffer(&self, id: BufferNodeId) -> &Tracked<wgpu::Buffer> {
        let (_, res) = resolve_buffer_resource(self.resources, id);

        if res.is_external {
            let ptr = res
                .external_buffer_ptr()
                .expect("External buffer missing pointer!");
            unsafe { &*ptr }
        } else {
            let physical_index = res.physical_index.expect("No physical memory!");
            self.pool.get_tracked_buffer(physical_index)
        }
    }

    /// Returns the raw `wgpu::Buffer` handle for the given node.
    #[must_use]
    pub fn get_buffer(&self, id: BufferNodeId) -> &wgpu::Buffer {
        let tracked = self.get_tracked_buffer(id);
        tracked
    }

    /// Returns the physical-buffer allocation UID for the given node.
    #[must_use]
    pub fn get_physical_buffer_uid(&self, id: BufferNodeId) -> u64 {
        let (_, res) = resolve_buffer_resource(self.resources, id);

        if res.is_external {
            self.get_tracked_buffer(id).id()
        } else {
            let physical_index = res.physical_index.expect("No physical memory!");
            self.pool.get_buffer_uid(physical_index)
        }
    }

    /// Returns a buffer binding truncated to the resource's logical size.
    #[must_use]
    pub fn get_buffer_binding(&self, id: BufferNodeId) -> wgpu::BufferBinding<'_> {
        let (_, res) = resolve_buffer_resource(self.resources, id);
        let desc = res.buffer_desc();
        let size = desc.logical_binding_size().unwrap_or_else(|| {
            panic!(
                "Buffer '{}' has zero logical size and cannot be bound",
                res.name
            )
        });

        wgpu::BufferBinding {
            buffer: self.get_buffer(id),
            offset: 0,
            size: Some(size),
        }
    }

    /// Returns a buffer slice truncated to the resource's logical size.
    #[must_use]
    pub fn get_buffer_slice(&self, id: BufferNodeId) -> wgpu::BufferSlice<'_> {
        let (_, res) = resolve_buffer_resource(self.resources, id);
        self.get_buffer(id).slice(0..res.buffer_desc().logical_size)
    }

    /// Returns `true` if the resource has a physical GPU allocation.
    ///
    /// A resource is considered allocated if it is external (always backed by
    /// caller-provided memory) or if the graph compiler assigned a physical
    /// pool slot.  Dead resources — those written but never read — will
    /// return `false`, allowing passes to skip bind-group creation and
    /// select leaner pipeline variants in [`PassNode::prepare`].
    #[inline]
    #[must_use]
    pub fn is_resource_allocated<T: GraphResourceType>(&self, id: ResourceNodeId<T>) -> bool {
        let res = &self.resources[id.index() as usize];
        res.is_external || res.physical_index.is_some()
    }
}

// ─── PrepareContext Helpers ────────────────────────────────────────────────────

/// Build the screen / transient bind group (Group 3), returning a
/// pointer-stable `&'a` reference.
///
/// Encapsulates key construction and cache lookup for the screen bind group
/// used by Opaque, Transparent, and SimpleForward passes.
///
/// Callers must destructure [`PrepareContext`] to obtain split borrows of
/// `global_bind_group_cache` and `device` before calling this function.
pub fn build_screen_bind_group<'a>(
    cache: &mut GlobalBindGroupCache,
    device: &wgpu::Device,
    sys: &SystemTextures,
    transmission_view: &Tracked<wgpu::TextureView>,
    ssao_view: &Tracked<wgpu::TextureView>,
    shadow_view: &Tracked<wgpu::TextureView>,
    shadow_cube_view: &Tracked<wgpu::TextureView>,
) -> &'a wgpu::BindGroup {
    let key = BindGroupKey::new(sys.screen_layout.id())
        .with_resource(transmission_view.id())
        .with_resource(sys.screen_sampler.id())
        .with_resource(ssao_view.id())
        .with_resource(shadow_view.id())
        .with_resource(sys.shadow_compare_sampler.id())
        .with_resource(shadow_cube_view.id());

    cache.get_or_create_bg(key, || {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Screen BindGroup (Group 3)"),
            layout: &sys.screen_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(transmission_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sys.screen_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(ssao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(shadow_cube_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&sys.shadow_compare_sampler),
                },
            ],
        })
    })
}

// ─── Execute Context ──────────────────────────────────────────────────────────

/// Immutable context available during the RDG **execute** phase.
///
/// Provides read-only access to the compiled render graph, physical resource
/// pool, pipeline cache, render lists, and any external views injected before
/// execution.
pub struct ExecuteContext<'a> {
    pub resources: &'a [ResourceRecord],
    pub pool: &'a TransientPool,
    pub device: &'a Device,
    pub queue: &'a Queue,
    pub pipeline_cache: &'a PipelineCache,
    pub global_bind_group_cache: &'a GlobalBindGroupCache,

    // ─── Scene Data (Phase 3: full RDG integration) ──────────────────
    /// GPU resource manager (read-only) — used by compute, post-processing,
    /// and skybox passes that have not yet been migrated to baked commands.
    pub mipmap_generator: &'a MipmapGenerator,

    /// Pre-baked draw command lists with all GPU handles resolved.
    ///
    /// Scene-drawing passes (opaque, transparent, shadow, prepass,
    /// simple-forward) consume these instead of performing per-command
    /// handle lookups via `resource_manager`.
    pub baked_lists: &'a BakedRenderLists<'a>,

    /// Full wgpu context — depth format, render path, etc.
    pub wgpu_ctx: &'a WgpuContext,

    /// Index of the currently executing pass within the compiled execution
    /// queue timeline.  Used by [`get_color_attachment`] and
    /// [`get_depth_stencil_attachment`] to auto-deduce `LoadOp` / `StoreOp`.
    pub current_timeline_index: usize,
}

impl ExecuteContext<'_> {
    /// Resolve a virtual [`TextureNodeId`] to its physical [`TextureView`].
    ///
    /// For external resources, the view is looked up in `external_views`.
    /// For transient resources, the view is obtained from the physical pool.
    #[must_use]
    pub fn get_texture_view(&self, id: TextureNodeId) -> &TextureView {
        let (_, res) = resolve_texture_resource(self.resources, id);

        if res.is_external {
            let ptr = res
                .external_texture_ptr()
                .expect("External texture missing view pointer!");
            unsafe { &*ptr }
        } else {
            let physical_index = res
                .physical_index
                .expect("Transient resource has no physical memory assigned!");
            self.pool.get_view(physical_index)
        }
    }

    /// Returns the [`Tracked<TextureView>`] for cache-key use during execute.
    #[must_use]
    pub fn get_tracked_texture_view(&self, id: TextureNodeId) -> &Tracked<wgpu::TextureView> {
        let (_, res) = resolve_texture_resource(self.resources, id);

        if res.is_external {
            let ptr = res
                .external_texture_ptr()
                .expect("External texture missing view pointer!");

            (unsafe { &*ptr }) as _
        } else {
            let physical_index = res
                .physical_index
                .expect("Resource has no physical memory!");
            self.pool.get_tracked_view(physical_index)
        }
    }

    /// Returns the raw [`wgpu::Texture`] handle for the given node.
    ///
    /// Useful for passes that need to create custom views at execute time
    /// (e.g. per-layer shadow map views from a 2D-array texture).
    #[must_use]
    pub fn get_texture(&self, id: TextureNodeId) -> &wgpu::Texture {
        let (_, res) = resolve_texture_resource(self.resources, id);

        if res.is_external {
            let ptr = res
                .external_texture_ptr()
                .expect("External texture missing view pointer!");
            let tracked_view = unsafe { &*ptr };
            tracked_view.texture()
        } else {
            let physical_index = res
                .physical_index
                .expect("Transient resource has no physical memory assigned!");
            self.pool.get_texture(physical_index)
        }
    }

    /// Safely resolve a [`TextureNodeId`] to its physical [`TextureView`].
    ///
    /// Returns `None` if the resource was culled by the graph compiler
    /// (i.e. it has no consumers and no physical allocation).  Passes
    /// should use this for optional MRT targets that may have been
    /// optimized out.
    #[must_use]
    pub fn try_get_texture_view(&self, id: TextureNodeId) -> Option<&TextureView> {
        let (_, res) = resolve_texture_resource(self.resources, id);

        if res.is_external {
            let ptr = res.external_texture_ptr()?;
            let tracked = unsafe { &*ptr };
            Some(&**tracked) // Deref Tracked 获得 wgpu::TextureView
        } else {
            res.physical_index.map(|idx| self.pool.get_view(idx))
        }
    }

    #[must_use]
    pub fn try_get_base_mip_view(&self, id: TextureNodeId) -> Option<&TextureView> {
        let (_, res) = resolve_texture_resource(self.resources, id);

        if res.is_external {
            let ptr = res.external_texture_ptr()?;
            let tracked = unsafe { &*ptr };
            Some(&**tracked) // Deref Tracked 获得 wgpu::TextureView
        } else {
            res.physical_index
                .map(|idx| self.pool.get_base_mip_view(idx))
        }
    }

    /// Returns the tracked buffer handle for the given node.
    #[must_use]
    pub fn get_tracked_buffer(&self, id: BufferNodeId) -> &Tracked<wgpu::Buffer> {
        let (_, res) = resolve_buffer_resource(self.resources, id);

        if res.is_external {
            let ptr = res
                .external_buffer_ptr()
                .expect("External buffer missing pointer!");
            unsafe { &*ptr }
        } else {
            let physical_index = res
                .physical_index
                .expect("Transient resource has no physical memory assigned!");
            self.pool.get_tracked_buffer(physical_index)
        }
    }

    /// Returns the raw [`wgpu::Buffer`] handle for the given node.
    #[must_use]
    pub fn get_buffer(&self, id: BufferNodeId) -> &wgpu::Buffer {
        let tracked = self.get_tracked_buffer(id);
        tracked
    }

    /// Returns a buffer binding truncated to the resource's logical size.
    #[must_use]
    pub fn get_buffer_binding(&self, id: BufferNodeId) -> wgpu::BufferBinding<'_> {
        let (_, res) = resolve_buffer_resource(self.resources, id);
        let desc = res.buffer_desc();
        let size = desc.logical_binding_size().unwrap_or_else(|| {
            panic!(
                "Buffer '{}' has zero logical size and cannot be bound",
                res.name
            )
        });

        wgpu::BufferBinding {
            buffer: self.get_buffer(id),
            offset: 0,
            size: Some(size),
        }
    }

    /// Returns a buffer slice truncated to the resource's logical size.
    #[must_use]
    pub fn get_buffer_slice(&self, id: BufferNodeId) -> wgpu::BufferSlice<'_> {
        let (_, res) = resolve_buffer_resource(self.resources, id);
        self.get_buffer(id).slice(0..res.buffer_desc().logical_size)
    }

    /// Returns the physical-buffer allocation UID for the given node.
    #[must_use]
    pub fn get_physical_buffer_uid(&self, id: BufferNodeId) -> u64 {
        let (_, res) = resolve_buffer_resource(self.resources, id);

        if res.is_external {
            self.get_tracked_buffer(id).id()
        } else {
            let physical_index = res
                .physical_index
                .expect("Transient resource has no physical memory assigned!");
            self.pool.get_buffer_uid(physical_index)
        }
    }

    /// Returns `true` if the resource is backed by physical GPU memory.
    ///
    /// Equivalent to [`RdgViewResolver::is_resource_allocated`] but
    /// available during the execute phase.
    #[inline]
    #[must_use]
    pub fn is_resource_allocated<T: GraphResourceType>(&self, id: ResourceNodeId<T>) -> bool {
        let res = &self.resources[id.index() as usize];
        res.is_external || res.physical_index.is_some()
    }

    /// Construct a `wgpu::RenderPassColorAttachment` with explicit load
    /// semantics and automatic `StoreOp` deduction.
    ///
    /// # Load Semantics (`RenderTargetOps`)
    ///
    /// | Variant    | GPU Effect | Notes |
    /// |------------|------------|-------|
    /// | `Clear(c)` | `LoadOp::Clear(c)` | Use when a known background is required. |
    /// | `Load`     | `LoadOp::Load`     | Only valid on resources with prior content (aliases or multi-write). |
    /// | `DontCare` | `LoadOp::Clear(BLACK)` | Full-screen replace — zero bandwidth on TBDR. |
    ///
    /// # Store Semantics (Automatic)
    ///
    /// - **`Discard`** when `last_use == current_timeline_index` and the
    ///   resource is not external.
    /// - **`Store`** otherwise.
    ///
    /// # Safety Validation
    ///
    /// In debug builds, using `RenderTargetOps::Load` on a freshly created
    /// transient resource (first write, non-alias, non-external) will
    /// **panic** — this catches uninitialised-memory reads that would
    /// produce visual artefacts and waste GPU bandwidth.
    ///
    /// # MSAA Resolve
    ///
    /// The optional `resolve_target` specifies a single-sample texture for
    /// hardware MSAA resolve.  If the target was culled (no allocation),
    /// it is silently ignored.
    ///
    /// Returns `None` if the primary resource was culled.
    #[must_use]
    pub fn get_color_attachment(
        &self,
        id: TextureNodeId,
        ops: RenderTargetOps,
        resolve_target: Option<TextureNodeId>,
    ) -> Option<wgpu::RenderPassColorAttachment<'_>> {
        let view = self.try_get_base_mip_view(id)?;

        let res = &self.resources[id.index() as usize];
        let ti = self.current_timeline_index;

        let is_first_write = res.first_use == ti && !res.is_external && res.alias_of.is_none();

        // Validate: Load on an uninitialised transient resource is always a bug.
        assert!(
            !(matches!(ops, RenderTargetOps::Load) && is_first_write),
            "RDG Validation Error: LoadOp::Load on freshly created transient \
             resource '{name}' (node {id:?}).  This reads uninitialised GPU \
             memory and wastes bandwidth.  Use RenderTargetOps::DontCare for \
             full-screen replace shaders, or RenderTargetOps::Clear(color) \
             when a specific background is needed.",
            name = res.name,
        );

        // For alias resources the caller should pass `Load` explicitly;
        // the graph guarantees content inheritance from the prior version.
        let load = if is_first_write {
            ops.to_wgpu_load_op()
        } else {
            // Aliases and subsequent writes always load prior content
            // unless the caller explicitly requests Clear/DontCare.
            match ops {
                RenderTargetOps::Load => wgpu::LoadOp::Load,
                other => other.to_wgpu_load_op(),
            }
        };

        let store = if res.last_use == ti && !res.is_external {
            wgpu::StoreOp::Discard
        } else {
            wgpu::StoreOp::Store
        };

        let resolve_view = resolve_target.and_then(|rt| self.try_get_base_mip_view(rt));

        Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: resolve_view,
            ops: wgpu::Operations { load, store },
            depth_slice: None,
        })
    }

    /// Auto-deduce `LoadOp` and `StoreOp` for a depth-stencil attachment.
    ///
    /// Rules mirror [`get_color_attachment`]:
    /// - First use of a non-alias, non-external resource →
    ///   `Clear(clear_depth)`, otherwise `Load`.
    /// - Last use on a non-external resource → `Discard`, otherwise `Store`.
    ///
    /// Returns `None` if the resource was culled.
    #[must_use]
    pub fn get_depth_stencil_attachment(
        &self,
        id: TextureNodeId,
        clear_depth: f32,
    ) -> Option<wgpu::RenderPassDepthStencilAttachment<'_>> {
        let view = self.try_get_texture_view(id)?;
        let res = &self.resources[id.index() as usize];
        let ti = self.current_timeline_index;

        let load = if res.first_use == ti && res.alias_of.is_none() {
            wgpu::LoadOp::Clear(clear_depth)
        } else {
            wgpu::LoadOp::Load
        };

        let store = if res.last_use == ti && !res.is_external {
            wgpu::StoreOp::Discard
        } else {
            wgpu::StoreOp::Store
        };

        Some(wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(wgpu::Operations { load, store }),
            stencil_ops: None,
        })
    }
}
