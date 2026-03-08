use crate::assets::AssetServer;
use crate::renderer::core::ResourceManager;
use crate::renderer::core::WgpuContext;
use crate::renderer::core::binding::GlobalBindGroupCache;
use crate::renderer::core::resources::MipmapGenerator;
use crate::renderer::core::resources::{SamplerRegistry, Tracked};
use crate::renderer::graph::frame::{BakedRenderLists, RenderLists};
use crate::renderer::graph::{ExtractedScene, RenderState};
use crate::renderer::pipeline::{PipelineCache, ShaderManager};
use rustc_hash::FxHashMap;
use wgpu::{Device, Queue, TextureView};

use super::allocator::{RdgTransientPool, SubViewKey};
use super::types::{ResourceRecord, TextureNodeId};

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
///
/// After this phase, Features hold only lightweight pipeline IDs.  The
/// per-frame ephemeral `PassNode` created by `Feature::add_to_graph()`
/// carries those IDs into the graph.
pub struct ExtractContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub pipeline_cache: &'a mut PipelineCache,
    pub shader_manager: &'a mut ShaderManager,
    pub sampler_registry: &'a mut SamplerRegistry,
    pub global_bind_group_cache: &'a mut GlobalBindGroupCache,
    pub resource_manager: &'a mut ResourceManager,
    pub wgpu_ctx: &'a WgpuContext,
    pub render_lists: &'a mut RenderLists,
    pub extracted_scene: &'a ExtractedScene,
    pub render_state: &'a RenderState,
    pub assets: &'a AssetServer,
}

// ─── Prepare Context (Transient-Only) ─────────────────────────────────────────

/// Minimal context available during the RDG **prepare** phase.
///
/// After the render graph has been compiled and transient resources allocated,
/// each pass's [`PassNode::prepare`] receives this context to assemble
/// `wgpu::BindGroup`s that reference RDG-managed transient textures.
///
/// This context is deliberately kept **pure**: it provides only the GPU
/// device, transient view resolver, sampler registry, and the global bind
/// group cache.  All persistent GPU resources (pipelines, material buffers,
/// geometry buffers) must be resolved during the earlier
/// `Feature::extract_and_prepare()` phase and carried into the PassNode
/// as lightweight cloned handles.
pub struct RdgPrepareContext<'a> {
    /// Transient resource view resolver.
    pub views: RdgViewResolver<'a>,
    /// GPU device for creating bind groups and sub-views.
    pub device: &'a wgpu::Device,
    /// GPU queue for immediate buffer uploads (rare in prepare).
    pub queue: &'a wgpu::Queue,
    /// Shared sampler registry (persistent, immutable during prepare).
    pub sampler_registry: &'a SamplerRegistry,
    /// Mutable cache for transient bind groups with TTL eviction.
    pub global_bind_group_cache: &'a mut GlobalBindGroupCache,
}

pub struct RdgViewResolver<'a> {
    pub resources: &'a [ResourceRecord],
    pub pool: &'a mut RdgTransientPool,
    pub external_resources: &'a FxHashMap<TextureNodeId, &'a Tracked<wgpu::TextureView>>,
}

impl<'a> RdgViewResolver<'a> {
    /// Resolve a virtual [`TextureNodeId`] to its physical [`Tracked<TextureView>`].
    ///
    /// For external resources, the view is looked up in `external_resources`.
    /// For transient resources, the **default** view is obtained from the pool.
    pub fn get_texture_view(&self, id: TextureNodeId) -> &Tracked<wgpu::TextureView> {
        let res = &self.resources[id.0 as usize];

        if res.is_external {
            self.external_resources
                .get(&id)
                .expect(&format!("External {} resource missing!", res.name))
        } else {
            let physical_index = res.physical_index.expect("No physical memory!");
            &self.pool.resources[physical_index].default_view
        }
    }

    /// Returns the physical-texture allocation UID for the given node.
    ///
    /// Useful for dirty-checking: if the UID hasn't changed between frames,
    /// the physical texture is the same and derived state can be reused.
    pub fn get_physical_texture_uid(&self, id: TextureNodeId) -> u64 {
        let res = &self.resources[id.0 as usize];
        let physical_index = res.physical_index.expect("No physical memory!");
        self.pool.get_uid(physical_index)
    }

    /// Returns the raw `wgpu::Texture` handle for the given node.
    ///
    /// Useful for passes that need to create custom views (e.g. Bloom mip chain).
    pub fn get_texture(&self, id: TextureNodeId) -> &wgpu::Texture {
        let res = &self.resources[id.0 as usize];
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
        key: SubViewKey,
    ) -> &Tracked<wgpu::TextureView> {
        let res = &self.resources[id.0 as usize];
        let physical_index = res.physical_index.expect("No physical memory!");
        self.pool.get_or_create_sub_view(physical_index, key)
    }

    pub fn get_sub_view(
        &self,
        id: TextureNodeId,
        key: &SubViewKey,
    ) -> Option<&Tracked<wgpu::TextureView>> {
        let res = &self.resources[id.0 as usize];
        let physical_index = res.physical_index.expect("No physical memory!");
        self.pool.get_sub_view(physical_index, key)
    }
}

// ─── Execute Context ──────────────────────────────────────────────────────────

/// Immutable context available during the RDG **execute** phase.
///
/// Provides read-only access to the compiled render graph, physical resource
/// pool, pipeline cache, render lists, and any external views injected before
/// execution.
pub struct RdgExecuteContext<'a> {
    pub resources: &'a [ResourceRecord],
    pub pool: &'a RdgTransientPool,
    pub device: &'a Device,
    pub queue: &'a Queue,
    pub pipeline_cache: &'a PipelineCache,
    pub global_bind_group_cache: &'a GlobalBindGroupCache,
    /// External views (e.g. swapchain backbuffer) injected before the execute loop.
    pub external_views: &'a FxHashMap<TextureNodeId, &'a TextureView>,
    /// Per-frame global bind group (group 0 in most shaders).
    /// Built by SceneCullPass; consumed by all scene and post-processing passes.
    pub global_bind_group: Option<&'a wgpu::BindGroup>,

    // ─── Scene Data (Phase 3: full RDG integration) ──────────────────
    /// GPU resource manager (read-only) — used by compute, post-processing,
    /// and skybox passes that have not yet been migrated to baked commands.
    pub mipmap_generator: &'a MipmapGenerator,

    /// Render lists populated by SceneCullPass — opaque/transparent draw commands.
    pub render_lists: &'a RenderLists,

    /// Pre-baked draw command lists with all GPU handles resolved.
    ///
    /// Scene-drawing passes (opaque, transparent, shadow, prepass,
    /// simple-forward) consume these instead of performing per-command
    /// handle lookups via `resource_manager`.
    pub baked_lists: &'a BakedRenderLists<'a>,

    /// Full wgpu context — depth format, render path, etc.
    pub wgpu_ctx: &'a WgpuContext,
    // /// Frame blackboard for cross-pass transient data (read-only during execute).
    // pub blackboard: &'a FrameBlackboard,
}

impl<'a> RdgExecuteContext<'a> {
    /// Resolve a virtual [`TextureNodeId`] to its physical [`TextureView`].
    ///
    /// For external resources, the view is looked up in `external_views`.
    /// For transient resources, the view is obtained from the physical pool.
    pub fn get_texture_view(&self, id: TextureNodeId) -> &TextureView {
        let res = &self.resources[id.0 as usize];
        if res.is_external {
            self.external_views
                .get(&id)
                .expect("External view was not provided to RdgExecuteContext!")
        } else {
            let physical_index = res
                .physical_index
                .expect("Transient resource has no physical memory assigned!");
            self.pool.get_view(physical_index)
        }
    }

    /// Returns the [`Tracked<TextureView>`] for cache-key use during execute.
    pub fn get_tracked_texture_view(&self, id: TextureNodeId) -> &Tracked<wgpu::TextureView> {
        let res = &self.resources[id.0 as usize];
        let physical_index = res
            .physical_index
            .expect("Resource has no physical memory!");
        self.pool.get_tracked_view(physical_index)
    }
}
