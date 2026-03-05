use crate::assets::AssetServer;
use crate::renderer::core::ResourceManager;
use crate::renderer::core::WgpuContext;
use crate::renderer::core::binding::GlobalBindGroupCache;
use crate::renderer::core::resources::{SamplerRegistry, Tracked};
use crate::renderer::graph::context::FrameResources;
use crate::renderer::graph::frame::{FrameBlackboard, RenderLists};
use crate::renderer::graph::transient_pool::TransientTexturePool;
use crate::renderer::graph::{ExtractedScene, RenderState};
use crate::renderer::pipeline::{PipelineCache, ShaderManager};
use crate::scene::Scene;
use crate::scene::camera::RenderCamera;
use rustc_hash::FxHashMap;
use wgpu::{Device, Queue, TextureView};

use super::allocator::RdgTransientPool;
use super::graph::RenderGraph;
use super::types::TextureNodeId;


// ─── Prepare Context ──────────────────────────────────────────────────────────

/// Mutable context available during the RDG **prepare** phase.
///
/// Provides both low-level rendering infrastructure (Device, Queue, PipelineCache,
/// ShaderManager, etc.) and high-level scene data needed by scene rendering passes.
///
/// # Push Model
///
/// Pass-specific parameters (TextureNodeId slots, configuration flags) are pushed
/// into [`PassNode`](super::node::PassNode) fields by the Composer before the
/// prepare loop. Shared scene infrastructure (RenderLists, FrameResources) is
/// provided through this context.
pub struct RdgPrepareContext<'a> {
    pub graph: &'a RenderGraph,
    pub pool: &'a RdgTransientPool,
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub pipeline_cache: &'a mut PipelineCache,
    pub sampler_registry: &'a mut SamplerRegistry,
    pub global_bind_group_cache: &'a mut GlobalBindGroupCache,
    pub shader_manager: &'a mut ShaderManager,
    pub external_resources: &'a FxHashMap<TextureNodeId, &'a Tracked<wgpu::TextureView>>,

    /// GPU resource manager — provides `CpuBuffer → GpuBuffer` resolution,
    /// texture upload, and global state access (layouts, bind groups).
    pub resource_manager: &'a mut ResourceManager,

    // ─── Scene Data (Phase 3: full RDG integration) ──────────────────

    /// Full wgpu context — depth format, MSAA samples, render path, etc.
    pub wgpu_ctx: &'a WgpuContext,

    /// Render lists populated by SceneCullPass (read-only during RDG prepare).
    pub render_lists: &'a mut RenderLists,

    /// Frame-level GPU resources (depth buffer, ping-pong textures, screen bind
    /// group layout, dummy views).
    pub frame_resources: &'a FrameResources,

    /// Extracted scene data (render items, scene defines, etc.).
    pub extracted_scene: &'a ExtractedScene,

    /// Render state (camera matrices, time, render_state_id, etc.).
    pub render_state: &'a RenderState,

    /// Live scene reference (for background settings, SSAO config, etc.).
    pub scene: &'a Scene,

    /// Active camera.
    pub camera: &'a RenderCamera,

    /// Asset server for geometry/material lookups.
    pub assets: &'a AssetServer,

    /// Old-system transient pool — still used by pre-RDG passes and for
    /// resources not yet migrated to RDG (e.g. feature_id, specular).
    pub transient_pool: &'a mut TransientTexturePool,

    /// Frame blackboard for cross-pass transient data (SSAO IDs, feature IDs, etc.).
    pub blackboard: &'a mut FrameBlackboard,
}

impl<'a> RdgPrepareContext<'a> {
    /// Resolve a virtual [`TextureNodeId`] to its physical [`Tracked<TextureView>`].
    ///
    /// For external resources, the view is looked up in `external_resources`.
    /// For transient resources, the view is obtained from the physical pool.
    pub fn get_texture_view(&self, id: TextureNodeId) -> &Tracked<wgpu::TextureView> {
        let res = &self.graph.resources[id.0 as usize];

        if res.is_external {
            self.external_resources
                .get(&id)
                .expect("External resource missing!")
        } else {
            let physical_index = res.physical_index.expect("No physical memory!");
            &self.pool.resources[physical_index].view
        }
    }
}

// ─── Execute Context ──────────────────────────────────────────────────────────

/// Immutable context available during the RDG **execute** phase.
///
/// Provides read-only access to the compiled render graph, physical resource
/// pool, pipeline cache, render lists, and any external views injected before
/// execution.
pub struct RdgExecuteContext<'a> {
    pub graph: &'a RenderGraph,
    pub pool: &'a RdgTransientPool,
    pub device: &'a Device,
    pub queue: &'a Queue,
    pub pipeline_cache: &'a PipelineCache,
    pub global_bind_group_cache: &'a GlobalBindGroupCache,
    /// External views (e.g. swapchain backbuffer) injected before the execute loop.
    pub external_views: FxHashMap<TextureNodeId, &'a TextureView>,
    /// Per-frame global bind group (group 0 in most shaders).
    /// Built by SceneCullPass; consumed by all scene and post-processing passes.
    pub global_bind_group: Option<&'a wgpu::BindGroup>,

    // ─── Scene Data (Phase 3: full RDG integration) ──────────────────

    /// GPU resource manager (read-only) — material/geometry lookups during draw.
    pub resource_manager: &'a ResourceManager,

    /// Render lists populated by SceneCullPass — opaque/transparent draw commands.
    pub render_lists: &'a RenderLists,

    /// Frame-level GPU resources (screen bind group layout, dummy views, etc.).
    pub frame_resources: &'a FrameResources,

    /// Old-system transient pool (read-only) — for resources not yet in RDG.
    pub transient_pool: &'a TransientTexturePool,

    /// Full wgpu context — depth format, render path, etc.
    pub wgpu_ctx: &'a WgpuContext,

    /// Frame blackboard for cross-pass transient data (read-only during execute).
    pub blackboard: &'a FrameBlackboard,
}

impl<'a> RdgExecuteContext<'a> {
    /// Resolve a virtual [`TextureNodeId`] to its physical [`TextureView`].
    ///
    /// For external resources, the view is looked up in `external_views`.
    /// For transient resources, the view is obtained from the physical pool.
    pub fn get_texture_view(&self, id: TextureNodeId) -> &TextureView {
        let res = &self.graph.resources[id.0 as usize];
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
}
