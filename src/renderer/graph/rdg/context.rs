use crate::assets::AssetServer;
use crate::renderer::core::ResourceManager;
use crate::renderer::core::WgpuContext;
use crate::renderer::core::binding::GlobalBindGroupCache;
use crate::renderer::core::resources::{SamplerRegistry, Tracked};
use crate::renderer::graph::frame::{RenderLists};
use crate::renderer::graph::{ExtractedScene, RenderState};
use crate::renderer::pipeline::{PipelineCache, ShaderManager};
use crate::scene::camera::RenderCamera;
use rustc_hash::FxHashMap;
use wgpu::{Device, Queue, TextureView};

use super::allocator::{RdgTransientPool, SubViewKey};
use super::graph::RenderGraph;
use super::types::TextureNodeId;


// ─── Prepare Context ──────────────────────────────────────────────────────────

/// Mutable context available during the RDG **prepare** phase.
///
/// Provides low-level rendering infrastructure (Device, Queue, PipelineCache,
/// ShaderManager, etc.) and extracted scene data needed by rendering passes.
///
/// # Push Model
///
/// Pass-specific parameters (TextureNodeId slots, configuration flags, uniform
/// buffer IDs) are pushed into [`PassNode`](super::node::PassNode) fields by
/// the Composer **before** the prepare loop. The context provides only shared,
/// read-only infrastructure. `Scene` is **not** available here — all scene
/// data must be pre-extracted into [`ExtractedScene`] or pushed via parameters.
pub struct RdgPrepareContext<'a> {
    // pub graph: &'a RenderGraph,
    pub views: RdgViewResolver<'a>,
    // pub pool: &'a mut RdgTransientPool,
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub pipeline_cache: &'a mut PipelineCache,
    pub sampler_registry: &'a mut SamplerRegistry,
    pub global_bind_group_cache: &'a mut GlobalBindGroupCache,
    pub shader_manager: &'a mut ShaderManager,
    // pub external_resources: &'a FxHashMap<TextureNodeId, &'a Tracked<wgpu::TextureView>>,

    /// GPU resource manager — provides `CpuBuffer → GpuBuffer` resolution,
    /// texture upload, and global state access (layouts, bind groups).
    pub resource_manager: &'a mut ResourceManager,

    // ─── Scene Data ──────────────────────────────────────────────────

    /// Full wgpu context — depth format, MSAA samples, render path, etc.
    pub wgpu_ctx: &'a WgpuContext,

    /// Render lists populated by SceneCullPass (read-only during RDG prepare).
    pub render_lists: &'a mut RenderLists,

    /// Extracted scene data (render items, scene defines, etc.).
    pub extracted_scene: &'a ExtractedScene,

    /// Render state (camera matrices, time, render_state_id, etc.).
    pub render_state: &'a RenderState,

    /// Active camera.
    pub camera: &'a RenderCamera,

    /// Asset server for geometry/material lookups.
    pub assets: &'a AssetServer,
}

pub struct RdgViewResolver<'a> {
    pub graph: &'a RenderGraph,
    pub pool: &'a mut RdgTransientPool,
    pub external_resources: &'a FxHashMap<TextureNodeId, &'a Tracked<wgpu::TextureView>>,
}

impl<'a> RdgViewResolver<'a> {
    /// Resolve a virtual [`TextureNodeId`] to its physical [`Tracked<TextureView>`].
    ///
    /// For external resources, the view is looked up in `external_resources`.
    /// For transient resources, the **default** view is obtained from the pool.
    pub fn get_texture_view(&self, id: TextureNodeId) -> &Tracked<wgpu::TextureView> {
        let res = &self.graph.resources[id.0 as usize];

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
        let res = &self.graph.resources[id.0 as usize];
        let physical_index = res.physical_index.expect("No physical memory!");
        self.pool.get_uid(physical_index)
    }

    /// Returns the raw `wgpu::Texture` handle for the given node.
    ///
    /// Useful for passes that need to create custom views (e.g. Bloom mip chain).
    pub fn get_texture(&self, id: TextureNodeId) -> &wgpu::Texture {
        let res = &self.graph.resources[id.0 as usize];
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
        let res = &self.graph.resources[id.0 as usize];
        let physical_index = res.physical_index.expect("No physical memory!");
        self.pool.get_or_create_sub_view(physical_index, key)
    }

    pub fn get_sub_view(&self, id: TextureNodeId, key: &SubViewKey) -> Option<&Tracked<wgpu::TextureView>> {
        let res = &self.graph.resources[id.0 as usize];
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
    pub graph: &'a RenderGraph,
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

    /// GPU resource manager (read-only) — material/geometry lookups during draw.
    pub resource_manager: &'a ResourceManager,

    /// Render lists populated by SceneCullPass — opaque/transparent draw commands.
    pub render_lists: &'a RenderLists,

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

    /// Returns the [`Tracked<TextureView>`] for cache-key use during execute.
    pub fn get_tracked_texture_view(&self, id: TextureNodeId) -> &Tracked<wgpu::TextureView> {
        let res = &self.graph.resources[id.0 as usize];
        let physical_index = res
            .physical_index
            .expect("Resource has no physical memory!");
        self.pool.get_tracked_view(physical_index)
    }
}
