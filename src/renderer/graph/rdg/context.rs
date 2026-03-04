use crate::renderer::core::ResourceManager;
use crate::renderer::core::binding::GlobalBindGroupCache;
use crate::renderer::core::resources::{SamplerRegistry, Tracked};
use crate::renderer::pipeline::{PipelineCache, ShaderManager};
use rustc_hash::FxHashMap;
use wgpu::{Device, Queue, TextureView};

use super::allocator::RdgTransientPool;
use super::graph::RenderGraph;
use super::types::TextureNodeId;

// ─── Prepare Context ──────────────────────────────────────────────────────────

/// Mutable context available during the RDG **prepare** phase.
///
/// Provides low-level rendering infrastructure (Device, Queue, PipelineCache,
/// ShaderManager, etc.) but **no** high-level scene data. Pass parameters are
/// pushed into [`PassNode`](super::node::PassNode) fields externally by the
/// Composer before the prepare loop begins.
///
/// # Phase 2 additions
///
/// - `resource_manager`: required for GPU buffer lookups (`ensure_buffer_id`,
///   `get_texture_view`) used by ToneMapping, Bloom, and SSAO passes that
///   need `CpuBuffer → GpuBuffer` resolution.
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
/// pool, pipeline cache, and any external views injected before execution.
///
/// # Phase 2 additions
///
/// - `global_bind_group`: the per-frame global bind group (camera, lighting,
///   environment) built by the old system's prepare phase. Required by passes
///   such as ToneMapping and SSAO whose shaders consume frame-level uniforms.
pub struct RdgExecuteContext<'a> {
    pub graph: &'a RenderGraph,
    pub pool: &'a RdgTransientPool,
    pub device: &'a Device,
    pub queue: &'a Queue,
    pub pipeline_cache: &'a PipelineCache,
    pub global_bind_group_cache: &'a GlobalBindGroupCache,
    /// External views (e.g. swapchain backbuffer, scene HDR color) injected
    /// before the execute loop.
    pub external_views: FxHashMap<TextureNodeId, &'a TextureView>,
    /// Per-frame global bind group (group 0 in most shaders).
    /// Built by the old system's `SceneCullPass → OpaquePass` chain;
    /// consumed by ToneMapping, SSAO and any other pass that reads
    /// camera / lighting / environment uniforms.
    pub global_bind_group: Option<&'a wgpu::BindGroup>,
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
