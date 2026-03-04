use crate::renderer::core::binding::GlobalBindGroupCache;
use crate::renderer::core::resources::{SamplerRegistry, Tracked};
use crate::renderer::pipeline::{PipelineCache, ShaderManager};
use rustc_hash::FxHashMap;
use wgpu::{Device, Queue, TextureView};

use super::allocator::RdgTransientPool;
use super::graph::RenderGraph;
use super::types::TextureNodeId;

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
}

impl<'a> RdgPrepareContext<'a> {
    pub fn get_physical_texture(&self, id: TextureNodeId) -> &Tracked<wgpu::TextureView> {
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

pub struct RdgExecuteContext<'a> {
    pub graph: &'a RenderGraph,
    pub pool: &'a RdgTransientPool,
    pub device: &'a Device,
    pub queue: &'a Queue,
    pub pipeline_cache: &'a PipelineCache,
    pub global_bind_group_cache: &'a GlobalBindGroupCache,
    // 外部资源（如 Swapchain 的 Backbuffer）需要在 Execute 前注入
    pub external_views: FxHashMap<TextureNodeId, &'a TextureView>,
}

impl<'a> RdgExecuteContext<'a> {
    /// 核心方法：PassNode 调它把虚拟 ID 换成真正的物理 View
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
