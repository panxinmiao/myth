//! Frame Composer
//!
//! `FrameComposer` provides a fluent API to build and execute the render pipeline.
//! It acts as the "glue" connecting the Prepare phase and the Execute phase.
//!
//! # Three-Phase Rendering Architecture
//!
//! 1. **Prepare**: Extract data and prepare GPU resources
//! 2. **Compose**: Add `RenderNode`s via the fluent API
//! 3. **Execute**: Acquire the Surface, build the `RenderGraph`, and submit GPU commands
//!
//! # Example
//!
//! ```ignore
//! // Fluent chained invocation
//! renderer.begin_frame(scene, &camera, assets, time)?
//!     .add_node(RenderStage::UI, &ui_pass)
//!     .add_node(RenderStage::PostProcess, &bloom_pass)
//!     .render();
//! ```

use super::frame::RenderLists;
use crate::assets::AssetServer;
use crate::render::RenderState;
use crate::renderer::core::binding::GlobalBindGroupCache;
use crate::renderer::core::{ResourceManager, WgpuContext};
use crate::renderer::graph::ExtractedScene;
use crate::renderer::graph::builder::FrameBuilder;
use crate::renderer::graph::context::{ExecuteContext, FrameResources, PrepareContext};
use crate::renderer::graph::frame::FrameBlackboard;
use crate::renderer::graph::node::RenderNode;
use crate::renderer::graph::stage::RenderStage;
use crate::renderer::graph::transient_pool::TransientTexturePool;
use crate::renderer::pipeline::PipelineCache;
use crate::renderer::pipeline::ShaderManager;
use crate::scene::Scene;
use crate::scene::camera::RenderCamera;

pub struct ComposerContext<'a> {
    pub wgpu_ctx: &'a mut WgpuContext,
    pub resource_manager: &'a mut ResourceManager,
    pub pipeline_cache: &'a mut PipelineCache,
    pub shader_manager: &'a mut ShaderManager,

    pub extracted_scene: &'a ExtractedScene,
    pub render_state: &'a RenderState,

    pub frame_resources: &'a FrameResources,
    pub transient_pool: &'a mut TransientTexturePool,
    pub global_bind_group_cache: &'a mut GlobalBindGroupCache,

    /// Render lists (populated by `SceneCullPass`)
    pub render_lists: &'a mut RenderLists,

    /// Frame blackboard (cross-pass transient data communication)
    pub blackboard: &'a mut FrameBlackboard,

    // External scene data
    pub scene: &'a mut Scene,
    pub camera: &'a RenderCamera,
    pub assets: &'a AssetServer,
    pub time: f32,

    pub rdg_graph: &'a mut crate::renderer::graph::rdg::graph::RenderGraph,
    pub test_fxaa_pass: &'a mut crate::renderer::graph::rdg::test_pass::RdgFxaaPass,
    pub rdg_pool: &'a mut crate::renderer::graph::rdg::allocator::RdgTransientPool,
    pub sampler_registry: &'a mut crate::renderer::core::resources::SamplerRegistry,
}

/// Frame Composer
///
/// Holds all context references needed to render a single frame and provides
/// a fluent API for adding render nodes.
///
/// # Design Notes
///
/// - **Clear responsibilities**: `FrameComposer` only handles context and flow control
/// - **Lifetime safety**: Lifetime `'a` locks the mutable borrow on `Renderer`
/// - **Deferred Surface acquisition**: The Surface is acquired only in `.render()` to minimize hold time
///
/// # Performance Considerations
///
/// - Internal `FrameBuilder` pre-allocates capacity for 16 nodes
/// - Sorting uses `FrameBuilder`'s efficient sorting mechanism
/// - All fields are references — no heap allocation overhead
pub struct FrameComposer<'a> {
    // GPU context
    ctx: ComposerContext<'a>,

    // Builder (collects render nodes)
    builder: FrameBuilder<'a>,
}

impl<'a> FrameComposer<'a> {
    /// Creates a new frame composer.
    ///
    /// Built-in passes (BRDF LUT, IBL, Forward) are injected automatically.
    pub(crate) fn new(builder: FrameBuilder<'a>, ctx: ComposerContext<'a>) -> Self {
        Self { ctx, builder }
    }

    /// Adds a custom render node at the specified stage.
    ///
    /// Supports method chaining.
    ///
    /// # Arguments
    ///
    /// - `stage`: Render stage (determines execution order)
    /// - `node`: Render node reference
    ///
    /// # Example
    ///
    /// ```ignore
    /// composer
    ///     .add_node(RenderStage::UI, &ui_pass)
    ///     .add_node(RenderStage::PostProcess, &bloom_pass)
    ///     .render();
    /// ```
    #[inline]
    #[must_use]
    pub fn add_node(mut self, stage: RenderStage, node: &'a mut dyn RenderNode) -> Self {
        self.builder.add_node(stage, node);
        self
    }

    /// Adds multiple nodes to the same stage in batch.
    ///
    /// # Example
    ///
    /// ```ignore
    /// composer
    ///     .add_nodes(RenderStage::PostProcess, &[&mut bloom, &mut fxaa, &mut tone_mapping])
    ///     .render();
    /// ```
    #[inline]
    #[must_use]
    pub fn add_nodes<I>(mut self, stage: RenderStage, nodes: I) -> Self
    where
        I: IntoIterator<Item = &'a mut dyn RenderNode>,
    {
        self.builder.add_nodes(stage, nodes);
        self
    }

    /// Executes the render pipeline and presents to the screen.
    ///
    /// This is the final step of the rendering workflow:
    /// 1. Acquire the Surface texture
    /// 2. Build the `RenderContext`
    /// 3. Convert the Builder into a sorted `RenderGraph`
    /// 4. Execute the render graph
    /// 5. Present
    ///
    /// # Note
    ///
    /// This method consumes `self`; the `FrameComposer` cannot be reused after calling it.
    pub fn render(self) {
        // 1. 获取真实 Surface
        let output = match self.ctx.wgpu_ctx.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost) => return,
            Err(e) => {
                log::error!("Render error: {e:?}");
                return;
            }
        };

        let view_format = self.ctx.wgpu_ctx.surface_view_format;
        let surface_view = output.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(view_format),
            ..Default::default()
        });
        
        // 提取物理屏幕尺寸
        let width = output.texture.width();
        let height = output.texture.height();

        // 3. 旧图
        let mut graph = self.builder.build();

        // 2. 构建 RDG 拓扑并规划内存
        // use crate::renderer::graph::rdg::graph::RenderGraph as RdgGraph;
        // use crate::renderer::graph::rdg::test_pass::RdgFxaaPass;
        use crate::renderer::graph::rdg::types::RdgTextureDesc;
        use crate::renderer::graph::rdg::context::{RdgPrepareContext, RdgExecuteContext};
        use rustc_hash::FxHashMap;
        
        // let mut rdg = RdgGraph::new();
        let rdg =  self.ctx.rdg_graph; // 直接使用 Renderer 中的 RDG 实例，避免重复创建和内存浪费
        rdg.begin_frame();

        let desc = RdgTextureDesc::new_2d(
            width, height, view_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
        );

        // 【大升级】：Bridge 不再是 external！交给 RDG 去在对象池里智能复用！
        let node_in = rdg.register_resource("Bridge_In", desc.clone(), false);
        // 输出端绑定真实的屏幕 (依然是外部资源)
        let node_out = rdg.register_resource("Surface_Out", desc.clone(), true);

        // let mut fxaa_pass = RdgFxaaPass::new();
        let fxaa_pass = self.ctx.test_fxaa_pass; // 直接使用 Renderer 中的测试 Pass 实例，避免重复创建和内存浪费
        fxaa_pass.input_tex = node_in;
        fxaa_pass.output_tex = node_out;
        rdg.add_pass(fxaa_pass);

        // 编译 RDG，触发物理内存池的 acquire 分配
        rdg.compile(self.ctx.rdg_pool, &self.ctx.wgpu_ctx.device);

        // 3. 从 RDG 分配好的物理池中，把纹理“借”出来给旧系统
        let bridge_idx = rdg.resources[node_in.0 as usize].physical_index.expect("RDG failed to allocate bridge");
        let bridge_tracked = &self.ctx.rdg_pool.resources[bridge_idx].view;


        // 4. 准备并执行旧图
        {
            let mut prepare_ctx = PrepareContext {
                // ... 保留你原有的 PrepareContext 参数填法 ...
                wgpu_ctx: &*self.ctx.wgpu_ctx,
                resource_manager: self.ctx.resource_manager,
                pipeline_cache: self.ctx.pipeline_cache,
                shader_manager: self.ctx.shader_manager,
                assets: self.ctx.assets,
                scene: self.ctx.scene,
                camera: self.ctx.camera,
                render_state: self.ctx.render_state,
                extracted_scene: self.ctx.extracted_scene,
                render_lists: self.ctx.render_lists,
                blackboard: self.ctx.blackboard,
                frame_resources: self.ctx.frame_resources,
                transient_pool: self.ctx.transient_pool,
                time: self.ctx.time,
                global_bind_group_cache: self.ctx.global_bind_group_cache,
                color_view_flip_flop: 0,
            };
            graph.prepare(&mut prepare_ctx);
        }

        // 4. 执行旧图
        // 注意：这里传入 &*bridge_tracked 代替旧的 &surface_view
        // 所有输出到 Surface 的操作，现在都会悄无声息地写入到我们的桥接纹理上！
        let execute_ctx = ExecuteContext::new(
            &*self.ctx.wgpu_ctx,
            &*self.ctx.resource_manager,
            &*bridge_tracked, 
            &*self.ctx.render_lists,
            &*self.ctx.blackboard,
            self.ctx.frame_resources,
            &*self.ctx.transient_pool,
            &*self.ctx.pipeline_cache,
        );
        graph.execute(&execute_ctx); // 旧的 graph 会自己提交一个 CommandEncoder 到 Queue

        // 5. 准备并执行 RDG 图
        // 因为 node_in 是内部资源，external_resources 现在只需要传空的 HashMap 了！
        let ext_res = FxHashMap::default();
        let mut rdg_prepare_ctx = RdgPrepareContext {
            graph: &rdg,
            pool: self.ctx.rdg_pool,
            device: &self.ctx.wgpu_ctx.device,
            queue: &self.ctx.wgpu_ctx.queue,
            pipeline_cache: self.ctx.pipeline_cache,
            sampler_registry: self.ctx.sampler_registry,
            global_bind_group_cache: self.ctx.global_bind_group_cache,
            shader_manager: self.ctx.shader_manager,
            external_resources: &ext_res,
        };

        for &pass_idx in &rdg.execution_queue {
            let pass = rdg.passes[pass_idx].get_pass_mut();
            pass.prepare(&mut rdg_prepare_ctx);
        }

        let mut ext_views = FxHashMap::default();
        ext_views.insert(node_out, &surface_view); // 仅映射真实的屏幕输出

        let rdg_execute_ctx = RdgExecuteContext {
            graph: &rdg,
            pool: self.ctx.rdg_pool,
            device: &self.ctx.wgpu_ctx.device,
            queue: &self.ctx.wgpu_ctx.queue,
            pipeline_cache: self.ctx.pipeline_cache,
            global_bind_group_cache: self.ctx.global_bind_group_cache,
            external_views: ext_views,
        };

        let mut rdg_encoder = self.ctx.wgpu_ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RDG Encoder"),
        });

        for &pass_idx in &rdg.execution_queue {
            let pass = rdg.passes[pass_idx].get_pass_mut();
            pass.execute(&rdg_execute_ctx, &mut rdg_encoder);
        }

        // 6. 统一收尾
        self.ctx.wgpu_ctx.queue.submit(Some(rdg_encoder.finish()));
        output.present();
        self.ctx.transient_pool.reset(); // 旧系统池清理
    }
}
