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
    pub rdg_pool: &'a mut crate::renderer::graph::rdg::allocator::RdgTransientPool,
    pub sampler_registry: &'a mut crate::renderer::core::resources::SamplerRegistry,

    // 鈹€鈹€鈹€ RDG Passes (Phase 2) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    pub rdg_fxaa_pass: &'a mut crate::renderer::graph::rdg::fxaa::RdgFxaaPass,
    pub rdg_tone_map_pass: &'a mut crate::renderer::graph::rdg::tone_mapping::RdgToneMapPass,
    pub rdg_bloom_pass: &'a mut crate::renderer::graph::rdg::bloom::RdgBloomPass,
    pub rdg_ssao_pass: &'a mut crate::renderer::graph::rdg::ssao::RdgSsaoPass,
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
/// - All fields are references 锟?no heap allocation overhead
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
    /// Phase 2 architecture: old graph renders scene (PrePass 锟?SSAO 锟?Opaque
    /// 锟?Transparent) to HDR `scene_color_view`, then RDG runs the full
    /// post-processing chain (Bloom 锟?ToneMapping 锟?FXAA) with topology sort,
    /// memory aliasing, and conditional wiring based on scene settings.
    ///
    /// This method consumes `self`; the `FrameComposer` cannot be reused.
    pub fn render(self) {
        // 鈹佲攣锟?1. Acquire Surface 鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣

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

        let width = output.texture.width();
        let height = output.texture.height();

        // 鈹佲攣锟?2. Old Graph: scene rendering only 鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣

        let mut graph = self.builder.build();
        let active_hdr_tracked;
        {
            let mut prepare_ctx = PrepareContext {
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
            active_hdr_tracked = prepare_ctx
                .get_resource_view(crate::renderer::graph::context::GraphResource::SceneColorInput)
                .clone();
        }

        // Execute old graph 锟?writes HDR scene color to
        // compatibility but no remaining old pass writes to it.
        let execute_ctx = ExecuteContext::new(
            self.ctx.wgpu_ctx,
            self.ctx.resource_manager,
            &surface_view,
            self.ctx.render_lists,
            self.ctx.blackboard,
            self.ctx.frame_resources,
            self.ctx.transient_pool,
            self.ctx.pipeline_cache,
        );
        graph.execute(&execute_ctx);

        // 鈹佲攣锟?3. Build RDG Post-Processing Chain 鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣

        use crate::renderer::HDR_TEXTURE_FORMAT;
        use crate::renderer::graph::rdg::context::{RdgExecuteContext, RdgPrepareContext};
        use crate::renderer::graph::rdg::types::RdgTextureDesc;
        use rustc_hash::FxHashMap;

        let rdg = self.ctx.rdg_graph;
        rdg.begin_frame();

        // External: HDR scene color (written by old system)
        let hdr_desc = RdgTextureDesc::new_2d(
            width,
            height,
            HDR_TEXTURE_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let scene_hdr = rdg.register_resource("Scene_HDR", hdr_desc.clone(), true);

        // External: surface output (final write target)
        let surface_desc = RdgTextureDesc::new_2d(
            width,
            height,
            view_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let surface_out = rdg.register_resource("Surface_Out", surface_desc.clone(), true);

        let bloom_enabled = self.ctx.scene.bloom.enabled;
        let fxaa_enabled = self.ctx.scene.fxaa.enabled;

        // 鈹€鈹€ Conditional Wiring 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        //
        // Bloom ON  + FXAA ON  锟?HDR 锟?Bloom 锟?HDR' 锟?ToneMap 锟?LDR 锟?FXAA 锟?Surface
        // Bloom ON  + FXAA OFF 锟?HDR 锟?Bloom 锟?HDR' 锟?ToneMap 锟?Surface
        // Bloom OFF + FXAA ON  锟?HDR 锟?ToneMap 锟?LDR 锟?FXAA 锟?Surface
        // Bloom OFF + FXAA OFF 锟?HDR 锟?ToneMap 锟?Surface

        let bloom_pass = self.ctx.rdg_bloom_pass;
        let tone_map_pass = self.ctx.rdg_tone_map_pass;
        let fxaa_pass = self.ctx.rdg_fxaa_pass;

        let tonemap_input = if bloom_enabled {
            let bloom_out = rdg.register_resource("Bloom_Out", hdr_desc.clone(), false);

            // let source_w = input_desc.size.width;
            // let source_h = input_desc.size.height;
            let (source_w, source_h) = self.ctx.wgpu_ctx.size();
            let bloom_w = (source_w / 2).max(1);
            let bloom_h = (source_h / 2).max(1);

            let max_possible = ((bloom_w.max(bloom_h) as f32).log2().floor() as u32) + 1;
            let mip_count = self
                .ctx
                .scene
                .bloom
                .max_mip_levels()
                .min(max_possible)
                .max(1);

            let mut bloom_chain_desc = RdgTextureDesc::new_2d(
                width,
                height,
                HDR_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );
            bloom_chain_desc.mip_level_count = mip_count; // <--- 关键：告诉 RDG 分配带 mip 的物理纹理

            // 注册为一个内部节点，RDG 会在 Bloom 结束后自动回收这块显存给别的 Pass 用！
            let node_bloom_chain = rdg.register_resource("Bloom_MipChain", bloom_chain_desc, false);

            bloom_pass.bloom_texture = node_bloom_chain;
            // bloom_pass.current_mip_count = mip_count;

            bloom_pass.input_tex = scene_hdr;
            bloom_pass.output_tex = bloom_out;
            bloom_pass.karis_average = self.ctx.scene.bloom.karis_average;
            // bloom_pass.max_mip_levels = self.ctx.scene.bloom.max_mip_levels();
            self.ctx
                .resource_manager
                .ensure_buffer(&self.ctx.scene.bloom.upsample_uniforms);
            self.ctx
                .resource_manager
                .ensure_buffer(&self.ctx.scene.bloom.composite_uniforms);
            bloom_pass.upsample_uniforms_cpu_id = self.ctx.scene.bloom.upsample_uniforms.id();
            bloom_pass.composite_uniforms_cpu_id = self.ctx.scene.bloom.composite_uniforms.id();

            rdg.add_pass(bloom_pass);
            bloom_out
        } else {
            scene_hdr
        };

        let tonemap_output = if fxaa_enabled {
            rdg.register_resource("LDR_Intermediate", surface_desc.clone(), false)
        } else {
            surface_out
        };

        // Wire ToneMap
        tone_map_pass.input_tex = tonemap_input;
        tone_map_pass.output_tex = tonemap_output;
        tone_map_pass.mode = self.ctx.scene.tone_mapping.mode;
        tone_map_pass.has_lut = self.ctx.scene.tone_mapping.lut_texture.is_some();
        let cpu_buffer = &self.ctx.scene.tone_mapping.uniforms;
        self.ctx.resource_manager.ensure_buffer(cpu_buffer);
        tone_map_pass.uniforms_cpu_id = cpu_buffer.id();
        tone_map_pass.lut_handle = self.ctx.scene.tone_mapping.lut_texture.clone();
        tone_map_pass.global_state_key = (self.ctx.render_state.id, self.ctx.scene.id());
        rdg.add_pass(tone_map_pass);

        // Wire FXAA (conditional)
        if fxaa_enabled {
            fxaa_pass.input_tex = tonemap_output;
            fxaa_pass.output_tex = surface_out;
            rdg.add_pass(fxaa_pass);
        }

        // 鈹佲攣锟?4. Compile & Execute RDG 鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣锟?

        rdg.compile(self.ctx.rdg_pool, &self.ctx.wgpu_ctx.device);

        // let scene_color_view = &self.ctx.frame_resources.scene_color_view[0];
        let ext_res: FxHashMap<_, _> = [(scene_hdr, &active_hdr_tracked)].into_iter().collect();

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
            resource_manager: self.ctx.resource_manager,
        };

        for &pass_idx in &rdg.execution_queue {
            let pass = rdg.passes[pass_idx].get_pass_mut();
            pass.prepare(&mut rdg_prepare_ctx);
        }

        // Execute phase 锟?map external views
        let mut ext_views: FxHashMap<_, &wgpu::TextureView> = FxHashMap::default();
        ext_views.insert(scene_hdr, &active_hdr_tracked);
        ext_views.insert(surface_out, &surface_view);

        let global_bg_ref = self.ctx.render_lists.gpu_global_bind_group.as_ref();

        let rdg_execute_ctx = RdgExecuteContext {
            graph: &rdg,
            pool: self.ctx.rdg_pool,
            device: &self.ctx.wgpu_ctx.device,
            queue: &self.ctx.wgpu_ctx.queue,
            pipeline_cache: self.ctx.pipeline_cache,
            global_bind_group_cache: self.ctx.global_bind_group_cache,
            external_views: ext_views,
            global_bind_group: global_bg_ref,
        };

        let mut rdg_encoder =
            self.ctx
                .wgpu_ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("RDG Post-Process Encoder"),
                });

        for &pass_idx in &rdg.execution_queue {
            let pass = rdg.passes[pass_idx].get_pass_mut();
            pass.execute(&rdg_execute_ctx, &mut rdg_encoder);
        }

        // 鈹佲攣锟?5. Submit & Present 鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣鈹佲攣

        self.ctx.wgpu_ctx.queue.submit(Some(rdg_encoder.finish()));
        output.present();
        self.ctx.transient_pool.reset();
    }
}
