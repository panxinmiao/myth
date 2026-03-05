//! Frame Composer
//!
//! `FrameComposer` orchestrates the entire rendering pipeline for a single frame.
//! It connects the old-system pre-passes (CullPass, ShadowPass, Compute) with
//! the new RDG-based scene rendering and post-processing chain.
//!
//! # Rendering Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │   Old Graph (CullPass, ShadowPass, …)   │  ← FrameBuilder / RenderGraph
//! ├─────────────────────────────────────────┤
//! │         Unified RDG Pipeline            │  ← RenderGraph (topology sort)
//! │  Prepass → SSAO → Opaque → Skybox →    │
//! │  TransmissionCopy → Transparent →       │
//! │  Bloom → ToneMap → FXAA → Surface      │
//! └─────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! renderer.begin_frame(scene, &camera, assets, time)?
//!     .add_node(RenderStage::UI, &ui_pass)
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

pub struct ComposerContext<'a> {    pub wgpu_ctx: &'a mut WgpuContext,
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

    // ─── RDG Passes ─────────────────────────────────────────────────────
    // Post-processing
    pub rdg_fxaa_pass: &'a mut crate::renderer::graph::rdg::fxaa::RdgFxaaPass,
    pub rdg_tone_map_pass: &'a mut crate::renderer::graph::rdg::tone_mapping::RdgToneMapPass,
    pub rdg_bloom_pass: &'a mut crate::renderer::graph::rdg::bloom::RdgBloomPass,
    pub rdg_ssao_pass: &'a mut crate::renderer::graph::rdg::ssao::RdgSsaoPass,
    // Scene rendering
    pub rdg_prepass: &'a mut crate::renderer::graph::rdg::prepass::RdgPrepass,
    pub rdg_opaque_pass: &'a mut crate::renderer::graph::rdg::opaque::RdgOpaquePass,
    pub rdg_skybox_pass: &'a mut crate::renderer::graph::rdg::skybox::RdgSkyboxPass,
    pub rdg_transparent_pass: &'a mut crate::renderer::graph::rdg::transparent::RdgTransparentPass,
    pub rdg_transmission_copy_pass: &'a mut crate::renderer::graph::rdg::transmission_copy::RdgTransmissionCopyPass,
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
/// - **Deferred Surface acquisition**: The Surface is acquired only in `.render()`
///   to minimize hold time
///
/// # Performance Considerations
///
/// - Internal `FrameBuilder` pre-allocates capacity for 16 nodes
/// - All fields are references — no heap allocation overhead
pub struct FrameComposer<'a> {
    ctx: ComposerContext<'a>,
    builder: FrameBuilder<'a>,
}

impl<'a> FrameComposer<'a> {
    /// Creates a new frame composer.
    pub(crate) fn new(builder: FrameBuilder<'a>, ctx: ComposerContext<'a>) -> Self {
        Self { ctx, builder }
    }

    /// Adds a custom render node at the specified stage (method chaining).
    #[inline]
    #[must_use]
    pub fn add_node(mut self, stage: RenderStage, node: &'a mut dyn RenderNode) -> Self {
        self.builder.add_node(stage, node);
        self
    }

    /// Adds multiple nodes to the same stage in batch.
    #[inline]
    #[must_use]
    pub fn add_nodes<I>(mut self, stage: RenderStage, nodes: I) -> Self
    where
        I: IntoIterator<Item = &'a mut dyn RenderNode>,
    {
        self.builder.add_nodes(stage, nodes);
        self
    }

    /// Executes the full rendering pipeline and presents to the screen.
    ///
    /// # Architecture
    ///
    /// 1. **Acquire Surface** — get the swap-chain back buffer
    /// 2. **Old Graph** — prepare + execute pre-RDG passes (CullPass fills
    ///    `RenderLists`, ShadowPass renders shadow maps, Compute passes
    ///    run BRDF LUT / IBL generation)
    /// 3. **Unified RDG** — build, compile, prepare, execute the full scene
    ///    rendering + post-processing chain via topology-sorted RDG
    /// 4. **Submit & Present** — submit GPU command buffer, present swap-chain
    ///
    /// Consumes `self`; the composer cannot be reused after render.
    pub fn render(self) {
        use crate::renderer::HDR_TEXTURE_FORMAT;
        use crate::renderer::graph::rdg::context::{RdgExecuteContext, RdgPrepareContext};
        use crate::renderer::graph::rdg::types::RdgTextureDesc;
        use rustc_hash::FxHashMap;

        // ━━━ 1. Acquire Surface ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

        // ━━━ 2. Old Graph: CullPass + ShadowPass + Compute ━━━━━━━━━━━━

        let mut graph = self.builder.build();
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
        }

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

        // ━━━ 3. Build Unified RDG (Scene + Post-Processing) ━━━━━━━━━━━

        let rdg = self.ctx.rdg_graph;
        rdg.begin_frame();

        // ── 3a. Register External Resources ────────────────────────────
        //
        // These are backed by persistent GPU textures from FrameResources
        // or the swap-chain surface. RDG treats them as fixed endpoints.

        let hdr_desc = RdgTextureDesc::new_2d(
            width,
            height,
            HDR_TEXTURE_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );

        let depth_desc = RdgTextureDesc::new_2d(
            width,
            height,
            self.ctx.wgpu_ctx.depth_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let surface_desc = RdgTextureDesc::new_2d(
            width,
            height,
            view_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let scene_color = rdg.register_resource("Scene_Color_HDR", hdr_desc.clone(), true);
        let scene_depth = rdg.register_resource("Scene_Depth", depth_desc, true);
        let surface_out = rdg.register_resource("Surface_Out", surface_desc.clone(), true);

        // ── 3b. Scene Configuration ────────────────────────────────────

        let ssao_enabled = self.ctx.scene.ssao.enabled;
        let needs_feature_id =
            self.ctx.scene.screen_space.enable_sss || self.ctx.scene.screen_space.enable_ssr;
        let needs_normal = ssao_enabled || needs_feature_id;
        let needs_prepass = needs_normal; // prepass is required for normals / feature ID
        let needs_skybox = self.ctx.scene.background.needs_skybox_pass();
        let needs_specular = self.ctx.scene.screen_space.enable_sss;
        let has_transmission = self.ctx.render_lists.use_transmission;
        let bloom_enabled = self.ctx.scene.bloom.enabled;
        let fxaa_enabled = self.ctx.scene.fxaa.enabled;

        // ── 3c. Register Transient Resources ───────────────────────────

        let scene_normals = if needs_normal {
            let desc = RdgTextureDesc::new_2d(
                width,
                height,
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );
            rdg.register_resource("Scene_Normals", desc, false)
        } else {
            // Unused sentinel — never read in setup() when !needs_normal
            super::rdg::types::TextureNodeId(u32::MAX)
        };

        let ssao_output = if ssao_enabled {
            let desc = RdgTextureDesc::new_2d(
                width,
                height,
                wgpu::TextureFormat::R8Unorm,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );
            rdg.register_resource("SSAO_Output", desc, false)
        } else {
            super::rdg::types::TextureNodeId(u32::MAX)
        };

        let transmission_tex = if has_transmission {
            let desc = RdgTextureDesc::new_2d(
                width,
                height,
                HDR_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST,
            );
            rdg.register_resource("Transmission_Tex", desc, false)
        } else {
            super::rdg::types::TextureNodeId(u32::MAX)
        };

        // ── 3d. Wire Scene Rendering Passes ────────────────────────────

        // Prepass (conditional: depth + optional normals + optional feature_id)
        if needs_prepass {
            let prepass = self.ctx.rdg_prepass;
            prepass.scene_depth = scene_depth;
            prepass.scene_normals = scene_normals;
            prepass.needs_normal = needs_normal;
            prepass.needs_feature_id = needs_feature_id;
            rdg.add_pass(prepass);
        }

        // SSAO (conditional: reads depth + normals, writes AO texture)
        if ssao_enabled {
            let ssao_pass = self.ctx.rdg_ssao_pass;
            ssao_pass.depth_tex = scene_depth;
            ssao_pass.normal_tex = scene_normals;
            ssao_pass.output_tex = ssao_output;

            let ssao_uniforms = &self.ctx.scene.ssao.uniforms;
            self.ctx.resource_manager.ensure_buffer(ssao_uniforms);
            ssao_pass.uniforms_cpu_id = ssao_uniforms.id();
            ssao_pass.global_state_key =
                (self.ctx.render_state.id, self.ctx.scene.id());
            rdg.add_pass(ssao_pass);
        }

        // Opaque
        {
            let opaque = self.ctx.rdg_opaque_pass;
            opaque.scene_color = scene_color;
            opaque.scene_depth = scene_depth;
            opaque.has_prepass = needs_prepass;
            opaque.clear_color = self.ctx.scene.background.clear_color();
            opaque.needs_specular = needs_specular;
            opaque.ssao_enabled = ssao_enabled;
            opaque.ssao_tex = ssao_output;
            rdg.add_pass(opaque);
        }

        // Skybox (conditional)
        if needs_skybox {
            let skybox = self.ctx.rdg_skybox_pass;
            skybox.scene_color = scene_color;
            skybox.scene_depth = scene_depth;
            rdg.add_pass(skybox);
        }

        // Transmission Copy (conditional: only when materials use transmission)
        if has_transmission {
            let tc = self.ctx.rdg_transmission_copy_pass;
            tc.scene_color = scene_color;
            tc.transmission_tex = transmission_tex;
            tc.active = true;
            rdg.add_pass(tc);
        }

        // Transparent
        {
            let transparent = self.ctx.rdg_transparent_pass;
            transparent.scene_color = scene_color;
            transparent.scene_depth = scene_depth;
            transparent.has_transmission = has_transmission;
            transparent.transmission_tex = transmission_tex;
            transparent.ssao_enabled = ssao_enabled;
            transparent.ssao_tex = ssao_output;
            rdg.add_pass(transparent);
        }

        // ── 3e. Wire Post-Processing Passes ───────────────────────────
        //
        // Chain: SceneColor → [Bloom] → ToneMap → [FXAA] → Surface
        //
        // Bloom ON  + FXAA ON  → HDR → Bloom → HDR' → ToneMap → LDR → FXAA → Surface
        // Bloom ON  + FXAA OFF → HDR → Bloom → HDR' → ToneMap → Surface
        // Bloom OFF + FXAA ON  → HDR → ToneMap → LDR → FXAA → Surface
        // Bloom OFF + FXAA OFF → HDR → ToneMap → Surface

        let tonemap_input = if bloom_enabled {
            let bloom_out = rdg.register_resource("Bloom_Out", hdr_desc.clone(), false);

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
            bloom_chain_desc.mip_level_count = mip_count;

            let bloom_chain = rdg.register_resource("Bloom_MipChain", bloom_chain_desc, false);

            let bloom_pass = self.ctx.rdg_bloom_pass;
            bloom_pass.bloom_texture = bloom_chain;
            bloom_pass.input_tex = scene_color;
            bloom_pass.output_tex = bloom_out;
            bloom_pass.karis_average = self.ctx.scene.bloom.karis_average;
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
            scene_color
        };

        let tonemap_output = if fxaa_enabled {
            rdg.register_resource("LDR_Intermediate", surface_desc.clone(), false)
        } else {
            surface_out
        };

        // ToneMap
        {
            let tone_map = self.ctx.rdg_tone_map_pass;
            tone_map.input_tex = tonemap_input;
            tone_map.output_tex = tonemap_output;
            tone_map.mode = self.ctx.scene.tone_mapping.mode;
            tone_map.has_lut = self.ctx.scene.tone_mapping.lut_texture.is_some();
            let cpu_buffer = &self.ctx.scene.tone_mapping.uniforms;
            self.ctx.resource_manager.ensure_buffer(cpu_buffer);
            tone_map.uniforms_cpu_id = cpu_buffer.id();
            tone_map.lut_handle = self.ctx.scene.tone_mapping.lut_texture.clone();
            tone_map.global_state_key =
                (self.ctx.render_state.id, self.ctx.scene.id());
            rdg.add_pass(tone_map);
        }

        // FXAA (conditional)
        if fxaa_enabled {
            let fxaa = self.ctx.rdg_fxaa_pass;
            fxaa.input_tex = tonemap_output;
            fxaa.output_tex = surface_out;
            rdg.add_pass(fxaa);
        }

        // ━━━ 4. Compile & Execute RDG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        rdg.compile(self.ctx.rdg_pool, &self.ctx.wgpu_ctx.device);

        // Map external resources to physical views for prepare + execute
        let scene_color_tracked = &self.ctx.frame_resources.scene_color_view[0];
        let depth_tracked = &self.ctx.frame_resources.depth_only_view;

        let ext_res: FxHashMap<_, _> = [
            (scene_color, scene_color_tracked),
            (scene_depth, depth_tracked),
        ]
        .into_iter()
        .collect();

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
            wgpu_ctx: &*self.ctx.wgpu_ctx,
            render_lists: self.ctx.render_lists,
            frame_resources: self.ctx.frame_resources,
            extracted_scene: self.ctx.extracted_scene,
            render_state: self.ctx.render_state,
            scene: self.ctx.scene,
            camera: self.ctx.camera,
            assets: self.ctx.assets,
            transient_pool: self.ctx.transient_pool,
            blackboard: self.ctx.blackboard,
        };

        for &pass_idx in &rdg.execution_queue {
            let pass = rdg.passes[pass_idx].get_pass_mut();
            pass.prepare(&mut rdg_prepare_ctx);
        }

        // Execute phase: map external views for rendering
        let mut ext_views: FxHashMap<_, &wgpu::TextureView> = FxHashMap::default();
        ext_views.insert(scene_color, scene_color_tracked);
        ext_views.insert(scene_depth, depth_tracked);
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
            resource_manager: self.ctx.resource_manager,
            render_lists: self.ctx.render_lists,
            frame_resources: self.ctx.frame_resources,
            transient_pool: self.ctx.transient_pool,
            wgpu_ctx: &*self.ctx.wgpu_ctx,
            blackboard: self.ctx.blackboard,
        };

        let mut rdg_encoder =
            self.ctx
                .wgpu_ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("RDG Unified Encoder"),
                });

        for &pass_idx in &rdg.execution_queue {
            let pass = rdg.passes[pass_idx].get_pass_mut();
            #[cfg(debug_assertions)]
            rdg_encoder.push_debug_group(pass.name());
            pass.execute(&rdg_execute_ctx, &mut rdg_encoder);
            #[cfg(debug_assertions)]
            rdg_encoder.pop_debug_group();
        }

        // ━━━ 5. Submit & Present ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        self.ctx.wgpu_ctx.queue.submit(Some(rdg_encoder.finish()));
        output.present();
        self.ctx.transient_pool.reset();
    }
}
