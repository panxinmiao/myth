//! Frame Composer
//!
//! `FrameComposer` orchestrates the entire rendering pipeline for a single
//! frame using the Declarative Render Graph (RDG). All GPU work — compute
//! pre-processing, shadow mapping, scene rendering, post-processing, and
//! custom user hooks — flows through a single unified RDG.
//!
//! # Rendering Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                  Unified RDG Pipeline                       │
//! │                                                             │
//! │  HighFidelity:                                              │
//! │  BRDF LUT → IBL → Shadow → Prepass → SSAO → Opaque →      │
//! │  Skybox → TransmissionCopy → Transparent → SSSSS →         │
//! │  Bloom → ToneMap → FXAA → [User Hooks] → Surface           │
//! │                                                             │
//! │  BasicForward:                                              │
//! │  BRDF LUT → IBL → Shadow → Skybox(prepare) →               │
//! │  SimpleForward → [User Hooks] → Surface                    │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! renderer.begin_frame(scene, &camera, assets, time)?
//!     .add_custom_pass(|rdg, bb| {
//!         ui_pass.target_tex = bb.surface_out;
//!         rdg.add_pass(&mut ui_pass);
//!     })
//!     .render();
//! ```

use crate::assets::AssetServer;
use crate::render::RenderState;
use crate::renderer::core::binding::GlobalBindGroupCache;
use crate::renderer::core::{ResourceManager, WgpuContext};
use crate::renderer::graph::ExtractedScene;
use crate::renderer::graph::context::FrameResources;
use crate::renderer::graph::frame::{RenderLists};
use crate::renderer::graph::rdg::blackboard::{GraphBlackboard, HookStage};
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
    pub global_bind_group_cache: &'a mut GlobalBindGroupCache,

    /// Render lists (populated by `SceneCullPass`)
    pub render_lists: &'a mut RenderLists,

    // /// Frame blackboard (cross-pass transient data communication)
    // pub blackboard: &'a mut FrameBlackboard,

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
    pub rdg_simple_forward_pass: &'a mut crate::renderer::graph::rdg::simple_forward::RdgSimpleForwardPass,
    pub rdg_sssss_pass: &'a mut crate::renderer::graph::rdg::sssss::RdgSssssPass,

    // Shadow + Compute (migrated from old system)
    pub rdg_shadow_pass: &'a mut crate::renderer::graph::rdg::shadow::RdgShadowPass,
    pub rdg_brdf_pass: &'a mut crate::renderer::graph::rdg::compute::RdgBrdfLutPass,
    pub rdg_ibl_pass: &'a mut crate::renderer::graph::rdg::compute::RdgIblComputePass,
}

/// Frame Composer
///
/// Holds all context references needed to render a single frame and provides
/// a fluent API for injecting custom RDG passes via hooks.
///
/// # Design Notes
///
/// - **Single unified graph**: All rendering (scene + post + custom) flows
///   through the RDG. No legacy graph system remains.
/// - **Hook-based extensibility**: External code (e.g. UI) injects passes
///   via [`add_custom_pass`](Self::add_custom_pass) closures that receive
///   the [`GraphBlackboard`] for type-safe resource wiring.
/// - **Lifetime safety**: Lifetime `'a` locks the mutable borrow on `Renderer`.
/// - **Deferred Surface acquisition**: The Surface is acquired only in
///   `.render()` to minimise hold time.
pub struct FrameComposer<'a> {
    ctx: ComposerContext<'a>,
    hooks: smallvec::SmallVec<[(HookStage, Box<dyn FnMut(&mut crate::renderer::graph::rdg::graph::RenderGraph, &GraphBlackboard) + 'a>); 4]>,
}

impl<'a> FrameComposer<'a> {
    /// Creates a new frame composer.
    pub(crate) fn new(ctx: ComposerContext<'a>) -> Self {
        Self {
            ctx,
            hooks: smallvec::SmallVec::new(),
        }
    }

    /// Registers a custom pass hook that will be invoked during graph building.
    ///
    /// The closure receives a mutable reference to the [`RenderGraph`] and
    /// the [`GraphBlackboard`] containing the frame's well-known resource slots.
    /// This is the primary extension point for UI overlays, debug visualisations,
    /// and other user-defined passes.
    ///
    /// Hooks registered with [`HookStage::AfterPostProcess`] run after all
    /// built-in post-processing (Bloom, ToneMap, FXAA) and are typically used
    /// for UI rendering.
    ///
    /// # Example
    ///
    /// ```ignore
    /// composer
    ///     .add_custom_pass(HookStage::AfterPostProcess, |rdg, bb| {
    ///         my_ui_pass.target_tex = bb.surface_out;
    ///         rdg.add_pass(&mut my_ui_pass);
    ///     })
    ///     .render();
    /// ```
    #[inline]
    #[must_use]
    pub fn add_custom_pass<F>(mut self, stage: HookStage, hook: F) -> Self
    where
        F: FnMut(&mut crate::renderer::graph::rdg::graph::RenderGraph, &GraphBlackboard) + 'a,
    {
        self.hooks.push((stage, Box::new(hook)));
        self
    }

    /// Executes the full rendering pipeline and presents to the screen.
    ///
    /// # Architecture
    ///
    /// 1. **Acquire Surface** — get the swap-chain back buffer
    /// 2. **Build RDG** — register resources, wire passes (compute, shadow,
    ///    scene, post-processing), invoke user hooks
    /// 3. **Compile & Execute** — topological sort, allocate transients,
    ///    prepare, execute, submit
    /// 4. **Present** — present swap-chain
    ///
    /// Consumes `self`; the composer cannot be reused after render.
    pub fn render(mut self) {
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

        // ━━━ 2. Build Unified RDG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        let rdg = self.ctx.rdg_graph;
        rdg.begin_frame();

        // ── 2a. Register Resources ──────────────────────────────────────
        // Only the swapchain surface is truly external.
        // Scene colour and depth are transient — owned and aliased by the RDG.

        let hdr_desc = RdgTextureDesc::new_2d(
            width,
            height,
            HDR_TEXTURE_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );

        let depth_desc = RdgTextureDesc::new(
            width,
            height,
            1,
            1,
            self.ctx.wgpu_ctx.msaa_samples,
            wgpu::TextureDimension::D2,
            self.ctx.wgpu_ctx.depth_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let surface_desc = RdgTextureDesc::new_2d(
            width,
            height,
            view_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let surface_out = rdg.register_resource("Surface_Out", surface_desc.clone(), true);

        // Transient scene resources — RDG manages allocation & aliasing
        let scene_color = rdg.register_resource("Scene_Color_HDR", hdr_desc.clone(), false);
        let scene_depth = rdg.register_resource("Scene_Depth", depth_desc.clone(), false);

        // ── 2b. Scene Configuration ────────────────────────────────────

        let is_high_fidelity = self.ctx.wgpu_ctx.render_path.supports_post_processing();
        let ssao_enabled = self.ctx.scene.ssao.enabled && is_high_fidelity;
        let needs_feature_id = is_high_fidelity
            && (self.ctx.scene.screen_space.enable_sss || self.ctx.scene.screen_space.enable_ssr);
        let needs_normal = ssao_enabled || needs_feature_id;
        // let needs_prepass = needs_normal;
        let needs_skybox = self.ctx.scene.background.needs_skybox_pass();
        let needs_specular = self.ctx.scene.screen_space.enable_sss && is_high_fidelity;
        let has_transmission = self.ctx.render_lists.use_transmission && is_high_fidelity;
        let bloom_enabled = self.ctx.scene.bloom.enabled && is_high_fidelity;
        let fxaa_enabled = self.ctx.scene.fxaa.enabled && is_high_fidelity;

        // ── 2c. Register Transient Resources ───────────────────────────

        let scene_normals = if needs_normal {
            let desc = RdgTextureDesc::new_2d(
                width,
                height,
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );
            rdg.register_resource("Scene_Normals", desc, false)
        } else {
            super::rdg::types::TextureNodeId(u32::MAX)
        };

        let feature_id = if needs_feature_id {
            let desc = RdgTextureDesc::new_2d(
                width,
                height,
                wgpu::TextureFormat::Rg8Uint,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );
            rdg.register_resource("Feature_ID", desc, false)
        } else {
            super::rdg::types::TextureNodeId(u32::MAX)
        };

        let specular_tex = if needs_specular {
            rdg.register_resource("Specular_MRT", hdr_desc.clone(), false)
        } else {
            super::rdg::types::TextureNodeId(u32::MAX)
        };

        let ssao_output = if ssao_enabled {
            let desc = RdgTextureDesc::new_2d(
                width / 2,
                height / 2,
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

        // ── 2d. Wire Compute + Shadow Passes ───────────────────────────

        rdg.add_pass(self.ctx.rdg_brdf_pass);
        rdg.add_pass(self.ctx.rdg_ibl_pass);
        rdg.add_pass(self.ctx.rdg_shadow_pass);

        // ── 2e. Wire Scene Rendering Passes ────────────────────────────

        // Build blackboard for hooks
        let blackboard = GraphBlackboard {
            scene_color,
            scene_depth,
            surface_out,
        };

        if is_high_fidelity {
            // HighFidelity pipeline: separate passes for opaque, skybox,
            // transparent with HDR targets and post-processing chain.

            // if needs_prepass {
            let prepass = self.ctx.rdg_prepass;
            prepass.scene_depth = scene_depth;
            prepass.scene_normals = scene_normals;
            prepass.needs_normal = needs_normal;
            prepass.needs_feature_id = needs_feature_id;
            prepass.feature_id = feature_id;
            rdg.add_pass(prepass);
            //}

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
                opaque.has_prepass = true;
                opaque.clear_color = self.ctx.extracted_scene.background.clear_color();
                opaque.needs_specular = needs_specular;
                opaque.specular_tex = specular_tex;
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

            // Transmission Copy (conditional)
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

            // SSSSS (Screen-Space Sub-Surface Scattering) — SSA-compliant
            // Reads scene_color, writes a NEW sssss_output node. current_color
            // cursor is advanced so downstream passes read the post-SSSSS result.
            if needs_specular {
                let sssss_temp = rdg.register_resource("SSSSS_Temp", hdr_desc.clone(), false);
                // let sssss_output = rdg.register_resource("SceneColor_PostSSSS", hdr_desc.clone(), false);

                let sssss = self.ctx.rdg_sssss_pass;
                // sssss.color_in = scene_color;
                sssss.scene_color = scene_color;
                sssss.temp_blur = sssss_temp;
                sssss.depth_in = scene_depth;
                sssss.normal_in = scene_normals;
                sssss.feature_id = feature_id;
                sssss.specular_tex = specular_tex;
                sssss.enabled = true;
                rdg.add_pass(sssss);

            }

            // ── Before-Post-Process Hooks ──────────────────────────────
            for (stage, hook) in &mut self.hooks {
                if *stage == HookStage::BeforePostProcess {
                    hook(rdg, &blackboard);
                }
            }

            // ── Post-Processing Chain ──────────────────────────────────

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

                let bloom_chain_desc = RdgTextureDesc::new(
                    bloom_w,
                    bloom_h,
                    1,
                    mip_count,
                    1,
                    wgpu::TextureDimension::D2,
                    HDR_TEXTURE_FORMAT,
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                );
                // bloom_chain_desc.mip_level_count = mip_count;

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
                bloom_pass.composite_uniforms_cpu_id =
                    self.ctx.scene.bloom.composite_uniforms.id();
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
        } else {
            // BasicForward pipeline: single-pass LDR rendering.

            // Skybox prepare is still needed (stores PreparedSkyboxDraw for
            // inline rendering inside SimpleForwardPass). The skybox pass is
            // wired into the graph as a side-effect node so its prepare()
            // runs, but execute() is a no-op since SimpleForwardPass draws
            // the skybox inline.
            if needs_skybox {
                let skybox = self.ctx.rdg_skybox_pass;
                skybox.scene_color = surface_out;
                skybox.scene_depth = scene_depth;
                rdg.add_pass(skybox);
            }

            let msaa_desc = RdgTextureDesc::new(
                width,
                height,
                1,
                1,
                self.ctx.wgpu_ctx.msaa_samples,
                wgpu::TextureDimension::D2,
                view_format,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );

            // SimpleForwardPass: opaque + skybox (inline) + transparent
            {
                let simple_fwd = self.ctx.rdg_simple_forward_pass;
                simple_fwd.surface_out = surface_out;
                simple_fwd.scene_depth = scene_depth;
                if self.ctx.wgpu_ctx.msaa_samples > 1{
                    let msaa = rdg.register_resource("Scene_Msaa", msaa_desc, false);
                    simple_fwd.msaa_view = Some(msaa);
                }else {
                    simple_fwd.msaa_view = None;
                }
                rdg.add_pass(simple_fwd);
            }
        }

        // ── After-Post-Process Hooks (UI, debug overlays) ──────────────
        for (stage, hook) in &mut self.hooks {
            if *stage == HookStage::AfterPostProcess {
                hook(rdg, &blackboard);
            }
        }

        // ━━━ 3. Compile & Execute RDG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        rdg.compile(self.ctx.rdg_pool, &self.ctx.wgpu_ctx.device);

        // Only the swapchain surface is truly external — all other textures
        // (scene_color, scene_depth, etc.) are RDG transient resources.
        let ext_res: FxHashMap<_, _> = FxHashMap::default();

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
            // blackboard: self.ctx.blackboard,
        };

        for &pass_idx in &rdg.execution_queue {
            let pass = rdg.passes[pass_idx].get_pass_mut();
            pass.prepare(&mut rdg_prepare_ctx);
        }

        // Execute phase: only the swapchain surface is external
        let mut ext_views: FxHashMap<_, &wgpu::TextureView> = FxHashMap::default();
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
            wgpu_ctx: &*self.ctx.wgpu_ctx,
            // blackboard: self.ctx.blackboard,
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

        // ━━━ 4. Submit & Present ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        self.ctx.wgpu_ctx.queue.submit(Some(rdg_encoder.finish()));
        output.present();
    }
}
