//! Frame Composer
//!
//! `FrameComposer` orchestrates the entire rendering pipeline for a single
//! frame using the Render Dependency Graph (RDG). All GPU work — compute
//! pre-processing, shadow mapping, scene rendering, post-processing, and
//! custom user hooks — flows through a single unified RDG.
//!
//! # Architecture
//!
//! The composer follows a **hybrid imperative Feature + declarative blackboard**
//! model:
//!
//! 1. **Extract & Prepare** — each active [`Feature`] initialises persistent
//!    GPU resources (pipelines, layouts, buffers) via [`ExtractContext`].
//! 2. **Build Graph** — Features inject transient pass-nodes via imperative
//!    `add_to_graph()` calls that return explicit [`TextureNodeId`]s. Plain
//!    pass-nodes (opaque, transparent, etc.) wire themselves through the
//!    blackboard name lookup system.
//! 3. **Compile & Execute** — topological sort, transient resource allocation,
//!    prepare, execute, submit.
//!
//! # Resource Ownership
//!
//! The Composer registers only the shared **backbone** resources
//! (`Scene_Color_HDR`, `Scene_Depth`, `Surface_Out`). All other transient
//! resources are created by their **producer Features / PassNodes** inside
//! `add_to_graph()` or `setup()`.
//!
//! # Rendering Pipeline
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                  Unified RDG Pipeline                       │
//! │                                                             │
//! │  HighFidelity:                                              │
//! │  BRDF LUT → IBL → Shadow → Prepass → SSAO → Opaque →      │
//! │  SSSSS → Skybox → TransmissionCopy → Transparent →         │
//! │  Bloom → ToneMap → FXAA → [User Hooks] → Surface           │
//! │                                                             │
//! │  BasicForward:                                              │
//! │  BRDF LUT → IBL → Shadow → Skybox(prepare) →               │
//! │  SimpleForward → [User Hooks] → Surface                    │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use rustc_hash::FxHashMap;

use crate::assets::AssetServer;
use crate::render::RenderState;
use crate::renderer::core::binding::GlobalBindGroupCache;
use crate::renderer::core::resources::Tracked;
use crate::renderer::core::{ResourceManager, WgpuContext};
use crate::renderer::graph::ExtractedScene;
use crate::renderer::graph::frame::RenderLists;
use crate::renderer::graph::rdg::blackboard::{GraphBlackboard, HookStage};
use crate::renderer::graph::rdg::context::RdgViewResolver;
use crate::renderer::graph::rdg::feature::ExtractContext;
use crate::renderer::graph::rdg::types::TextureNodeId;
use crate::renderer::pipeline::PipelineCache;
use crate::renderer::pipeline::ShaderManager;
use crate::scene::Scene;
use crate::scene::camera::RenderCamera;

/// All references needed to render a single frame.
///
/// Carries mutable borrows of GPU infrastructure, Features, and plain
/// PassNodes. Built by [`Renderer::begin_frame`], consumed by
/// [`FrameComposer::render`].
pub struct ComposerContext<'a> {
    pub wgpu_ctx: &'a mut WgpuContext,
    pub resource_manager: &'a mut ResourceManager,
    pub pipeline_cache: &'a mut PipelineCache,
    pub shader_manager: &'a mut ShaderManager,

    pub extracted_scene: &'a ExtractedScene,
    pub render_state: &'a RenderState,

    pub global_bind_group_cache: &'a mut GlobalBindGroupCache,

    pub render_lists: &'a mut RenderLists,

    // External scene data
    pub scene: &'a mut Scene,
    pub camera: &'a RenderCamera,
    pub assets: &'a AssetServer,
    pub time: f32,

    pub rdg_graph: &'a mut crate::renderer::graph::rdg::graph::RenderGraph,
    pub rdg_pool: &'a mut crate::renderer::graph::rdg::allocator::RdgTransientPool,
    pub sampler_registry: &'a mut crate::renderer::core::resources::SamplerRegistry,

    // ─── Feature-based passes ───────────────────────────────────────
    pub fxaa_feature: &'a mut crate::renderer::graph::rdg::fxaa::FxaaFeature,
    pub tone_map_feature: &'a mut crate::renderer::graph::rdg::tone_mapping::ToneMapFeature,
    pub bloom_feature: &'a mut crate::renderer::graph::rdg::bloom::BloomFeature,
    pub ssao_feature: &'a mut crate::renderer::graph::rdg::ssao::SsaoFeature,
    pub prepass_feature: &'a mut crate::renderer::graph::rdg::prepass::PrepassFeature,
    pub skybox_feature: &'a mut crate::renderer::graph::rdg::skybox::SkyboxFeature,
    pub sssss_feature: &'a mut crate::renderer::graph::rdg::sssss::SssssFeature,
    pub shadow_feature: &'a mut crate::renderer::graph::rdg::shadow::ShadowFeature,
    pub brdf_feature: &'a mut crate::renderer::graph::rdg::compute::BrdfLutFeature,
    pub ibl_feature: &'a mut crate::renderer::graph::rdg::compute::IblFeature,

    // ─── Plain PassNode passes (no prepare_resources) ───────────────
    pub rdg_opaque_pass: &'a mut crate::renderer::graph::rdg::opaque::RdgOpaquePass,
    pub rdg_transparent_pass: &'a mut crate::renderer::graph::rdg::transparent::RdgTransparentPass,
    pub rdg_transmission_copy_pass:
        &'a mut crate::renderer::graph::rdg::transmission_copy::RdgTransmissionCopyPass,
    pub rdg_simple_forward_pass:
        &'a mut crate::renderer::graph::rdg::simple_forward::RdgSimpleForwardPass,
}

/// Frame Composer — builds and executes the unified render graph.
///
/// Provides a fluent API for injecting custom RDG passes via hooks.
/// Consumed by [`render()`] which takes ownership.
pub struct FrameComposer<'a> {
    ctx: ComposerContext<'a>,
    external_res: FxHashMap<TextureNodeId, &'a Tracked<wgpu::TextureView>>,
    hooks: smallvec::SmallVec<
        [(
            HookStage,
            Box<
                dyn FnMut(&mut crate::renderer::graph::rdg::graph::RenderGraph, &GraphBlackboard)
                    + 'a,
            >,
        ); 4],
    >,
}

impl<'a> FrameComposer<'a> {
    pub(crate) fn new(ctx: ComposerContext<'a>) -> Self {
        Self {
            ctx,
            external_res: FxHashMap::default(),
            hooks: smallvec::SmallVec::new(),
        }
    }

    /// Register a custom pass hook at the specified stage.
    ///
    /// The closure receives the [`RenderGraph`] and [`GraphBlackboard`] so it
    /// can wire pass-nodes that read/write the frame's well-known resources.
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

    /// Execute the full rendering pipeline and present to the screen.
    ///
    /// # Phases
    ///
    /// 1. **Extract & Prepare** — Features compile pipelines, upload buffers
    /// 2. **Acquire Surface** — get the swap-chain back buffer
    /// 3. **Build RDG** — register backbone resources, wire Features and PassNodes
    /// 4. **Compile** — topological sort, allocate transients
    /// 5. **Prepare** — assemble per-frame bind groups (transient resource views)
    /// 6. **Execute** — record and submit GPU commands
    /// 7. **Present** — present swap-chain
    ///
    /// Consumes `self`; the composer cannot be reused after render.
    #[allow(clippy::too_many_lines)]
    pub fn render(mut self) {
        use crate::renderer::HDR_TEXTURE_FORMAT;
        use crate::renderer::graph::rdg::bloom::BloomParams;
        use crate::renderer::graph::rdg::context::{RdgExecuteContext, RdgPrepareContext};
        use crate::renderer::graph::rdg::feature::SkyboxConfig;
        use crate::renderer::graph::rdg::tone_mapping::ToneMapParams;
        use crate::renderer::graph::rdg::types::RdgTextureDesc;

        // ━━━ 0. Scene Configuration ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        //
        // Compute rendering flags before destructuring self.ctx.

        let is_high_fidelity = self.ctx.wgpu_ctx.render_path.supports_post_processing();
        let ssao_enabled = self.ctx.scene.ssao.enabled && is_high_fidelity;
        let needs_feature_id = is_high_fidelity
            && (self.ctx.scene.screen_space.enable_sss || self.ctx.scene.screen_space.enable_ssr);
        let needs_normal = ssao_enabled || needs_feature_id;
        let needs_skybox = self.ctx.scene.background.needs_skybox_pass();
        let needs_specular = self.ctx.scene.screen_space.enable_sss && is_high_fidelity;
        let has_transmission = self.ctx.render_lists.use_transmission && is_high_fidelity;
        let bloom_enabled = self.ctx.scene.bloom.enabled && is_high_fidelity;
        let fxaa_enabled = self.ctx.scene.fxaa.enabled && is_high_fidelity;

        let scene_id_val = self.ctx.scene.id();
        let render_state_id = self.ctx.render_state.id;
        let global_state_key = (render_state_id, scene_id_val);

        let bg_uniforms_gpu_id = self
            .ctx
            .resource_manager
            .ensure_buffer_id(&self.ctx.scene.background.uniforms);
        let bg_uniforms_cpu_id = self.ctx.scene.background.uniforms.id();
        let bg_mode = self.ctx.scene.background.mode.clone();

        let clear_color = self.ctx.extracted_scene.background.clear_color();
        let depth_format = self.ctx.wgpu_ctx.depth_format;
        let surface_view_format = self.ctx.wgpu_ctx.surface_view_format;

        // Pre-ensure GPU buffers for uniforms that Features will reference.
        if ssao_enabled {
            self.ctx
                .resource_manager
                .ensure_buffer(&self.ctx.scene.ssao.uniforms);
        }
        if bloom_enabled {
            self.ctx
                .resource_manager
                .ensure_buffer(&self.ctx.scene.bloom.upsample_uniforms);
            self.ctx
                .resource_manager
                .ensure_buffer(&self.ctx.scene.bloom.composite_uniforms);
        }
        {
            let cpu_buffer = &self.ctx.scene.tone_mapping.uniforms;
            self.ctx.resource_manager.ensure_buffer(cpu_buffer);
        }

        // ━━━ 1. Extract & Prepare Features ━━━━━━━━━━━━━━━━━━━━━━━━━━━
        //
        // Build an [`ExtractContext`] and call each Feature's
        // `extract_and_prepare()` to initialise persistent GPU resources
        // (pipelines, layouts, buffers). This runs *before* graph assembly.
        //
        // Return values (e.g. tone_map pipeline ID) are captured via locals
        // declared before the block so they survive the ExtractContext scope.

        let mut tone_map_pipeline_id = None;

        {
            let mut extract_ctx = ExtractContext {
                device: &self.ctx.wgpu_ctx.device,
                queue: &self.ctx.wgpu_ctx.queue,
                pipeline_cache: self.ctx.pipeline_cache,
                shader_manager: self.ctx.shader_manager,
                sampler_registry: self.ctx.sampler_registry,
                global_bind_group_cache: self.ctx.global_bind_group_cache,
                resource_manager: self.ctx.resource_manager,
                wgpu_ctx: &*self.ctx.wgpu_ctx,
                render_lists: self.ctx.render_lists,
                extracted_scene: self.ctx.extracted_scene,
                render_state: self.ctx.render_state,
                assets: self.ctx.assets,
            };

            // Compute & shadow (always active)
            self.ctx.brdf_feature.extract_and_prepare(&mut extract_ctx);
            self.ctx.ibl_feature.extract_and_prepare(&mut extract_ctx);
            self.ctx
                .shadow_feature
                .extract_and_prepare(&mut extract_ctx);

            if is_high_fidelity {
                self.ctx.prepass_feature.extract_and_prepare(
                    &mut extract_ctx,
                    needs_normal,
                    needs_feature_id,
                );

                if ssao_enabled {
                    let ssao_uniforms_cpu_id = self.ctx.scene.ssao.uniforms.id();
                    self.ctx.ssao_feature.extract_and_prepare(
                        &mut extract_ctx,
                        ssao_uniforms_cpu_id,
                        global_state_key,
                    );
                }

                if needs_specular {
                    self.ctx
                        .sssss_feature
                        .extract_and_prepare(&mut extract_ctx);
                }

                if bloom_enabled {
                    self.ctx
                        .bloom_feature
                        .extract_and_prepare(&mut extract_ctx);
                }

                // Tone mapping — capture pipeline ID for graph building phase
                let tm_params = ToneMapParams {
                    mode: self.ctx.scene.tone_mapping.mode,
                    has_lut: self.ctx.scene.tone_mapping.lut_texture.is_some(),
                    uniforms_cpu_id: self.ctx.scene.tone_mapping.uniforms.id(),
                    lut_handle: self.ctx.scene.tone_mapping.lut_texture,
                    global_state_key,
                    output_format: surface_view_format,
                };
                tone_map_pipeline_id = Some(
                    self.ctx
                        .tone_map_feature
                        .extract_and_prepare(&mut extract_ctx, &tm_params),
                );

                if fxaa_enabled {
                    self.ctx.fxaa_feature.extract_and_prepare(
                        &mut extract_ctx,
                        self.ctx.scene.fxaa.quality(),
                        surface_view_format,
                    );
                }
            }

            // Skybox (both paths)
            if needs_skybox {
                let skybox_config = SkyboxConfig {
                    background_mode: bg_mode.clone(),
                    bg_uniforms_cpu_id,
                    bg_uniforms_gpu_id,
                    scene_id: scene_id_val,
                    color_format: if is_high_fidelity {
                        HDR_TEXTURE_FORMAT
                    } else {
                        surface_view_format
                    },
                    depth_format,
                };
                self.ctx
                    .skybox_feature
                    .extract_and_prepare(&mut extract_ctx, &skybox_config);
            }
        }

        // ━━━ 2. Acquire Surface ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        let output = match self.ctx.wgpu_ctx.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost) => return,
            Err(e) => {
                log::error!("Render error: {e:?}");
                return;
            }
        };

        let surface_view = output.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(surface_view_format),
            ..Default::default()
        });
        let width = output.texture.width();
        let height = output.texture.height();

        // ━━━ 3. Build Unified RDG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        let rdg = self.ctx.rdg_graph;
        rdg.begin_frame(crate::renderer::graph::rdg::graph::FrameConfig {
            width,
            height,
            depth_format,
            msaa_samples: self.ctx.wgpu_ctx.msaa_samples,
            surface_format: surface_view_format,
            hdr_format: HDR_TEXTURE_FORMAT,
        });

        // ── 3a. Register Backbone Resources ─────────────────────────────

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
            depth_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let surface_desc = RdgTextureDesc::new_2d(
            width,
            height,
            surface_view_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let surface_out = rdg.register_resource("Surface_Out", surface_desc.clone(), true);
        let scene_color = rdg.register_resource("Scene_Color_HDR", hdr_desc, false);
        let scene_depth = rdg.register_resource("Scene_Depth", depth_desc, false);

        // ── 3b. Wire Feature Passes + Plain PassNodes ───────────────────

        // Compute & Shadow (always active, side-effect nodes)
        self.ctx.brdf_feature.add_to_graph(rdg);
        self.ctx.ibl_feature.add_to_graph(rdg);
        self.ctx.shadow_feature.add_to_graph(rdg);

        // Blackboard for hooks and plain PassNodes
        let blackboard = GraphBlackboard {
            current_color: scene_color,
            scene_depth,
            surface_out,
        };

        if is_high_fidelity {
            // ── HighFidelity Pipeline ───────────────────────────────────

            // Prepass
            let prepass_out = self.ctx.prepass_feature.add_to_graph(
                rdg,
                scene_depth,
                needs_normal,
                needs_feature_id,
            );

            // SSAO → registers "SSAO_Output" for opaque blackboard lookup
            if ssao_enabled {
                let ssao_uniforms_cpu_id = self.ctx.scene.ssao.uniforms.id();
                let _ssao_out =
                    self.ctx
                        .ssao_feature
                        .add_to_graph(rdg, &prepass_out, ssao_uniforms_cpu_id);
            }

            // Opaque (plain PassNode — uses blackboard: "Scene_Color_HDR",
            // "Scene_Depth", optionally reads "SSAO_Output", creates "Specular_MRT")
            self.ctx.rdg_opaque_pass.has_prepass = true;
            self.ctx.rdg_opaque_pass.needs_specular = needs_specular;
            self.ctx.rdg_opaque_pass.clear_color = clear_color;
            rdg.add_pass(self.ctx.rdg_opaque_pass);

            // SSSSS
            if needs_specular {
                let normals = prepass_out.normal.expect("SSSSS requires normal prepass");
                let feature_id = prepass_out
                    .feature_id
                    .expect("SSSSS requires feature ID prepass");
                let specular_tex = rdg
                    .find_resource("Specular_MRT")
                    .expect("Opaque must create Specular_MRT when needs_specular");
                self.ctx.sssss_feature.add_to_graph(
                    rdg,
                    scene_color,
                    scene_depth,
                    normals,
                    feature_id,
                    specular_tex,
                );
            }

            // Skybox
            if needs_skybox {
                self.ctx
                    .skybox_feature
                    .add_to_graph(rdg, scene_color, scene_depth);
            }

            // Transmission Copy (plain PassNode)
            if has_transmission {
                self.ctx.rdg_transmission_copy_pass.active = true;
                rdg.add_pass(self.ctx.rdg_transmission_copy_pass);
            }

            // Transparent (plain PassNode)
            rdg.add_pass(self.ctx.rdg_transparent_pass);

            // Before-Post-Process Hooks
            for (stage, hook) in &mut self.hooks {
                if *stage == HookStage::BeforePostProcess {
                    hook(rdg, &blackboard);
                }
            }

            // ── Post-Processing Chain ──────────────────────────────────

            let tonemap_input = if bloom_enabled {
                let bloom_params = BloomParams {
                    karis_average: self.ctx.scene.bloom.karis_average,
                    max_mip_levels: self.ctx.scene.bloom.max_mip_levels(),
                    upsample_uniforms_cpu_id: self.ctx.scene.bloom.upsample_uniforms.id(),
                    composite_uniforms_cpu_id: self.ctx.scene.bloom.composite_uniforms.id(),
                };
                self.ctx
                    .bloom_feature
                    .add_to_graph(rdg, scene_color, bloom_params)
            } else {
                scene_color
            };

            let tonemap_output = if fxaa_enabled {
                rdg.register_resource("LDR_Intermediate", surface_desc.clone(), false)
            } else {
                surface_out
            };

            // Tone mapping — use pipeline ID captured in phase 1
            {
                let tone_map_params = ToneMapParams {
                    mode: self.ctx.scene.tone_mapping.mode,
                    has_lut: self.ctx.scene.tone_mapping.lut_texture.is_some(),
                    uniforms_cpu_id: self.ctx.scene.tone_mapping.uniforms.id(),
                    lut_handle: self.ctx.scene.tone_mapping.lut_texture,
                    global_state_key,
                    output_format: surface_view_format,
                };
                self.ctx.tone_map_feature.add_to_graph(
                    rdg,
                    tonemap_input,
                    tonemap_output,
                    tone_map_pipeline_id.expect("Tone map pipeline must be prepared"),
                    &tone_map_params,
                );
            }

            // FXAA
            if fxaa_enabled {
                self.ctx
                    .fxaa_feature
                    .add_to_graph(rdg, tonemap_output, surface_out);
            }
        } else {
            // ── BasicForward Pipeline ──────────────────────────────────

            // Skybox prepare (for PreparedSkyboxDraw inline rendering)
            if needs_skybox {
                self.ctx
                    .skybox_feature
                    .add_to_graph(rdg, surface_out, scene_depth);
            }

            // SimpleForward (plain PassNode)
            self.ctx.rdg_simple_forward_pass.clear_color = clear_color;
            rdg.add_pass(self.ctx.rdg_simple_forward_pass);
        }

        // After-Post-Process Hooks (UI, debug overlays)
        for (stage, hook) in &mut self.hooks {
            if *stage == HookStage::AfterPostProcess {
                hook(rdg, &blackboard);
            }
        }

        // ━━━ 4. Compile & Execute RDG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        rdg.compile(self.ctx.rdg_pool, &self.ctx.wgpu_ctx.device);

        // ── 4a. RDG Prepare: transient-only BindGroup assembly ─────────
        //
        // Only the swapchain surface is truly external — all other textures
        // are RDG transient resources.
        self.external_res.clear();

        let mut rdg_prepare_ctx = RdgPrepareContext {
            views: RdgViewResolver {
                graph: &rdg,
                pool: self.ctx.rdg_pool,
                external_resources: &self.external_res,
            },
            device: &self.ctx.wgpu_ctx.device,
            queue: &self.ctx.wgpu_ctx.queue,
            pipeline_cache: self.ctx.pipeline_cache,
            sampler_registry: self.ctx.sampler_registry,
            global_bind_group_cache: self.ctx.global_bind_group_cache,
            resource_manager: self.ctx.resource_manager,
            render_lists: self.ctx.render_lists,
        };

        for &pass_idx in &rdg.execution_queue {
            let pass = rdg.passes[pass_idx].get_pass_mut();
            pass.prepare(&mut rdg_prepare_ctx);
        }

        // ── 4b. Execute ────────────────────────────────────────────────

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
            external_views: &ext_views,
            global_bind_group: global_bg_ref,
            resource_manager: self.ctx.resource_manager,
            render_lists: self.ctx.render_lists,
            wgpu_ctx: &*self.ctx.wgpu_ctx,
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
    }
}
