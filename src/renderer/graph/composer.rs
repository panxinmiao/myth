//! Frame Composer
//!
//! `FrameComposer` orchestrates the entire rendering pipeline for a single
//! frame using the Declarative Render Graph (RDG). All GPU work — compute
//! pre-processing, shadow mapping, scene rendering, post-processing, and
//! custom user hooks — flows through a single unified RDG.
//!
//! # Resource Ownership (Explicit Wiring)
//!
//! The Composer registers only the **external** `Surface_Out` resource and
//! the routing-level `LDR_Intermediate`.  All scene-level transient resources
//! (`Scene_Color_HDR`, `Scene_Depth`, MSAA intermediates, specular MRT, etc.)
//! are created by their **producer passes** inside `add_to_graph()`.  Each
//! pass returns typed output structs (`PrepassOutputs`, `OpaqueOutputs`, …)
//! carrying `TextureNodeId` values that the Composer threads to downstream
//! consumers — no blackboard lookups remain for mutable resources.
//!
//! # Rendering Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │                    Unified RDG Pipeline                        │
//! │                                                                │
//! │  HighFidelity:                                                 │
//! │  BRDF LUT → IBL → Shadow → Prepass → SSAO → Opaque →         │
//! │  SSSS → Skybox → TransmissionCopy → Transparent →             │
//! │  [Bloom_System: Extract → DS_1..N → US_N..0 → Composite] →   │
//! │  ToneMap → FXAA → [User Hooks] → Surface                     │
//! │                                                                │
//! │  BasicForward:                                                 │
//! │  BRDF LUT → IBL → Shadow → Skybox(prepare) →                  │
//! │  SimpleForward → [User Hooks] → Surface                       │
//! └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! renderer.begin_frame(scene, &camera, assets, time)?
//!     .add_custom_pass(HookStage::AfterPostProcess, |rdg, bb| {
//!         let new_surface = rdg.add_pass("UI_Pass", |builder| {
//!             let out = builder.mutate_and_export(bb.surface_out, "Surface_With_UI");
//!             (ui_node, out)
//!         });
//!         GraphBlackboard { surface_out: new_surface, ..bb }
//!     })
//!     .render();
//! ```

use rustc_hash::FxHashMap;

use crate::assets::AssetServer;
use crate::render::RenderState;
use crate::renderer::core::binding::GlobalBindGroupCache;
use crate::renderer::core::gpu::{SamplerRegistry, Tracked};
use crate::renderer::core::{ResourceManager, WgpuContext};
use crate::renderer::graph::ExtractedScene;
use crate::renderer::graph::core::{
    ExecuteContext, FrameArena, GraphBlackboard, HookStage, PrepareContext, RenderGraph, TextureDesc,
    TextureNodeId, TransientPool, ViewResolver,
};
use crate::renderer::graph::core::GraphStorage;
use crate::renderer::graph::frame::{PreparedSkyboxDraw, RenderLists};
use crate::renderer::graph::passes::{
    BloomFeature, BrdfLutFeature, FxaaFeature, IblComputeFeature, MsaaSyncFeature, OpaqueFeature,
    PrepassFeature, ShadowFeature, SimpleForwardFeature, SkyboxFeature, SsaoFeature, SsssFeature,
    ToneMappingFeature, TransmissionCopyFeature, TransparentFeature,
};
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

    pub global_bind_group_cache: &'a mut GlobalBindGroupCache,

    /// Render lists (populated by `SceneCullPass`)
    pub render_lists: &'a mut RenderLists,

    // External scene data
    pub scene: &'a mut Scene,
    pub camera: &'a RenderCamera,
    pub assets: &'a AssetServer,
    pub time: f32,

    pub graph_storage: &'a mut GraphStorage,
    pub transient_pool: &'a mut TransientPool,
    pub sampler_registry: &'a mut SamplerRegistry,
    pub frame_arena: &'a FrameArena,

    // ─── RDG Features ────────────────────────────────────────────────────
    // Post-processing
    pub fxaa_pass: &'a mut FxaaFeature,
    pub tone_map_pass: &'a mut ToneMappingFeature,
    pub bloom_pass: &'a mut BloomFeature,
    pub ssao_pass: &'a mut SsaoFeature,
    // Scene rendering
    pub prepass: &'a mut PrepassFeature,
    pub opaque_pass: &'a mut OpaqueFeature,
    pub skybox_pass: &'a mut SkyboxFeature,
    pub transparent_pass: &'a mut TransparentFeature,
    pub transmission_copy_pass: &'a mut TransmissionCopyFeature,
    pub simple_forward_pass: &'a mut SimpleForwardFeature,
    pub ssss_pass: &'a mut SsssFeature,
    pub msaa_sync_pass: &'a mut MsaaSyncFeature,

    // Shadow + Compute
    pub shadow_pass: &'a mut ShadowFeature,
    pub brdf_pass: &'a mut BrdfLutFeature,
    pub ibl_pass: &'a mut IblComputeFeature,
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
/// - **Pure dataflow chain**: Every `Feature::add_to_graph` returns a
///   [`TextureNodeId`], enabling a strict functional pipeline where each
///   pass's output feeds the next pass's input — no side-effect writes.
/// - **Flattened macro-nodes**: Complex multi-step effects (e.g. Bloom)
///   are decomposed into individual RDG passes, exposing fine-grained
///   dependencies to the compiler for optimal barrier and memory aliasing.
/// - **Hook-based extensibility**: External code (e.g. UI) injects passes
///   via [`add_custom_pass`](Self::add_custom_pass) closures that receive
///   the [`GraphBlackboard`] for type-safe resource wiring.
/// - **Lifetime safety**: Lifetime `'a` locks the mutable borrow on `Renderer`.
/// - **Deferred Surface acquisition**: The Surface is acquired only in
///   `.render()` to minimise hold time.
pub struct FrameComposer<'a> {
    ctx: ComposerContext<'a>,
    external_res: FxHashMap<TextureNodeId, &'a Tracked<wgpu::TextureView>>,
    #[allow(clippy::type_complexity)]
    hooks: smallvec::SmallVec<
        [(
            HookStage,
            Option<Box<dyn FnOnce(&mut RenderGraph<'a>, GraphBlackboard) -> GraphBlackboard + 'a>>,
        ); 4],
    >,
}

impl<'a> FrameComposer<'a> {
    /// Creates a new frame composer.
    pub(crate) fn new(ctx: ComposerContext<'a>) -> Self {
        Self {
            ctx,
            external_res: FxHashMap::default(),
            hooks: smallvec::SmallVec::new(),
        }
    }

    /// Returns a reference to the wgpu device.
    #[inline]
    #[must_use]
    pub fn device(&self) -> &wgpu::Device {
        &self.ctx.wgpu_ctx.device
    }

    /// Returns a reference to the resource manager.
    ///
    /// Useful for user-land passes that need to resolve engine resources
    /// (e.g. texture handles) before the RDG prepare phase.
    #[inline]
    #[must_use]
    pub fn resource_manager(&self) -> &ResourceManager {
        self.ctx.resource_manager
    }

    /// Registers a custom pass hook that will be invoked during graph building.
    ///
    /// The closure receives a mutable reference to the [`RenderGraph`] and
    /// the [`GraphBlackboard`] containing the frame's well-known resource slots.
    /// It must return an (optionally updated) [`GraphBlackboard`] — the Rust
    /// type system enforces that every hook path returns a valid blackboard.
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
    ///         let new_surface = rdg.add_pass("MyPass", |builder| {
    ///             let out = builder.mutate_and_export(bb.surface_out, "Surface_MyPass");
    ///             (MyPassNode { target: out }, out)
    ///         });
    ///         GraphBlackboard { surface_out: new_surface, ..bb }
    ///     })
    ///     .render();
    /// ```
    #[inline]
    #[must_use]
    pub fn add_custom_pass<F>(mut self, stage: HookStage, hook: F) -> Self
    where
        F: FnOnce(&mut RenderGraph<'a>, GraphBlackboard) -> GraphBlackboard + 'a,
    {
        self.hooks.push((stage, Some(Box::new(hook))));
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

        let mut graph = RenderGraph::new(
            self.ctx.graph_storage,
            self.ctx.frame_arena,
            crate::renderer::graph::core::graph::FrameConfig {
                width,
                height,
                depth_format: self.ctx.wgpu_ctx.depth_format,
                msaa_samples: self.ctx.wgpu_ctx.msaa_samples,
                surface_format: view_format,
                hdr_format: crate::renderer::HDR_TEXTURE_FORMAT,
            },
            self.ctx.pipeline_cache,
        );

        // ── 2a. Register Resources ──────────────────────────────────────
        // Only the swapchain surface is truly external.
        // Scene colour and depth are transient — owned and aliased by the RDG.

        let is_high_fidelity = self.ctx.wgpu_ctx.render_path.supports_post_processing();
        let msaa_samples = self.ctx.wgpu_ctx.msaa_samples;
        let is_msaa = msaa_samples > 1;

        let surface_desc = TextureDesc::new_2d(
            width,
            height,
            view_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let surface_out = graph.register_resource("Surface_Out", surface_desc, true);

        // ── 2b. Scene Configuration ────────────────────────────────────

        let ssao_enabled = self.ctx.scene.ssao.enabled && is_high_fidelity;
        let needs_feature_id = is_high_fidelity
            && (self.ctx.scene.screen_space.enable_sss || self.ctx.scene.screen_space.enable_ssr);
        let needs_normal = ssao_enabled || needs_feature_id;
        let needs_skybox = self.ctx.scene.background.needs_skybox_pass();
        let ssss_enabled = self.ctx.scene.screen_space.enable_sss;
        let has_transmission = self.ctx.render_lists.use_transmission;
        let bloom_enabled = self.ctx.scene.bloom.enabled && is_high_fidelity;
        let fxaa_enabled = self.ctx.scene.fxaa.enabled && is_high_fidelity;

        // ── 2c. Wire Compute + Shadow Passes ───────────────────────────
        graph.with_group("Compute", |g| {
            if self.ctx.resource_manager.needs_brdf_compute {
                self.ctx.brdf_pass.add_to_graph(g);
            }

            if let Some(source) = self.ctx.resource_manager.pending_ibl_source.take() {
                self.ctx.ibl_pass.add_to_graph(g, source);
            }
        });

        let shadow_tex = if self.ctx.extracted_scene.has_shadow_casters() {
            graph.with_group("Shadow", |g| self.ctx.shadow_pass.add_to_graph(g))
        } else {
            None
        };

        // ── 2d. Wire Scene Rendering Passes (explicit data-flow) ──────
        //
        // Each pass's `add_to_graph` creates its own transient resources
        // internally and returns typed output structs.  The Composer
        // threads `TextureNodeId` values from producer to consumer —
        // no blackboard lookups remain for mutable resources.

        // Track scene_color / scene_depth for the GraphBlackboard (hooks).
        let mut bb_scene_color = surface_out;
        let mut bb_scene_depth = surface_out;

        let mut current_surface = surface_out;

        if is_high_fidelity {
            // ────────────────────────────────────────────────────────────
            // HighFidelity pipeline: separate passes, explicit wiring.
            // ────────────────────────────────────────────────────────────

            // ── Scene Rendering Group ──────────────────────────────────

            let (post_transparent_color, prepass_depth, _ssao_output) =
                graph.with_group("Scene", |g| {
                    // 1. Prepass
                    let prepass_out =
                        self.ctx
                            .prepass
                            .add_to_graph(g, needs_normal, needs_feature_id);

                    // 2. SSAO
                    let ssao_output = if ssao_enabled {
                        Some(
                            self.ctx.ssao_pass.add_to_graph(
                                g,
                                prepass_out.scene_depth,
                                prepass_out
                                    .scene_normals
                                    .expect("SSAO requires scene normals from Prepass"),
                            ),
                        )
                    } else {
                        None
                    };

                    // 3. Opaque
                    let opaque_has_prepass = !is_msaa;
                    let opaque_out = self.ctx.opaque_pass.add_to_graph(
                        g,
                        prepass_out.scene_depth,
                        opaque_has_prepass,
                        self.ctx.extracted_scene.background.clear_color(),
                        ssss_enabled,
                        ssao_output,
                        shadow_tex,
                    );

                    let mut active_color = opaque_out.active_color;
                    let active_depth = opaque_out.active_depth;
                    let mut scene_color_hdr = opaque_out.scene_color_hdr;

                    // 4. SSSS
                    if ssss_enabled {
                        scene_color_hdr = self.ctx.ssss_pass.add_to_graph(
                            g,
                            scene_color_hdr,
                            prepass_out.scene_depth,
                            prepass_out.scene_normals.unwrap(),
                            prepass_out.feature_id.unwrap(),
                            opaque_out.specular_mrt.unwrap(),
                        );

                        if is_msaa {
                            active_color = self.ctx.msaa_sync_pass.add_to_graph(g, scene_color_hdr);
                        } else {
                            active_color = scene_color_hdr;
                        }
                    }

                    // 5. Skybox
                    if needs_skybox {
                        active_color =
                            self.ctx
                                .skybox_pass
                                .add_to_graph(g, active_color, active_depth);
                    }

                    // 6. Transmission Copy
                    let transmission_tex = if has_transmission {
                        let tx_source = if is_msaa {
                            scene_color_hdr
                        } else {
                            active_color
                        };
                        Some(
                            self.ctx
                                .transmission_copy_pass
                                .add_to_graph(g, tx_source, true),
                        )
                    } else {
                        None
                    };

                    // 7. Transparent
                    let post_transparent_color = self.ctx.transparent_pass.add_to_graph(
                        g,
                        active_color,
                        active_depth,
                        transmission_tex,
                        ssao_output,
                        shadow_tex,
                    );

                    (post_transparent_color, prepass_out.scene_depth, ssao_output)
                });

            bb_scene_color = post_transparent_color;
            bb_scene_depth = prepass_depth;

            // ── Before-Post-Process Hooks ──────────────────────────────
            {
                let mut blackboard = GraphBlackboard {
                    scene_color: post_transparent_color,
                    scene_depth: prepass_depth,
                    surface_out,
                };
                for (stage, hook_opt) in &mut self.hooks {
                    if *stage == HookStage::BeforePostProcess {
                        if let Some(hook) = hook_opt.take() {
                            blackboard = hook(&mut graph, blackboard);
                        }
                    }
                }
            }

            // ── Post-Processing Group ──────────────────────────────────

            current_surface = graph.with_group("PostProcess", |g| {
                // Bloom (internally flattened into Bloom_System subgroup)
                let tonemap_input = if bloom_enabled {
                    self.ctx.bloom_pass.add_to_graph(
                        g,
                        post_transparent_color,
                        self.ctx.scene.bloom.karis_average,
                        self.ctx.scene.bloom.max_mip_levels(),
                    )
                } else {
                    post_transparent_color
                };

                // ToneMapping: HDR → LDR
                let mut surface = if fxaa_enabled {
                    // Route through an intermediate LDR texture for FXAA input
                    let ldr = g.register_resource("LDR_Intermediate", surface_desc, false);
                    self.ctx.tone_map_pass.add_to_graph(g, tonemap_input, ldr)
                } else {
                    self.ctx
                        .tone_map_pass
                        .add_to_graph(g, tonemap_input, current_surface)
                };

                // FXAA: anti-alias the LDR result onto the surface
                if fxaa_enabled {
                    let ldr_intermediate = surface;
                    surface = self
                        .ctx
                        .fxaa_pass
                        .add_to_graph(g, ldr_intermediate, current_surface);
                }

                surface
            });
        } else {
            // BasicForward pipeline: single-pass LDR rendering.

            let prepared_skybox = if needs_skybox {
                let skybox_pipeline = self.ctx.skybox_pass.current_pipeline;
                let skybox_bind_group = &self.ctx.skybox_pass.current_bind_group;

                if let (Some(pipeline_id), Some(bg)) = (skybox_pipeline, skybox_bind_group) {
                    Some(PreparedSkyboxDraw {
                        pipeline: self.ctx.pipeline_cache.get_render_pipeline(pipeline_id),
                        bind_group: bg,
                    })
                } else {
                    None
                }
            } else {
                None
            };

            graph.with_group("BasicForward", |g| {
                self.ctx.simple_forward_pass.add_to_graph(
                    g,
                    surface_out,
                    self.ctx.extracted_scene.background.clear_color(),
                    prepared_skybox,
                    shadow_tex,
                );
            });
        }

        // ── After-Post-Process Hooks (UI, debug overlays) ──────────────
        {
            let mut blackboard = GraphBlackboard {
                scene_color: bb_scene_color,
                scene_depth: bb_scene_depth,
                surface_out: current_surface,
            };
            for (stage, hook_opt) in &mut self.hooks {
                if *stage == HookStage::AfterPostProcess {
                    if let Some(hook) = hook_opt.take() {
                        blackboard = hook(&mut graph, blackboard);
                    }
                }
            }
        }

        // ━━━ 3. Compile & Execute RDG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        graph.compile(self.ctx.transient_pool, &self.ctx.wgpu_ctx.device);

        // ─── 3a. RDG Prepare: transient-only BindGroup assembly ────────
        //
        // Only the swapchain surface is truly external — all other textures
        // (scene_color, scene_depth, etc.) are RDG transient resources.
        self.external_res.clear();

        let mut prepare_ctx = PrepareContext {
            views: ViewResolver {
                resources: &graph.storage.resources,
                pool: self.ctx.transient_pool,
                external_resources: &self.external_res,
            },
            device: &self.ctx.wgpu_ctx.device,
            queue: &self.ctx.wgpu_ctx.queue,
            sampler_registry: self.ctx.sampler_registry,
            global_bind_group_cache: self.ctx.global_bind_group_cache,
        };

        for &pass_idx in &graph.storage.execution_queue {
            let pass = graph.storage.passes[pass_idx].get_pass_mut();
            pass.prepare(&mut prepare_ctx);
        }

        // ─── 3c. Bake render commands ──────────────────────────────────
        //
        // Resolve every asset handle (geometry, material, pipeline) to its
        // physical wgpu reference.  After this point the execute phase is
        // "blind" — it processes only pre-resolved GPU state.
        let prepass_config = if is_high_fidelity {
            Some(crate::renderer::graph::bake::PrepassBakeConfig {
                local_cache: self.ctx.prepass.local_cache(),
                needs_normal: self.ctx.prepass.needs_normal(),
                needs_feature_id: self.ctx.prepass.needs_feature_id(),
            })
        } else {
            None
        };

        let baked_lists = crate::renderer::graph::bake::bake_render_lists(
            self.ctx.render_lists,
            self.ctx.resource_manager,
            self.ctx.pipeline_cache,
            &prepass_config,
        );

        // ─── 3d. Execute ───────────────────────────────────────────────
        let mut ext_views: FxHashMap<_, &wgpu::TextureView> = FxHashMap::default();
        ext_views.insert(surface_out, &surface_view);

        let mut execute_ctx = ExecuteContext {
            resources: &graph.storage.resources,
            pool: self.ctx.transient_pool,
            device: &self.ctx.wgpu_ctx.device,
            queue: &self.ctx.wgpu_ctx.queue,
            pipeline_cache: self.ctx.pipeline_cache,
            global_bind_group_cache: self.ctx.global_bind_group_cache,
            external_views: &ext_views,
            mipmap_generator: &self.ctx.resource_manager.mipmap_generator,
            baked_lists: &baked_lists,
            wgpu_ctx: &*self.ctx.wgpu_ctx,
            current_timeline_index: 0,
        };

        let mut encoder =
            self.ctx
                .wgpu_ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Unified Encoder"),
                });

        for (timeline_index, &pass_idx) in graph.storage.execution_queue.iter().enumerate() {
            execute_ctx.current_timeline_index = timeline_index;
            #[cfg(debug_assertions)]
            encoder.push_debug_group(graph.storage.passes[pass_idx].name);
            graph.storage.passes[pass_idx]
                .get_pass_mut()
                .execute(&execute_ctx, &mut encoder);
            #[cfg(debug_assertions)]
            encoder.pop_debug_group();
        }

        // ━━━ 4. Submit & Present ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        self.ctx.wgpu_ctx.queue.submit(Some(encoder.finish()));
        output.present();
    }
}
