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
//!             let out = builder.mutate_texture(bb.surface_out, "Surface_With_UI");
//!             (ui_node, out)
//!         });
//!         GraphBlackboard { surface_out: new_surface, ..bb }
//!     })
//!     .render();
//! ```

use crate::core::binding::GlobalBindGroupCache;
use crate::core::gpu::Tracked;
use crate::core::{ResourceManager, WgpuContext};
use crate::graph::ExtractedScene;
use crate::graph::RenderState;
use crate::graph::core::GraphStorage;
use crate::graph::core::graph::FrameConfig;
use crate::graph::core::{
    ExecuteContext, FrameArena, GraphBlackboard, HookStage, PrepareContext, RenderGraph,
    TextureDesc, TransientPool, ViewResolver,
};
use crate::graph::frame::{PreparedSkyboxDraw, RenderLists};
use crate::graph::passes::utils::add_msaa_resolve_pass;
use crate::graph::passes::{
    BloomFeature, BrdfLutFeature, CasFeature, FxaaFeature, IblComputeFeature, MsaaSyncFeature,
    OpaqueFeature, PrepassFeature, ShadowFeature, SimpleForwardFeature, SkyboxFeature, SsaoFeature,
    SsssFeature, TaaFeature, ToneMappingFeature, TransmissionCopyFeature, TransparentFeature,
};
use crate::pipeline::PipelineCache;
use crate::pipeline::ShaderManager;
use crate::renderer::FrameTime;
use myth_assets::AssetServer;
use myth_scene::Scene;
use myth_scene::camera::RenderCamera;

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
    pub frame_time: FrameTime,

    pub graph_storage: &'a mut GraphStorage,
    pub transient_pool: &'a mut TransientPool,
    // pub sampler_registry: &'a mut SamplerRegistry,
    pub frame_arena: &'a FrameArena,

    // ─── RDG Features ────────────────────────────────────────────────────
    // Post-processing
    pub fxaa_pass: &'a mut FxaaFeature,
    pub taa_pass: &'a mut TaaFeature,
    pub cas_pass: &'a mut CasFeature,
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

    // Debug view (compile-time gated)
    #[cfg(feature = "debug_view")]
    pub debug_view_pass: &'a mut crate::graph::passes::DebugViewFeature,
}

pub struct GraphBuilderContext<'a, 'g> {
    pub graph: &'g mut RenderGraph<'a>,
    pub pipeline_cache: &'a PipelineCache,
    pub frame_config: &'g FrameConfig,
}

impl GraphBuilderContext<'_, '_> {
    #[cfg(feature = "rdg_inspector")]
    pub fn with_group<F, R>(&mut self, group_name: &'static str, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        self.graph.push_group(group_name);
        let result = f(self);
        self.graph.pop_group();
        result
    }

    /// Zero-cost fallback when the inspector is disabled.
    #[cfg(not(feature = "rdg_inspector"))]
    #[inline]
    pub fn with_group<F, R>(&mut self, _group_name: &'static str, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        f(self)
    }
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
    frame_config: FrameConfig,
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
    pub(crate) fn new(ctx: ComposerContext<'a>, size: (u32, u32)) -> Self {
        let frame_config = FrameConfig {
            width: size.0,
            height: size.1,
            depth_format: ctx.wgpu_ctx.depth_format,
            msaa_samples: ctx.wgpu_ctx.msaa_samples,
            surface_format: ctx.wgpu_ctx.surface_view_format,
            hdr_format: crate::HDR_TEXTURE_FORMAT,
        };

        Self {
            ctx,
            frame_config,
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
    ///             let out = builder.mutate_texture(bb.surface_out, "Surface_MyPass");
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

        let mut graph = RenderGraph::new(self.ctx.graph_storage, self.ctx.frame_arena);

        let surface_desc = TextureDesc::new_2d(
            width,
            height,
            view_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let surface_view_tracked = Tracked::with_id(surface_view, 0);

        let surface_out =
            graph.import_external_resource("Surface_View", surface_desc, &surface_view_tracked);

        let mut graph_ctx = GraphBuilderContext {
            graph: &mut graph,
            pipeline_cache: self.ctx.pipeline_cache,
            frame_config: &self.frame_config,
        };

        // ── 2a. Register Resources ──────────────────────────────────────
        // Only the swapchain surface is truly external.
        // Scene colour and depth are transient — owned and aliased by the RDG.

        let is_high_fidelity = self.ctx.wgpu_ctx.render_path.supports_post_processing();
        let msaa_samples = self.ctx.wgpu_ctx.msaa_samples;
        let is_msaa = msaa_samples > 1;

        // ── 2b. Scene Configuration ────────────────────────────────────

        let ssao_enabled = self.ctx.scene.ssao.enabled && is_high_fidelity;
        let needs_feature_id = is_high_fidelity
            && (self.ctx.scene.screen_space.enable_sss || self.ctx.scene.screen_space.enable_ssr);
        let needs_normal = ssao_enabled || needs_feature_id;
        let needs_skybox = self.ctx.scene.background.needs_skybox_pass();
        let ssss_enabled = self.ctx.scene.screen_space.enable_sss;
        let has_transmission = self.ctx.render_lists.use_transmission;
        let bloom_enabled = self.ctx.scene.bloom.enabled && is_high_fidelity;
        // let fxaa_enabled = self.ctx.wgpu_ctx.fxaa_enabled && is_high_fidelity;
        // let taa_enabled = self.ctx.wgpu_ctx.taa_enabled && is_high_fidelity;

        // ── 2c. Wire Compute + Shadow Passes ───────────────────────────
        graph_ctx.with_group("Compute", |c| {
            if self.ctx.resource_manager.needs_brdf_compute {
                self.ctx.brdf_pass.add_to_graph(c);
            }

            if let Some(source) = self.ctx.resource_manager.pending_ibl_source.take() {
                self.ctx.ibl_pass.add_to_graph(c, source);
            }
        });

        let shadow_tex = if self.ctx.extracted_scene.has_shadow_casters() {
            graph_ctx.with_group("Shadow", |c| self.ctx.shadow_pass.add_to_graph(c))
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
        let mut bb_scene_color = None;
        let mut bb_scene_depth = None;

        // Debug view: capture intermediate texture IDs for safe resolution.
        #[cfg(feature = "debug_view")]
        let mut dbg_normals: Option<crate::graph::core::TextureNodeId> = None;
        #[cfg(feature = "debug_view")]
        let mut dbg_velocity: Option<crate::graph::core::TextureNodeId> = None;
        #[cfg(feature = "debug_view")]
        let mut dbg_ssao: Option<crate::graph::core::TextureNodeId> = None;
        #[cfg(feature = "debug_view")]
        let mut dbg_bloom_color: Option<crate::graph::core::TextureNodeId> = None;

        let mut current_surface = surface_out;

        if is_high_fidelity {
            // ────────────────────────────────────────────────────────────
            // HighFidelity pipeline: separate passes, explicit wiring.
            // ────────────────────────────────────────────────────────────

            // ── Scene Rendering Group ──────────────────────────────────

            let taa_enabled = self.ctx.camera.aa_mode.is_taa();

            let cas_enabled = if let Some(s) = self.ctx.camera.aa_mode.taa_settings() {
                s.sharpen_intensity > 0.0
            } else {
                false
            };

            let fxaa_enabled = self.ctx.camera.aa_mode.is_fxaa();

            let (mut active_color, mut scene_depth) = graph_ctx.with_group("Scene", |c| {
                // 1. Prepass
                let prepass_out =
                    self.ctx
                        .prepass
                        .add_to_graph(c, needs_normal, needs_feature_id, taa_enabled);

                let scene_depth = prepass_out.scene_depth;

                // 2. SSAO
                let ssao_output = if ssao_enabled {
                    Some(
                        self.ctx.ssao_pass.add_to_graph(
                            c,
                            scene_depth,
                            prepass_out
                                .scene_normals
                                .expect("SSAO requires scene normals from Prepass"),
                        ),
                    )
                } else {
                    None
                };

                // 3. Opaque
                let opaque_out = self.ctx.opaque_pass.add_to_graph(
                    c,
                    scene_depth,
                    self.ctx.extracted_scene.background.clear_color(),
                    ssss_enabled,
                    ssao_output,
                    shadow_tex,
                );

                let mut active_color = opaque_out.active_color;

                // 4. SSSS
                if ssss_enabled {
                    if is_msaa {
                        let hdr_desc = TextureDesc::new_2d(
                            c.frame_config.width,
                            c.frame_config.height,
                            crate::HDR_TEXTURE_FORMAT,
                            wgpu::TextureUsages::RENDER_ATTACHMENT
                                | wgpu::TextureUsages::TEXTURE_BINDING
                                | wgpu::TextureUsages::COPY_SRC,
                        );
                        // If MSAA is enabled, resolve the opaque output to an intermediate non-MSAA texture for SSSS input.
                        active_color = add_msaa_resolve_pass(c, active_color, hdr_desc);
                    }

                    active_color = self.ctx.ssss_pass.add_to_graph(
                        c,
                        active_color,
                        prepass_out.scene_depth,
                        prepass_out.scene_normals.unwrap(),
                        prepass_out.feature_id.unwrap(),
                        opaque_out.specular_mrt.unwrap(),
                    );

                    if is_msaa {
                        // If MSAA is enabled, synchronize the SSSS output back to an MSAA texture for downstream passes (Skybox, Transparent) that expect MSAA input.
                        // This avoids redundant MSAA resolve + re-multisample operations.
                        active_color = self.ctx.msaa_sync_pass.add_to_graph(c, active_color);
                    }
                }

                // 5. Skybox
                if needs_skybox {
                    active_color =
                        self.ctx
                            .skybox_pass
                            .add_to_graph(c, active_color, opaque_out.active_depth);
                }

                // ── 6. TAA Resolve ────────────────────────────────────────────
                // Resolve temporal anti-aliasing before bloom/tone-mapping.
                // The resolved colour replaces post_transparent_color for
                // downstream post-processing.
                if taa_enabled && let Some(velocity) = prepass_out.velocity_buffer {
                    c.with_group("TAA_System", |c| {
                        active_color =
                            self.ctx
                                .taa_pass
                                .add_to_graph(c, active_color, velocity, scene_depth);

                        // ── 6b. CAS (Contrast Adaptive Sharpening) ────────────
                        // Recover fine detail lost to temporal filtering.
                        if cas_enabled {
                            active_color = self.ctx.cas_pass.add_to_graph(c, active_color);
                        }
                    });
                }

                // 7. Transmission Copy
                let transmission_tex = if has_transmission {
                    Some(
                        self.ctx
                            .transmission_copy_pass
                            .add_to_graph(c, active_color),
                    )
                } else {
                    None
                };

                // 8. Transparent
                let active_color = self.ctx.transparent_pass.add_to_graph(
                    c,
                    active_color,
                    opaque_out.active_depth,
                    transmission_tex,
                    ssao_output,
                    shadow_tex,
                );

                // Capture intermediate IDs for debug view resolution.
                #[cfg(feature = "debug_view")]
                {
                    dbg_normals = prepass_out.scene_normals;
                    dbg_velocity = prepass_out.velocity_buffer;
                    dbg_ssao = ssao_output;
                }

                (active_color, scene_depth)
            });

            // ── Before-Post-Process Hooks ──────────────────────────────
            {
                let mut blackboard = GraphBlackboard {
                    scene_color: Some(active_color),
                    scene_depth: Some(scene_depth),
                    surface_out,
                };
                for (stage, hook_opt) in &mut self.hooks {
                    if *stage == HookStage::BeforePostProcess
                        && let Some(hook) = hook_opt.take()
                    {
                        blackboard = hook(graph_ctx.graph, blackboard);
                    }
                }

                active_color = blackboard.scene_color.unwrap_or(active_color);
                scene_depth = blackboard.scene_depth.unwrap_or(scene_depth);
            }

            // ── Post-Processing Group ──────────────────────────────────
            current_surface = graph_ctx.with_group("PostProcess", |ctx| {
                // Bloom (internally flattened into Bloom_System subgroup)
                if bloom_enabled {
                    active_color = self.ctx.bloom_pass.add_to_graph(
                        ctx,
                        active_color,
                        self.ctx.scene.bloom.karis_average,
                        self.ctx.scene.bloom.max_mip_levels(),
                    );
                }

                #[cfg(feature = "debug_view")]
                {
                    dbg_bloom_color = if bloom_enabled {
                        Some(active_color)
                    } else {
                        None
                    };
                }

                // ToneMapping: HDR → LDR
                let mut surface = if fxaa_enabled {
                    // Route through an intermediate LDR texture for FXAA input
                    let ldr = ctx
                        .graph
                        .register_resource("LDR_Intermediate", surface_desc, false);
                    self.ctx.tone_map_pass.add_to_graph(ctx, active_color, ldr)
                } else {
                    self.ctx
                        .tone_map_pass
                        .add_to_graph(ctx, active_color, current_surface)
                };

                // FXAA: anti-alias the LDR result onto the surface
                if fxaa_enabled {
                    let ldr_intermediate = surface;
                    surface =
                        self.ctx
                            .fxaa_pass
                            .add_to_graph(ctx, ldr_intermediate, current_surface);
                }

                bb_scene_color = Some(active_color);
                bb_scene_depth = Some(scene_depth);

                surface
            });

            // ── Debug View Override ────────────────────────────────────
            // Resolve the semantic DebugViewTarget to a concrete
            // TextureNodeId, then blit it onto the surface.  Targets
            // whose producer was disabled (e.g. SSAO off) safely
            // resolve to None — no pass is injected.
            #[cfg(feature = "debug_view")]
            {
                use crate::graph::render_state::DebugViewTarget;
                let target = self.ctx.render_state.debug_view_target;
                let source: Option<crate::graph::core::TextureNodeId> = match target {
                    DebugViewTarget::None => None,
                    // SceneDepth is Depth32Float — incompatible with float
                    // texture sampling.  A dedicated depth-copy-to-color
                    // pass will be added in a future iteration.
                    DebugViewTarget::SceneDepth => None,
                    DebugViewTarget::SceneNormal => dbg_normals,
                    DebugViewTarget::Velocity => dbg_velocity,
                    DebugViewTarget::SsaoRaw => dbg_ssao,
                    DebugViewTarget::BloomMip0 => dbg_bloom_color,
                };

                if let Some(src) = source {
                    current_surface =
                        self.ctx
                            .debug_view_pass
                            .add_to_graph(&mut graph_ctx, src, current_surface);
                }
            }
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

            graph_ctx.with_group("BasicForward", |c| {
                self.ctx.simple_forward_pass.add_to_graph(
                    c,
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
                if *stage == HookStage::AfterPostProcess
                    && let Some(hook) = hook_opt.take()
                {
                    blackboard = hook(&mut graph, blackboard);
                }
            }
        }

        // ━━━ 3. Compile & Execute RDG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        graph.compile(self.ctx.transient_pool, &self.ctx.wgpu_ctx.device);

        // ─── 3a. RDG Prepare: transient-only BindGroup assembly ────────
        //
        // Only the swapchain surface is truly external — all other textures
        // (scene_color, scene_depth, etc.) are RDG transient resources.

        let mut prepare_ctx = PrepareContext {
            views: ViewResolver {
                resources: &graph.storage.resources,
                pool: self.ctx.transient_pool,
            },
            device: &self.ctx.wgpu_ctx.device,
            queue: &self.ctx.wgpu_ctx.queue,
            sampler_registry: &self.ctx.resource_manager.sampler_registry,
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
            Some(crate::graph::bake::PrepassBakeConfig {
                local_cache: self.ctx.prepass.local_cache(),
                needs_normal: self.ctx.prepass.needs_normal(),
                needs_feature_id: self.ctx.prepass.needs_feature_id(),
                needs_velocity: self.ctx.prepass.needs_velocity(),
            })
        } else {
            None
        };

        let baked_lists = crate::graph::bake::bake_render_lists(
            self.ctx.render_lists,
            self.ctx.resource_manager,
            self.ctx.pipeline_cache,
            &prepass_config,
        );

        // ─── 3d. Execute ───────────────────────────────────────────────

        let mut execute_ctx = ExecuteContext {
            resources: &graph.storage.resources,
            pool: self.ctx.transient_pool,
            device: &self.ctx.wgpu_ctx.device,
            queue: &self.ctx.wgpu_ctx.queue,
            pipeline_cache: self.ctx.pipeline_cache,
            global_bind_group_cache: self.ctx.global_bind_group_cache,
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
