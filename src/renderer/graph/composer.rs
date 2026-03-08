//! Frame Composer
//!
//! `FrameComposer` orchestrates the entire rendering pipeline for a single
//! frame using the Declarative Render Graph (RDG). All GPU work — compute
//! pre-processing, shadow mapping, scene rendering, post-processing, and
//! custom user hooks — flows through a single unified RDG.
//!
//! # Resource Ownership
//!
//! The Composer registers only the shared **backbone** resources
//! (`Scene_Color_HDR`, `Scene_Depth`, `Surface_Out`) and the routing-level
//! `LDR_Intermediate`. All other transient resources are created by their
//! **producer passes** inside `PassNode::setup()` via
//! `builder.create_and_export()`, making each pass self-contained.
//!
//! # Rendering Architecture
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

use rustc_hash::FxHashMap;

use crate::assets::AssetServer;
use crate::render::RenderState;
use crate::renderer::core::binding::GlobalBindGroupCache;
use crate::renderer::core::resources::Tracked;
use crate::renderer::core::{ResourceManager, WgpuContext};
use crate::renderer::graph::ExtractedScene;
use crate::renderer::graph::frame::{PreparedSkyboxDraw, RenderLists};
use crate::renderer::graph::rdg::blackboard::{GraphBlackboard, HookStage};
use crate::renderer::graph::rdg::context::RdgViewResolver;
use crate::renderer::graph::rdg::graph::RenderGraph;
use crate::renderer::graph::rdg::types::TextureNodeId;
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

    // ─── RDG Features ────────────────────────────────────────────────────
    // Post-processing
    pub rdg_fxaa_pass: &'a mut crate::renderer::graph::rdg::fxaa::FxaaFeature,
    pub rdg_tone_map_pass: &'a mut crate::renderer::graph::rdg::tone_mapping::ToneMapFeature,
    pub rdg_bloom_pass: &'a mut crate::renderer::graph::rdg::bloom::BloomFeature,
    pub rdg_ssao_pass: &'a mut crate::renderer::graph::rdg::ssao::SsaoFeature,
    // Scene rendering
    pub rdg_prepass: &'a mut crate::renderer::graph::rdg::prepass::PrepassFeature,
    pub rdg_opaque_pass: &'a mut crate::renderer::graph::rdg::opaque::OpaqueFeature,
    pub rdg_skybox_pass: &'a mut crate::renderer::graph::rdg::skybox::SkyboxFeature,
    pub rdg_transparent_pass: &'a mut crate::renderer::graph::rdg::transparent::TransparentFeature,
    pub rdg_transmission_copy_pass:
        &'a mut crate::renderer::graph::rdg::transmission_copy::TransmissionCopyFeature,
    pub rdg_simple_forward_pass:
        &'a mut crate::renderer::graph::rdg::simple_forward::SimpleForwardFeature,
    pub rdg_sssss_pass: &'a mut crate::renderer::graph::rdg::sssss::SssssFeature,

    // Shadow + Compute
    pub rdg_shadow_pass: &'a mut crate::renderer::graph::rdg::shadow::ShadowFeature,
    pub rdg_brdf_pass: &'a mut crate::renderer::graph::rdg::compute::BrdfLutFeature,
    pub rdg_ibl_pass: &'a mut crate::renderer::graph::rdg::compute::IblComputeFeature,
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
    external_res: FxHashMap<TextureNodeId, &'a Tracked<wgpu::TextureView>>,
    hooks: smallvec::SmallVec<
        [(
            HookStage,
            Box<
                dyn FnMut(&mut RenderGraph, &mut GraphBlackboard)
                    + 'a,
            >,
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
    pub fn device(&self) -> &wgpu::Device {
        &self.ctx.wgpu_ctx.device
    }

    /// Returns a reference to the resource manager.
    ///
    /// Useful for user-land passes that need to resolve engine resources
    /// (e.g. texture handles) before the RDG prepare phase.
    #[inline]
    pub fn resource_manager(&self) -> &ResourceManager {
        self.ctx.resource_manager
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
        F: FnMut(&mut RenderGraph, &mut GraphBlackboard) + 'a,
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
        rdg.begin_frame(crate::renderer::graph::rdg::graph::FrameConfig {
            width,
            height,
            depth_format: self.ctx.wgpu_ctx.depth_format,
            msaa_samples: self.ctx.wgpu_ctx.msaa_samples,
            surface_format: view_format,
            hdr_format: HDR_TEXTURE_FORMAT,
        });

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
        let scene_color = rdg.register_resource("Scene_Color_HDR", hdr_desc, false);
        let scene_depth = rdg.register_resource("Scene_Depth", depth_desc, false);

        // ── 2b. Scene Configuration ────────────────────────────────────

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

        // ── 2c. Wire Compute + Shadow Passes ───────────────────────────

        self.ctx.rdg_brdf_pass.add_to_graph(rdg);
        self.ctx.rdg_ibl_pass.add_to_graph(rdg);
        self.ctx.rdg_shadow_pass.add_to_graph(rdg);

        // ── 2d. Wire Scene Rendering Passes ────────────────────────────

        // Build blackboard for hooks
        let mut blackboard = GraphBlackboard {
            scene_color,
            scene_depth,
            surface_out,
        };

        if is_high_fidelity {
            // HighFidelity pipeline: separate passes for opaque, skybox,
            // transparent with HDR targets and post-processing chain.

            // Prepass — creates Scene_Depth (write), Scene_Normals, Feature_ID
            self.ctx
                .rdg_prepass
                .add_to_graph(rdg, needs_normal, needs_feature_id);

            // SSAO
            if ssao_enabled {
                self.ctx.rdg_ssao_pass.add_to_graph(
                    rdg,
                );
            }

            // Opaque — writes Scene_Color_HDR, Scene_Depth; reads SSAO
            self.ctx.rdg_opaque_pass.add_to_graph(
                rdg,
                true, // has_prepass in HighFidelity
                self.ctx.extracted_scene.background.clear_color(),
                needs_specular,
            );

            // SSSSS — screen-space subsurface scattering
            if needs_specular {
                self.ctx.rdg_sssss_pass.add_to_graph(rdg);
            }

            // Skybox (conditional)
            if needs_skybox {
                self.ctx
                    .rdg_skybox_pass
                    .add_to_graph(rdg, scene_color, scene_depth);
            }

            // Transmission Copy — creates Transmission_Tex
            if has_transmission {
                self.ctx
                    .rdg_transmission_copy_pass
                    .add_to_graph(rdg, true);
            }

            // Transparent — reads color, depth, transmission, ssao via blackboard
            self.ctx.rdg_transparent_pass.add_to_graph(rdg);

            // ── Before-Post-Process Hooks ──────────────────────────────
            for (stage, hook) in &mut self.hooks {
                if *stage == HookStage::BeforePostProcess {
                    hook(rdg, &mut blackboard);
                }
            }

            // ── Post-Processing Chain ──────────────────────────────────

            let tonemap_input = if bloom_enabled {
                self.ctx.rdg_bloom_pass.add_to_graph(
                    rdg,
                    self.ctx.scene.bloom.karis_average,
                    self.ctx.scene.bloom.max_mip_levels(),
                )
            } else {
                scene_color
            };

            let tonemap_output = if fxaa_enabled {
                rdg.register_resource("LDR_Intermediate", surface_desc.clone(), false)
            } else {
                surface_out
            };

            // ToneMap
            self.ctx.rdg_tone_map_pass.add_to_graph(
                rdg,
                tonemap_input,
                tonemap_output,
            );

            // FXAA — reads LDR_Intermediate, writes Surface_Out
            if fxaa_enabled {
                self.ctx
                    .rdg_fxaa_pass
                    .add_to_graph(rdg, tonemap_output, surface_out);
            }
        } else {
            // BasicForward pipeline: single-pass LDR rendering.

            // Skybox prepare (inline rendering in SimpleForward)

            let prepared_skybox = if needs_skybox {
                let skybox_pipeline = self.ctx.rdg_skybox_pass.current_pipeline;
                let skybox_bind_group = &self.ctx.rdg_skybox_pass.current_bind_group;

                if let (Some(pipeline_id), Some(bg)) = (skybox_pipeline, skybox_bind_group) {
                    Some(PreparedSkyboxDraw {
                        pipeline_id,
                        bind_group: bg.clone(),
                    })
                } else {
                    None
                }
            } else {
                None
            };
    
            // SimpleForward — writes Surface_Out, Scene_Depth
            self.ctx.rdg_simple_forward_pass.add_to_graph(
                rdg,
                self.ctx.extracted_scene.background.clear_color(),
                prepared_skybox,
            );
        }

        // ── After-Post-Process Hooks (UI, debug overlays) ──────────────
        for (stage, hook) in &mut self.hooks {
            if *stage == HookStage::AfterPostProcess {
                hook(rdg, &mut blackboard);
            }
        }

        // ━━━ 3. Compile & Execute RDG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        rdg.compile(self.ctx.rdg_pool, &self.ctx.wgpu_ctx.device);

        // ─── 3a. Feature extract_and_prepare ────────────────────────────
        //
        // All Feature::extract_and_prepare() calls have already run in
        // `Renderer::begin_frame()`, before the graph was built.
        // This ensures persistent GPU resources (pipelines, layouts, bind
        // groups) are fully resolved before any PassNode references them.

        // ─── 3b. RDG Prepare: transient-only BindGroup assembly ────────
        //
        // Only the swapchain surface is truly external — all other textures
        // (scene_color, scene_depth, etc.) are RDG transient resources.
        self.external_res.clear();

        let mut rdg_prepare_ctx = RdgPrepareContext {
            views: RdgViewResolver {
                resources: &rdg.resources,
                pool: self.ctx.rdg_pool,
                external_resources: &self.external_res,
            },
            device: &self.ctx.wgpu_ctx.device,
            queue: &self.ctx.wgpu_ctx.queue,
            sampler_registry: self.ctx.sampler_registry,
            global_bind_group_cache: self.ctx.global_bind_group_cache,
        };

        for &pass_idx in &rdg.execution_queue {
            let pass = rdg.passes[pass_idx].get_pass_mut();
            pass.prepare(&mut rdg_prepare_ctx);
        }

        // ─── 3c. Bake render commands ──────────────────────────────────
        //
        // Resolve every asset handle (geometry, material, pipeline) to its
        // physical wgpu reference.  After this point the execute phase is
        // "blind" — it processes only pre-resolved GPU state.
        let prepass_config = if is_high_fidelity {
            Some(crate::renderer::graph::bake::PrepassBakeConfig {
                local_cache: self.ctx.rdg_prepass.local_cache(),
                needs_normal: self.ctx.rdg_prepass.needs_normal(),
                needs_feature_id: self.ctx.rdg_prepass.needs_feature_id(),
            })
        } else {
            None
        };

        let baked_lists = crate::renderer::graph::bake::bake_render_lists(
            self.ctx.render_lists,
            self.ctx.resource_manager,
            self.ctx.pipeline_cache,
            prepass_config,
        );

        // ─── 3d. Execute ───────────────────────────────────────────────
        let mut ext_views: FxHashMap<_, &wgpu::TextureView> = FxHashMap::default();
        ext_views.insert(surface_out, &surface_view);

        let rdg_execute_ctx = RdgExecuteContext {
            resources: &rdg.resources,
            pool: self.ctx.rdg_pool,
            device: &self.ctx.wgpu_ctx.device,
            queue: &self.ctx.wgpu_ctx.queue,
            pipeline_cache: self.ctx.pipeline_cache,
            global_bind_group_cache: self.ctx.global_bind_group_cache,
            external_views: &ext_views,
            mipmap_generator: &self.ctx.resource_manager.mipmap_generator,
            baked_lists: &baked_lists,
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

        // ━━━ 4. Submit & Present ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        self.ctx.wgpu_ctx.queue.submit(Some(rdg_encoder.finish()));
        output.present();
    }
}