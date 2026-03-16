//! Rendering System
//!
//! This module handles all GPU rendering operations using a layered architecture:
//!
//! - **[`core`]**: wgpu context wrapper (Device, Queue, Surface, `ResourceManager`)
//! - **[`graph`]**: Declarative Render Graph (RDG) frame organization
//! - **[`pipeline`]**: Shader compilation and pipeline caching (L1/L2 cache strategy)
//!
//! # Architecture Overview
//!
//! ```text
//! ┌───────────────────────────────────────────────┐
//! │                  FrameComposer                  │
//! │          (High-level rendering API)             │
//! ├───────────────────────────────────────────────┤
//! │  Declarative RDG  │     RenderFrame           │
//! │   (Pass execution)│  (Scene extraction)       │
//! ├───────────────────────────────────────────────┤
//! │   PipelineCache    │    ResourceManager        │
//! │  (Shader/PSO cache) │  (GPU resource lifecycle) │
//! ├───────────────────────────────────────────────┤
//! │                   WgpuContext                   │
//! │            (Device, Queue, Surface)             │
//! └───────────────────────────────────────────────┘
//! ```
//!
//! # Rendering Pipeline
//!
//! Each frame goes through these phases:
//!
//! 1. **Extract**: Scene data is extracted into GPU-friendly format
//! 2. **Prepare**: Resources are uploaded/updated on GPU
//! 3. **Queue**: Render commands are sorted and batched
//! 4. **Render**: RDG passes execute via topological order
//!
//! # Example
//!
//! ```rust,ignore
//! // Start a frame
//! if let Some(composer) = renderer.begin_frame(scene, camera, assets, time) {
//!     composer.render();
//! }
//! ```

pub mod core;
pub mod graph;
pub mod pipeline;
pub mod settings;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::assets::AssetServer;
use crate::errors::Result;
use crate::prelude::AntiAliasingMode;
use crate::renderer::core::binding::GlobalBindGroupCache;
use crate::renderer::core::gpu::SamplerRegistry;
use crate::renderer::graph::composer::ComposerContext;
use crate::renderer::graph::core::allocator::TransientPool;
use crate::renderer::graph::core::arena::FrameArena;
use crate::renderer::graph::core::graph::GraphStorage;
use crate::renderer::graph::frame::RenderLists;
#[cfg(feature = "debug_view")]
use crate::renderer::graph::passes::DebugViewFeature;
use crate::renderer::graph::passes::{
    BloomFeature, BrdfLutFeature, CasFeature, FxaaFeature, IblComputeFeature, MsaaSyncFeature,
    OpaqueFeature, PrepassFeature, ShadowFeature, SimpleForwardFeature, SkyboxFeature, SsaoFeature,
    SsssFeature, TaaFeature, ToneMappingFeature, TransmissionCopyFeature, TransparentFeature,
};
use crate::scene::Scene;
use crate::scene::camera::RenderCamera;

use self::core::{ResourceManager, WgpuContext};
use self::graph::{FrameComposer, RenderFrame};
use self::pipeline::PipelineCache;
use self::pipeline::ShaderManager;
use self::settings::{RenderPath, RendererSettings};

/// HDR texture format.
///
/// Format used for high dynamic range render targets and intermediate buffers.
pub const HDR_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The main renderer responsible for GPU rendering operations.
///
/// The renderer manages the complete rendering pipeline including:
/// - GPU context (device, queue, surface)
/// - Resource management (buffers, textures, bind groups)
/// - Pipeline caching (shader compilation, PSO creation)
/// - Frame rendering (scene extraction, command submission)
///
/// # Lifecycle
///
/// 1. Create with [`Renderer::new`] (no GPU resources allocated)
/// 2. Initialize GPU with [`Renderer::init`]
/// 3. Render frames with [`Renderer::begin_frame`]
/// 4. Clean up with [`Renderer::maybe_prune`]
pub struct Renderer {
    size: (u32, u32),
    settings: RendererSettings,
    context: Option<RendererState>,
}

/// Internal renderer state
struct RendererState {
    wgpu_ctx: WgpuContext,
    resource_manager: ResourceManager,
    pipeline_cache: PipelineCache,
    shader_manager: ShaderManager,

    render_frame: RenderFrame,
    /// Render lists (separated from `render_frame` to avoid borrow conflicts)
    render_lists: RenderLists,
    // /// Frame blackboard (cross-pass transient data communication, cleared each frame)
    // blackboard: FrameBlackboard,
    global_bind_group_cache: GlobalBindGroupCache,

    // ===== RDG (Declarative Render Graph) =====
    pub(crate) graph_storage: GraphStorage,
    pub(crate) sampler_registry: SamplerRegistry,
    pub(crate) transient_pool: TransientPool,
    pub(crate) frame_arena: FrameArena,

    // Post-processing passes
    pub(crate) fxaa_pass: FxaaFeature,
    pub(crate) taa_pass: TaaFeature,
    pub(crate) cas_pass: CasFeature,
    pub(crate) tone_map_pass: ToneMappingFeature,
    pub(crate) bloom_pass: BloomFeature,
    pub(crate) ssao_pass: SsaoFeature,

    // Scene rendering passes
    pub(crate) prepass: PrepassFeature,
    pub(crate) opaque_pass: OpaqueFeature,
    pub(crate) skybox_pass: SkyboxFeature,
    pub(crate) transparent_pass: TransparentFeature,
    pub(crate) transmission_copy_pass: TransmissionCopyFeature,
    pub(crate) simple_forward_pass: SimpleForwardFeature,
    pub(crate) ssss_pass: SsssFeature,
    pub(crate) msaa_sync_pass: MsaaSyncFeature,

    // Shadow + Compute passes (migrated from old system)
    pub(crate) shadow_pass: ShadowFeature,
    pub(crate) brdf_pass: BrdfLutFeature,
    pub(crate) ibl_pass: IblComputeFeature,

    // Debug view (compile-time gated)
    #[cfg(feature = "debug_view")]
    pub(crate) debug_view_pass: DebugViewFeature,
}

impl Renderer {
    /// Phase 1: Create configuration (no GPU resources yet).
    ///
    /// This only stores the render settings. GPU resources are
    /// allocated when [`init`](Self::init) is called.
    #[must_use]
    pub fn new(settings: RendererSettings) -> Self {
        Self {
            settings,
            context: None,
            size: (0, 0),
        }
    }

    /// Returns the current surface size in pixels as `(width, height)`.
    #[inline]
    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Phase 2: Initialize GPU context with window handle.
    ///
    /// This method:
    /// 1. Creates the wgpu instance and adapter
    /// 2. Requests a device with required features/limits
    /// 3. Configures the surface for presentation
    /// 4. Initializes resource manager and pipeline cache
    pub async fn init<W>(&mut self, window: W, width: u32, height: u32) -> Result<()>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        if self.context.is_some() {
            return Ok(());
        }

        self.size = (width, height);

        // 1. Create WGPU context
        let wgpu_ctx = WgpuContext::new(window, &self.settings, width, height).await?;

        // 2. Initialize resource manager
        let resource_manager =
            ResourceManager::new(wgpu_ctx.device.clone(), wgpu_ctx.queue.clone());

        // 3. Create render frame manager
        let render_frame = RenderFrame::new();

        // 5. Create global bind group cache
        let global_bind_group_cache = GlobalBindGroupCache::new();

        let sampler_registry = SamplerRegistry::new(&wgpu_ctx.device);

        // Shadow + compute passes (need device ref before wgpu_ctx moves)
        let shadow_pass = ShadowFeature::new(&wgpu_ctx.device);
        let brdf_pass = BrdfLutFeature::new(&wgpu_ctx.device);
        let ibl_pass = IblComputeFeature::new(&wgpu_ctx.device);

        // 6. Assemble state
        self.context = Some(RendererState {
            wgpu_ctx,
            resource_manager,
            pipeline_cache: PipelineCache::new(),
            shader_manager: ShaderManager::new(),

            render_frame,
            render_lists: RenderLists::new(),
            global_bind_group_cache,

            // RDG
            graph_storage: GraphStorage::new(),
            sampler_registry,
            transient_pool: TransientPool::new(),
            frame_arena: FrameArena::new(),
            fxaa_pass: FxaaFeature::new(),
            taa_pass: TaaFeature::new(),
            cas_pass: CasFeature::new(),
            tone_map_pass: ToneMappingFeature::new(),
            bloom_pass: BloomFeature::new(),
            ssao_pass: SsaoFeature::new(),

            // RDG Scene Passes
            prepass: PrepassFeature::new(),
            opaque_pass: OpaqueFeature::new(),
            skybox_pass: SkyboxFeature::new(),
            transparent_pass: TransparentFeature::new(),
            transmission_copy_pass: TransmissionCopyFeature::new(),
            simple_forward_pass: SimpleForwardFeature::new(),
            ssss_pass: SsssFeature::new(),
            msaa_sync_pass: MsaaSyncFeature::new(),

            // Shadow + Compute passes (migrated from old system)
            shadow_pass,
            brdf_pass,
            ibl_pass,

            #[cfg(feature = "debug_view")]
            debug_view_pass: DebugViewFeature::new(),
        });

        // Propagate screen bind group info to features that need it.
        if let Some(ref mut state) = self.context {
            let screen_info =
                crate::renderer::core::gpu::ScreenBindGroupInfo::from_resource_manager(
                    &state.resource_manager,
                );
            state.opaque_pass.set_screen_info(screen_info.clone());
            state.transparent_pass.set_screen_info(screen_info.clone());
            state.simple_forward_pass.set_screen_info(screen_info);
        }

        log::info!("Renderer Initialized");
        Ok(())
    }

    pub fn resize(&mut self, width: u32, height: u32, _scale_factor: f32) {
        self.size = (width, height);
        if let Some(state) = &mut self.context {
            state.wgpu_ctx.resize(width, height);
            // Invalidate all cached bind groups — texture views are now stale.
            state.global_bind_group_cache.clear();
        }
    }

    /// Begins building a new frame for rendering.
    ///
    /// Returns a [`FrameComposer`] that provides a chainable API for
    /// configuring the render pipeline via custom pass hooks.
    ///
    /// # Usage
    ///
    /// ```rust,ignore
    /// // Method 1: Use default built-in passes
    /// if let Some(composer) = renderer.begin_frame(scene, camera, assets, time) {
    ///     composer.render();
    /// }
    ///
    /// // Method 2: With custom hooks (e.g., UI overlay)
    /// if let Some(composer) = renderer.begin_frame(scene, camera, assets, time) {
    ///     composer
    ///         .add_custom_pass(HookStage::AfterPostProcess, |graph, bb| {
    ///             ui_pass.target_tex = bb.surface_out;
    ///             graph.add_pass(&mut ui_pass);
    ///         })
    ///         .render();
    /// }
    /// ```
    ///
    /// # Returns
    ///
    /// Returns `Some(FrameComposer)` if frame preparation succeeds,
    /// or `None` if rendering should be skipped (e.g., window size is 0).
    pub fn begin_frame<'a>(
        &'a mut self,
        scene: &'a mut Scene,
        camera: &'a RenderCamera,
        assets: &'a AssetServer,
        time: f32,
    ) -> Option<FrameComposer<'a>> {
        if self.size.0 == 0 || self.size.1 == 0 {
            return None;
        }

        let state = self.context.as_mut()?;

        // ── Frame Arena Lifecycle ───────────────────────────────────────
        // Reset the arena in O(1) — all previous PassNodes are trivially
        // forgotten (no Drop needed).
        state.frame_arena.reset();

        // Advance the bind-group cache's frame counter for TTL tracking.
        state.global_bind_group_cache.begin_frame();

        // ── Phase 1: Extract scene, build shadow views, prepare global ──
        let surface_size = state.wgpu_ctx.size();
        state.render_frame.extract_and_prepare(
            &mut state.resource_manager,
            scene,
            camera,
            assets,
            time,
            &mut state.render_lists,
            surface_size,
        );

        let requested_msaa = camera.aa_mode.msaa_sample_count();
        if state.wgpu_ctx.msaa_samples != requested_msaa {
            state.wgpu_ctx.msaa_samples = requested_msaa;
            state.wgpu_ctx.pipeline_settings_version += 1;
        }

        // ── Phase 2: Cull + sort + command generation ───────────────────
        crate::renderer::graph::culling::cull_and_sort(
            &state.render_frame.extracted_scene,
            &state.render_frame.render_state,
            &state.wgpu_ctx,
            &mut state.resource_manager,
            &mut state.pipeline_cache,
            &mut state.shader_manager,
            &mut state.render_lists,
            camera,
            assets,
        );

        // ── Phase 2.5: Feature extract & prepare ────────────────────────
        //
        // Resolve persistent GPU resources (pipelines, layouts, bind groups)
        // BEFORE the render graph is built. This ensures all Features are
        // fully prepared when their ephemeral PassNodes are created.
        {
            use crate::renderer::HDR_TEXTURE_FORMAT;
            use crate::renderer::graph::core::context::ExtractContext;

            let view_format = state.wgpu_ctx.surface_view_format;
            let is_hf = state.wgpu_ctx.render_path.supports_post_processing();
            let scene_id_val = scene.id();
            let render_state_id = state.render_frame.render_state.id;
            let global_state_key = (render_state_id, scene_id_val);

            let ssao_enabled = scene.ssao.enabled && is_hf;
            let needs_feature_id =
                is_hf && (scene.screen_space.enable_sss || scene.screen_space.enable_ssr);
            let needs_normal = ssao_enabled || needs_feature_id;
            let needs_skybox = scene.background.needs_skybox_pass();
            let bloom_enabled = scene.bloom.enabled && is_hf;

            let mut extract_ctx = ExtractContext {
                device: &state.wgpu_ctx.device,
                queue: &state.wgpu_ctx.queue,
                pipeline_cache: &mut state.pipeline_cache,
                shader_manager: &mut state.shader_manager,
                sampler_registry: &mut state.sampler_registry,
                global_bind_group_cache: &mut state.global_bind_group_cache,
                resource_manager: &mut state.resource_manager,
                wgpu_ctx: &state.wgpu_ctx,
                render_lists: &mut state.render_lists,
                extracted_scene: &state.render_frame.extracted_scene,
                render_state: &state.render_frame.render_state,
                assets,
            };

            // Always: compute + shadow
            state.brdf_pass.extract_and_prepare(&mut extract_ctx);
            state.ibl_pass.extract_and_prepare(&mut extract_ctx);
            state.shadow_pass.extract_and_prepare(&mut extract_ctx);

            // Skybox (both pipelines)
            if needs_skybox {
                let color_format = if is_hf {
                    HDR_TEXTURE_FORMAT
                } else {
                    view_format
                };
                state.skybox_pass.extract_and_prepare(
                    &mut extract_ctx,
                    &scene.background.mode,
                    &scene.background.uniforms,
                    global_state_key,
                    color_format,
                );
            }

            if is_hf {
                match &camera.aa_mode {
                    AntiAliasingMode::TAA(settings) => {
                        state.taa_pass.extract_and_prepare(
                            &mut extract_ctx,
                            settings.feedback_weight,
                            self.size,
                            HDR_TEXTURE_FORMAT,
                        );
                        if settings.sharpen_intensity > 0.0 {
                            state.cas_pass.extract_and_prepare(
                                &mut extract_ctx,
                                settings.sharpen_intensity,
                                HDR_TEXTURE_FORMAT,
                            );
                        }
                    }
                    AntiAliasingMode::FXAA(settings) | AntiAliasingMode::MSAA_FXAA(_, settings) => {
                        state.fxaa_pass.target_quality = settings.quality();
                        state
                            .fxaa_pass
                            .extract_and_prepare(&mut extract_ctx, view_format);
                    }
                    _ => {}
                }

                state.prepass.extract_and_prepare(
                    &mut extract_ctx,
                    needs_normal,
                    needs_feature_id,
                    matches!(camera.aa_mode, AntiAliasingMode::TAA(..)),
                );

                if ssao_enabled {
                    state
                        .ssao_pass
                        .extract_and_prepare(&mut extract_ctx, &scene.ssao.uniforms);
                }

                state.ssss_pass.extract_and_prepare(&mut extract_ctx);

                // MSAA Sync — needed when SSSS modifies the resolved HDR
                // buffer and subsequent passes re-enter the MSAA context.
                let msaa = state.wgpu_ctx.msaa_samples;
                let needs_specular = scene.screen_space.enable_sss;
                if msaa > 1 && needs_specular {
                    state
                        .msaa_sync_pass
                        .extract_and_prepare(&mut extract_ctx, msaa);
                }

                if bloom_enabled {
                    state.bloom_pass.extract_and_prepare(
                        &mut extract_ctx,
                        &scene.bloom.upsample_uniforms,
                        &scene.bloom.composite_uniforms,
                    );
                }

                state.tone_map_pass.extract_and_prepare(
                    &mut extract_ctx,
                    scene.tone_mapping.mode,
                    view_format,
                    global_state_key,
                    &scene.tone_mapping.uniforms,
                    scene.tone_mapping.lut_texture,
                );

                // Debug View — prepare pipeline & uniforms when active
                #[cfg(feature = "debug_view")]
                {
                    use crate::renderer::graph::render_state::DebugViewTarget;
                    let target = state.render_frame.render_state.debug_view_target;
                    if target != DebugViewTarget::None {
                        state.debug_view_pass.extract_and_prepare(
                            &mut extract_ctx,
                            view_format,
                            target.view_mode(),
                        );
                    }
                }
            }
        }

        // ── Phase 3: Build ComposerContext ──────────────────────────────
        let ctx = ComposerContext {
            wgpu_ctx: &mut state.wgpu_ctx,
            resource_manager: &mut state.resource_manager,
            pipeline_cache: &mut state.pipeline_cache,
            shader_manager: &mut state.shader_manager,

            extracted_scene: &state.render_frame.extracted_scene,
            render_state: &state.render_frame.render_state,

            global_bind_group_cache: &mut state.global_bind_group_cache,

            render_lists: &mut state.render_lists,

            // blackboard: &mut state.blackboard,
            scene,
            camera,
            assets,
            time,

            graph_storage: &mut state.graph_storage,
            transient_pool: &mut state.transient_pool,
            sampler_registry: &mut state.sampler_registry,
            frame_arena: &state.frame_arena,
            fxaa_pass: &mut state.fxaa_pass,
            taa_pass: &mut state.taa_pass,
            cas_pass: &mut state.cas_pass,
            tone_map_pass: &mut state.tone_map_pass,
            bloom_pass: &mut state.bloom_pass,
            ssao_pass: &mut state.ssao_pass,

            prepass: &mut state.prepass,
            opaque_pass: &mut state.opaque_pass,
            skybox_pass: &mut state.skybox_pass,
            transparent_pass: &mut state.transparent_pass,
            transmission_copy_pass: &mut state.transmission_copy_pass,
            simple_forward_pass: &mut state.simple_forward_pass,
            ssss_pass: &mut state.ssss_pass,
            msaa_sync_pass: &mut state.msaa_sync_pass,

            shadow_pass: &mut state.shadow_pass,
            brdf_pass: &mut state.brdf_pass,
            ibl_pass: &mut state.ibl_pass,

            #[cfg(feature = "debug_view")]
            debug_view_pass: &mut state.debug_view_pass,
        };

        // Return FrameComposer, defer Surface acquisition to render() call
        Some(FrameComposer::new(ctx, self.size))
    }

    /// Performs periodic resource cleanup.
    ///
    /// Should be called after each frame to release unused GPU resources.
    /// Uses internal heuristics to avoid expensive cleanup every frame.
    pub fn maybe_prune(&mut self) {
        if let Some(state) = &mut self.context {
            state.render_frame.maybe_prune(&mut state.resource_manager);
            // Evict stale bind groups that haven't been touched recently.
            state.global_bind_group_cache.garbage_collect();
        }
    }

    // === Runtime Settings API ===

    /// Returns the current [`RenderPath`].
    #[inline]
    pub fn render_path(&self) -> &RenderPath {
        &self.settings.path
    }

    /// Switches the active render path at runtime.
    ///
    /// Changes the pipeline topology between [`RenderPath::BasicForward`]
    /// and [`RenderPath::HighFidelity`].  The AA mode is unaffected — use
    /// [`set_aa_mode`](Self::set_aa_mode) to change it independently.
    /// The change takes effect on the **next frame**.
    pub fn set_render_path(&mut self, path: RenderPath) {
        if self.settings.path != path {
            self.settings.path = path;
            if let Some(state) = &mut self.context {
                state.wgpu_ctx.render_path = path;
                state.wgpu_ctx.pipeline_settings_version += 1;
            }
        }
    }

    /// Returns a reference to the current renderer settings.
    #[inline]
    pub fn settings(&self) -> &RendererSettings {
        &self.settings
    }

    /// Sets the active debug view target.
    ///
    /// When set to anything other than `None`, the FrameComposer will
    /// replace the post-process output with a fullscreen visualisation
    /// of the selected intermediate texture (if available).
    #[cfg(feature = "debug_view")]
    pub fn set_debug_view_target(
        &mut self,
        target: crate::renderer::graph::render_state::DebugViewTarget,
    ) {
        if let Some(state) = &mut self.context {
            state.render_frame.render_state.debug_view_target = target;
        }
    }

    /// Returns the current debug view target.
    #[cfg(feature = "debug_view")]
    pub fn debug_view_target(&self) -> crate::renderer::graph::render_state::DebugViewTarget {
        self.context
            .as_ref()
            .map(|s| s.render_frame.render_state.debug_view_target)
            .unwrap_or_default()
    }

    // === Public Methods: For External Plugins (e.g., UI Pass) ===

    /// Returns a reference to the wgpu Device.
    ///
    /// Useful for external plugins to initialize GPU resources.
    pub fn device(&self) -> Option<&wgpu::Device> {
        self.context.as_ref().map(|s| &s.wgpu_ctx.device)
    }

    /// Returns a reference to the wgpu Queue.
    ///
    /// Useful for external plugins to submit commands.
    pub fn queue(&self) -> Option<&wgpu::Queue> {
        self.context.as_ref().map(|s| &s.wgpu_ctx.queue)
    }

    /// Returns the surface texture format.
    ///
    /// Useful for external plugins to configure render pipelines.
    pub fn surface_format(&self) -> Option<wgpu::TextureFormat> {
        self.context.as_ref().map(|s| s.wgpu_ctx.config.format)
    }

    /// Returns a reference to the `WgpuContext`.
    ///
    /// For external plugins that need access to low-level GPU resources.
    /// Only available after renderer initialization.
    pub fn wgpu_ctx(&self) -> Option<&WgpuContext> {
        self.context.as_ref().map(|s| &s.wgpu_ctx)
    }

    pub fn dump_graph_mermaid(&self) -> Option<String> {
        self.context
            .as_ref()
            .map(|s| s.graph_storage.dump_mermaid())
    }
}
