//! Rendering System
//!
//! The main [`Renderer`] struct orchestrating GPU rendering operations.

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::core::binding::GlobalBindGroupCache;
use crate::graph::composer::ComposerContext;
use crate::graph::core::allocator::TransientPool;
use crate::graph::core::arena::FrameArena;
use crate::graph::core::graph::GraphStorage;
use crate::graph::frame::RenderLists;
#[cfg(feature = "debug_view")]
use crate::graph::passes::DebugViewFeature;
#[cfg(feature = "3dgs")]
use crate::graph::passes::GaussianSplattingFeature;
use crate::graph::passes::{
    AtmosphereFeature, BloomFeature, BrdfLutFeature, CasFeature, EquirectToCubeFeature,
    FxaaFeature, IblComputeFeature, MsaaSyncFeature, OpaqueFeature, PrepassFeature, ShadowFeature,
    SimpleForwardFeature, SkyboxFeature, SsaoFeature, SsssFeature, TaaFeature, ToneMappingFeature,
    TransmissionCopyFeature, TransparentFeature,
};
use myth_assets::AssetServer;
use myth_core::Result;
use myth_scene::Scene;
use myth_scene::background::BackgroundMode;
use myth_scene::camera::RenderCamera;

use crate::core::{ResourceManager, WgpuContext};
use crate::graph::{FrameComposer, RenderFrame};
use crate::pipeline::PipelineCache;
use crate::pipeline::ShaderManager;
use crate::settings::{RenderPath, RendererInitConfig, RendererSettings};

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
    init_config: RendererInitConfig,
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
    // pub(crate) sampler_registry: SamplerRegistry,
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
    pub(crate) equirect_to_cube_pass: EquirectToCubeFeature,
    pub(crate) ibl_pass: IblComputeFeature,
    pub(crate) atmosphere_pass: AtmosphereFeature,

    #[cfg(feature = "3dgs")]
    // Gaussian Splatting
    pub(crate) gaussian_splatting_pass: GaussianSplattingFeature,

    // Debug view (compile-time gated)
    #[cfg(feature = "debug_view")]
    pub(crate) debug_view_pass: DebugViewFeature,

    /// Cached staging buffer for synchronous `readback_pixels()`.
    /// Re-used across calls when the required size has not changed.
    cached_readback_buffer: Option<wgpu::Buffer>,
    /// Size (in bytes) of the cached readback buffer.
    cached_readback_buffer_size: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FrameTime {
    pub time: f32,
    pub delta_time: f32,
    pub frame_count: u64,
}

impl Renderer {
    /// Phase 1: Create configuration (no GPU resources yet).
    ///
    /// This only stores the render settings. GPU resources are
    /// allocated when [`init`](Self::init) is called.
    ///
    /// # Arguments
    ///
    /// * `init_config` - Static GPU/device configuration (consumed at init time)
    /// * `settings` - Runtime rendering settings (can be changed later via [`update_settings`](Self::update_settings))
    #[must_use]
    pub fn new(init_config: RendererInitConfig, settings: RendererSettings) -> Self {
        Self {
            init_config,
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

        let wgpu_ctx =
            WgpuContext::new(window, &self.init_config, &self.settings, width, height).await?;

        self.assemble_state(wgpu_ctx);
        log::info!("Renderer initialized (windowed)");
        Ok(())
    }

    /// Phase 2 (headless): Initialize GPU context without a window.
    ///
    /// Creates an offscreen render target of the specified dimensions. No
    /// window surface is created, making this suitable for server-side
    /// rendering, automated testing, and GPU readback workflows.
    ///
    /// # Arguments
    ///
    /// * `width` — Render target width in pixels.
    /// * `height` — Render target height in pixels.
    /// * `format` — Desired pixel format. Pass `None` for the default
    ///   `Rgba8Unorm` (sRGB). Use `Some(Rgba16Float)` for HDR readback.
    pub async fn init_headless(
        &mut self,
        width: u32,
        height: u32,
        format: Option<myth_resources::PixelFormat>,
    ) -> Result<()> {
        if self.context.is_some() {
            return Ok(());
        }

        self.size = (width, height);

        let wgpu_format = format.map(|f| f.to_wgpu(myth_resources::ColorSpace::Srgb));

        let wgpu_ctx = WgpuContext::new_headless(
            &self.init_config,
            &self.settings,
            width,
            height,
            wgpu_format,
        )
        .await?;

        self.assemble_state(wgpu_ctx);
        log::info!("Renderer initialized (headless {width}×{height})");
        Ok(())
    }

    /// Assembles the internal renderer state from a fully initialised GPU context.
    fn assemble_state(&mut self, wgpu_ctx: WgpuContext) {
        let resource_manager = ResourceManager::new(
            wgpu_ctx.device.clone(),
            wgpu_ctx.queue.clone(),
            self.settings.anisotropy_clamp,
        );

        let render_frame = RenderFrame::new();
        let global_bind_group_cache = GlobalBindGroupCache::new();

        let shadow_pass = ShadowFeature::new(&wgpu_ctx.device);
        let brdf_pass = BrdfLutFeature::new(&wgpu_ctx.device);
        let equirect_to_cube_pass = EquirectToCubeFeature::new(&wgpu_ctx.device);
        let ibl_pass = IblComputeFeature::new(&wgpu_ctx.device);

        self.context = Some(RendererState {
            wgpu_ctx,
            resource_manager,
            pipeline_cache: PipelineCache::new(),
            shader_manager: ShaderManager::new(),

            render_frame,
            render_lists: RenderLists::new(),
            global_bind_group_cache,

            graph_storage: GraphStorage::new(),
            transient_pool: TransientPool::new(),
            frame_arena: FrameArena::new(),
            fxaa_pass: FxaaFeature::new(),
            taa_pass: TaaFeature::new(),
            cas_pass: CasFeature::new(),
            tone_map_pass: ToneMappingFeature::new(),
            bloom_pass: BloomFeature::new(),
            ssao_pass: SsaoFeature::new(),

            prepass: PrepassFeature::new(),
            opaque_pass: OpaqueFeature::new(),
            skybox_pass: SkyboxFeature::new(),
            transparent_pass: TransparentFeature::new(),
            transmission_copy_pass: TransmissionCopyFeature::new(),
            simple_forward_pass: SimpleForwardFeature::new(),
            ssss_pass: SsssFeature::new(),
            msaa_sync_pass: MsaaSyncFeature::new(),

            shadow_pass,
            brdf_pass,
            equirect_to_cube_pass,
            ibl_pass,
            atmosphere_pass: AtmosphereFeature::new(),

            #[cfg(feature = "3dgs")]
            gaussian_splatting_pass: GaussianSplattingFeature::new(),

            #[cfg(feature = "debug_view")]
            debug_view_pass: DebugViewFeature::new(),

            cached_readback_buffer: None,
            cached_readback_buffer_size: 0,
        });
    }

    pub fn resize(&mut self, width: u32, height: u32) {
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
        camera: RenderCamera,
        assets: &'a AssetServer,
        frame_time: FrameTime,
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
            &camera,
            assets,
            frame_time,
            &mut state.render_lists,
            surface_size,
        );

        let requested_msaa = camera.aa_mode.msaa_sample_count();
        if state.wgpu_ctx.msaa_samples != requested_msaa {
            state.wgpu_ctx.msaa_samples = requested_msaa;
            state.wgpu_ctx.pipeline_settings_version += 1;
        }

        // ── Phase 2: Cull + sort + command generation ───────────────────
        crate::graph::culling::cull_and_sort(
            &state.render_frame.extracted_scene,
            &state.render_frame.render_state,
            &state.wgpu_ctx,
            &mut state.resource_manager,
            &mut state.pipeline_cache,
            &mut state.shader_manager,
            &mut state.render_lists,
            &camera,
            assets,
        );

        // ── Phase 2.5: Feature extract & prepare ────────────────────────
        //
        // Resolve persistent GPU resources (pipelines, layouts, bind groups)
        // BEFORE the render graph is built. This ensures all Features are
        // fully prepared when their ephemeral PassNodes are created.
        {
            use crate::HDR_TEXTURE_FORMAT;
            use crate::graph::core::context::ExtractContext;

            let view_format = state.wgpu_ctx.surface_view_format;
            let is_hf = state.wgpu_ctx.render_path.supports_post_processing();
            let scene_id_val = scene.id();
            let render_state_id = state.render_frame.render_state.id;
            let global_state_key = (render_state_id, scene_id_val);

            let ssao_enabled = scene.ssao.enabled && is_hf;
            let needs_feature_id =
                is_hf && (scene.screen_space.enable_sss || scene.screen_space.enable_ssr);

            // Sync camera debug settings → RenderState before borrowing it.
            #[cfg(feature = "debug_view")]
            {
                let dv = camera.debug_view;
                state.render_frame.render_state.debug_view_mode = dv.mode;
                state.render_frame.render_state.debug_view_scale = dv.custom_scale;
            }

            #[cfg(feature = "debug_view")]
            let (dbg_needs_normal, dbg_needs_velocity) = {
                use crate::graph::render_state::DebugViewTarget;
                let target =
                    DebugViewTarget::from_mode(state.render_frame.render_state.debug_view_mode);
                (
                    target == DebugViewTarget::SceneNormal,
                    target == DebugViewTarget::Velocity,
                )
            };

            #[cfg(not(feature = "debug_view"))]
            let (dbg_needs_normal, dbg_needs_velocity) = (false, false);

            let needs_normal = ssao_enabled || needs_feature_id || dbg_needs_normal;
            let needs_velocity = camera.aa_mode.is_taa() || dbg_needs_velocity;

            // let needs_normal = ssao_enabled || needs_feature_id;
            let needs_skybox = scene.background.needs_skybox_pass();
            let bloom_enabled = scene.bloom.enabled && is_hf;

            let mut extract_ctx = ExtractContext {
                device: &state.wgpu_ctx.device,
                queue: &state.wgpu_ctx.queue,
                pipeline_cache: &mut state.pipeline_cache,
                shader_manager: &mut state.shader_manager,
                global_bind_group_cache: &mut state.global_bind_group_cache,
                resource_manager: &mut state.resource_manager,
                wgpu_ctx: &state.wgpu_ctx,
                render_lists: &mut state.render_lists,
                extracted_scene: &state.render_frame.extracted_scene,
                render_state: &state.render_frame.render_state,
                render_camera: &camera,
                assets,
            };

            // Always: compute + shadow
            state.brdf_pass.extract_and_prepare(&mut extract_ctx);
            if scene.environment.has_env_map() {
                state
                    .equirect_to_cube_pass
                    .extract_and_prepare(&mut extract_ctx, scene);
            }
            state
                .ibl_pass
                .extract_and_prepare(&mut extract_ctx, scene.id());
            state.shadow_pass.extract_and_prepare(&mut extract_ctx);

            // Procedural atmosphere (LUT + cubemap + PMREM compute)
            let procedural_skybox_resources =
                if let BackgroundMode::Procedural(params) = &scene.background.mode {
                    state
                        .atmosphere_pass
                        .extract_and_prepare(&mut extract_ctx, scene.id(), params);
                    state
                        .atmosphere_pass
                        .procedural_skybox_resources(scene.id())
                } else {
                    None
                };

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
                    procedural_skybox_resources,
                );
            }

            #[cfg(feature = "3dgs")]
            // Gaussian Splatting
            if scene.has_gaussian_clouds() {
                let mut cloud_entries = Vec::new();
                for (node_handle, cloud_handle) in &scene.gaussian_clouds {
                    if let Some(cloud) = assets.gaussian_clouds.get(*cloud_handle) {
                        let world_matrix = scene
                            .get_node(node_handle)
                            .map(|node| glam::Mat4::from(*node.world_matrix()))
                            .unwrap_or(glam::Mat4::IDENTITY);
                        cloud_entries.push((*cloud_handle, cloud, world_matrix));
                    }
                }
                if !cloud_entries.is_empty() {
                    state
                        .gaussian_splatting_pass
                        .extract_and_prepare(&mut extract_ctx, &cloud_entries);
                }
            }

            if is_hf {
                if let Some(taa_settins) = camera.aa_mode.taa_settings() {
                    state.taa_pass.extract_and_prepare(
                        &mut extract_ctx,
                        taa_settins.feedback_weight,
                        self.size,
                        HDR_TEXTURE_FORMAT,
                    );

                    if taa_settins.sharpen_intensity > 0.0 {
                        state.cas_pass.extract_and_prepare(
                            &mut extract_ctx,
                            taa_settins.sharpen_intensity,
                            HDR_TEXTURE_FORMAT,
                        );
                    }
                }

                if let Some(fxaa_settings) = camera.aa_mode.fxaa_settings() {
                    state.fxaa_pass.target_quality = fxaa_settings.quality();
                    state
                        .fxaa_pass
                        .extract_and_prepare(&mut extract_ctx, view_format);
                }

                state.prepass.extract_and_prepare(
                    &mut extract_ctx,
                    needs_normal,
                    needs_feature_id,
                    needs_velocity,
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
                    use crate::graph::passes::debug_view::DebugViewUniforms;
                    use crate::graph::render_state::DebugViewTarget;

                    let dv = camera.debug_view;
                    let target = DebugViewTarget::from_mode(dv.mode);
                    if target != DebugViewTarget::None {
                        let params = DebugViewUniforms {
                            view_mode: target.view_mode(),
                            custom_scale: dv.custom_scale,
                            z_near: camera.near,
                            z_far: if camera.far.is_infinite() {
                                10000.0
                            } else {
                                camera.far
                            },
                        };
                        let is_depth = target == DebugViewTarget::SceneDepth;
                        state.debug_view_pass.extract_and_prepare(
                            &mut extract_ctx,
                            view_format,
                            params,
                            is_depth,
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
            frame_time,

            graph_storage: &mut state.graph_storage,
            transient_pool: &mut state.transient_pool,
            // sampler_registry: &mut state.sampler_registry,
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
            equirect_to_cube_pass: &mut state.equirect_to_cube_pass,
            ibl_pass: &mut state.ibl_pass,
            atmosphere_pass: &mut state.atmosphere_pass,

            #[cfg(feature = "3dgs")]
            gaussian_splatting_pass: &mut state.gaussian_splatting_pass,

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

    /// Returns a reference to the current runtime renderer settings.
    #[inline]
    pub fn settings(&self) -> &RendererSettings {
        &self.settings
    }

    /// Returns a reference to the init-time configuration.
    #[inline]
    pub fn init_config(&self) -> &RendererInitConfig {
        &self.init_config
    }

    /// Applies new runtime settings, performing an internal diff to update
    /// only the parts that actually changed.
    ///
    /// This is the **single entry point** for all runtime configuration
    /// changes. Callers (UI panels, scripting layers, etc.) should maintain
    /// their own [`RendererSettings`] instance, mutate it, and pass it here.
    pub fn update_settings(&mut self, new_settings: RendererSettings) {
        if self.settings == new_settings {
            return;
        }

        let old = std::mem::replace(&mut self.settings, new_settings);

        if let Some(state) = &mut self.context {
            // VSync
            if old.vsync != self.settings.vsync {
                state.wgpu_ctx.set_vsync(self.settings.vsync);
            }

            // Render path
            if old.path != self.settings.path {
                state.wgpu_ctx.render_path = self.settings.path;
                state.wgpu_ctx.pipeline_settings_version += 1;
                log::info!("RenderPath changed to {:?}", self.settings.path);
            }

            // Anisotropy
            if old.anisotropy_clamp != self.settings.anisotropy_clamp {
                state
                    .resource_manager
                    .sampler_registry
                    .set_global_anisotropy(self.settings.anisotropy_clamp);
                log::info!(
                    "Anisotropy clamp changed to {}",
                    self.settings.anisotropy_clamp
                );
            }
        }
    }

    /// Switches the active render path at runtime.
    ///
    /// Convenience wrapper around [`update_settings`](Self::update_settings)
    /// for changing only the render path.
    pub fn set_render_path(&mut self, path: RenderPath) {
        if self.settings.path != path {
            let mut new = self.settings.clone();
            new.path = path;
            self.update_settings(new);
        }
    }

    /// Sets the active debug view mode.
    ///
    /// When set to anything other than `None`, the FrameComposer will
    /// either replace the post-process output with a fullscreen
    /// visualisation of the selected intermediate texture (post-process
    /// modes), or inject shader defines to short-circuit PBR lighting
    /// and output raw material attributes (material-override modes).
    #[cfg(feature = "debug_view")]
    pub fn set_debug_view_mode(&mut self, mode: myth_scene::camera::DebugViewMode) {
        if let Some(state) = &mut self.context {
            state.render_frame.render_state.debug_view_mode = mode;
        }
    }

    /// Returns the current debug view mode.
    #[cfg(feature = "debug_view")]
    pub fn debug_view_mode(&self) -> myth_scene::camera::DebugViewMode {
        self.context
            .as_ref()
            .map(|s| s.render_frame.render_state.debug_view_mode)
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

    /// Returns the surface/render-target texture format.
    ///
    /// In windowed mode this is the swap-chain format; in headless mode it
    /// is the offscreen texture format. Returns `None` before initialisation.
    pub fn surface_format(&self) -> Option<wgpu::TextureFormat> {
        self.context
            .as_ref()
            .map(|s| s.wgpu_ctx.surface_view_format)
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

    // === Custom Shader Registration API ===

    /// Registers a custom WGSL shader template with the given name.
    ///
    /// The source string is pre-processed by the minijinja template engine at
    /// compile time, so `{$ include "chunks/camera_uniforms.wgsl" $}` and
    /// similar directives are fully supported.
    ///
    /// # Usage
    ///
    /// ```rust,ignore
    /// renderer.register_shader_template(
    ///     "custom_unlit",
    ///     include_str!("shaders/custom_unlit.wgsl"),
    /// );
    /// ```
    ///
    /// After registration, any material declared with
    /// `#[myth_material(shader = "custom_unlit")]` will use this template.
    ///
    /// # Panics
    ///
    /// Panics if the renderer has not been initialized via [`init`](Self::init).
    pub fn register_shader_template(&mut self, name: &str, source: &str) {
        let state = self
            .context
            .as_mut()
            .expect("Renderer must be initialized before registering shader templates");
        state.shader_manager.register_template(name, source);
    }

    /// Returns `true` if the renderer is in headless (offscreen) mode.
    #[inline]
    #[must_use]
    pub fn is_headless(&self) -> bool {
        self.context
            .as_ref()
            .is_some_and(|s| s.wgpu_ctx.is_headless())
    }

    /// Reads back the current headless render target as raw pixel data.
    ///
    /// The returned `Vec<u8>` contains tightly-packed pixel data whose per-pixel
    /// byte count matches the headless texture format (e.g. 4 bytes for RGBA8,
    /// 8 bytes for RGBA16Float). Row ordering is top-to-bottom.
    ///
    /// A staging buffer is cached internally and re-used across calls as long
    /// as the required size has not changed, eliminating per-frame allocation.
    ///
    /// This method submits a GPU copy command and blocks the calling thread
    /// until the transfer completes.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The renderer has not been initialised
    /// - No headless render target exists (windowed mode)
    /// - The GPU buffer mapping fails
    pub fn readback_pixels(&mut self) -> Result<Vec<u8>> {
        let state = self
            .context
            .as_mut()
            .ok_or(myth_core::RenderError::NotInitialized)?;

        let texture = state
            .wgpu_ctx
            .headless_texture
            .as_ref()
            .ok_or(myth_core::RenderError::NoHeadlessTarget)?;

        let width = state.wgpu_ctx.target_width;
        let height = state.wgpu_ctx.target_height;
        let format = state.wgpu_ctx.surface_view_format;

        let bytes_per_pixel = format.block_copy_size(None).ok_or_else(|| {
            myth_core::RenderError::ReadbackFailed(format!(
                "unsupported readback format: {format:?}"
            ))
        })?;

        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let buffer_size = u64::from(padded_bytes_per_row) * u64::from(height);

        // Re-use the cached buffer when the required capacity matches.
        if state.cached_readback_buffer_size != buffer_size {
            state.cached_readback_buffer = Some(state.wgpu_ctx.device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("Readback Buffer"),
                    size: buffer_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                },
            ));
            state.cached_readback_buffer_size = buffer_size;
        }

        let readback_buffer = state.cached_readback_buffer.as_ref().unwrap();

        let mut encoder =
            state
                .wgpu_ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Readback Encoder"),
                });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        state
            .wgpu_ctx
            .queue
            .submit(std::iter::once(encoder.finish()));

        // Block until the GPU finishes the copy and the buffer is mappable.
        let buffer_slice = readback_buffer.slice(..);

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        state
            .wgpu_ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| myth_core::RenderError::ReadbackFailed(e.to_string()))?;

        rx.recv()
            .map_err(|e| myth_core::RenderError::ReadbackFailed(e.to_string()))?
            .map_err(|e| myth_core::RenderError::ReadbackFailed(e.to_string()))?;

        // Strip per-row padding and produce a tightly-packed pixel buffer.
        let mapped = buffer_slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((width * height * bytes_per_pixel) as usize);
        for row in 0..height {
            let start = (row * padded_bytes_per_row) as usize;
            let end = start + unpadded_bytes_per_row as usize;
            pixels.extend_from_slice(&mapped[start..end]);
        }
        drop(mapped);
        readback_buffer.unmap();

        Ok(pixels)
    }

    /// Creates a [`ReadbackStream`] backed by the headless render target.
    ///
    /// The stream pre-allocates `buffer_count` staging buffers that rotate in
    /// a ring, enabling fully non-blocking GPU→CPU readback suitable for
    /// video recording and AI training-data pipelines.
    ///
    /// # Errors
    ///
    /// Returns an error if the renderer is not initialised or not in headless
    /// mode, or if the texture format does not support readback.
    pub fn create_readback_stream(
        &self,
        buffer_count: usize,
        max_stash_size: usize,
    ) -> Result<crate::core::ReadbackStream> {
        let state = self
            .context
            .as_ref()
            .ok_or(myth_core::RenderError::NotInitialized)?;

        if state.wgpu_ctx.headless_texture.is_none() {
            return Err(myth_core::RenderError::NoHeadlessTarget.into());
        }

        let width = state.wgpu_ctx.target_width;
        let height = state.wgpu_ctx.target_height;
        let format = state.wgpu_ctx.surface_view_format;

        let stream = crate::core::ReadbackStream::new(
            &state.wgpu_ctx.device,
            width,
            height,
            format,
            buffer_count,
            max_stash_size,
        )?;

        Ok(stream)
    }

    /// Drives pending GPU callbacks without blocking.
    ///
    /// Call this once per frame in a readback-stream loop so that `map_async`
    /// callbacks fire and frames become available via
    /// [`ReadbackStream::try_recv`].
    pub fn poll_device(&self) {
        if let Some(state) = &self.context {
            let _ = state.wgpu_ctx.device.poll(wgpu::PollType::Poll);
        }
    }

    /// Returns a reference to the headless render target texture, if present.
    #[must_use]
    pub fn headless_texture(&self) -> Option<&wgpu::Texture> {
        self.context
            .as_ref()
            .and_then(|s| s.wgpu_ctx.headless_texture.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use glam::{Affine3A, Quat, Vec3};
    use myth_assets::AssetServer;
    use myth_scene::Scene;
    use myth_scene::background::BackgroundMode;
    use myth_scene::camera::Camera;

    fn init_headless_renderer() -> Renderer {
        let mut renderer =
            Renderer::new(RendererInitConfig::default(), RendererSettings::default());
        pollster::block_on(renderer.init_headless(128, 128, None))
            .expect("headless renderer init failed");
        renderer
    }

    fn make_camera() -> RenderCamera {
        let mut camera = Camera::new_perspective(45.0, 1.0, 0.1);
        camera.update_view_projection(&Affine3A::from_translation(Vec3::new(0.0, 0.0, 5.0)));
        camera.extract_render_camera()
    }

    fn render_frame(
        renderer: &mut Renderer,
        scene: &mut Scene,
        camera: RenderCamera,
        assets: &AssetServer,
        frame_index: u64,
    ) {
        renderer
            .begin_frame(
                scene,
                camera,
                assets,
                FrameTime {
                    time: frame_index as f32 / 60.0,
                    delta_time: 1.0 / 60.0,
                    frame_count: frame_index,
                },
            )
            .expect("frame composer must exist")
            .render();
    }

    #[test]
    fn procedural_environment_updates_keep_global_bind_group_stable() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        scene.background.set_mode(BackgroundMode::procedural());
        let camera = make_camera();

        render_frame(&mut renderer, &mut scene, camera, &assets, 0);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let scene_id = scene.id();
        let render_state_id = state.render_frame.render_state.id;
        let global_state = state
            .resource_manager
            .get_global_state(render_state_id, scene_id)
            .expect("global state missing after first render");
        let first_bind_group_id = global_state.bind_group_id;
        let first_env = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("scene gpu environment missing after first render");
        let first_base_cube_id = first_env.base_cube_view.id();
        let first_pmrem_id = first_env.pmrem_view.id();

        if let BackgroundMode::Procedural(params) = &mut scene.background.mode {
            params.set_sun_intensity(25.0);
        } else {
            panic!("expected procedural background");
        }

        render_frame(&mut renderer, &mut scene, camera, &assets, 1);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let global_state = state
            .resource_manager
            .get_global_state(render_state_id, scene_id)
            .expect("global state missing after procedural update");
        let updated_env = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("scene gpu environment missing after procedural update");

        assert_eq!(
            first_bind_group_id, global_state.bind_group_id,
            "procedural parameter changes should not rebuild the global bind group"
        );
        assert_eq!(
            first_base_cube_id,
            updated_env.base_cube_view.id(),
            "procedural parameter changes should keep the persistent base cube view stable"
        );
        assert_eq!(
            first_pmrem_id,
            updated_env.pmrem_view.id(),
            "procedural parameter changes should keep the persistent PMREM view stable"
        );
    }

    #[test]
    fn resizing_environment_maps_recreates_persistent_views() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        scene.background.set_mode(BackgroundMode::procedural());
        let camera = make_camera();

        render_frame(&mut renderer, &mut scene, camera, &assets, 0);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let scene_id = scene.id();
        let render_state_id = state.render_frame.render_state.id;
        let global_state = state
            .resource_manager
            .get_global_state(render_state_id, scene_id)
            .expect("global state missing after first render");
        let first_bind_group_id = global_state.bind_group_id;
        let first_env = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("scene gpu environment missing after first render");
        let first_base_cube_id = first_env.base_cube_view.id();
        let first_pmrem_id = first_env.pmrem_view.id();

        scene.environment.set_base_cube_size(256);
        scene.environment.set_pmrem_size(128);

        render_frame(&mut renderer, &mut scene, camera, &assets, 1);

        let state = renderer.context.as_ref().expect("renderer state missing");
        let global_state = state
            .resource_manager
            .get_global_state(render_state_id, scene_id)
            .expect("global state missing after resize");
        let resized_env = state
            .resource_manager
            .gpu_environment(scene_id)
            .expect("scene gpu environment missing after resize");

        assert_ne!(
            first_bind_group_id, global_state.bind_group_id,
            "environment map resizing should rebuild the global bind group"
        );
        assert_ne!(
            first_base_cube_id,
            resized_env.base_cube_view.id(),
            "base cube resize should recreate the persistent base cube view"
        );
        assert_ne!(
            first_pmrem_id,
            resized_env.pmrem_view.id(),
            "PMREM resize should recreate the persistent PMREM view"
        );
        assert_eq!(256, resized_env.base_cube_texture.width());
        assert_eq!(128, resized_env.pmrem_texture.width());
    }

    #[test]
    fn small_sun_rotation_does_not_rebake_procedural_environment() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        scene.background.set_mode(BackgroundMode::procedural());
        let camera = make_camera();

        render_frame(&mut renderer, &mut scene, camera, &assets, 0);

        let scene_id = scene.id();
        let initial_source_version = renderer
            .context
            .as_ref()
            .expect("renderer state missing")
            .resource_manager
            .gpu_environment(scene_id)
            .expect("scene gpu environment missing after first render")
            .source_version;

        let initial_sun_direction =
            if let BackgroundMode::Procedural(params) = &scene.background.mode {
                params.sun_direction
            } else {
                panic!("expected procedural background");
            };

        if let BackgroundMode::Procedural(params) = &mut scene.background.mode {
            let slight_rotation = Quat::from_rotation_x(0.1_f32.to_radians());
            params.set_sun_direction((slight_rotation * initial_sun_direction).normalize());
        }

        render_frame(&mut renderer, &mut scene, camera, &assets, 1);

        let after_small_rotation = renderer
            .context
            .as_ref()
            .expect("renderer state missing")
            .resource_manager
            .gpu_environment(scene_id)
            .expect("scene gpu environment missing after small sun rotation")
            .source_version;

        assert_eq!(
            initial_source_version, after_small_rotation,
            "sub-threshold sun motion should not trigger a procedural environment rebake"
        );

        if let BackgroundMode::Procedural(params) = &mut scene.background.mode {
            let larger_rotation = Quat::from_rotation_x(1.0_f32.to_radians());
            params.set_sun_direction((larger_rotation * initial_sun_direction).normalize());
        }

        render_frame(&mut renderer, &mut scene, camera, &assets, 2);

        let after_large_rotation = renderer
            .context
            .as_ref()
            .expect("renderer state missing")
            .resource_manager
            .gpu_environment(scene_id)
            .expect("scene gpu environment missing after large sun rotation")
            .source_version;

        assert_ne!(
            after_small_rotation, after_large_rotation,
            "sun motion beyond the bake threshold should trigger a procedural environment rebake"
        );
    }

    #[test]
    fn procedural_starbox_triggers_environment_rebake() {
        let mut renderer = init_headless_renderer();
        let assets = AssetServer::new();
        let mut scene = Scene::new();
        scene.background.set_mode(BackgroundMode::procedural());
        let camera = make_camera();

        render_frame(&mut renderer, &mut scene, camera, &assets, 0);

        let scene_id = scene.id();
        let initial_source_version = renderer
            .context
            .as_ref()
            .expect("renderer state missing")
            .resource_manager
            .gpu_environment(scene_id)
            .expect("scene gpu environment missing after first render")
            .source_version;

        let starbox = assets.checkerboard(32, 8);
        if let BackgroundMode::Procedural(params) = &mut scene.background.mode {
            params.set_starbox_texture(Some(starbox.into()));
            params.set_star_intensity(2.0);
        }

        render_frame(&mut renderer, &mut scene, camera, &assets, 1);

        let updated_source_version = renderer
            .context
            .as_ref()
            .expect("renderer state missing")
            .resource_manager
            .gpu_environment(scene_id)
            .expect("scene gpu environment missing after starbox update")
            .source_version;

        assert_ne!(
            initial_source_version, updated_source_version,
            "adding a procedural starbox should trigger an environment rebake"
        );
    }
}
