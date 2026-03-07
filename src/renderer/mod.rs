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
use crate::renderer::core::binding::GlobalBindGroupCache;
use crate::renderer::core::resources::SamplerRegistry;
use crate::renderer::graph::composer::ComposerContext;
use crate::renderer::graph::frame::RenderLists;
use crate::renderer::graph::rdg::allocator::RdgTransientPool;
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
    pub(crate) rdg_graph: crate::renderer::graph::rdg::graph::RenderGraph,
    pub(crate) sampler_registry: SamplerRegistry,
    pub(crate) rdg_pool: RdgTransientPool,

    // Post-processing passes
    pub(crate) rdg_fxaa_pass: crate::renderer::graph::rdg::fxaa::FxaaFeature,
    pub(crate) rdg_tone_map_pass: crate::renderer::graph::rdg::tone_mapping::ToneMapFeature,
    pub(crate) rdg_bloom_pass: crate::renderer::graph::rdg::bloom::BloomFeature,
    pub(crate) rdg_ssao_pass: crate::renderer::graph::rdg::ssao::SsaoFeature,

    // Scene rendering passes
    pub(crate) rdg_prepass: crate::renderer::graph::rdg::prepass::PrepassFeature,
    pub(crate) rdg_opaque_pass: crate::renderer::graph::rdg::opaque::OpaqueFeature,
    pub(crate) rdg_skybox_pass: crate::renderer::graph::rdg::skybox::SkyboxFeature,
    pub(crate) rdg_transparent_pass: crate::renderer::graph::rdg::transparent::TransparentFeature,
    pub(crate) rdg_transmission_copy_pass:
        crate::renderer::graph::rdg::transmission_copy::TransmissionCopyFeature,
    pub(crate) rdg_simple_forward_pass:
        crate::renderer::graph::rdg::simple_forward::SimpleForwardFeature,
    pub(crate) rdg_sssss_pass: crate::renderer::graph::rdg::sssss::SssssFeature,

    // Shadow + Compute passes (migrated from old system)
    pub(crate) rdg_shadow_pass: crate::renderer::graph::rdg::shadow::ShadowFeature,
    pub(crate) rdg_brdf_pass: crate::renderer::graph::rdg::compute::BrdfLutFeature,
    pub(crate) rdg_ibl_pass: crate::renderer::graph::rdg::compute::IblComputeFeature,
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
        let rdg_shadow_pass =
            crate::renderer::graph::rdg::shadow::ShadowFeature::new(&wgpu_ctx.device);
        let rdg_brdf_pass =
            crate::renderer::graph::rdg::compute::BrdfLutFeature::new(&wgpu_ctx.device);
        let rdg_ibl_pass =
            crate::renderer::graph::rdg::compute::IblComputeFeature::new(&wgpu_ctx.device);

        // 6. Assemble state
        self.context = Some(RendererState {
            wgpu_ctx,
            resource_manager,
            pipeline_cache: PipelineCache::new(),
            shader_manager: ShaderManager::new(),

            render_frame,
            render_lists: RenderLists::new(),
            // blackboard: FrameBlackboard::new(),
            global_bind_group_cache,

            // RDG
            rdg_graph: crate::renderer::graph::rdg::graph::RenderGraph::new(),
            sampler_registry,
            rdg_pool: RdgTransientPool::new(),
            rdg_fxaa_pass: crate::renderer::graph::rdg::fxaa::FxaaFeature::new(),
            rdg_tone_map_pass: crate::renderer::graph::rdg::tone_mapping::ToneMapFeature::new(),
            rdg_bloom_pass: crate::renderer::graph::rdg::bloom::BloomFeature::new(),
            rdg_ssao_pass: crate::renderer::graph::rdg::ssao::SsaoFeature::new(),

            // RDG Scene Passes
            rdg_prepass: crate::renderer::graph::rdg::prepass::PrepassFeature::new(),
            rdg_opaque_pass: crate::renderer::graph::rdg::opaque::OpaqueFeature::new(),
            rdg_skybox_pass: crate::renderer::graph::rdg::skybox::SkyboxFeature::new(),
            rdg_transparent_pass: crate::renderer::graph::rdg::transparent::TransparentFeature::new(),
            rdg_transmission_copy_pass:
                crate::renderer::graph::rdg::transmission_copy::TransmissionCopyFeature::new(),
            rdg_simple_forward_pass:
                crate::renderer::graph::rdg::simple_forward::SimpleForwardFeature::new(),
            rdg_sssss_pass: crate::renderer::graph::rdg::sssss::SssssFeature::new(),

            // Shadow + Compute passes (migrated from old system)
            rdg_shadow_pass,
            rdg_brdf_pass,
            rdg_ibl_pass,
        });

        log::info!("Renderer Initialized");
        Ok(())
    }

    pub fn resize(&mut self, width: u32, height: u32, _scale_factor: f32) {
        self.size = (width, height);
        if let Some(state) = &mut self.context {
            state.wgpu_ctx.resize(width, height);
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
    ///         .add_custom_pass(HookStage::AfterPostProcess, |rdg, bb| {
    ///             ui_pass.target_tex = bb.surface_out;
    ///             rdg.add_pass(&mut ui_pass);
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
            use crate::renderer::graph::rdg::context::ExtractContext;
            use crate::renderer::HDR_TEXTURE_FORMAT;

            let view_format = state.wgpu_ctx.surface_view_format;
            let depth_format = state.wgpu_ctx.depth_format;
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
            let fxaa_enabled = scene.fxaa.enabled && is_hf;

            // GPU buffer uploads (before resource_manager enters ExtractContext)
            let bg_uniforms_gpu_id =
                state.resource_manager.ensure_buffer_id(&scene.background.uniforms);
            let bg_uniforms_cpu_id = scene.background.uniforms.id();
            let bg_mode = scene.background.mode.clone();

            if bloom_enabled {
                state
                    .resource_manager
                    .ensure_buffer(&scene.bloom.upsample_uniforms);
                state
                    .resource_manager
                    .ensure_buffer(&scene.bloom.composite_uniforms);
            }
            if ssao_enabled {
                state.resource_manager.ensure_buffer(&scene.ssao.uniforms);
            }
            state
                .resource_manager
                .ensure_buffer(&scene.tone_mapping.uniforms);
            if let Some(lut_handle) = scene.tone_mapping.lut_texture {
                state.resource_manager.prepare_texture(assets, lut_handle);
            }

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
            state.rdg_brdf_pass.extract_and_prepare(&mut extract_ctx);
            state.rdg_ibl_pass.extract_and_prepare(&mut extract_ctx);
            state.rdg_shadow_pass.extract_and_prepare(&mut extract_ctx);

            // Skybox (both pipelines)
            if needs_skybox {
                let color_format = if is_hf { HDR_TEXTURE_FORMAT } else { view_format };
                state.rdg_skybox_pass.extract_and_prepare(
                    &mut extract_ctx,
                    bg_mode,
                    bg_uniforms_cpu_id,
                    bg_uniforms_gpu_id,
                    scene_id_val,
                    color_format,
                    depth_format,
                );
            }

            if is_hf {
                state
                    .rdg_prepass
                    .extract_and_prepare(&mut extract_ctx, needs_normal, needs_feature_id);

                if ssao_enabled {
                    state.rdg_ssao_pass.extract_and_prepare(&mut extract_ctx);
                }

                state.rdg_sssss_pass.extract_and_prepare(&mut extract_ctx);

                if bloom_enabled {
                    state.rdg_bloom_pass.extract_and_prepare(&mut extract_ctx);
                }

                state.rdg_tone_map_pass.extract_and_prepare(
                    &mut extract_ctx,
                    scene.tone_mapping.mode,
                    view_format,
                    scene.tone_mapping.lut_texture.is_some(),
                    global_state_key,
                );

                if fxaa_enabled {
                    state
                        .rdg_fxaa_pass
                        .extract_and_prepare(&mut extract_ctx, view_format);
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

            rdg_graph: &mut state.rdg_graph,
            rdg_pool: &mut state.rdg_pool,
            sampler_registry: &mut state.sampler_registry,
            rdg_fxaa_pass: &mut state.rdg_fxaa_pass,
            rdg_tone_map_pass: &mut state.rdg_tone_map_pass,
            rdg_bloom_pass: &mut state.rdg_bloom_pass,
            rdg_ssao_pass: &mut state.rdg_ssao_pass,

            rdg_prepass: &mut state.rdg_prepass,
            rdg_opaque_pass: &mut state.rdg_opaque_pass,
            rdg_skybox_pass: &mut state.rdg_skybox_pass,
            rdg_transparent_pass: &mut state.rdg_transparent_pass,
            rdg_transmission_copy_pass: &mut state.rdg_transmission_copy_pass,
            rdg_simple_forward_pass: &mut state.rdg_simple_forward_pass,
            rdg_sssss_pass: &mut state.rdg_sssss_pass,

            rdg_shadow_pass: &mut state.rdg_shadow_pass,
            rdg_brdf_pass: &mut state.rdg_brdf_pass,
            rdg_ibl_pass: &mut state.rdg_ibl_pass,
        };

        // Return FrameComposer, defer Surface acquisition to render() call
        Some(FrameComposer::new(ctx))
    }

    /// Performs periodic resource cleanup.
    ///
    /// Should be called after each frame to release unused GPU resources.
    /// Uses internal heuristics to avoid expensive cleanup every frame.
    pub fn maybe_prune(&mut self) {
        if let Some(state) = &mut self.context {
            state.render_frame.maybe_prune(&mut state.resource_manager);
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
    /// This is the canonical way to transition between [`RenderPath::BasicForward`]
    /// and [`RenderPath::HighFidelity`] after initialization. The change takes
    /// effect on the **next frame**.
    ///
    /// Internally this:
    /// 1. Updates the stored settings and derived `WgpuContext` state
    ///    (`msaa_samples`, `render_path`).
    /// 2. Increments the pipeline settings version (invalidates the L1 cache).
    /// 3. Forces recreation of frame resources (render targets change format/sample count).
    /// 4. Clears the L2 pipeline cache.
    pub fn set_render_path(&mut self, path: RenderPath) {
        if self.settings.path != path {
            self.settings.path = path;
            if let Some(state) = &mut self.context {
                state.wgpu_ctx.msaa_samples = self.settings.msaa_samples();
                state.wgpu_ctx.render_path = path;
                state.wgpu_ctx.pipeline_settings_version += 1;
            }
        }
    }

    /// Returns the effective MSAA sample count.
    ///
    /// Always returns **1** in [`RenderPath::HighFidelity`] mode.
    #[inline]
    pub fn msaa_samples(&self) -> u32 {
        self.settings.msaa_samples()
    }

    /// Sets the MSAA sample count at runtime.
    ///
    /// This is only meaningful in [`RenderPath::BasicForward`] mode. In
    /// [`RenderPath::HighFidelity`] mode the call is ignored and a warning
    /// is logged — use FXAA (or future TAA) for anti-aliasing instead.
    ///
    /// Common values: 1 (disabled), 4, 8.
    pub fn set_msaa_samples(&mut self, samples: u32) {
        match &self.settings.path {
            RenderPath::BasicForward { msaa_samples } => {
                if *msaa_samples != samples {
                    let samples = samples.clamp(1, 8);
                    self.settings.path = RenderPath::BasicForward {
                        msaa_samples: samples,
                    };
                    if let Some(state) = &mut self.context {
                        state.wgpu_ctx.msaa_samples = samples;
                        state.wgpu_ctx.render_path = self.settings.path;
                        state.wgpu_ctx.pipeline_settings_version += 1;
                    }
                }
            }
            RenderPath::HighFidelity => {
                if samples != 1 {
                    log::warn!(
                        "set_msaa_samples({samples}) ignored: hardware MSAA is disabled in \
                         HighFidelity mode. Use FXAA for anti-aliasing."
                    );
                }
            }
        }
    }

    /// Returns a reference to the current renderer settings.
    #[inline]
    pub fn settings(&self) -> &RendererSettings {
        &self.settings
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
}
