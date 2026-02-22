//! Rendering System
//!
//! This module handles all GPU rendering operations using a layered architecture:
//!
//! - **[`core`]**: wgpu context wrapper (Device, Queue, Surface, `ResourceManager`)
//! - **[`graph`]**: Render frame organization (`RenderFrame`, `RenderNode`, `FrameBuilder`)
//! - **[`pipeline`]**: Shader compilation and pipeline caching (L1/L2 cache strategy)
//!
//! # Architecture Overview
//!
//! ```text
//! ┌───────────────────────────────────────────────┐
//! │                  FrameComposer                  │
//! │          (High-level rendering API)             │
//! ├───────────────────────────────────────────────┤
//! │     RenderGraph     │     RenderFrame           │
//! │   (Node execution)  │  (Scene extraction)       │
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
//! 4. **Render**: Commands are executed via render passes
//!
//! # Example
//!
//! ```rust,ignore
//! // Start a frame
//! if let Some(composer) = renderer.begin_frame(scene, camera, assets, time) {
//!     composer
//!         .add_node(RenderStage::UI, &ui_pass)
//!         .render();
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
use crate::renderer::graph::composer::ComposerContext;
use crate::renderer::graph::context::FrameResources;
use crate::renderer::graph::frame::RenderLists;
use crate::renderer::graph::passes::{
    BRDFLutComputePass, BloomPass, FxaaPass, IBLComputePass, OpaquePass, SceneCullPass, ShadowPass,
    SimpleForwardPass, SkyboxPass, ToneMapPass, TransmissionCopyPass, TransparentPass,
};
use crate::renderer::graph::transient_pool::{TransientTextureDesc, TransientTexturePool};
use crate::scene::Scene;
use crate::scene::camera::RenderCamera;
use crate::{FrameBuilder, RenderStage};

use self::core::{ResourceManager, WgpuContext};
use self::graph::{FrameComposer, RenderFrame};
use self::pipeline::PipelineCache;
use self::settings::RenderSettings;

/// HDR 纹理格式
///
/// 用于高动态范围渲染目标, 中间缓冲区的格式
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
    settings: RenderSettings,
    context: Option<RendererState>,
    size: (u32, u32),
}

/// Internal renderer state
struct RendererState {
    wgpu_ctx: WgpuContext,
    resource_manager: ResourceManager,
    pipeline_cache: PipelineCache,

    render_frame: RenderFrame,
    /// 渲染列表（与 `render_frame` 分离以避免借用冲突）
    render_lists: RenderLists,

    frame_resources: FrameResources,
    transient_pool: TransientTexturePool,
    global_bind_group_cache: GlobalBindGroupCache,

    // ===== Built-in passes =====

    // Data Preparation
    pub(crate) cull_pass: SceneCullPass,
    pub(crate) shadow_pass: ShadowPass,

    // Simple Path (LDR)
    pub(crate) simple_forward_pass: SimpleForwardPass,

    // PBR Path (HDR)
    pub(crate) opaque_pass: OpaquePass,
    pub(crate) transparent_pass: TransparentPass,
    pub(crate) transmission_copy_pass: TransmissionCopyPass,

    // Skybox / Background
    pub(crate) skybox_pass: SkyboxPass,

    // Compute Passes
    pub(crate) brdf_pass: BRDFLutComputePass,
    pub(crate) ibl_pass: IBLComputePass,

    // Post Processing
    pub(crate) bloom_pass: BloomPass,
    pub(crate) tone_mapping_pass: ToneMapPass,
    pub(crate) fxaa_pass: FxaaPass,
}

impl Renderer {
    /// Phase 1: Create configuration (no GPU resources yet).
    ///
    /// This only stores the render settings. GPU resources are
    /// allocated when [`init`](Self::init) is called.
    #[must_use]
    pub fn new(settings: RenderSettings) -> Self {
        Self {
            settings,
            context: None,
            size: (0, 0),
        }
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

        // 4. Create frame resources
        let frame_resources = FrameResources::new(
            &wgpu_ctx,
            // &self.settings,
            (width, height),
        );

        // 5. Create global bind group cache
        let global_bind_group_cache = GlobalBindGroupCache::new();

        // Build passes
        // Data Preparation
        let cull_pass = SceneCullPass::new();
        let shadow_pass = ShadowPass::new(&wgpu_ctx.device);

        // Simple Path (LDR)
        let simple_forward_pass = SimpleForwardPass::new(self.settings.clear_color);

        // PBR Path (HDR)
        let opaque_pass = OpaquePass::new(self.settings.clear_color);
        let transparent_pass = TransparentPass::new();
        let transmission_copy_pass = TransmissionCopyPass::new();

        // Compute Passes
        let brdf_pass = BRDFLutComputePass::new(&wgpu_ctx.device);
        let ibl_pass = IBLComputePass::new(&wgpu_ctx.device);

        // Post Processing
        let bloom_pass = BloomPass::new(&wgpu_ctx.device);
        let tone_mapping_pass = ToneMapPass::new(&wgpu_ctx.device);
        let fxaa_pass = FxaaPass::new(&wgpu_ctx.device);

        // Skybox / Background
        let skybox_pass = SkyboxPass::new(&wgpu_ctx.device);

        // 6. Assemble state
        self.context = Some(RendererState {
            wgpu_ctx,
            resource_manager,
            pipeline_cache: PipelineCache::new(),

            render_frame,
            render_lists: RenderLists::new(),

            frame_resources,
            transient_pool: TransientTexturePool::new(),
            global_bind_group_cache,

            cull_pass,
            shadow_pass,
            simple_forward_pass,
            opaque_pass,
            transparent_pass,
            transmission_copy_pass,
            brdf_pass,
            ibl_pass,
            bloom_pass,
            tone_mapping_pass,
            fxaa_pass,
            skybox_pass,
        });

        log::info!("Renderer Initialized");
        Ok(())
    }

    pub fn resize(&mut self, width: u32, height: u32, _scale_factor: f32) {
        self.size = (width, height);
        if let Some(state) = &mut self.context {
            state.wgpu_ctx.resize(width, height);
            state
                .frame_resources
                .resize(&state.wgpu_ctx, (width, height));
        }
    }

    /// Begins building a new frame for rendering.
    ///
    /// Returns a [`FrameComposer`] that provides a chainable API for
    /// configuring the render pipeline.
    ///
    /// # Usage
    ///
    /// ```rust,ignore
    /// // Method 1: Use default built-in passes
    /// if let Some(composer) = renderer.begin_frame(scene, camera, assets, time) {
    ///     composer.render();
    /// }
    ///
    /// // Method 2: Custom pipeline with chained nodes
    /// if let Some(composer) = renderer.begin_frame(scene, camera, assets, time) {
    ///     composer
    ///         .add_node(RenderStage::UI, &ui_pass)
    ///         .add_node(RenderStage::PostProcess, &bloom_pass)
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

        // Prepare phase: extract scene, prepare built-in passes
        state.render_frame.extract_and_prepare(
            &mut state.resource_manager,
            scene,
            camera,
            assets,
            time,
        );

        let mut frame_builder = FrameBuilder::new();

        // ========================================
        // 1. Compute passes (conditional)
        // ========================================
        if state.resource_manager.needs_brdf_compute {
            frame_builder.add_node(RenderStage::PreProcess, &mut state.brdf_pass);
        }
        if state.resource_manager.pending_ibl_source.is_some() {
            frame_builder.add_node(RenderStage::PreProcess, &mut state.ibl_pass);
        }

        // ========================================
        // 2. Scene culling
        // ========================================
        frame_builder.add_node(RenderStage::PreProcess, &mut state.cull_pass);

        // ========================================
        // 2. 阴影
        // ========================================
        frame_builder.add_node(RenderStage::ShadowMap, &mut state.shadow_pass);

        // ========================================
        // 3. 路径选择：HDR (PBR Path) vs LDR (Simple Path)
        // ========================================
        //
        // PBR Path: OpaquePass → [TransmissionCopyPass] → TransparentPass → ToneMapPass
        // Simple Path: SimpleForwardPass (直接输出到 Surface)
        //
        let use_hdr_path = self.settings.enable_hdr;

        if use_hdr_path {
            state
                .frame_resources
                .ensure_transmission_resource(&state.wgpu_ctx.device);
            // === PBR Path (HDR) ===

            // Opaque rendering
            frame_builder.add_node(RenderStage::Opaque, &mut state.opaque_pass);

            // Skybox / Background (after opaque, before transparent)
            if scene.background.needs_skybox_pass() {
                frame_builder.add_node(RenderStage::Skybox, &mut state.skybox_pass);
            }

            // Transmission copy (conditional)
            // 注意：TransmissionCopyPass 内部会检查 use_transmission 标志
            // 如果场景中没有使用 Transmission 的材质，此 Pass 会提前返回
            frame_builder.add_node(RenderStage::Opaque, &mut state.transmission_copy_pass);

            // Transparent rendering
            frame_builder.add_node(RenderStage::Transparent, &mut state.transparent_pass);

            // Bloom (conditional — only when enabled in Scene.bloom)
            if scene.bloom.enabled {
                frame_builder.add_node(RenderStage::PostProcess, &mut state.bloom_pass);
            }

            // FXAA routing: when enabled, ToneMap outputs to a transient LDR texture
            // which FXAA then reads and writes to the surface.
            // When disabled, ToneMap writes directly to the surface.
            if scene.fxaa.enabled {
                let ldr_tex_id = state.transient_pool.allocate(
                    &state.wgpu_ctx.device,
                    &TransientTextureDesc {
                        width: state.wgpu_ctx.size().0,
                        height: state.wgpu_ctx.size().1,
                        format: state.wgpu_ctx.surface_view_format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        mip_level_count: 1,
                        label: "FXAA LDR Intermediate",
                    },
                );

                state.tone_mapping_pass.output_texture_id = Some(ldr_tex_id);
                state.fxaa_pass.input_texture_id = Some(ldr_tex_id);
            } else {
                state.tone_mapping_pass.output_texture_id = None;
                state.fxaa_pass.input_texture_id = None;
            }

            // Tone mapping (HDR → LDR)
            frame_builder.add_node(RenderStage::PostProcess, &mut state.tone_mapping_pass);

            // FXAA (conditional — only when enabled in Scene.fxaa)
            if scene.fxaa.enabled {
                frame_builder.add_node(RenderStage::PostProcess, &mut state.fxaa_pass);
            }
        } else {
            // === Simple Path (LDR) ===
            // Prepare skybox for potential inline drawing by SimpleForwardPass.
            // SkyboxPass::prepare() stores pipeline/bind_group in render_lists.
            // SkyboxPass::run() is a no-op in LDR mode (checked via enable_hdr).
            if scene.background.needs_skybox_pass() {
                frame_builder.add_node(RenderStage::PreProcess, &mut state.skybox_pass);
            }
            frame_builder.add_node(RenderStage::Opaque, &mut state.simple_forward_pass);
        }

        // ========================================
        // 3. 构建 ComposerContext
        // ========================================
        let ctx = ComposerContext {
            wgpu_ctx: &mut state.wgpu_ctx,
            resource_manager: &mut state.resource_manager,
            pipeline_cache: &mut state.pipeline_cache,

            extracted_scene: &state.render_frame.extracted_scene,
            render_state: &state.render_frame.render_state,

            frame_resources: &mut state.frame_resources,
            transient_pool: &mut state.transient_pool,
            global_bind_group_cache: &mut state.global_bind_group_cache,

            render_lists: &mut state.render_lists,

            scene,
            camera,
            assets,
            time,
        };

        // Return FrameComposer, defer Surface acquisition to render() call
        Some(FrameComposer::new(frame_builder, ctx))
    }

    /// Performs periodic resource cleanup.
    ///
    /// Should be called after each frame to release unused GPU resources.
    /// Uses internal heuristics to avoid expensive cleanup every frame.
    pub fn maybe_prune(&mut self) {
        if let Some(state) = &mut self.context {
            state.render_frame.maybe_prune(&mut state.resource_manager);
            // Trim transient pool textures idle for > 60 frames (~1 second at 60 fps)
            state.transient_pool.trim(60);
        }
    }

    // === Runtime Settings API ===

    /// Returns whether HDR rendering is currently enabled.
    ///
    /// HDR rendering uses floating-point render targets and tone mapping
    /// for high dynamic range lighting.
    #[inline]
    pub fn is_hdr_enabled(&self) -> bool {
        self.settings.enable_hdr
    }

    /// Enables or disables HDR rendering at runtime.
    ///
    /// When HDR is enabled:
    /// - Scene is rendered to a floating-point HDR buffer
    /// - Tone mapping pass converts HDR to LDR for display
    /// - `ToneMappingSettings` in Scene become effective
    ///
    /// When HDR is disabled:
    /// - Scene is rendered directly to the screen
    /// - No tone mapping is applied
    ///
    /// Note: This change takes effect on the next frame.
    pub fn set_hdr_enabled(&mut self, enabled: bool) {
        if self.settings.enable_hdr != enabled {
            self.settings.enable_hdr = enabled;
            if let Some(state) = &mut self.context {
                state.wgpu_ctx.enable_hdr = enabled;
                // Increment pipeline settings version to invalidate L1 cache
                state.wgpu_ctx.pipeline_settings_version += 1;
                // Force recreation of frame resources (HDR textures)
                let size = state.wgpu_ctx.size();
                state.frame_resources.force_recreate(&state.wgpu_ctx, size);
                // Clear L2 cache since settings have changed
                state.pipeline_cache.clear();
            }
        }
    }

    /// Returns the current MSAA sample count.
    ///
    /// Common values: 1 (disabled), 4, 8.
    #[inline]
    pub fn msaa_samples(&self) -> u32 {
        self.settings.msaa_samples
    }

    /// Sets the MSAA sample count at runtime.
    ///
    /// Higher values provide better anti-aliasing quality but use more GPU memory
    /// and bandwidth. Common values are:
    /// - 1: MSAA disabled
    /// - 4: Good balance of quality and performance
    /// - 8: High quality (may impact performance)
    ///
    /// Note: This requires recreation of render targets and takes effect on resize
    /// or the next frame.
    pub fn set_msaa_samples(&mut self, samples: u32) {
        if self.settings.msaa_samples != samples {
            let samples = samples.clamp(1, 8);
            self.settings.msaa_samples = samples;
            if let Some(state) = &mut self.context {
                state.wgpu_ctx.msaa_samples = samples;
                // Increment pipeline settings version to invalidate L1 cache
                state.wgpu_ctx.pipeline_settings_version += 1;
                // Force recreation of frame resources
                let size = state.wgpu_ctx.size();
                state.frame_resources.force_recreate(&state.wgpu_ctx, size);
                // Clear L2 cache since settings have changed
                state.pipeline_cache.clear();
            }
        }
    }

    /// Returns a reference to the current render settings.
    #[inline]
    pub fn settings(&self) -> &RenderSettings {
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
