//! Rendering System
//!
//! This module handles all GPU rendering operations using a layered architecture:
//!
//! - **[`core`]**: wgpu context wrapper (Device, Queue, Surface, ResourceManager)
//! - **[`graph`]**: Render frame organization (RenderFrame, RenderNode, FrameBuilder)
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

use crate::renderer::core::binding::GlobalBindGroupCache;
use crate::renderer::graph::composer::ComposerContext;
use crate::{FrameBuilder, RenderStage};
use crate::assets::AssetServer;
use crate::errors::Result;
use crate::renderer::graph::context::FrameResources;
use crate::scene::Scene;
use crate::scene::camera::RenderCamera;

use self::core::{ResourceManager, WgpuContext};
use self::graph::{FrameComposer, RenderFrame};
use self::pipeline::PipelineCache;
use self::settings::RenderSettings;

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
    frame_resources: FrameResources,
    global_bind_group_cache: GlobalBindGroupCache,
}

impl Renderer {
    /// Phase 1: Create configuration (no GPU resources yet).
    ///
    /// This only stores the render settings. GPU resources are
    /// allocated when [`init`](Self::init) is called.
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
        let render_frame = RenderFrame::new(wgpu_ctx.device.clone());

        // 4. Create frame resources
        let frame_resources = FrameResources::new(
            &wgpu_ctx.device,
            &self.settings,
            (width, height),
        );

        // 5. Create global bind group cache
        let global_bind_group_cache = GlobalBindGroupCache::new();

        // 4. Assemble state
        self.context = Some(RendererState {
            wgpu_ctx,
            resource_manager,
            pipeline_cache: PipelineCache::new(),
            render_frame,
            frame_resources,
            global_bind_group_cache,
        });

        log::info!("Renderer Initialized");
        Ok(())
    }

    pub fn resize(&mut self, width: u32, height: u32, _scale_factor: f32) {
        self.size = (width, height);
        if let Some(state) = &mut self.context {
            state.wgpu_ctx.resize(width, height);
            state.frame_resources.resize(&state.wgpu_ctx.device,  &self.settings, (width, height));
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
            //&mut state.wgpu_ctx,
            &mut state.resource_manager,
            //&mut state.pipeline_cache,
            scene,
            camera,
            assets,
            time,
        );

        let RendererState {
            wgpu_ctx,
            resource_manager,
            pipeline_cache,
            render_frame,     // 将被 Builder 借用
            frame_resources,  // 将被 Context 借用
            global_bind_group_cache,
        } = state;

        let mut frame_builder = FrameBuilder::new();

        frame_builder
            .add_node(RenderStage::PreProcess, &mut render_frame.brdf_pass)
            .add_node(RenderStage::PreProcess, &mut render_frame.ibl_pass)
            .add_node(RenderStage::Opaque, &mut render_frame.forward_pass);

        let ctx = ComposerContext {
            wgpu_ctx,
            resource_manager,
            pipeline_cache,
            render_frame,
            frame_resources,
            global_bind_group_cache,

            scene,
            camera,
            assets,
            time,
        };

        // Return FrameComposer, defer Surface acquisition to render() call
        Some(FrameComposer::new(
            frame_builder,
            ctx,
        ))
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

    /// Returns a reference to the WgpuContext.
    ///
    /// For external plugins that need access to low-level GPU resources.
    /// Only available after renderer initialization.
    pub fn wgpu_ctx(&self) -> Option<&WgpuContext> {
        self.context.as_ref().map(|s| &s.wgpu_ctx)
    }
}
