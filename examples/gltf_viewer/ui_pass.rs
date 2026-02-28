//! UI Pass Plugin Module
//!
//! An external plugin example demonstrating how to integrate [egui](https://github.com/emilk/egui)
//! into the Myth rendering engine. This module is **not** part of the engine core — it lives
//! entirely in user land. In the future it may be promoted to an official built-in UI solution.
//!
//! # Architecture
//!
//! `UiPass` implements the [`RenderNode`] trait so it can be injected into the engine's
//! transient render graph. The lifecycle is split into clearly separated phases:
//!
//! | Phase | Receiver | Responsibility |
//! |-------|----------|----------------|
//! | **CPU / App** | `&mut self` | Input forwarding, egui frame begin/end, tessellation |
//! | **Prepare** | `&mut self` | Deferred texture registration, buffer uploads, delta application |
//! | **Run** | `&self` | Record a single render pass — purely read-only |
//!
//! # Per-Frame Usage
//!
//! ```text
//! handle_input()          // forward winit events to egui
//! begin_frame(window)     // start an egui pass
//! … build UI via context() …
//! end_frame(window)       // tessellate shapes, capture texture delta
//! ── RenderGraph ──
//! prepare(ctx)            // upload textures & geometry to GPU
//! run(ctx, encoder)       // record the egui render pass
//! ```

use rustc_hash::FxHashMap;
use wgpu::{Device, TextureFormat};
use winit::event::WindowEvent;
use winit::window::Window;

use myth::{
    assets::TextureHandle,
    renderer::graph::{ExecuteContext, PrepareContext, RenderNode},
};

/// Egui-based UI render pass.
///
/// Wraps the full egui lifecycle (input → frame → tessellation → GPU upload → draw)
/// behind the engine's [`RenderNode`] interface.
///
/// # Deferred Texture Registration
///
/// Engine textures ([`TextureHandle`]) are not immediately available to egui.
/// Call [`request_texture`](Self::request_texture) during the UI-build phase;
/// the actual GPU registration happens in [`prepare`](RenderNode::prepare) once the
/// resource manager confirms the texture is uploaded.
pub struct UiPass {
    /// Shared egui context (cheap to clone — reference-counted internally).
    egui_ctx: egui::Context,
    /// Bridges winit events into egui raw input.
    state: egui_winit::State,
    /// Egui's wgpu backend — owns GPU pipelines, textures, and vertex buffers.
    renderer: egui_wgpu::Renderer,

    /// Tessellated draw data produced by [`end_frame`](Self::end_frame),
    /// consumed by [`prepare`](RenderNode::prepare) and [`run`](RenderNode::run).
    clipped_primitives: Vec<egui::ClippedPrimitive>,
    /// Texture create/update/free operations accumulated during the egui frame.
    textures_delta: egui::TexturesDelta,
    /// Current viewport size and DPI, kept in sync via [`resize`](Self::resize).
    screen_descriptor: egui_wgpu::ScreenDescriptor,

    // ── Deferred texture registration ──────────────────────────────────────
    /// Handles waiting for GPU-side readiness (drained each prepare phase).
    texture_requests: Vec<TextureHandle>,
    /// Successfully registered engine textures → egui texture IDs.
    texture_map: FxHashMap<TextureHandle, egui::TextureId>,
    /// Tracks the GPU image ID per handle for staleness detection.
    gpu_resource_ids: FxHashMap<TextureHandle, u64>,
}

impl UiPass {
    /// Creates a new UI pass.
    ///
    /// Initializes the egui context, winit integration state, and wgpu renderer.
    /// The initial screen descriptor is derived from `window`'s inner size and
    /// scale factor.
    pub fn new(device: &Device, output_format: TextureFormat, window: &Window) -> Self {
        let size = window.inner_size();
        let egui_ctx = egui::Context::default();

        let id = egui_ctx.viewport_id();
        let state = egui_winit::State::new(egui_ctx.clone(), id, window, None, None, None);

        let renderer =
            egui_wgpu::Renderer::new(device, output_format, egui_wgpu::RendererOptions::default());

        Self {
            egui_ctx,
            state,
            renderer,
            clipped_primitives: Vec::new(),
            textures_delta: egui::TexturesDelta::default(),
            screen_descriptor: egui_wgpu::ScreenDescriptor {
                size_in_pixels: [size.width, size.height],
                pixels_per_point: window.scale_factor() as f32,
            },
            texture_requests: Vec::new(),
            texture_map: FxHashMap::default(),
            gpu_resource_ids: FxHashMap::default(),
        }
    }

    // ── CPU-side public API ────────────────────────────────────────────────

    /// Forwards a winit window event to egui.
    ///
    /// Returns `true` if egui consumed the event (the application should skip
    /// its own handling). Mouse-button releases are always reported as
    /// unconsumed so that orbit controls etc. can detect "drag end".
    pub fn handle_input(&mut self, window: &Window, event: &WindowEvent) -> bool {
        let response = self.state.on_window_event(window, event);

        if let WindowEvent::MouseInput {
            state: winit::event::ElementState::Released,
            ..
        } = event
        {
            return false;
        }

        response.consumed
    }

    /// Requests that an engine texture be made available in egui.
    ///
    /// If the texture has already been registered, its [`egui::TextureId`] is
    /// returned immediately. Otherwise the handle is enqueued for deferred
    /// registration during the next [`prepare`](RenderNode::prepare) phase.
    #[allow(dead_code)]
    pub fn request_texture(&mut self, handle: TextureHandle) -> Option<egui::TextureId> {
        if let Some(&id) = self.texture_map.get(&handle) {
            return Some(id);
        }

        if !self.texture_requests.contains(&handle) {
            self.texture_requests.push(handle);
        }

        None
    }

    /// Unregisters a previously registered engine texture from egui.
    #[allow(dead_code)]
    pub fn free_texture(&mut self, handle: TextureHandle) {
        if let Some(id) = self.texture_map.remove(&handle) {
            self.gpu_resource_ids.remove(&handle);
            self.renderer.free_texture(&id);
        }
    }

    /// Registers a raw wgpu texture view with egui, returning the assigned
    /// [`egui::TextureId`]. Useful for off-screen render targets or
    /// externally managed textures.
    #[allow(dead_code)]
    pub fn register_native_texture(
        &mut self,
        device: &wgpu::Device,
        view: &wgpu::TextureView,
        filter: wgpu::FilterMode,
    ) -> egui::TextureId {
        self.renderer.register_native_texture(device, view, filter)
    }

    /// Begins a new egui frame. Call once per frame **before** building UI.
    pub fn begin_frame(&mut self, window: &Window) {
        let raw_input = self.state.take_egui_input(window);
        self.egui_ctx.begin_pass(raw_input);
    }

    /// Ends the current egui frame.
    ///
    /// Tessellates all accumulated shapes into [`ClippedPrimitive`]s and
    /// captures the [`TexturesDelta`] for the prepare phase.
    /// Also forwards platform output (cursor icon, clipboard, IME) back to winit.
    pub fn end_frame(&mut self, window: &Window) {
        let egui::FullOutput {
            shapes,
            textures_delta,
            platform_output,
            ..
        } = self.egui_ctx.end_pass();

        self.state.handle_platform_output(window, platform_output);
        self.textures_delta = textures_delta;
        self.clipped_primitives = self
            .egui_ctx
            .tessellate(shapes, self.egui_ctx.pixels_per_point());
    }

    /// Returns the shared [`egui::Context`] for building UI widgets.
    pub fn context(&self) -> &egui::Context {
        &self.egui_ctx
    }

    /// Updates the screen descriptor after a window resize.
    pub fn resize(&mut self, width: u32, height: u32, scale_factor: f32) {
        self.screen_descriptor.size_in_pixels = [width, height];
        self.screen_descriptor.pixels_per_point = scale_factor;
    }

    /// Returns `true` if egui wants exclusive keyboard focus (e.g. a text field is active).
    #[allow(dead_code)]
    pub fn wants_keyboard_input(&self) -> bool {
        self.egui_ctx.egui_wants_keyboard_input()
    }

    /// Returns `true` if egui wants exclusive pointer focus (e.g. hovering a widget).
    #[allow(dead_code)]
    pub fn wants_pointer_input(&self) -> bool {
        self.egui_ctx.egui_wants_pointer_input()
    }
}

/// [`RenderNode`] implementation for `UiPass`.
///
/// # Phase Responsibilities
///
/// ## Prepare (`&mut self`)
///
/// 1. **Deferred texture registration** — drain pending [`request_texture`](UiPass::request_texture)
///    requests, look up GPU images via [`ResourceManager`](myth::renderer::core::ResourceManager),
///    and register them as native egui textures.
/// 2. **Texture delta application** — upload new/updated egui textures to the GPU.
/// 3. **Geometry buffer upload** — write tessellated vertex/index data through a temporary
///    command encoder (submitted immediately).
/// 4. **Stale texture cleanup** — free textures that egui no longer references.
///
/// ## Run (`&self`)
///
/// Records a single `wgpu::RenderPass` that draws the tessellated primitives onto the
/// swap-chain surface. No mutable state is touched — fully compatible with the engine's
/// read-only execute phase.
impl RenderNode for UiPass {
    fn name(&self) -> &str {
        "UI Pass (egui)"
    }

    fn prepare(&mut self, ctx: &mut PrepareContext) {
        // 1. Process deferred texture registration requests.
        let pending_requests: Vec<TextureHandle> = self.texture_requests.drain(..).collect();

        if !pending_requests.is_empty() {
            let mut remaining_requests = Vec::new();
            let resources = &*ctx.resource_manager;

            for handle in pending_requests {
                let mut registered = false;

                if let Some(binding) = resources.get_texture_binding(handle) {
                    if let Some(gpu_image) = resources.get_image(binding.cpu_image_id) {
                        let id = self.renderer.register_native_texture(
                            &ctx.wgpu_ctx.device,
                            &gpu_image.default_view,
                            wgpu::FilterMode::Linear,
                        );
                        self.texture_map.insert(handle, id);
                        self.gpu_resource_ids.insert(handle, gpu_image.id);
                        registered = true;
                    }
                }

                if !registered {
                    remaining_requests.push(handle);
                }
            }

            self.texture_requests = remaining_requests;
        }

        let device = &ctx.wgpu_ctx.device;
        let queue = &ctx.wgpu_ctx.queue;

        // 2. Upload new / updated egui-managed textures.
        for (id, delta) in &self.textures_delta.set {
            self.renderer.update_texture(device, queue, *id, delta);
        }

        // 3. Upload vertex & index buffers via a temporary encoder.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("egui buffer upload"),
        });
        let user_cmd_bufs = self.renderer.update_buffers(
            device,
            queue,
            &mut encoder,
            &self.clipped_primitives,
            &self.screen_descriptor,
        );
        let mut cmd_bufs: Vec<wgpu::CommandBuffer> = Vec::with_capacity(1 + user_cmd_bufs.len());
        cmd_bufs.push(encoder.finish());
        cmd_bufs.extend(user_cmd_bufs);
        queue.submit(cmd_bufs);

        // 4. Free textures that egui no longer needs.
        for id in &self.textures_delta.free {
            self.renderer.free_texture(id);
            self.texture_map.retain(|_, v| v != id);
        }

        // 5. Clear the delta so it is not re-processed next frame.
        self.textures_delta.set.clear();
        self.textures_delta.free.clear();
    }

    fn run(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let mut rpass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ctx.surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            })
            .forget_lifetime();

        self.renderer.render(
            &mut rpass,
            &self.clipped_primitives,
            &self.screen_descriptor,
        );
    }
}
