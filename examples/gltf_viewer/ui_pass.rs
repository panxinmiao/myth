//! UI Pass Plugin Module
//!
//! An external plugin example demonstrating how to integrate
//! [egui](https://github.com/emilk/egui) into the Myth rendering engine.
//! This module is **not** part of the engine core — it lives entirely in user
//! land.
//!
//! # Architecture
//!
//! `UiPass` implements the RDG [`PassNode`] trait so it can be injected into
//! the declarative render graph via a [`CustomPassHook`]. The lifecycle is
//! split into clearly separated phases:
//!
//! | Phase | Receiver | Responsibility |
//! |-------|----------|----------------|
//! | **CPU / App** | `&mut self` | Input forwarding, egui frame begin/end, tessellation |
//! | **Prepare** | `&mut self` | Deferred texture registration, buffer uploads, delta application |
//! | **Execute** | `&self` | Record a single render pass — purely read-only |
//!
//! # Per-Frame Usage
//!
//! ```text
//! handle_input()          // forward winit events to egui
//! begin_frame(window)     // start an egui pass
//! … build UI via context() …
//! end_frame(window)       // tessellate shapes, capture texture delta
//! ── RDG ──
//! prepare(ctx)            // upload textures & geometry to GPU
//! execute(ctx, encoder)   // record the egui render pass
//! ```
//!
//! # Integration with RDG
//!
//! The pass reads `target_tex` from the [`GraphBlackboard`]'s `surface_out`
//! slot. It uses `LoadOp::Load` so that 3D content rendered by preceding
//! passes is preserved beneath the UI overlay.

use rustc_hash::FxHashMap;
use wgpu::{Device, TextureFormat};
use winit::event::WindowEvent;
use winit::window::Window;

use myth::{
    assets::TextureHandle,
    renderer::{
        core::ResourceManager,
        graph::core::{
            context::{ExecuteContext, PrepareContext},
            node::PassNode,
            types::{RenderTargetOps, TextureNodeId},
        },
    },
};

/// Egui-based UI render pass (RDG).
///
/// Wraps the full egui lifecycle (input → frame → tessellation → GPU upload →
/// draw) behind the RDG [`PassNode`] interface.
///
/// # Deferred Texture Registration
///
/// Engine textures ([`TextureHandle`]) are not immediately available to egui.
/// Call [`request_texture`](Self::request_texture) during the UI-build phase;
/// the actual GPU registration happens in [`prepare`](PassNode::prepare) once
/// the resource manager confirms the texture is uploaded.
pub struct UiPass {
    /// Shared egui context (cheap to clone — reference-counted internally).
    egui_ctx: egui::Context,
    /// Bridges winit events into egui raw input.
    state: egui_winit::State,
    /// Egui's wgpu backend — owns GPU pipelines, textures, and vertex buffers.
    renderer: egui_wgpu::Renderer,

    /// Tessellated draw data produced by [`end_frame`](Self::end_frame),
    /// consumed by [`prepare`](PassNode::prepare) and
    /// [`execute`](PassNode::execute).
    clipped_primitives: Vec<egui::ClippedPrimitive>,
    /// Texture create/update/free operations accumulated during the egui frame.
    textures_delta: egui::TexturesDelta,
    /// Current viewport size and DPI, kept in sync via [`resize`](Self::resize).
    screen_descriptor: egui_wgpu::ScreenDescriptor,

    /// RDG resource slot: the final swap-chain output target.
    /// Set by the custom pass hook from [`GraphBlackboard::surface_out`].
    // pub target_tex: TextureNodeId,

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
    /// Initialises the egui context, winit integration state, and wgpu
    /// renderer. The initial screen descriptor is derived from `window`'s
    /// inner size and scale factor.
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
            // target_tex: TextureNodeId(0),
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
    /// registration during the next [`prepare`](PassNode::prepare) phase.
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

    /// Resolves pending texture registration requests against the resource
    /// manager.
    ///
    /// Must be called **before** the RDG prepare phase (e.g. in the app's
    /// `compose_frame`) so that the PassNode's `prepare()` doesn't need
    /// access to the `ResourceManager`.
    pub fn resolve_textures(&mut self, device: &wgpu::Device, resource_manager: &ResourceManager) {
        let pending_requests: Vec<TextureHandle> = self.texture_requests.drain(..).collect();
        if pending_requests.is_empty() {
            return;
        }

        let mut remaining_requests = Vec::new();
        for handle in pending_requests {
            let mut registered = false;

            if let Some(binding) = resource_manager.get_texture_binding(handle) {
                if let Some(gpu_image) = resource_manager.get_image(binding.cpu_image_id) {
                    let id = self.renderer.register_native_texture(
                        device,
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
}

pub struct UiPassNode<'a> {
    pub pass: &'a mut UiPass,
    pub target_tex: TextureNodeId,
}

/// [`PassNode`] implementation for `UiPass`.
///
/// # Phase Responsibilities
///
/// ## Prepare (`&mut self`)
///
/// 1. **Texture delta application** — upload new/updated egui textures to the GPU.
/// 2. **Geometry buffer upload** — write tessellated vertex/index data through a temporary
///    command encoder (submitted immediately).
/// 3. **Stale texture cleanup** — free textures that egui no longer references.
///
/// ## Execute (`&self`)
///
/// Records a single `wgpu::RenderPass` that draws the tessellated primitives onto
/// the RDG-resolved surface. No mutable state is touched — fully compatible with the
/// engine's read-only execute phase.
impl<'a> PassNode<'a> for UiPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let device = ctx.device;
        let queue = ctx.queue;

        // 2. Upload new / updated egui-managed textures.
        for (id, delta) in &self.pass.textures_delta.set {
            self.pass.renderer.update_texture(device, queue, *id, delta);
        }

        // 3. Upload vertex & index buffers via a temporary encoder.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("egui buffer upload"),
        });
        let user_cmd_bufs = self.pass.renderer.update_buffers(
            device,
            queue,
            &mut encoder,
            &self.pass.clipped_primitives,
            &self.pass.screen_descriptor,
        );
        let mut cmd_bufs: Vec<wgpu::CommandBuffer> = Vec::with_capacity(1 + user_cmd_bufs.len());
        cmd_bufs.push(encoder.finish());
        cmd_bufs.extend(user_cmd_bufs);
        queue.submit(cmd_bufs);

        // 4. Free textures that egui no longer needs.
        for id in &self.pass.textures_delta.free {
            self.pass.renderer.free_texture(id);
            self.pass.texture_map.retain(|_, v| v != id);
        }

        // 5. Clear the delta so it is not re-processed next frame.
        self.pass.textures_delta.set.clear();
        self.pass.textures_delta.free.clear();
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        // Resolve the final swap-chain surface view from the RDG.
        let rtt = ctx.get_color_attachment(self.target_tex, RenderTargetOps::Load, None);
        let mut rpass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui Pass"),
                color_attachments: &[rtt],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            })
            .forget_lifetime();

        self.pass.renderer.render(
            &mut rpass,
            &self.pass.clipped_primitives,
            &self.pass.screen_descriptor,
        );
    }
}
