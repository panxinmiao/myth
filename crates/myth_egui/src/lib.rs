//! Myth-native egui integration.

mod renderer;
mod texture_registry;

use myth_assets::TextureHandle;
use myth_render::core::ResourceManager;
use myth_render::graph::core::{
    context::{ExecuteContext, PrepareContext},
    node::PassNode,
    types::{RenderTargetOps, TextureNodeId},
};
use texture_registry::TextureRegistry;
use winit::{event::WindowEvent, window::Window};

pub use egui;
pub use renderer::{Renderer, RendererOptions, ScreenDescriptor};

/// Egui-based UI render pass integrated with Myth's render graph.
pub struct UiPass {
    egui_ctx: egui::Context,
    state: egui_winit::State,
    renderer: Renderer,
    clipped_primitives: Vec<egui::ClippedPrimitive>,
    textures_delta: egui::TexturesDelta,
    screen_descriptor: ScreenDescriptor,
    textures: TextureRegistry,
}

impl UiPass {
    #[must_use]
    pub fn new(device: &wgpu::Device, output_format: wgpu::TextureFormat, window: &Window) -> Self {
        let size = window.inner_size();
        let egui_ctx = egui::Context::default();
        let viewport_id = egui_ctx.viewport_id();
        let state = egui_winit::State::new(egui_ctx.clone(), viewport_id, window, None, None, None);
        let renderer = Renderer::new(device, output_format, RendererOptions::default());

        Self {
            egui_ctx,
            state,
            renderer,
            clipped_primitives: Vec::new(),
            textures_delta: egui::TexturesDelta::default(),
            screen_descriptor: ScreenDescriptor {
                size_in_pixels: [size.width, size.height],
                pixels_per_point: window.scale_factor() as f32,
            },
            textures: TextureRegistry::default(),
        }
    }

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

    /// Returns a stable egui texture id for a Myth texture handle.
    ///
    /// The returned id is immediately usable by egui. Until the Myth texture
    /// is available on the GPU, it is backed by a small placeholder texture.
    /// Call [`Self::resolve_textures`] before adding the UI pass to the graph
    /// so the id can be rebound to the latest Myth texture view and sampler.
    pub fn texture_id(&mut self, handle: TextureHandle) -> egui::TextureId {
        self.textures.texture_id(handle, &mut self.renderer)
    }

    /// Alias for [`Self::texture_id`] that reads naturally at call sites that
    /// explicitly register Myth textures before drawing them in egui.
    pub fn register_texture(&mut self, handle: TextureHandle) -> egui::TextureId {
        self.texture_id(handle)
    }

    #[allow(dead_code)]
    pub fn request_texture(&mut self, handle: TextureHandle) -> Option<egui::TextureId> {
        Some(self.texture_id(handle))
    }

    #[allow(dead_code)]
    pub fn free_texture(&mut self, handle: TextureHandle) {
        self.textures.free(handle, &mut self.renderer);
    }

    #[allow(dead_code)]
    pub fn register_native_texture(
        &mut self,
        device: &wgpu::Device,
        view: &wgpu::TextureView,
        filter: wgpu::FilterMode,
    ) -> egui::TextureId {
        self.renderer.register_native_texture(device, view, filter)
    }

    pub fn begin_frame(&mut self, window: &Window) {
        let raw_input = self.state.take_egui_input(window);
        self.egui_ctx.begin_pass(raw_input);
    }

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

    #[must_use]
    pub fn context(&self) -> &egui::Context {
        &self.egui_ctx
    }

    pub fn resize(&mut self, width: u32, height: u32, scale_factor: f32) {
        self.screen_descriptor.size_in_pixels = [width, height];
        self.screen_descriptor.pixels_per_point = scale_factor;
    }

    #[allow(dead_code)]
    #[must_use]
    pub fn wants_keyboard_input(&self) -> bool {
        self.egui_ctx.egui_wants_keyboard_input()
    }

    #[allow(dead_code)]
    #[must_use]
    pub fn wants_pointer_input(&self) -> bool {
        self.egui_ctx.egui_wants_pointer_input()
    }

    pub fn resolve_textures(&mut self, device: &wgpu::Device, resource_manager: &ResourceManager) {
        self.textures
            .resolve(device, resource_manager, &mut self.renderer);
    }
}

pub struct UiPassNode<'a> {
    pub pass: &'a mut UiPass,
    pub target_tex: TextureNodeId,
}

impl<'a> PassNode<'a> for UiPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let device = ctx.device;
        let queue = ctx.queue;

        for (id, delta) in &self.pass.textures_delta.set {
            self.pass.renderer.update_texture(device, queue, *id, delta);
        }

        self.pass.renderer.update_buffers(
            device,
            queue,
            &self.pass.clipped_primitives,
            &self.pass.screen_descriptor,
        );

        for id in &self.pass.textures_delta.free {
            self.pass.renderer.free_texture(id);
            self.pass.textures.free_by_id(*id);
        }

        self.pass.textures_delta.set.clear();
        self.pass.textures_delta.free.clear();
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        if self.pass.screen_descriptor.size_in_pixels.contains(&0) {
            return;
        }

        let Some(color_attachment) =
            ctx.get_color_attachment(self.target_tex, RenderTargetOps::Load, None)
        else {
            return;
        };

        let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("myth egui pass"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let mut render_pass = render_pass.forget_lifetime();

        self.pass.renderer.render(
            &mut render_pass,
            &self.pass.clipped_primitives,
            &self.pass.screen_descriptor,
        );
    }
}
