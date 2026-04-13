//! Egui render-pass integration utilities for Myth applications.

use myth_assets::TextureHandle;
use myth_render::{
    core::ResourceManager,
    graph::core::{
        context::{ExecuteContext, PrepareContext},
        node::PassNode,
        types::{RenderTargetOps, TextureNodeId},
    },
};
use rustc_hash::FxHashMap;
use wgpu::{Device, TextureFormat};
use winit::event::WindowEvent;
use winit::window::Window;

/// Egui-based UI render pass.
pub struct UiPass {
    egui_ctx: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    clipped_primitives: Vec<egui::ClippedPrimitive>,
    textures_delta: egui::TexturesDelta,
    screen_descriptor: egui_wgpu::ScreenDescriptor,
    texture_requests: Vec<TextureHandle>,
    texture_map: FxHashMap<TextureHandle, egui::TextureId>,
    gpu_resource_ids: FxHashMap<TextureHandle, u64>,
}

impl UiPass {
    #[must_use]
    pub fn new(device: &Device, output_format: TextureFormat, window: &Window) -> Self {
        let size = window.inner_size();
        let egui_ctx = egui::Context::default();
        let viewport_id = egui_ctx.viewport_id();
        let state = egui_winit::State::new(egui_ctx.clone(), viewport_id, window, None, None, None);
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

    #[allow(dead_code)]
    pub fn free_texture(&mut self, handle: TextureHandle) {
        if let Some(id) = self.texture_map.remove(&handle) {
            self.gpu_resource_ids.remove(&handle);
            self.renderer.free_texture(&id);
        }
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
        let pending_requests: Vec<TextureHandle> = self.texture_requests.drain(..).collect();
        if pending_requests.is_empty() {
            return;
        }

        let mut remaining_requests = Vec::new();
        for handle in pending_requests {
            let mut registered = false;

            if let Some(binding) = resource_manager.get_texture_binding(handle)
                && let Some(gpu_image) = resource_manager.get_image(binding.image_handle)
            {
                let id = self.renderer.register_native_texture(
                    device,
                    &gpu_image.default_view,
                    wgpu::FilterMode::Linear,
                );
                self.texture_map.insert(handle, id);
                self.gpu_resource_ids.insert(handle, gpu_image.id);
                registered = true;
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

impl<'a> PassNode<'a> for UiPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let device = ctx.device;
        let queue = ctx.queue;

        for (id, delta) in &self.pass.textures_delta.set {
            self.pass.renderer.update_texture(device, queue, *id, delta);
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("egui buffer upload"),
        });
        self.pass.renderer.update_buffers(
            device,
            queue,
            &mut encoder,
            &self.pass.clipped_primitives,
            &self.pass.screen_descriptor,
        );

        for id in &self.pass.textures_delta.free {
            self.pass.renderer.free_texture(id);
            self.pass.texture_map.retain(|_, value| value != id);
        }

        self.pass.textures_delta.set.clear();
        self.pass.textures_delta.free.clear();
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let color_attachment =
            ctx.get_color_attachment(self.target_tex, RenderTargetOps::Load, None);
        let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui Pass"),
            color_attachments: &[color_attachment],
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