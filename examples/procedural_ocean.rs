//! [gallery]
//! name = "Procedural Ocean"
//! category = "Environment"
//! description = "A fullscreen procedural ocean rendered through a reusable custom pass, with live controls and a neutral output chain for the reference look."
//! order = 428
//!

use std::any::Any;

use myth::prelude::*;
use myth::renderer::graph::core::{GraphBlackboard, HookStage};
use myth_dev_utils::{FpsCounter, OceanRenderer};
use myth_egui::{UiPass, UiPassNode, egui};
use winit::event::WindowEvent;

struct ProceduralOceanDemo {
    fps_counter: FpsCounter,
    controls: OrbitControls,
    ocean: OceanRenderer,
    ui_pass: UiPass,
}

impl AppHandler for ProceduralOceanDemo {
    fn init(engine: &mut Engine, window: &dyn Window) -> Self {
        let mut ocean = OceanRenderer::new(&mut engine.renderer);
        ocean.apply_preset(myth_dev_utils::OceanPreset::Reference);
        ocean.set_camera_source(myth_dev_utils::OceanCameraSource::Reference);

        let wgpu_ctx = engine
            .renderer
            .wgpu_ctx()
            .expect("renderer must be initialized before ocean demo setup");
        let winit_window = window
            .as_any()
            .downcast_ref::<winit::window::Window>()
            .expect("Expected winit window backend");

        let ui_pass = UiPass::new(&wgpu_ctx.device, wgpu_ctx.surface_view_format, winit_window);

        let scene = engine.scene_manager.create_active();
        scene.environment.set_ambient_light(Vec3::splat(0.0));
        scene.bloom.set_enabled(false);
        scene.tone_mapping.set_mode(myth::ToneMappingMode::Linear);

        let camera = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .node(&camera)
            .set_position(0.0, 2.5, 6.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(camera);

        Self {
            fps_counter: FpsCounter::new(),
            controls: OrbitControls::new(Vec3::new(0.0, 2.2, 8.8), Vec3::new(0.0, 1.0, -0.8)),
            ocean,
            ui_pass,
        }
    }

    fn on_event(&mut self, _engine: &mut Engine, window: &dyn Window, event: &dyn Any) -> bool {
        let Some(event) = event.downcast_ref::<WindowEvent>() else {
            return false;
        };

        let winit_window = window
            .as_any()
            .downcast_ref::<winit::window::Window>()
            .expect("Expected winit window backend");

        if self.ui_pass.handle_input(winit_window, event) {
            return true;
        }

        if let WindowEvent::Resized(size) = event {
            self.ui_pass
                .resize(size.width, size.height, window.scale_factor());
        }

        false
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        self.ocean.advance_time(frame.dt);

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        }

        let winit_window = window
            .as_any()
            .downcast_ref::<winit::window::Window>()
            .expect("Expected winit window backend");
        self.ui_pass.begin_frame(winit_window);

        let egui_ctx = self.ui_pass.context().clone();
        egui::Window::new("Ocean Controls")
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-20.0, 20.0))
            .resizable(false)
            .show(&egui_ctx, |ui| {
                ui.label("Reusable fullscreen ocean helper");
                ui.separator();
                self.ocean.ui(ui);
            });

        self.ui_pass.end_frame(winit_window);

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Procedural Ocean | FPS: {:.1}", fps));
        }
    }

    fn render(&mut self, engine: &mut Engine, _window: &dyn Window) {
        let (width, height) = engine.renderer.size();
        if let Some(scene) = engine.scene_manager.active_scene_mut() {
            self.ocean
                .sync_gpu_with_scene(&engine.renderer, scene, width, height);
        } else {
            self.ocean.sync_gpu(&engine.renderer, width, height);
        }

        let Some(composer) = engine.compose_frame() else {
            return;
        };

        self.ui_pass
            .resolve_textures(composer.device(), composer.resource_manager());

        let ocean = &self.ocean;
        let ui_pass = &mut self.ui_pass;

        composer
            .add_custom_pass(HookStage::BeforePostProcess, move |rdg, blackboard| {
                ocean.apply_standalone(rdg, blackboard, width, height)
            })
            .add_custom_pass(HookStage::AfterPostProcess, move |rdg, blackboard| {
                let new_surface = rdg.add_pass("Procedural_Ocean_UI", |builder| {
                    let out =
                        builder.mutate_texture(blackboard.surface_out, "Surface_With_Ocean_UI");
                    let node = UiPassNode {
                        pass: ui_pass,
                        target_tex: out,
                    };
                    (node, out)
                });

                GraphBlackboard {
                    surface_out: new_surface,
                    ..blackboard
                }
            })
            .render();
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_title("Procedural Ocean")
        .with_settings(RendererSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<ProceduralOceanDemo>()
}
