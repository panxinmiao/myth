//! [gallery]
//! name = "Sky Ocean"
//! category = "Environment"
//! description = "Procedural sky and reusable ocean compositing fused into a single day-night scene with synchronized solar-time controls."
//! order = 430
//!

use std::any::Any;

use myth::prelude::*;
use myth::renderer::graph::core::{GraphBlackboard, HookStage};
use myth_dev_utils::{FpsCounter, OceanCameraSource, OceanLightSource, OceanPreset, OceanRenderer};
use myth_egui::{UiPass, UiPassNode, egui};
use winit::event::WindowEvent;

const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "examples/assets/",
};
const SCENE_CAMERA_FOV_DEGREES: f32 = 45.0;

struct SkyOceanDemo {
    controls: OrbitControls,
    cycle: DayNightCycle,
    fps_counter: FpsCounter,
    ocean: OceanRenderer,
    ui_pass: UiPass,
}

impl AppHandler for SkyOceanDemo {
    fn init(engine: &mut Engine, window: &dyn Window) -> Self {
        let mut ocean = OceanRenderer::new(&mut engine.renderer);
        ocean.apply_preset(OceanPreset::Cinematic);
        ocean.set_camera_source(OceanCameraSource::SceneMainCamera);
        ocean.set_light_source(OceanLightSource::ProceduralSky);

        let wgpu_ctx = engine
            .renderer
            .wgpu_ctx()
            .expect("renderer must be initialized before sky ocean demo setup");
        let winit_window = window
            .as_any()
            .downcast_ref::<winit::window::Window>()
            .expect("Expected winit window backend");
        let ui_pass = UiPass::new(&wgpu_ctx.device, wgpu_ctx.surface_view_format, winit_window);

        let scene = engine.scene_manager.create_active();
        let starbox = engine.assets.load_texture(
            format!("{}envs/Milky_Way_panorama.jpg", ASSET_PATH),
            ColorSpace::Srgb,
            true,
        );
        let moon_albedo =
            engine
                .assets
                .load_texture(format!("{}moon.jpg", ASSET_PATH), ColorSpace::Srgb, true);

        let mut sky = ProceduralSkyParams::golden_hour();
        sky.set_starbox_texture(starbox);
        sky.set_moon_texture(moon_albedo);
        scene
            .background
            .set_mode(BackgroundMode::procedural_with(sky));

        scene.environment.set_ambient_light(Vec3::splat(0.012));
        scene.bloom.set_enabled(true);
        scene.bloom.set_strength(0.03);
        scene.bloom.set_radius(0.006);
        scene.bloom.set_karis_average(true);
        scene
            .tone_mapping
            .set_mode(myth::ToneMappingMode::AgX(myth::AgxLook::Punchy));
        scene.tone_mapping.set_exposure(1.08);

        let mut sun_light = Light::new_directional(Vec3::new(1.0, 0.94, 0.82), 3.0);
        sun_light.cast_shadows = true;
        if let Some(shadow) = sun_light.shadow.as_mut() {
            shadow.max_shadow_distance = 36.0;
        }
        let sun_light_node = scene.add_light(sun_light);

        let moon_light = Light::new_directional(Vec3::new(0.62, 0.72, 1.0), 0.12);
        let moon_light_node = scene.add_light(moon_light);

        let cycle = DayNightCycle::new(16.5, 32.0)
            .with_sun(sun_light_node)
            .with_moon(moon_light_node)
            .with_time_speed(0.1);

        let mut camera = Camera::new_perspective(SCENE_CAMERA_FOV_DEGREES, 16.0 / 9.0, 0.1);
        camera.set_aa_mode(AntiAliasingMode::msaa());

        let cam_node_id = scene.add_camera(camera);
        scene
            .node(&cam_node_id)
            .set_position(0.0, 2.0, 7.5)
            .look_at(Vec3::new(0.0, 0.9, -18.0));
        scene.active_camera = Some(cam_node_id);

        Self {
            controls: OrbitControls::new(Vec3::new(0.0, 2.0, 7.5), Vec3::new(0.0, 0.9, -18.0)),
            cycle,
            fps_counter: FpsCounter::new(),
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
        self.cycle.time_of_day = self.cycle.time_of_day.rem_euclid(24.0);

        let winit_window = window
            .as_any()
            .downcast_ref::<winit::window::Window>()
            .expect("Expected winit window backend");
        self.ui_pass.begin_frame(winit_window);

        let egui_ctx = self.ui_pass.context().clone();
        egui::Window::new("Sky + Ocean")
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-20.0, 20.0))
            .resizable(false)
            .show(&egui_ctx, |ui| {
                ui.label("Pure procedural sky and ocean composition with synchronized day-night lighting.");
                ui.separator();
                ui.checkbox(&mut self.cycle.auto_tick, "Auto animate");
                ui.add(
                    egui::Slider::new(&mut self.cycle.time_of_day, 0.0..=24.0).text("Solar Time"),
                );
                ui.add(
                    egui::Slider::new(&mut self.cycle.time_speed, -2.0..=2.0)
                        .text("Hours / second"),
                );
                ui.label(format!("Day count: {:.2}", self.cycle.day_count));
                ui.separator();
                self.ocean.ui(ui);
            });

        self.ui_pass.end_frame(winit_window);

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        if self.ocean.camera_source() == OceanCameraSource::Reference {
            let (position, target, fov_radians) = self.ocean.reference_camera_view();
            if let Some((transform, camera)) = scene.query_main_camera_bundle() {
                transform.position = position;
                transform.look_at(target, Vec3::Y);
                camera.set_fov(fov_radians);
            }
            self.controls.set_target(target);
            self.controls.set_position(position);
        } else if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            camera.set_fov_degrees(SCENE_CAMERA_FOV_DEGREES);
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        }

        self.cycle.update(scene, &engine.input, frame.dt);

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!(
                "Sky Ocean | Solar Time: {:.2} | FPS: {:.1}",
                self.cycle.time_of_day, fps
            ));
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
                ocean.apply_composite(rdg, blackboard, width, height)
            })
            .add_custom_pass(HookStage::AfterPostProcess, move |rdg, blackboard| {
                let new_surface = rdg.add_pass("Sky_Ocean_UI", |builder| {
                    let out =
                        builder.mutate_texture(blackboard.surface_out, "Surface_With_SkyOcean_UI");
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
        .with_title("Sky Ocean")
        .with_settings(RendererSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<SkyOceanDemo>()
}
