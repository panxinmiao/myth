//! [gallery]
//! name = "Procedural Sky"
//! category = "Environment"
//! description = "Interactive day-night atmosphere demo with celestial controls and UI overlay."
//! order = 240
//!

use std::any::Any;

use myth::prelude::*;
use myth_dev_utils::*;
use myth_resources::Key;
use winit::event::WindowEvent;

const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "examples/assets/",
};

/// Procedural Sky Demo
///
/// Demonstrates the Hillaire 2020 atmosphere system with a SceneLogic-driven
/// day/night cycle, moon light, and hybrid star field.
struct ProceduralSkyDemo {
    cam_node_id: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
    ui_pass: UiPass,
    cycle: DayNightCycle,
    helmet_prefab: PrefabHandle,
    helmet_loaded: bool,
}

fn print_help() {
    println!("╔══════════════════════════════════════════╗");
    println!("║          Procedural Sky Demo             ║");
    println!("╟──────────────────────────────────────────╢");
    println!("║ Space: Toggle day/night cycle auto-tick  ║");
    println!("║ Up/Down: Increase/decrease time speed    ║");
    println!("║ Mouse Drag: Orbit camera                 ║");
    println!("║ Scroll: Zoom                             ║");
    println!("╚══════════════════════════════════════════╝");
}

impl AppHandler for ProceduralSkyDemo {
    fn init(engine: &mut Engine, window: &dyn Window) -> Self {
        let wgpu_ctx = engine
            .renderer
            .wgpu_ctx()
            .expect("Renderer not initialized");
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

        let mut sun_light = Light::new_directional(Vec3::new(1.0, 0.95, 0.8), 3.0);
        sun_light.cast_shadows = true;
        let sun_light_node = scene.add_light(sun_light);

        let moon_light = Light::new_directional(Vec3::new(0.62, 0.72, 1.0), 0.12);
        let moon_light_node = scene.add_light(moon_light);

        let cycle = DayNightCycle::new(16.5, 35.0)
            .with_sun(sun_light_node)
            .with_moon(moon_light_node)
            .with_time_speed(0.35);

        // scene.add_logic(cycle);

        let helmet_prefab = engine.assets.load_gltf(format!(
            "{}DamagedHelmet/glTF/DamagedHelmet.gltf",
            ASSET_PATH
        ));

        scene.bloom.set_enabled(true);
        scene.bloom.set_strength(0.02);
        scene.bloom.set_radius(0.005);
        scene.bloom.set_karis_average(true);

        scene
            .tone_mapping
            .set_mode(myth::ToneMappingMode::AgX(myth::AgxLook::Punchy));

        // Camera
        let mut camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        camera.set_aa_mode(AntiAliasingMode::msaa());

        let cam_node_id = scene.add_camera(camera);
        scene
            .node(&cam_node_id)
            .set_position(0.0, 0.5, 4.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam_node_id);

        print_help();

        Self {
            cam_node_id,
            controls: OrbitControls::new(Vec3::new(0.0, 0.5, 4.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
            ui_pass,
            cycle,
            helmet_prefab,
            helmet_loaded: false,
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
        let winit_window = window
            .as_any()
            .downcast_ref::<winit::window::Window>()
            .expect("Expected winit window backend");
        self.ui_pass.begin_frame(winit_window);
        let egui_ctx = self.ui_pass.context().clone();
        egui::Window::new("Sky Controls")
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-20.0, 20.0))
            .resizable(false)
            .show(&egui_ctx, |ui| {
                ui.label("Procedural atmosphere and celestial motion");
                ui.separator();
                ui.checkbox(&mut self.cycle.auto_tick, "Auto animate");
                ui.add(
                    egui::Slider::new(&mut self.cycle.time_of_day, 0.0..=24.0).text("Solar time"),
                );
                ui.add(
                    egui::Slider::new(&mut self.cycle.time_speed, -2.0..=2.0)
                        .text("Hours / second"),
                );
                ui.label(format!("Day count: {:.2}", self.cycle.day_count));
                ui.label("Keyboard shortcuts remain available for quick tweaks.");
            });
        self.ui_pass.end_frame(winit_window);

        self.cycle.time_of_day = self.cycle.time_of_day.rem_euclid(24.0);
        let assets = engine.assets.clone();
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        if !self.helmet_loaded
            && let Some(prefab) = assets.prefabs.get(self.helmet_prefab)
        {
            let root = scene.instantiate(prefab.as_ref());
            scene.node(&root).set_position(0.0, 0.0, 0.0);
            self.helmet_loaded = true;
        }

        // Orbit camera
        if let Some(cam_node) = scene.get_node_mut(self.cam_node_id) {
            self.controls
                .update(&mut cam_node.transform, &engine.input, 45.0, frame.dt);
        }

        if engine.input.get_key_down(Key::Space) {
            self.cycle.auto_tick = !self.cycle.auto_tick;
        }

        if engine.input.get_key_down(Key::ArrowUp) {
            self.cycle.time_speed += 0.1;
        }

        if engine.input.get_key_down(Key::ArrowDown) {
            self.cycle.time_speed -= 0.1;
        }

        self.cycle.update(scene, &engine.input, frame.dt);

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!(
                "Procedural Sky Day/Night Time: {:.2}, Speed: {:.2} (hour/s) - FPS: {:.0}",
                self.cycle.time_of_day, self.cycle.time_speed, fps
            ));
        }
    }

    fn render(&mut self, engine: &mut Engine, _window: &dyn Window) {
        use myth::renderer::graph::core::{GraphBlackboard, HookStage};

        let Some(composer) = engine.compose_frame() else {
            return;
        };

        self.ui_pass
            .resolve_textures(composer.device(), composer.resource_manager());

        let ui_pass = &mut self.ui_pass;
        composer
            .add_custom_pass(HookStage::AfterPostProcess, move |rdg, blackboard| {
                let new_surface = rdg.add_pass("ProceduralSkyUI", |builder| {
                    let out = builder.mutate_texture(blackboard.surface_out, "Surface_With_UI");
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
        .with_settings(RendererSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<ProceduralSkyDemo>()
}
