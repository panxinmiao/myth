//! [gallery]
//! name = "Ocean Composite Scene"
//! category = "Environment"
//! description = "A depth-aware procedural ocean composited behind a lit 3D lookout scene through the reusable ocean helper."
//! order = 432
//!

use std::any::Any;
use std::f32::consts::FRAC_PI_2;

use myth::prelude::*;
use myth::renderer::graph::core::{GraphBlackboard, HookStage};
use myth_dev_utils::{FpsCounter, OceanPreset, OceanRenderer};
use myth_egui::{UiPass, UiPassNode, egui};
use winit::event::WindowEvent;

struct OceanCompositeSceneDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
    ocean: OceanRenderer,
    ui_pass: UiPass,
    beacon_ring: NodeHandle,
    beacon_core: NodeHandle,
    accent_light: NodeHandle,
    shard_nodes: Vec<NodeHandle>,
    time: f32,
}

impl AppHandler for OceanCompositeSceneDemo {
    fn init(engine: &mut Engine, window: &dyn Window) -> Self {
        let mut ocean = OceanRenderer::new(&mut engine.renderer);
        ocean.apply_preset(OceanPreset::Cinematic);

        let wgpu_ctx = engine
            .renderer
            .wgpu_ctx()
            .expect("renderer must be initialized before ocean composite demo setup");
        let winit_window = window
            .as_any()
            .downcast_ref::<winit::window::Window>()
            .expect("Expected winit window backend");
        let ui_pass = UiPass::new(&wgpu_ctx.device, wgpu_ctx.surface_view_format, winit_window);

        let scene = engine.scene_manager.create_active();
        scene.background.set_mode(BackgroundMode::gradient(
            Vec4::new(0.03, 0.04, 0.07, 1.0),
            Vec4::new(0.005, 0.007, 0.015, 1.0),
        ));
        scene.environment.set_ambient_light(Vec3::splat(0.014));
        scene.bloom.set_enabled(true);
        scene.bloom.set_strength(0.07);
        scene.bloom.set_radius(0.008);
        scene.tone_mapping.set_exposure(1.16);

        let stone_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.14, 0.14, 0.16, 1.0))
                .with_roughness(0.72)
                .with_metalness(0.08),
        );
        let metal_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.20, 0.22, 0.24, 1.0))
                .with_roughness(0.24)
                .with_metalness(0.78),
        );
        let emissive_material = engine.assets.materials.add(
            PhysicalMaterial::new(Vec4::new(0.08, 0.10, 0.14, 1.0))
                .with_emissive(Vec3::new(0.22, 0.92, 1.0), 3.2)
                .with_roughness(0.18)
                .with_metalness(0.16),
        );

        let platform = scene.spawn_box(7.8, 0.42, 3.0, stone_material, &engine.assets);
        scene
            .node(&platform)
            .set_position(0.0, -0.38, -1.3)
            .set_shadows(true, true);

        for &(x, y, z, sx, sy, sz) in &[
            (-2.8, 1.2, -1.5, 0.48, 2.6, 0.48),
            (2.8, 1.2, -1.5, 0.48, 2.6, 0.48),
            (-1.6, 0.6, -0.2, 0.36, 1.4, 0.36),
            (1.6, 0.6, -0.2, 0.36, 1.4, 0.36),
        ] {
            let column = scene.spawn_box(sx, sy, sz, stone_material, &engine.assets);
            scene
                .node(&column)
                .set_position(x, y, z)
                .set_shadows(true, true);
        }

        let beam = scene.spawn_box(5.8, 0.18, 0.28, metal_material, &engine.assets);
        scene
            .node(&beam)
            .set_position(0.0, 2.28, -1.5)
            .set_shadows(true, true);

        let beacon_ring = scene.spawn_torus(1.2, 0.12, metal_material, &engine.assets);
        scene
            .node(&beacon_ring)
            .set_position(0.0, 1.45, 0.0)
            .set_rotation(Quat::from_rotation_x(FRAC_PI_2 * 0.35))
            .set_shadows(true, true);

        let beacon_core = scene.spawn_sphere(0.44, emissive_material, &engine.assets);
        scene
            .node(&beacon_core)
            .set_position(0.0, 1.45, 0.0)
            .set_shadows(false, false);

        let mut shard_nodes = Vec::new();
        for index in 0..5 {
            let shard = scene.spawn_box(0.22, 1.1, 0.22, emissive_material, &engine.assets);
            scene.node(&shard).set_shadows(false, false);
            shard_nodes.push(shard);
            let angle = index as f32 * 1.2566371;
            scene
                .node(&shard)
                .set_position(angle.cos() * 1.9, 1.4, angle.sin() * 1.9);
        }

        let mut sun = Light::new_directional(Vec3::new(1.0, 0.86, 0.72), 2.4);
        sun.cast_shadows = true;
        if let Some(shadow) = sun.shadow.as_mut() {
            shadow.max_shadow_distance = 28.0;
        }
        let sun = scene.add_light(sun);
        scene
            .node(&sun)
            .set_position(-8.0, 9.0, 5.0)
            .look_at(Vec3::new(0.0, 0.8, 0.0));

        let accent_light = scene.add_light(Light::new_point(Vec3::new(0.18, 0.92, 1.0), 2.4, 18.0));
        scene.node(&accent_light).set_position(0.0, 2.4, 2.8);

        let camera = scene.add_camera(Camera::new_perspective(45.0, 16.0 / 9.0, 0.1));
        scene
            .node(&camera)
            .set_position(0.0, 2.2, 8.8)
            .look_at(Vec3::new(0.0, 1.0, -0.8));
        scene.active_camera = Some(camera);

        Self {
            controls: OrbitControls::new(Vec3::new(0.0, 2.2, 8.8), Vec3::new(0.0, 1.0, -0.8)),
            fps_counter: FpsCounter::new(),
            ocean,
            ui_pass,
            beacon_ring,
            beacon_core,
            accent_light,
            shard_nodes,
            time: 0.0,
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
        self.time += frame.dt;
        self.ocean.advance_time(frame.dt);

        let winit_window = window
            .as_any()
            .downcast_ref::<winit::window::Window>()
            .expect("Expected winit window backend");
        self.ui_pass.begin_frame(winit_window);

        let egui_ctx = self.ui_pass.context().clone();
        egui::Window::new("Ocean Composite")
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-20.0, 20.0))
            .resizable(false)
            .show(&egui_ctx, |ui| {
                ui.label("Depth-aware ocean behind the scene");
                ui.separator();
                self.ocean.ui(ui);
            });

        self.ui_pass.end_frame(winit_window);

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        scene.node(&self.beacon_ring).set_rotation(
            Quat::from_rotation_x(FRAC_PI_2 * 0.35)
                * Quat::from_rotation_y(self.time * 0.36)
                * Quat::from_rotation_z(self.time * 0.18),
        );
        scene
            .node(&self.beacon_core)
            .set_position(0.0, 1.45 + (self.time * 1.6).sin() * 0.12, 0.0)
            .set_rotation(Quat::from_rotation_y(-self.time * 0.8));

        for (index, shard) in self.shard_nodes.iter().enumerate() {
            let orbit = self.time * (0.58 + index as f32 * 0.05) + index as f32 * 1.2566371;
            scene
                .node(shard)
                .set_position(
                    orbit.cos() * 1.9,
                    1.45 + (self.time * 2.0 + index as f32).sin() * 0.24,
                    orbit.sin() * 1.9,
                )
                .set_rotation(
                    Quat::from_rotation_y(-orbit * 1.2)
                        * Quat::from_rotation_x(self.time * 0.9 + index as f32 * 0.3),
                );
        }

        if let Some(node) = scene.get_node_mut(self.accent_light) {
            node.transform.position = Vec3::new(
                self.time.cos() * 2.8,
                2.4 + (self.time * 1.8).sin() * 0.4,
                1.8 + self.time.sin() * 2.0,
            );
        }

        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        }

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Ocean Composite Scene | FPS: {:.1}", fps));
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
                let new_surface = rdg.add_pass("Ocean_Composite_UI", |builder| {
                    let out = builder
                        .mutate_texture(blackboard.surface_out, "Surface_With_OceanComposite_UI");
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
        .with_title("Ocean Composite Scene")
        .with_settings(RendererSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<OceanCompositeSceneDemo>()
}
