//! Skybox / Background Demo
//!
//! Demonstrates all background modes and both rendering paths (HDR / LDR).
//!
//! # Controls
//!
//! | Key | Action |
//! |-----|--------|
//! | `1` | Solid color background (hardware clear, no skybox pass) |
//! | `2` | Gradient background (procedural sky) |
//! | `3` | Equirectangular HDR panorama as skybox |
//! | `H` | Toggle HDR / LDR rendering path |
//! | Mouse drag | Orbit camera |
//! | Scroll | Zoom |

use myth::prelude::*;
use myth::resources::Key;
use myth::utils::FpsCounter;

/// Which demo mode is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DemoMode {
    SolidColor,
    Gradient,
    Equirectangular,
}

impl DemoMode {
    fn label(self) -> &'static str {
        match self {
            Self::SolidColor => "Solid Color",
            Self::Gradient => "Gradient",
            Self::Equirectangular => "Equirectangular HDR",
        }
    }
}

struct SkyboxDemo {
    cam_node: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,

    /// Current background mode
    mode: DemoMode,
    /// Whether HDR path is active
    hdr_enabled: bool,
    /// HDR environment texture handle (reused for equirectangular skybox)
    env_texture: TextureHandle,
}

impl SkyboxDemo {
    /// Applies the current `DemoMode` to the scene background.
    fn apply_mode(&self, scene: &mut Scene) {
        scene.background = match self.mode {
            DemoMode::SolidColor => BackgroundMode::color(0.1, 0.1, 0.15),
            DemoMode::Gradient => BackgroundMode::gradient(
                Vec4::new(0.05, 0.05, 0.25, 1.0), // deep blue top
                Vec4::new(0.7, 0.45, 0.2, 1.0),   // warm orange bottom
            ),
            DemoMode::Equirectangular => {
                BackgroundMode::equirectangular(self.env_texture, 1.0)
            }
        };
    }

    fn print_help() {
        println!("╔══════════════════════════════════════╗");
        println!("║          Skybox Demo Controls        ║");
        println!("╠══════════════════════════════════════╣");
        println!("║  1 — Solid color (hardware clear)    ║");
        println!("║  2 — Gradient (procedural sky)       ║");
        println!("║  3 — Equirectangular HDR panorama    ║");
        println!("║  H — Toggle HDR / LDR path           ║");
        println!("║  Mouse drag / Scroll — Orbit / Zoom  ║");
        println!("╚══════════════════════════════════════╝");
    }
}

impl AppHandler for SkyboxDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        // --- Load HDR environment texture (used for both IBL and equirectangular skybox) ---
        let env_texture = engine
            .assets
            .load_hdr_texture("examples/assets/blouberg_sunrise_2_1k.hdr")
            .expect("Failed to load HDR environment map");

        // --- Scene setup ---
        let scene = engine.scene_manager.create_active();
        scene.add_light(Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0));

        // Environment map for image-based lighting (IBL)
        scene.environment.set_env_map(Some(env_texture));
        scene.environment.set_intensity(1.0);

        // Default to gradient background
        let mode = DemoMode::Gradient;

        // --- Load reference model ---
        let gltf_path =
            std::path::Path::new("examples/assets/DamagedHelmet/glTF/DamagedHelmet.gltf");
        let prefab =
            GltfLoader::load(gltf_path, engine.assets.clone()).expect("Failed to load glTF model");
        let node = scene.instantiate(&prefab);
        scene.node(&node).set_scale(1.0).set_position(0.0, 0.0, 0.0);

        // --- Camera ---
        let cam_node = scene.add_camera(Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1));
        scene
            .node(&cam_node)
            .set_position(0.0, 0.0, 3.5)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam_node);

        let hdr_enabled = engine.renderer.is_hdr_enabled();

        Self::print_help();

        let demo = Self {
            cam_node,
            controls: OrbitControls::new(Vec3::new(0.0, 0.0, 3.5), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
            mode,
            hdr_enabled,
            env_texture,
        };

        // Apply initial mode
        demo.apply_mode(scene);

        demo
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        // --- Mode switching ---
        let mut mode_changed = false;

        if engine.input.get_key_down(Key::Key1) && self.mode != DemoMode::SolidColor {
            self.mode = DemoMode::SolidColor;
            mode_changed = true;
        }
        if engine.input.get_key_down(Key::Key2) && self.mode != DemoMode::Gradient {
            self.mode = DemoMode::Gradient;
            mode_changed = true;
        }
        if engine.input.get_key_down(Key::Key3) && self.mode != DemoMode::Equirectangular {
            self.mode = DemoMode::Equirectangular;
            mode_changed = true;
        }

        if mode_changed {
            self.apply_mode(scene);
            println!("[Mode] → {}", self.mode.label());
        }

        // --- HDR / LDR toggle ---
        if engine.input.get_key_down(Key::H) {
            self.hdr_enabled = !self.hdr_enabled;
            engine.renderer.set_hdr_enabled(self.hdr_enabled);
            let path = if self.hdr_enabled { "HDR" } else { "LDR" };
            println!("[Path] → {path}");
        }

        // --- Orbit camera ---
        if let Some(cam_node) = scene.get_node_mut(self.cam_node) {
            self.controls
                .update(&mut cam_node.transform, &engine.input, 45.0, frame.dt);
        }

        // --- Title bar ---
        if let Some(fps) = self.fps_counter.update() {
            let path = if self.hdr_enabled { "HDR" } else { "LDR" };
            window.set_title(&format!(
                "Skybox Demo — {} | {} | FPS: {fps:.0}",
                self.mode.label(),
                path,
            ));
        }
    }
}

fn main() -> myth::Result<()> {
    env_logger::init();
    App::new()
        .with_settings(RenderSettings {
            vsync: false,
            enable_hdr: false,
            ..Default::default()
        })
        .run::<SkyboxDemo>()
}
