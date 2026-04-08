use myth::prelude::*;
use myth::resources::Key;
use myth::utils::FpsCounter;

/// Procedural Sky Demo
///
/// Demonstrates the Hillaire 2020 atmosphere system with dynamic IBL.
/// The sun orbits automatically; press Space to pause/resume.
struct ProceduralSkyDemo {
    cam_node_id: NodeHandle,
    light_node_id: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
    time: f32,
    paused: bool,
}

impl AppHandler for ProceduralSkyDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();

        // Procedural sky with golden-hour defaults
        scene
            .background
            .set_mode(BackgroundMode::procedural());

        // Directional light aligned with initial sun direction
        let sun_dir = scene
            .background
            .procedural_sky_params()
            .map(|p| p.sun_direction)
            .unwrap_or(Vec3::new(0.0, 0.2, -1.0).normalize());

        let light = Light::new_directional(Vec3::new(1.0, 0.95, 0.8), 3.0);
        let light_node_id = scene.add_light(light);
        if let Some(node) = scene.get_node_mut(light_node_id) {
            node.transform.position = sun_dir * 10.0;
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        // Load the DamagedHelmet model as a reference object
        let gltf_path =
            std::path::Path::new("examples/assets/DamagedHelmet/glTF/DamagedHelmet.gltf");
        let prefab =
            GltfLoader::load(gltf_path, engine.assets.clone()).expect("Failed to load glTF model");
        let root = scene.instantiate(&prefab);
        scene.node(&root).set_position(0.0, 0.0, 0.0);

        // Camera
        let cam_node_id = scene.add_camera(Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1));
        scene
            .node(&cam_node_id)
            .set_position(0.0, 0.5, 4.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam_node_id);

        Self {
            cam_node_id,
            light_node_id,
            controls: OrbitControls::new(Vec3::new(0.0, 0.5, 4.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
            time: 0.0,
            paused: false,
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        // Toggle pause with Space
        if engine.input.get_key_down(Key::Space) {
            self.paused = !self.paused;
        }

        // Animate the sun direction around a circular arc
        if !self.paused {
            self.time += frame.dt * 0.15;
        }

        let elevation = self.time.sin();
        let azimuth = self.time * 0.7;
        let sun_dir = Vec3::new(azimuth.cos() * elevation.cos(), elevation, azimuth.sin() * elevation.cos()).normalize();

        if let Some(params) = scene.background.procedural_sky_params_mut() {
            params.set_sun_direction(sun_dir);
        }

        // Keep the directional light pointing opposite the sun direction
        if let Some(node) = scene.get_node_mut(self.light_node_id) {
            node.transform.position = sun_dir * 10.0;
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        // Orbit camera
        if let Some(cam_node) = scene.get_node_mut(self.cam_node_id) {
            self.controls
                .update(&mut cam_node.transform, &engine.input, 45.0, frame.dt);
        }

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!(
                "Procedural Sky Demo - FPS: {:.0}{}",
                fps,
                if self.paused { " [PAUSED]" } else { "" }
            ));
        }
    }
}

fn main() -> myth::Result<()> {
    env_logger::init();
    App::new()
        .with_settings(RendererSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<ProceduralSkyDemo>()
}
