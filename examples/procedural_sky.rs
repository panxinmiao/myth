use myth::prelude::*;
use myth::utils::FpsCounter;

/// Procedural Sky Demo
///
/// Demonstrates the Hillaire 2020 atmosphere system with a SceneLogic-driven
/// day/night cycle, moon light, and hybrid star field.
struct ProceduralSkyDemo {
    cam_node_id: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
}

impl AppHandler for ProceduralSkyDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();

        let starbox = engine.assets.load_texture(
            "examples/assets/envs/Milky_Way_panorama.jpg",
            ColorSpace::Srgb,
            true,
        );

        let mut sky = ProceduralSkyParams::sunset();
        sky.set_starbox_texture(Some(starbox.into()));
        sky.set_star_intensity(0.2);
        sky.set_moon_intensity(0.35);
        scene
            .background
            .set_mode(BackgroundMode::procedural_with(sky));

        // Directional light aligned with initial sun direction
        let sun_dir = scene
            .background
            .procedural_sky_params()
            .map(|p| p.sun_direction)
            .unwrap_or(Vec3::new(0.0, 0.2, -1.0).normalize());
        let moon_dir = scene
            .background
            .procedural_sky_params()
            .map(|p| p.moon_direction)
            .unwrap_or(-sun_dir);

        let mut sun_light = Light::new_directional(Vec3::new(1.0, 0.95, 0.8), 3.0);
        sun_light.cast_shadows = true;
        let sun_light_node = scene.add_light(sun_light);
        if let Some(node) = scene.get_node_mut(sun_light_node) {
            node.transform.position = sun_dir * 10.0;
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        let moon_light = Light::new_directional(Vec3::new(0.62, 0.72, 1.0), 0.08);
        let moon_light_node = scene.add_light(moon_light);
        if let Some(node) = scene.get_node_mut(moon_light_node) {
            node.transform.position = moon_dir * 10.0;
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        scene.add_logic(
            DayNightCycle::new(18.5, 35.0)
                .with_sun(sun_light_node)
                .with_moon(moon_light_node)
                .with_time_speed(0.35)
                .with_auto_tick(false),
        );

        // Load the DamagedHelmet model as a reference object
        let gltf_path =
            std::path::Path::new("examples/assets/DamagedHelmet/glTF/DamagedHelmet.gltf");
        let prefab =
            GltfLoader::load(gltf_path, engine.assets.clone()).expect("Failed to load glTF model");
        let root = scene.instantiate(&prefab);
        scene.node(&root).set_position(0.0, 0.0, 0.0);

        scene.bloom.set_enabled(true);
        scene.bloom.set_strength(0.02);
        scene.bloom.set_radius(0.005);
        scene.bloom.set_karis_average(true);

        scene.tone_mapping
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

        Self {
            cam_node_id,
            controls: OrbitControls::new(Vec3::new(0.0, 0.5, 4.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        // Orbit camera
        if let Some(cam_node) = scene.get_node_mut(self.cam_node_id) {
            self.controls
                .update(&mut cam_node.transform, &engine.input, 45.0, frame.dt);
        }

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Procedural Sky Day/Night - FPS: {:.0}", fps));
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
