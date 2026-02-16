use myth::prelude::*;
use myth::utils::FpsCounter;

/// HDR Environment Map Demo
struct HdrEnvDemo {
    cam_node_id: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
}

impl AppHandler for HdrEnvDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let env_texture_handle = engine
            .assets
            .load_hdr_texture("examples/assets/blouberg_sunrise_2_1k.hdr")
            .expect("Failed to load HDR environment map");

        engine.scene_manager.create_active();
        let scene = engine.scene_manager.active_scene_mut().unwrap();

        scene.environment.set_env_map(Some(env_texture_handle));
        scene.environment.set_intensity(1.0);

        let light = Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0);
        scene.add_light(light);

        let gltf_path =
            std::path::Path::new("examples/assets/DamagedHelmet/glTF/DamagedHelmet.gltf");
        println!("Loading glTF model from: {}", gltf_path.display());

        let prefab =
            GltfLoader::load(gltf_path, engine.assets.clone()).expect("Failed to load glTF model");
        let gltf_node = scene.instantiate(&prefab);

        println!("Successfully loaded root node: {:?}", gltf_node);

        if let Some(node) = scene.get_node_mut(gltf_node) {
            node.transform.scale = Vec3::splat(1.0);
            node.transform.position = Vec3::new(0.0, 0.0, 0.0);
        }

        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let cam_node_id = scene.add_camera(camera);

        if let Some(node) = scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 0.0, 3.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        scene.active_camera = Some(cam_node_id);

        println!("HDR environment map loaded successfully!");
        println!("The HDR image is automatically converted to a CubeMap for IBL rendering.");

        Self {
            cam_node_id,
            controls: OrbitControls::new(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        if let Some(cam_node) = scene.get_node_mut(self.cam_node_id) {
            self.controls
                .update(&mut cam_node.transform, &engine.input, 45.0, frame.dt);
        }

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("HDR Environment Demo - FPS: {:.0}", fps));
        }
    }
}

fn main() -> myth::Result<()> {
    env_logger::init();
    App::new()
        .with_settings(RenderSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<HdrEnvDemo>()
}
