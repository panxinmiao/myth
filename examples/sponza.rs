use myth::prelude::*;
use myth::utils::FpsCounter;

struct HttpGltfExample {
    cam_node_id: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
    loaded: bool,
    _model_root: Option<NodeHandle>,
}

impl AppHandler for HttpGltfExample {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let map_path = "examples/assets/royal_esplanade_2k.hdr.jpg";

        let env_texture_handle = engine
            .assets
            .load_texture(map_path, ColorSpace::Srgb, false)
            .expect("Failed to load environment map");

        engine.scene_manager.create_active();
        let scene = engine.scene_manager.active_scene_mut().unwrap();

        scene.environment.set_env_map(Some(env_texture_handle));
        // scene.ssao.set_enabled(false);

        let mut dir_light = Light::new_directional(Vec3::ONE, 5.0);
        dir_light.cast_shadows = true;
        if let Some(shadow) = dir_light.shadow.as_mut() {
            shadow.map_size = 2048;
        }
        let light_node = scene.add_light(dir_light);

        if let Some(node) = scene.get_node_mut(light_node) {
            node.transform.position = Vec3::new(2.0, 12.0, 6.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let cam_node_id = scene.add_camera(camera);

        if let Some(node) = scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(6.0, 4.0, 0.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        scene.active_camera = Some(cam_node_id);

        Self {
            cam_node_id,
            controls: OrbitControls::new(Vec3::new(6.0, 4.0, 0.0), Vec3::new(0.0, 2.0, 0.0)),
            fps_counter: FpsCounter::new(),
            loaded: false,
            _model_root: None,
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        if !self.loaded {
            self.loaded = true;

            println!("Loading glTF model from network...");
            let url = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/refs/heads/main/Models/Sponza/glTF/Sponza.gltf";

            let scene = engine.scene_manager.active_scene_mut().unwrap();

            match GltfLoader::load_sync(url, engine.assets.clone()) {
                Ok(prefab) => {
                    scene.instantiate(&prefab);
                    println!("Successfully loaded model from network!");
                }
                Err(e) => {
                    eprintln!("Failed to load model: {}", e);
                }
            }
        }

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        if let Some(cam_node) = scene.get_node_mut(self.cam_node_id) {
            self.controls
                .update(&mut cam_node.transform, &engine.input, 45.0, frame.dt);
        }

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Sponza Lighting Example - FPS: {:.0}", fps));
        }
    }
}

fn main() -> myth::Result<()> {
    env_logger::init();

    println!("=== Sponza Lighting Example ===");

    App::new()
        .with_settings(RendererSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<HttpGltfExample>()
}
