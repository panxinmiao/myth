use std::sync::Arc;

use myth::prelude::*;
use myth::utils::FpsCounter;
use winit::window::Window;

struct HttpGltfExample {
    cam_node_id: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
    loaded: bool,
    _model_root: Option<NodeHandle>,
}

impl AppHandler for HttpGltfExample {
    fn init(engine: &mut MythEngine, _window: &Arc<Window>) -> Self {
        let env_texture_handle = engine
            .assets
            .load_cube_texture(
                [
                    "examples/assets/Park2/posx.jpg",
                    "examples/assets/Park2/negx.jpg",
                    "examples/assets/Park2/posy.jpg",
                    "examples/assets/Park2/negy.jpg",
                    "examples/assets/Park2/posz.jpg",
                    "examples/assets/Park2/negz.jpg",
                ],
                ColorSpace::Srgb,
                true,
            )
            .expect("Failed to load environment map");

        engine.scene_manager.create_active();
        let scene = engine.scene_manager.active_scene_mut().unwrap();

        scene.environment.set_env_map(Some(env_texture_handle));

        let light = Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0);
        scene.add_light(light);

        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let cam_node_id = scene.add_camera(camera);

        if let Some(node) = scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 0.5, 2.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        scene.active_camera = Some(cam_node_id);

        Self {
            cam_node_id,
            controls: OrbitControls::new(Vec3::new(0.0, 0.5, 2.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
            loaded: false,
            _model_root: None,
        }
    }

    fn update(&mut self, engine: &mut MythEngine, window: &Arc<Window>, frame: &FrameState) {
        if !self.loaded {
            self.loaded = true;

            println!("Loading glTF model from network...");
            let url = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/refs/heads/main/Models/ABeautifulGame/glTF/ABeautifulGame.gltf";

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
            window.set_title(&format!("HTTP glTF Loading - FPS: {:.0}", fps));
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("=== HTTP glTF Loading Example ===");
    println!("This example demonstrates loading glTF models from HTTP URLs.");
    println!("Loading: ABeautifulGame from Khronos glTF-Sample-Assets");

    App::new()
        .with_settings(RenderSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<HttpGltfExample>()
}
