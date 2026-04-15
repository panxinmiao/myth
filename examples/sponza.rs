//! [gallery]
//! name = "Sponza (SSAO & Shadows)"
//! category = "Scenes & glTF"
//! description = "Large streamed glTF scene used to stress-test lighting and traversal."
//! order = 620
//!

use myth::prelude::*;
use myth_dev_utils::FpsCounter;

const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "examples/assets/",
};

struct HttpGltfExample {
    cam_node_id: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
    model_prefab: PrefabHandle,
    model_resolved: bool,
    model_root: Option<NodeHandle>,
}

impl AppHandler for HttpGltfExample {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let map_path = format!("{}envs/blouberg_sunrise_2_1k.hdr", ASSET_PATH);

        let env_texture_handle = engine
            .assets
            .load_texture(map_path, ColorSpace::Srgb, false);

        engine.scene_manager.create_active();
        let scene = engine.scene_manager.active_scene_mut().unwrap();

        scene.environment.set_env_map(Some(env_texture_handle));
        scene
            .background
            .set_mode(BackgroundMode::equirectangular(env_texture_handle, 1.0));

        scene.ssao.enabled = true;

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

        let mut camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        camera.set_aa_mode(AntiAliasingMode::TAA_FXAA(
            TaaSettings::default(),
            FxaaSettings::default(),
        ));
        let cam_node_id = scene.add_camera(camera);

        if let Some(node) = scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(6.0, 4.0, 0.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        scene.active_camera = Some(cam_node_id);

        let url = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/refs/heads/main/Models/Sponza/glTF/Sponza.gltf";
        println!("Loading glTF model from network...");
        let model_prefab = engine.assets.load_gltf(url);

        Self {
            cam_node_id,
            controls: OrbitControls::new(Vec3::new(6.0, 4.0, 0.0), Vec3::new(0.0, 2.0, 0.0)),
            fps_counter: FpsCounter::new(),
            model_prefab,
            model_resolved: false,
            model_root: None,
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        let assets = engine.assets.clone();

        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        if !self.model_resolved {
            if let Some(prefab) = assets.prefabs.get(self.model_prefab) {
                let root = scene.instantiate(prefab.as_ref());
                self.model_root = Some(root);
                self.model_resolved = true;
                println!("Successfully loaded model from network!");
            } else if let Some(err) = assets.prefabs.get_error(self.model_prefab) {
                eprintln!("Failed to load model: {err}");
                self.model_resolved = true;
            }
        }

        if let Some(cam_node) = scene.get_node_mut(self.cam_node_id) {
            self.controls
                .update(&mut cam_node.transform, &engine.input, 45.0, frame.dt);
        }

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Sponza Lighting Example - FPS: {:.0}", fps));
        }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    println!("=== Sponza Lighting Example ===");

    App::new()
        .with_settings(RendererSettings {
            vsync: false,
            ..Default::default()
        })
        .run::<HttpGltfExample>()
}
