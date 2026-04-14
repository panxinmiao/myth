//! [gallery]
//! name = "glTF Helmet"
//! category = "Scenes & glTF"
//! description = "Loads the DamagedHelmet sample and frames it with orbit controls."
//! order = 610
//!

use myth::prelude::*;
use myth_dev_utils::FpsCounter;

const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "examples/assets/",
};

/// glTF PBR Helmet Example
struct HelmetGltf {
    cam_node_id: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
    helmet_prefab: PrefabHandle,
    helmet_loaded: bool,
}

impl AppHandler for HelmetGltf {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let map_path = format!("{}envs/royal_esplanade_2k.hdr.jpg", ASSET_PATH);
        let env_texture_handle = engine
            .assets
            .load_texture(map_path, ColorSpace::Srgb, false);

        let scene = engine.scene_manager.create_active();
        scene.environment.set_env_map(Some(env_texture_handle));
        scene.add_light(Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0));

        let helmet_source = format!("{}DamagedHelmet/glTF/DamagedHelmet.gltf", ASSET_PATH);
        println!("Loading glTF model from: {helmet_source}");
        let helmet_prefab = engine.assets.load_gltf(helmet_source);

        let cam_node_id = scene.add_camera(Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1));
        scene
            .node(&cam_node_id)
            .set_position(0.0, 0.0, 3.0)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(cam_node_id);

        Self {
            cam_node_id,
            controls: OrbitControls::new(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
            helmet_prefab,
            helmet_loaded: false,
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        let assets = engine.assets.clone();
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        if !self.helmet_loaded {
            if let Some(prefab) = assets.prefabs.get(self.helmet_prefab) {
                let gltf_node = scene.instantiate(prefab.as_ref());
                println!("Successfully loaded root node: {:?}", gltf_node);
                self.helmet_loaded = true;
            } else if let Some(err) = assets.prefabs.get_error(self.helmet_prefab) {
                eprintln!("Failed to load helmet glTF: {err}");
                self.helmet_loaded = true;
            }
        }

        if let Some(cam_node) = scene.get_node_mut(self.cam_node_id) {
            self.controls
                .update(&mut cam_node.transform, &engine.input, 45.0, frame.dt);
        }

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("glTF PBR Demo - FPS: {:.0}", fps));
        }
    }
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new()
        .with_settings(RendererSettings {
            path: RenderPath::BasicForward,
            vsync: false,
            ..Default::default()
        })
        .run::<HelmetGltf>()
}
