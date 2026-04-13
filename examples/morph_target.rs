//! [gallery]
//! name = "Morph Target"
//! category = "Animation"
//! description = "Morph target playback demo with animation mixer inspection."
//! order = 520
//!

use myth::prelude::*;
use myth::utils::FpsCounter;

#[cfg(not(target_arch = "wasm32"))]
const ASSET_PATH: &str = "examples/assets/";
#[cfg(target_arch = "wasm32")]
const ASSET_PATH: &str = match option_env!("MYTH_ASSET_PATH") {
    Some(path) => path,
    None => "assets/",
};

/// Morph Target
struct MorphTargetDemo {
    cam_node_id: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
    model_prefab: PrefabHandle,
    model_loaded: bool,
}

impl AppHandler for MorphTargetDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();

        let light = Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 2.0);
        scene.add_light(light);
        scene.environment.set_ambient_light(Vec3::splat(0.01));

        let env_texture_handle = engine.assets.load_texture(
            format!("{}envs/royal_esplanade_2k.hdr.jpg", ASSET_PATH),
            ColorSpace::Srgb,
            false,
        );

        scene.environment.set_env_map(Some(env_texture_handle));

        let model_source = format!("{}facecap.glb", ASSET_PATH);
        println!("Loading glTF model from: {model_source}");
        let model_prefab = engine.assets.load_gltf(model_source);

        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let cam_node_id = scene.add_camera(camera);
        if let Some(node) = scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 0.0, 4.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }
        scene.active_camera = Some(cam_node_id);

        Self {
            cam_node_id,
            controls: OrbitControls::new(Vec3::new(0.0, 0.0, 4.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
            model_prefab,
            model_loaded: false,
        }
    }

    fn update(&mut self, engine: &mut Engine, window: &dyn Window, frame: &FrameState) {
        let assets = engine.assets.clone();
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        if !self.model_loaded {
            if let Some(prefab) = assets.prefabs.get(self.model_prefab) {
                let gltf_node = scene.instantiate(prefab.as_ref());

                println!("Successfully loaded root node: {:?}", gltf_node);

                if let Some(mixer) = scene.animation_mixers.get_mut(gltf_node) {
                    println!("Loaded animations:");

                    let animations = mixer.list_animations();

                    for anim_name in &animations {
                        println!(" - {}", anim_name);
                    }
                    mixer.play("Key|Take 001|BaseLayer");
                }

                for (node_handle, mesh) in scene.meshes.iter() {
                    if let Some(geometry) = assets.geometries.get(mesh.geometry)
                        && geometry.has_morph_targets()
                    {
                        println!(
                            "Node {:?} has mesh with {} morph targets, {} vertices per target",
                            node_handle,
                            geometry.morph_target_count(),
                            geometry.morph_vertex_count()
                        );
                    }
                }

                self.model_loaded = true;
            } else if let Some(err) = assets.prefabs.get_error(self.model_prefab) {
                eprintln!("Failed to load morph target demo model: {err}");
                self.model_loaded = true;
            }
        }

        if let Some(cam_node) = scene.get_node_mut(self.cam_node_id) {
            self.controls
                .update(&mut cam_node.transform, &engine.input, 45.0, frame.dt);
        }

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Morph Target Demo - FPS: {:.1}", fps));
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
        .run::<MorphTargetDemo>()
}
