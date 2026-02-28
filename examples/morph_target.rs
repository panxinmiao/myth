use myth::prelude::*;
use myth::utils::FpsCounter;

/// Morph Target
struct MorphTargetDemo {
    cam_node_id: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
}

impl AppHandler for MorphTargetDemo {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let scene = engine.scene_manager.create_active();

        let light = Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 2.0);
        scene.add_light(light);
        scene.environment.set_ambient_light(Vec3::splat(0.01));

        let env_texture_handle = engine
            .assets
            .load_texture(
                "examples/assets/royal_esplanade_2k.hdr.jpg",
                ColorSpace::Srgb,
                false,
            )
            .expect("Failed to load environment map");

        scene.environment.set_env_map(Some(env_texture_handle));

        let gltf_path = std::path::Path::new("examples/assets/facecap.glb");
        println!("Loading glTF model from: {}", gltf_path.display());

        let prefab =
            GltfLoader::load(gltf_path, engine.assets.clone()).expect("Failed to load glTF model");
        let gltf_node = scene.instantiate(&prefab);

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
            if let Some(geometry) = engine.assets.geometries.get(mesh.geometry)
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
            window.set_title(&format!("Morph Target Demo - FPS: {:.1}", fps));
        }
    }
}

fn main() -> myth::Result<()> {
    env_logger::init();
    App::new()
        .with_settings(RendererSettings {
            path: RenderPath::BasicForward { msaa_samples: 1 },
            vsync: false,
            ..Default::default()
        })
        .run::<MorphTargetDemo>()
}
