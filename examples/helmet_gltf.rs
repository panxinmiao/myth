use myth::prelude::*;
use myth::utils::FpsCounter;

/// glTF PBR Helmet Example
struct HelmetGltf {
    cam_node_id: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
}

impl AppHandler for HelmetGltf {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        let map_path = "examples/assets/royal_esplanade_2k.hdr.jpg";
        let env_texture_handle = engine
            .assets
            .load_texture(map_path, ColorSpace::Srgb, false)
            .expect("Failed to load environment map");

        let scene = engine.scene_manager.create_active();
        scene.environment.set_env_map(Some(env_texture_handle));
        scene.add_light(Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0));

        let gltf_path =
            std::path::Path::new("examples/assets/DamagedHelmet/glTF/DamagedHelmet.gltf");
        println!("Loading glTF model from: {}", gltf_path.display());

        let prefab =
            GltfLoader::load(gltf_path, engine.assets.clone()).expect("Failed to load glTF model");
        let gltf_node = scene.instantiate(&prefab);
        println!("Successfully loaded root node: {:?}", gltf_node);

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
            window.set_title(&format!("glTF PBR Demo - FPS: {:.0}", fps));
        }
    }
}

fn main() -> myth::Result<()> {
    env_logger::init();
    App::new()
        .with_settings(RendererSettings {
            path: RenderPath::BasicForward { msaa_samples: 1 },
            vsync: false,
            clear_color: wgpu::Color {
                r: 0.03,
                g: 0.03,
                b: 0.03,
                a: 1.0,
            },
            ..Default::default()
        })
        .run::<HelmetGltf>()
}
