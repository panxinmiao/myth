use std::sync::Arc;

use glam::Vec3;
use three::app::winit::{App, AppHandler};
use three::engine::FrameState;
use three::scene::{Camera, NodeHandle, light};
use three::{OrbitControls, ThreeEngine};
use three::utils::fps_counter::FpsCounter;
use three::assets::GltfLoader;
use three::renderer::settings::RenderSettings;
use winit::window::Window;

/// glTF PBR 头盔示例
struct HelmetGltf {
    cam_node_id: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
}

impl AppHandler for HelmetGltf {
    fn init(engine: &mut ThreeEngine, _window: &Arc<Window>) -> Self {
        // 1. 加载环境贴图 (PBR 需要 IBL)
        let env_texture_handle = engine.assets.load_cube_texture(
            [
                "examples/assets/Park2/posx.jpg",
                "examples/assets/Park2/negx.jpg",
                "examples/assets/Park2/posy.jpg",
                "examples/assets/Park2/negy.jpg",
                "examples/assets/Park2/posz.jpg",
                "examples/assets/Park2/negz.jpg",
            ],
            three::ColorSpace::Srgb,
            true
        ).expect("Failed to load environment map");

        engine.scene_manager.create_active();
        let scene = engine.scene_manager.active_scene_mut().unwrap();

        scene.environment.set_env_map(Some(env_texture_handle));

        // 2. 添加灯光
        let light = light::Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0);
        scene.add_light(light);
        // 3. 加载 glTF 模型
        let gltf_path = std::path::Path::new("examples/assets/DamagedHelmet/glTF/DamagedHelmet.gltf");
        println!("Loading glTF model from: {}", gltf_path.display());
        
        let prefab = GltfLoader::load(
            gltf_path,
            engine.assets.clone()
        ).expect("Failed to load glTF model");
        let gltf_node = scene.instantiate(&prefab);

        println!("Successfully loaded root node: {:?}", gltf_node);

        // 4. 设置相机
        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let cam_node_id = scene.add_camera(camera);

        if let Some(node) = scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 0.0, 3.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        scene.active_camera = Some(cam_node_id);

        Self {
            cam_node_id,
            controls: OrbitControls::new(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
        }
    }

    fn update(&mut self, engine: &mut ThreeEngine, window: &Arc<Window>, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else{
            return;
        };
        // 轨道控制器
        if let Some(cam_node) = scene.get_node_mut(self.cam_node_id) {
            self.controls.update(&mut cam_node.transform, &engine.input, 45.0, frame.dt);
        }

        // FPS 显示
        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("glTF PBR Demo - FPS: {:.0}", fps));
        }
    }

}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    App::new()
        .with_settings(RenderSettings { vsync: false, ..Default::default() })
        .run::<HelmetGltf>()
}
