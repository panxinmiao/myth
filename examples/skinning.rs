use std::sync::Arc;
use std::path::Path; // 引入 Path
use std::env; // 引入 env

use glam::Vec3;
use three::app::winit::{App, AppHandler};
use three::scene::{Camera, light};
use three::OrbitControls;
use three::utils::fps_counter::FpsCounter;
use three::assets::GltfLoader;
use three::renderer::settings::RenderSettings;
use three::engine::{FrameState, ThreeEngine};
use winit::window::Window;

/// 骨骼动画示例
/// 
/// 用法: skinning [path_to_model.glb]
struct SkinningDemo {
    controls: OrbitControls,
    fps_counter: FpsCounter,
}

impl AppHandler for SkinningDemo {
    fn init(engine: &mut ThreeEngine, _window: &Arc<Window>) -> Self {
        // === 1. 解析启动参数 ===
        let args: Vec<String> = env::args().collect();
        
        // 默认模型路径 (如果没有传参数，就用这个)
        let default_path = "examples/assets/Michelle.glb"; 
        
        let gltf_path_str = if args.len() > 1 {
            &args[1]
        } else {
            println!("Tip: You can pass a model path as an argument.");
            println!("Usage: cargo run --example skinning -- <path_to_gltf>");
            println!("No path provided, loading default: {}", default_path);
            default_path
        };

        let gltf_path = Path::new(gltf_path_str);

        // === 2. 加载环境贴图 (保持不变) ===
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

        let scene = engine.scene_manager.create_active();

        scene.environment.set_env_map(Some(env_texture_handle));

        // === 3. 添加灯光 ===
        let light = light::Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0);
        scene.add_light(light);
        // === 4. 加载 glTF 模型 ===
        println!("Loading glTF model from: {:?}", gltf_path);
        
        // 这里加一个简单的错误处理，防止路径错误直接崩溃不好调试
        let prefab = match GltfLoader::load(
            gltf_path,
            engine.assets.clone()
        ) {
            Ok(res) => res,
            Err(e) => {
                eprintln!("Error loading model '{}': {}", gltf_path.display(), e);
                // 加载失败时返回一个空场景或 fallback，这里简单起见 panic
                panic!("Failed to load model");
            }
        };
        let gltf_node = scene.instantiate(&prefab);

        println!("Successfully loaded root node: {:?}", gltf_node);

        //  查询动画列表
        if let Some(mixer) = scene.animation_mixers.get_mut(gltf_node) {
            println!("Loaded animations:");

            let animations = mixer.list_animations();

            for anim_name in &animations {
                println!(" - {}", anim_name);
            }

            mixer.play("SambaDance");

        }

        // === 5. 设置相机 ===
        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let cam_node_id = scene.add_camera(camera);
        
        if let Some(node) = scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 1.5, 4.0); // 稍微抬高一点视角
            node.transform.look_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
        }
        scene.active_camera = Some(cam_node_id);

        Self {
            controls: OrbitControls::new(Vec3::new(0.0, 1.5, 4.0), Vec3::new(0.0, 1.0, 0.0)),
            fps_counter: FpsCounter::new(),
        }
    }

    fn update(&mut self, engine: &mut ThreeEngine, window: &Arc<Window>, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else{
            return;
        };

        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls.update(transform, &engine.input, camera.fov, frame.dt);
        }

        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Skinning Animation | FPS: {:.2}", fps));
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    App::new()
        // 开启 VSync 可以避免 GPU 满载啸叫，但在测试性能时可以关掉
        .with_settings(RenderSettings { vsync: false, enable_hdr: false, ..Default::default() }) 
        .run::<SkinningDemo>()
}