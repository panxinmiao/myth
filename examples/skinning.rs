use std::sync::Arc;
use std::path::Path; // 引入 Path
use std::env; // 引入 env

use glam::Vec3;
use three::app::{App, AppContext, AppHandler};
use three::scene::{Camera, light};
use three::OrbitControls;
use three::utils::fps_counter::FpsCounter;
use three::assets::GltfLoader;
use three::{AnimationMixer, AnimationAction, Binder};
use three::renderer::settings::RenderSettings;

/// 骨骼动画示例
/// 
/// 用法: skinning [path_to_model.glb]
struct SkinningDemo {
    mixer: AnimationMixer,
    controls: OrbitControls,
    fps_counter: FpsCounter,
}

impl AppHandler for SkinningDemo {
    fn init(ctx: &mut AppContext) -> Self {
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
        // 注意：实际项目中建议把 assets 路径也做成可配置的，或者打包在 binary 旁
        let env_texture_handle = ctx.assets.load_cube_texture_from_files(
            [
                "examples/assets/Park2/posx.jpg",
                "examples/assets/Park2/negx.jpg",
                "examples/assets/Park2/posy.jpg",
                "examples/assets/Park2/negy.jpg",
                "examples/assets/Park2/posz.jpg",
                "examples/assets/Park2/negz.jpg",
            ],
            three::ColorSpace::Srgb
        ).expect("Failed to load environment map");

        let env_texture = ctx.assets.get_texture_mut(env_texture_handle).unwrap();
        env_texture.generate_mipmaps = true;
        ctx.scene.environment.set_env_map(Some((env_texture_handle, &env_texture)));

        // === 3. 添加灯光 ===
        let light = light::Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0);
        ctx.scene.add_light(light);

        // === 4. 加载 glTF 模型 ===
        println!("Loading glTF model from: {:?}", gltf_path);
        
        // 这里加一个简单的错误处理，防止路径错误直接崩溃不好调试
        let (loaded_nodes, animations) = match GltfLoader::load(
            gltf_path,
            ctx.assets,
            ctx.scene
        ) {
            Ok(res) => res,
            Err(e) => {
                eprintln!("Error loading model '{}': {}", gltf_path.display(), e);
                // 加载失败时返回一个空场景或 fallback，这里简单起见 panic
                panic!("Failed to load model");
            }
        };

        println!("Successfully loaded {} root nodes", loaded_nodes.len());
        println!("Total animations found: {}", animations.len());

        // === 5. 设置动画混合器 ===
        let mut mixer = AnimationMixer::new();
        
        if let Some(clip) = animations.into_iter().next() {
            println!("Playing animation: {} (duration: {:.2}s)", clip.name, clip.duration);
            
            // 简单的假设：动画应用在第一个加载的根节点上
            if let Some(&root_node) = loaded_nodes.first() {
                let clip = Arc::new(clip);
                let bindings = Binder::bind(ctx.scene, root_node, &clip);
                
                if bindings.is_empty() {
                    println!("Warning: No bindings created for animation. Node names mismatch?");
                } else {
                    println!("Created {} bindings", bindings.len());
                    let mut action = AnimationAction::new(clip);
                    action.bindings = bindings;
                    mixer.add_action(action);
                }
            }
        }

        // === 6. 设置相机 ===
        // TODO: 真正的 Viewer 通常会根据模型包围盒(AABB)自动计算相机位置
        // 这里暂时用固定位置
        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1, 100.0);
        let cam_node_id = ctx.scene.add_camera(camera);
        
        if let Some(node) = ctx.scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 1.5, 4.0); // 稍微抬高一点视角
            node.transform.look_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
        }
        ctx.scene.active_camera = Some(cam_node_id);

        Self {
            mixer,
            // 这里的 center 最好也是模型的中心
            controls: OrbitControls::new(Vec3::new(0.0, 1.5, 4.0), Vec3::new(0.0, 1.0, 0.0)),
            fps_counter: FpsCounter::new(),
        }
    }

    fn update(&mut self, ctx: &mut AppContext) {
        self.mixer.update(ctx.dt, ctx.scene);

        if let Some((transform, camera)) = ctx.scene.query_main_camera_bundle() {
            self.controls.update(transform, ctx.input, camera.fov.to_degrees(), ctx.dt);
        }

        if let Some(fps) = self.fps_counter.update() {
            ctx.window.set_title(&format!("Skinning Animation | FPS: {:.2}", fps));
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    App::new()
        // 开启 VSync 可以避免 GPU 满载啸叫，但在测试性能时可以关掉
        .with_settings(RenderSettings { vsync: false, ..Default::default() }) 
        .run::<SkinningDemo>()
}