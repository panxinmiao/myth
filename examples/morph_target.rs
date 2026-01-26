use std::sync::Arc;
use glam::Vec3;
use three::app::winit::{App, AppHandler};
use three::engine::FrameState;
use three::scene::{Camera, NodeHandle, light};
use three::{OrbitControls, ThreeEngine};
use three::utils::fps_counter::FpsCounter;
use three::assets::GltfLoader;
use three::{AnimationMixer, AnimationAction, Binder};
use three::animation::binding::TargetPath;
use three::renderer::settings::RenderSettings;
use winit::window::Window;

/// Morph Target (变形目标) 动画示例
struct MorphTargetDemo {
    mixer: AnimationMixer,
    cam_node_id: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
}

impl AppHandler for MorphTargetDemo {
    fn init(engine: &mut ThreeEngine, _window: &Arc<Window>) -> Self {
        let scene = engine.scene_manager.create_active();
        // 1. 添加灯光和环境
        let light = light::Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 2.0);
        scene.add_light(light);
        scene.environment.set_ambient_color(Vec3::splat(0.3));
        // 加载环境贴图
        let env_texture_handle = engine.assets.load_cube_texture_from_files(
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

        let env_texture = engine.assets.get_texture_mut(env_texture_handle).unwrap();
        env_texture.generate_mipmaps = true;
        scene.environment.set_env_map(Some((env_texture_handle.into(), &env_texture)));

        // 2. 加载 glTF 模型 (带 Morph Target)
        let gltf_path = std::path::Path::new("examples/assets/facecap.glb");
        println!("Loading glTF model from: {}", gltf_path.display());
        
        let (loaded_nodes, animations) = GltfLoader::load(
            gltf_path,
            &mut engine.assets,
            scene
        ).expect("Failed to load glTF model");

        println!("Successfully loaded {} root nodes", loaded_nodes.len());
        println!("Total animations found: {}", animations.len());

        // 输出动画信息
        for (i, anim) in animations.iter().enumerate() {
            println!("Animation {}: '{}' with {} tracks", i, anim.name, anim.tracks.len());
            
            let morph_tracks: Vec<_> = anim.tracks.iter()
                .filter(|t| t.meta.target == TargetPath::Weights)
                .collect();
            
            if !morph_tracks.is_empty() {
                println!("  - Found {} morph target weight tracks", morph_tracks.len());
            }
        }

        // 输出 Mesh Morph Target 信息
        for (node_handle, mesh) in scene.meshes.iter() {
            if let Some(geometry) = engine.assets.get_geometry(mesh.geometry) {
                if geometry.has_morph_targets() {
                    println!("Node {:?} has mesh with {} morph targets, {} vertices per target",
                        node_handle,
                        geometry.morph_target_count,
                        geometry.morph_vertex_count
                    );
                }
            }
        }

        // 3. 设置动画混合器
        let mut mixer = AnimationMixer::new();
        
        if let Some(clip) = animations.into_iter().next() {
            println!("Playing animation: {} (duration: {:.2}s)", clip.name, clip.duration);
            
            let root_node = loaded_nodes.first().copied().unwrap();
            let clip = Arc::new(clip);
            let bindings = Binder::bind(scene, root_node, &clip);
            
            println!("Created {} bindings", bindings.len());
            
            let mut action = AnimationAction::new(clip);
            action.bindings = bindings;
            mixer.add_action(action);
        }

        // 4. 设置相机
        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let cam_node_id = scene.add_camera(camera);
        if let Some(node) = scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 0.0, 4.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }
        scene.active_camera = Some(cam_node_id);

        Self {
            mixer,
            cam_node_id,
            controls: OrbitControls::new(Vec3::new(0.0, 0.0, 4.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
        }
    }

    fn update(&mut self, engine: &mut ThreeEngine, window: &Arc<Window>, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else{
            return;
        };
        // 更新动画
        self.mixer.update(frame.dt, scene);
        // 轨道控制器
        if let Some(cam_node) = scene.get_node_mut(self.cam_node_id) {
            self.controls.update(&mut cam_node.transform, &engine.input, 45.0, frame.dt);
        }

        // FPS 显示
        if let Some(fps) = self.fps_counter.update() {
            window.set_title(&format!("Morph Target Demo - FPS: {:.1}", fps));
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    App::new()
        .with_settings(RenderSettings { vsync: false, ..Default::default() })
        .run::<MorphTargetDemo>()
}
