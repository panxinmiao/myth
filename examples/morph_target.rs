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

/// Morph Target (变形目标) 动画示例
struct MorphTargetDemo {
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
            three::ColorSpace::Srgb,
            true
        ).expect("Failed to load environment map");

        let env_texture = engine.assets.textures.get(env_texture_handle).unwrap();

        scene.environment.set_env_map(Some((env_texture_handle.into(), &env_texture)));

        // 2. 加载 glTF 模型 (带 Morph Target)
        let gltf_path = std::path::Path::new("examples/assets/facecap.glb");
        println!("Loading glTF model from: {}", gltf_path.display());
        
        let prefab = GltfLoader::load(
            gltf_path,
            &engine.assets
        ).expect("Failed to load glTF model");
        let gltf_node = scene.instantiate(&prefab);

        println!("Successfully loaded root node: {:?}", gltf_node);

        //  查询动画列表
        if let Some(mixer) = scene.animation_mixers.get_mut(gltf_node) {
            println!("Loaded animations:");

            let animations = mixer.list_animations();

            for anim_name in &animations {
                println!(" - {}", anim_name);
            }
            mixer.play("Key|Take 001|BaseLayer");

        }


        // 输出 Mesh Morph Target 信息
        for (node_handle, mesh) in scene.meshes.iter() {
            if let Some(geometry) = engine.assets.geometries.get(mesh.geometry) {
                if geometry.has_morph_targets() {
                    println!("Node {:?} has mesh with {} morph targets, {} vertices per target",
                        node_handle,
                        geometry.morph_target_count,
                        geometry.morph_vertex_count
                    );
                }
            }
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
            cam_node_id,
            controls: OrbitControls::new(Vec3::new(0.0, 0.0, 4.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
        }
    }

    fn update(&mut self, engine: &mut ThreeEngine, window: &Arc<Window>, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else{
            return;
        };

        if let Some(cam_node) = scene.get_node_mut(self.cam_node_id) {
            self.controls.update(&mut cam_node.transform, &engine.input, 45.0, frame.dt);
        }

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
