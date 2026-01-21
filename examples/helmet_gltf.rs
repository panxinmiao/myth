use glam::Vec3;
use three::app::{App, AppContext, AppHandler};
use three::scene::{Camera, NodeIndex, light};
use three::OrbitControls;
use three::utils::fps_counter::FpsCounter;
use three::assets::GltfLoader;
use three::renderer::settings::RenderSettings;

/// glTF PBR 头盔示例
struct HelmetGltf {
    cam_node_id: NodeIndex,
    controls: OrbitControls,
    fps_counter: FpsCounter,
}

impl AppHandler for HelmetGltf {
    fn init(ctx: &mut AppContext) -> Self {
        // 1. 加载环境贴图 (PBR 需要 IBL)
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
        ctx.scene.environment.set_env_map(Some((env_texture_handle.into(), &env_texture)));

        // 2. 添加灯光
        let light = light::Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0);
        ctx.scene.add_light(light);

        // 3. 加载 glTF 模型
        let gltf_path = std::path::Path::new("examples/assets/DamagedHelmet/glTF/DamagedHelmet.gltf");
        println!("Loading glTF model from: {}", gltf_path.display());
        
        let (loaded_nodes, _animations) = GltfLoader::load(
            gltf_path,
            ctx.assets,
            ctx.scene
        ).expect("Failed to load glTF model");

        println!("Successfully loaded {} root nodes", loaded_nodes.len());

        // 4. 调整模型位置/缩放
        if let Some(&root_node_idx) = loaded_nodes.first() {
            if let Some(node) = ctx.scene.get_node_mut(root_node_idx) {
                node.transform.scale = Vec3::splat(1.0);
                node.transform.position = Vec3::new(0.0, 0.0, 0.0);
                println!("Model root node: {}", node.name);
            }
        }

        // 5. 设置相机
        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let cam_node_id = ctx.scene.add_camera(camera);

        if let Some(node) = ctx.scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 0.0, 3.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }

        ctx.scene.active_camera = Some(cam_node_id);

        Self {
            cam_node_id,
            controls: OrbitControls::new(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
        }
    }

    fn update(&mut self, ctx: &mut AppContext) {
        // 轨道控制器
        if let Some(cam_node) = ctx.scene.get_node_mut(self.cam_node_id) {
            self.controls.update(&mut cam_node.transform, ctx.input, 45.0, ctx.dt);
        }

        // FPS 显示
        if let Some(fps) = self.fps_counter.update() {
            ctx.window.set_title(&format!("glTF PBR Demo - FPS: {:.0}", fps));
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    App::new()
        .with_settings(RenderSettings { vsync: false, ..Default::default() })
        .run::<HelmetGltf>()
}
