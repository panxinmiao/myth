use glam::{Vec3, Vec4, Quat};
use three::app::{App, AppContext, AppHandler};
use three::resources::{Geometry, MeshStandardMaterial, Mesh, Texture};
use three::scene::{Camera, NodeHandle, light};
use three::utils::fps_counter::FpsCounter;
use three::OrbitControls;
use three::renderer::settings::RenderSettings;

/// PBR 材质立方体示例
struct PbrBox {
    cube_node_id: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
}

impl AppHandler for PbrBox {
    fn init(ctx: &mut AppContext) -> Self {
        // 1. 准备资源
        let geometry = Geometry::new_box(2.0, 2.0, 2.0);
        let texture = Texture::create_checkerboard(Some("checker"), 512, 512, 64);
        
        // 创建具体材质类型，便于访问类型特定的方法
        let mut standard_mat = MeshStandardMaterial::new(Vec4::new(1.0, 1.0, 1.0, 1.0));

        let tex_handle = ctx.assets.add_texture(texture);
        standard_mat.map.texture = Some(tex_handle);
        
        let geo_handle = ctx.assets.add_geometry(geometry);
        // 在最后需要时才转换为通用 Material 类型
        let mat_handle = ctx.assets.add_material(standard_mat);

        let scene =ctx.scenes.create_active();
        //let scene = ctx.scenes.active_scene_mut().unwrap();

        // 2. 创建 Mesh 并加入场景
        let mesh = Mesh::new(geo_handle, mat_handle);
        let cube_node_id = scene.add_mesh(mesh);
        // 3. 添加灯光
        let light = light::Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 0.0);
        scene.add_light(light);

        // 4. 加载环境贴图
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
        scene.environment.set_env_map(Some((env_texture_handle.into(), &env_texture)));

        // 5. 设置相机
        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let cam_node_id = scene.add_camera(camera);
        
        if let Some(node) = scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 3.0, 10.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }
        
        scene.active_camera = Some(cam_node_id);

        Self {
            cube_node_id,
            controls: OrbitControls::new(Vec3::new(0.0, 3.0, 10.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
        }
    }

    fn update(&mut self, ctx: &mut AppContext) {
        let Some(scene) = ctx.scenes.active_scene_mut() else{
            return;
        };
        // 旋转立方体
        if let Some(node) = scene.get_node_mut(self.cube_node_id) {
            let rot_y = Quat::from_rotation_y(0.02 * 60.0 * ctx.dt);
            let rot_x = Quat::from_rotation_x(0.01 * 60.0 * ctx.dt);
            node.transform.rotation = node.transform.rotation * rot_y * rot_x;
        }

        // 轨道控制器
        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls.update(transform, ctx.input, camera.fov.to_degrees(), ctx.dt);
        }

        // FPS 显示
        if let Some(fps) = self.fps_counter.update() {
            ctx.window.set_title(&format!("Box PBR | FPS: {:.2}", fps));
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    App::new()
        .with_settings(RenderSettings { vsync: false, ..Default::default() })
        .run::<PbrBox>()
}