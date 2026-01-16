use glam::{Vec3, Vec4, Quat};
use three::app::App;
use three::resources::{Geometry, Material, Mesh, Texture};
use three::scene::{Camera};
use three::scene::light;
use three::utils::fps_counter::{FpsCounter};

fn main() -> anyhow::Result<()> {
    env_logger::init();

    // 1. 初始化引擎 App
    let mut app = App::new().with_settings(three::render::settings::RenderSettings {
        vsync: false,
        ..Default::default()
    });

    // 2. 准备资源 (Geometry, Texture, Material)
    let geometry = Geometry::new_box(2.0, 2.0, 2.0);
    let texture: Texture = Texture::create_checkerboard(Some("checker"), 512, 512, 64);
    let mut mat = Material::new_standard(Vec4::new(1.0, 1.0, 1.0, 1.0));

    // 3. 将资源添加到 AssetServer，获取 Handle
    let tex_handle = app.assets.add_texture(texture);


    if let Some(standard) = mat.as_standard_mut() {
        standard.set_map(Some(tex_handle));
    }
    
    let geo_handle = app.assets.add_geometry(geometry);
    let mat_handle = app.assets.add_material(mat.into());

    // 4. 创建 Mesh 并加入场景
    let mesh = Mesh::new(geo_handle, mat_handle);

    let cube_node_id = app.scene.add_mesh(mesh);

    let light = light::Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 0.0);
    app.scene.add_light(light);


    // 加载环境贴图
    let env_texture_handle = app.assets.load_cube_texture_from_files(
        [
            "examples/assets/Park2/posx.jpg",
            "examples/assets/Park2/negx.jpg",
            "examples/assets/Park2/posy.jpg",
            "examples/assets/Park2/negy.jpg",
            "examples/assets/Park2/posz.jpg",
            "examples/assets/Park2/negz.jpg",
        ],
        three::ColorSpace::Srgb
    )?;

    let env_texture = app.assets.get_texture_mut(env_texture_handle).unwrap();

    env_texture.generate_mipmaps = true;

    app.scene.environment.set_env_map(Some((env_texture_handle, &env_texture)));

    // 5. 设置相机
    // 5.1 创建相机组件 (纯投影数据)
    let camera = Camera::new_perspective(
        45.0, 
        1280.0 / 720.0, // 默认窗口大小的长宽比
        0.1, 
        100.0
    );
    
    // 5.2 将相机加入场景 (自动创建 Node)
    let cam_node_id = app.scene.add_camera(camera);
    
    // 5.3 设置相机节点的位置和朝向
    if let Some(node) = app.scene.get_node_mut(cam_node_id) {
        node.position = Vec3::new(0.0, 3.0, 10.0);
        // 使用我们刚实现的 look_at 方法
        node.look_at(Vec3::ZERO, Vec3::Y);
    }
    
    // 5.4 激活相机
    app.scene.active_camera = Some(cam_node_id);

    let mut controls = three::OrbitControls::new(Vec3::new(0.0, 3.0, 10.0), Vec3::ZERO);

    let mut fps_counter = FpsCounter::new();

    // 6. 设置 Update 回调 (处理旋转动画)
    // move 闭包捕获 cube_node_id
    app.set_update_fn(move |window, scene, _assets, input, _time, dt| {
        if let Some(node) = scene.get_node_mut(cube_node_id) {
            // 每帧旋转
            let rot_y = Quat::from_rotation_y(0.02 * 60.0 * dt);
            let rot_x = Quat::from_rotation_x(0.01 * 60.0 * dt);
            
            // 累加旋转，update_matrix_world 会自动处理矩阵更新
            node.rotation = node.rotation * rot_y * rot_x;
        }

        // 使用新的组件查询 API
        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            controls.update(transform, input, camera.fov.to_degrees(), dt);
        }

        if let Some(fps) = fps_counter.update() {
            let title = format!("Box PBR | FPS: {:.2}", fps);
            window.set_title(&title);
        }
    });

    // 7. 运行
    app.run()?;
    Ok(())
}