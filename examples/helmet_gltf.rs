use glam::Vec3;
use three::app::App;
use three::scene::Camera;
use three::scene::light;
use three::OrbitControls;
use three::utils::fps_counter::FpsCounter;
use three::assets::GltfLoader;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    // 1. 初始化引擎 App
    let mut app = App::new().with_settings(three::render::settings::RenderSettings {
        vsync: false,
        ..Default::default()
    });

    // 2. 加载环境贴图 (PBR 需要 IBL)
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

    // 3. 添加灯光 (可选，PBR 主要依赖 IBL，但也可以加直接光)
    let light = light::Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 1.0);
    app.scene.add_light(light);

    // 4. 加载 glTF 模型
    let gltf_path = std::path::Path::new("examples/assets/DamagedHelmet/glTF/DamagedHelmet.gltf");
    
    println!("Loading glTF model from: {}", gltf_path.display());
    
    let loaded_nodes = GltfLoader::load(
        gltf_path,
        &mut app.assets,
        &mut app.scene
    )?;

    println!("Successfully loaded {} root nodes", loaded_nodes.len());

    // 5. 调整模型位置/缩放（如果需要）
    if let Some(&root_node_idx) = loaded_nodes.first() {
        if let Some(node) = app.scene.get_node_mut(root_node_idx) {
            node.transform.scale = Vec3::splat(1.0);
            node.transform.position = Vec3::new(0.0, 0.0, 0.0);
            println!("Model root node: {}", node.name);
        }
    }

    // 6. 设置相机
    let camera = Camera::new_perspective(
        45.0,
        1280.0 / 720.0,
        0.1,
        100.0
    );

    let cam_node_id = app.scene.add_camera(camera);

    if let Some(node) = app.scene.get_node_mut(cam_node_id) {
        node.position = Vec3::new(0.0, 0.0, 3.0);
        node.look_at(Vec3::ZERO, Vec3::Y);
    }

    app.scene.active_camera = Some(cam_node_id);

    // 7. 轨道控制器
    let mut controls = OrbitControls::new(Vec3::ZERO, 3.0);
    let mut fps_counter = FpsCounter::new();

    // 8. 更新回调 (旋转模型或相机控制)
    app.set_update_fn(move |window, scene, _assets, input, _time, dt| {
        // 轨道控制器
        if let Some(cam_node) = scene.get_node_mut(cam_node_id) {
            controls.update(&mut cam_node.transform, input, 45.0, dt);
        }

        // FPS 计数
        if let Some(fps) = fps_counter.update() {
            window.set_title(&format!("glTF PBR Demo - FPS: {:.0}", fps));
        }
    });

    // 9. 运行
    app.run()?;
    
    Ok(())
}
