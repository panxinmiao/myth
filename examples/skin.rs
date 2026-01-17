use glam::{Affine3A, Quat, Vec3, Vec2, Vec4};
use three::Node;
use three::app::App;
use three::resources::{Attribute, Geometry, Material, Mesh};
use three::scene::skeleton::{BindMode, Skeleton};
use three::scene::{Camera};
use three::scene::light;
use three::utils::fps_counter::{FpsCounter};
use wgpu::VertexFormat;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    // 1. 初始化引擎 App
    let mut app = App::new().with_settings(three::renderer::settings::RenderSettings {
        vsync: false,
        ..Default::default()
    });



    // 1. 创建几何体 (圆柱体模拟)
    // 只有 4 个顶点：两个在底部(y=0)，两个在顶部(y=2)
    let positions = vec![
        Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0), // 底部
        Vec3::new(-0.5, 2.0, 0.0), Vec3::new(0.5, 2.0, 0.0), // 顶部
    ];
    // 索引
    let indices: Vec<u16> = vec![0, 1, 2, 1, 3, 2]; 

    let pos_attr = Attribute::new_planar(&positions, VertexFormat::Float32x3);

    let mut geometry = Geometry::new();
    geometry.set_attribute("position", pos_attr);
    geometry.set_indices(&indices);
    
    // === 关键：手动设置蒙皮数据 ===
    // 底部顶点 (0, 1) 受 Bone0 (根骨骼) 100% 控制
    // 顶部顶点 (2, 3) 受 Bone1 (子骨骼) 100% 控制
    let joints: Vec<[u16; 4]> = vec![
        [0, 0, 0, 0], [0, 0, 0, 0], // 底部 -> Bone 0
        [1, 0, 0, 0], [1, 0, 0, 0], // 顶部 -> Bone 1
    ];
    let weights: Vec<[f32; 4]> = vec![
        [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    ];

    let joints_attr = Attribute::new_planar(&joints, VertexFormat::Uint16x4);
    let weights_attr = Attribute::new_planar(&weights, VertexFormat::Float32x4);
    
    geometry.set_attribute("joints", joints_attr);
    geometry.set_attribute("weights", weights_attr);    


    // 加点法线和UV防止 Shader 报错
    geometry.set_attribute("normal", Attribute::new_planar(&vec![Vec3::Y; 4], VertexFormat::Float32x3)); 
    geometry.set_attribute("uv", Attribute::new_planar(&vec![Vec2::ZERO; 4], VertexFormat::Float32x2));
    
    // 加点法线和UV防止 Shader 报错
    // geometry.set_attribute("normal", vec![Vec3::Y; 4]); 
    // geometry.set_attribute("uv", vec![Vec2::ZERO; 4]);

    let geo_handle = app.assets.add_geometry(geometry);
    let mat_handle = app.assets.add_material(Material::new_basic(Vec4::new(1.0, 1.0, 1.0, 1.0)));

    // 2. 创建骨骼节点结构
    // Root -> Bone1
    let mut root_node = Node::new("Bone_Root");

    root_node.transform.position = Vec3::new(0.0, 0.0, 0.0);

    let root_id = app.scene.add_node(root_node);

    let mut bone1_node = Node::new("Bone_Top");

    bone1_node.transform.position = Vec3::new(0.0, 2.0, 0.0); // 骨骼位于顶部位置
 
    let bone1_id = app.scene.add_to_parent(bone1_node, root_id);
    
    // 3. 创建 Skeleton 资源
    // Inverse Bind Matrices: 抵消骨骼的初始变换
    // Bone0 在原点，IBM 是 Identity
    // Bone1 在 (0,2,0)，IBM 是 Translation(0,-2,0)
    let ibm0 = Affine3A::IDENTITY;
    let ibm1 = Affine3A::from_translation(Vec3::new(0.0, -2.0, 0.0));

    let skeleton = Skeleton::new(
        "MySkeleton",
        vec![root_id, bone1_id], // 对应 joints 0 和 1
        vec![ibm0, ibm1] // 转为 Skeleton 内部存储格式
    );

    let skel_id = app.scene.add_skeleton(skeleton);

    // let skel_id = app.scene.add_skeleton(skeleton);


    // 4. 创建 Mesh 节点并绑定
    let mesh = Mesh::new(geo_handle, mat_handle);

    let mesh_idx = app.scene.add_mesh(mesh);

    let mesh_node = app.scene.get_node_mut(mesh_idx).unwrap();

    mesh_node.bind_skeleton(skel_id, BindMode::Attached);



    let light = light::Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 0.0);
    app.scene.add_light(light);

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
        node.transform.position = Vec3::new(0.0, 3.0, 10.0);
        // 使用我们刚实现的 look_at 方法
        node.transform.look_at(Vec3::ZERO, Vec3::Y);
    }
    
    // 5.4 激活相机
    app.scene.active_camera = Some(cam_node_id);

    let mut controls = three::OrbitControls::new(Vec3::new(0.0, 3.0, 10.0), Vec3::ZERO);

    let mut fps_counter = FpsCounter::new();

    // 6. 设置 Update 回调 (处理旋转动画)
    // move 闭包捕获 bone1_id
    app.set_update_fn(move |window, scene, _assets, input, time, dt| {

        if let Some(node) = scene.get_node_mut(bone1_id) {
            let angle = time.sin() * 1.0; // +/- 1 radian
            node.transform.rotation = Quat::from_rotation_z(angle);
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