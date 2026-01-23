use glam::{Affine3A, Quat, Vec3, Vec2, Vec4};
use three::app::{App, AppContext, AppHandler};
use three::resources::{Attribute, Geometry, Material, Mesh};
use three::scene::skeleton::{BindMode, Skeleton};
use three::scene::{Camera, NodeHandle, light};
use three::utils::fps_counter::FpsCounter;
use three::OrbitControls;
use three::renderer::settings::RenderSettings;
use wgpu::VertexFormat;

/// 骨骼蒙皮示例
/// 
/// 演示如何手动创建骨骼和蒙皮网格
struct SkinDemo {
    bone1_id: NodeHandle,
    controls: OrbitControls,
    fps_counter: FpsCounter,
}

impl AppHandler for SkinDemo {
    fn init(ctx: &mut AppContext) -> Self {
        // 1. 创建几何体 (简单的矩形模拟手臂)
        let positions = vec![
            Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0), // 底部
            Vec3::new(-0.5, 2.0, 0.0), Vec3::new(0.5, 2.0, 0.0), // 顶部
        ];
        let indices: Vec<u16> = vec![0, 1, 2, 1, 3, 2]; 

        let pos_attr = Attribute::new_planar(&positions, VertexFormat::Float32x3);

        let mut geometry = Geometry::new();
        geometry.set_attribute("position", pos_attr);
        geometry.set_indices(&indices);
        
        // 关键：手动设置蒙皮数据
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

        // 添加法线和UV防止 Shader 报错
        geometry.set_attribute("normal", Attribute::new_planar(&vec![Vec3::Y; 4], VertexFormat::Float32x3)); 
        geometry.set_attribute("uv", Attribute::new_planar(&vec![Vec2::ZERO; 4], VertexFormat::Float32x2));

        let geo_handle = ctx.assets.add_geometry(geometry);
        let mat_handle = ctx.assets.add_material(Material::new_basic(Vec4::new(1.0, 1.0, 1.0, 1.0)));

        let scene = ctx.scenes.create_active();

        // 2. 创建骨骼节点结构 (Root -> Bone1)
        let root_id = scene.create_node_with_name("Bone_Root");
        if let Some(node) = scene.get_node_mut(root_id) {
            node.transform.position = Vec3::new(0.0, 0.0, 0.0);
        }

        let bone1_id = scene.create_node_with_name("Bone_Top");
        if let Some(node) = scene.get_node_mut(bone1_id) {
            node.transform.position = Vec3::new(0.0, 2.0, 0.0);
        }
        scene.attach(bone1_id, root_id);
        
        // 3. 创建 Skeleton 资源
        let ibm0 = Affine3A::IDENTITY;
        let ibm1 = Affine3A::from_translation(Vec3::new(0.0, -2.0, 0.0));

        let skeleton = Skeleton::new(
            "MySkeleton",
            vec![root_id, bone1_id],
            vec![ibm0, ibm1],
            0,
        );
        let skel_id = scene.add_skeleton(skeleton);

        // 4. 创建 Mesh 节点并绑定骨骼
        let mesh = Mesh::new(geo_handle, mat_handle);
        let mesh_node_id = scene.create_node_with_name("SkinnedMesh");
        scene.set_mesh(mesh_node_id, mesh);
        scene.bind_skeleton(mesh_node_id, skel_id, BindMode::Attached);

        // 5. 添加灯光
        let light = light::Light::new_directional(Vec3::new(1.0, 1.0, 1.0), 0.0);
        scene.add_light(light);

        // 6. 设置相机
        let camera = Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1);
        let cam_node_id = scene.add_camera(camera);
        
        if let Some(node) = scene.get_node_mut(cam_node_id) {
            node.transform.position = Vec3::new(0.0, 3.0, 10.0);
            node.transform.look_at(Vec3::ZERO, Vec3::Y);
        }
        
        scene.active_camera = Some(cam_node_id);

        Self {
            bone1_id,
            controls: OrbitControls::new(Vec3::new(0.0, 3.0, 10.0), Vec3::ZERO),
            fps_counter: FpsCounter::new(),
        }
    }

    fn update(&mut self, ctx: &mut AppContext) {

        let Some(scene) = ctx.scenes.active_scene_mut() else{
            return;
        };
        // 摆动骨骼
        if let Some(node) = scene.get_node_mut(self.bone1_id) {
            let angle = ctx.time.sin() * 1.0;
            node.transform.rotation = Quat::from_rotation_z(angle);
        }

        // 轨道控制器
        if let Some((transform, camera)) = scene.query_main_camera_bundle() {
            self.controls.update(transform, ctx.input, camera.fov.to_degrees(), ctx.dt);
        }

        // FPS 显示
        if let Some(fps) = self.fps_counter.update() {
            ctx.window.set_title(&format!("Skin Demo | FPS: {:.2}", fps));
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    App::new()
        .with_settings(RenderSettings { vsync: false, ..Default::default() })
        .run::<SkinDemo>()
}