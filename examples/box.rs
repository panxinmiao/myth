use std::sync::{Arc};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};
use glam::{Vec3, Vec4, Quat, Vec3A}; // 引入 Quat

// 引入你的引擎模块
use three::renderer::Renderer;
use three::core::scene::Scene;
use three::core::camera::Camera;
use three::core::geometry::{Geometry}; // 不再需要 Attribute, MeshBasicMaterial 等细节
use three::core::material::{Material}; // 使用统一的 Material 包装
use three::core::mesh::Mesh;
use three::core::texture::Texture;
use thunderdome::Index; // 需要保存 Node ID

pub struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    scene: Scene,
    camera: Camera,
    
    // 新增：保存旋转立方体的 Node ID，以便在 update 中访问
    cube_node_id: Option<Index>,
}

impl App {
    pub fn new() -> Self {
        let mut scene = Scene::new();

        let geometry = Geometry::new_box(2.0, 2.0, 2.0);
        let texture = Arc::new(Texture::create_checkerboard("checker", 512, 512, 64));

        let mut basic_mat = Material::new_basic(Vec4::new(1.0, 1.0, 1.0, 1.0));

        basic_mat.map = Some(texture);
        
        let mesh = Mesh::from_resource(geometry, basic_mat.into());
        
        // 3. 添加到场景并保存 ID
        let mesh_mut = scene.add_mesh(mesh, None);
        let node_id = mesh_mut.node_id.unwrap();
        
        // 4. 设置相机
        let mut camera = Camera::new_perspective(
            45.0, 
            1.0, 
            0.1, 
            100.0
        );
        camera.transform.translation = Vec3A::new(0.0, 3.0, 10.0); //稍微抬高一点视角
        camera.look_at(Vec3::ZERO, Vec3::Y);

        Self {
            window: None,
            renderer: None,
            scene,
            camera,
            cube_node_id: Some(node_id), // 保存立方体的 Node ID
        }
    }
    
    // ... run 方法保持不变 ...
    pub fn run(&mut self) {
        let event_loop = EventLoop::new().unwrap();
        event_loop.set_control_flow(ControlFlow::Poll); 
        event_loop.run_app(self).unwrap();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // ... 初始化逻辑保持不变 ...
        // 只是标题改一下
        if self.window.is_none() {
             let window_attributes = Window::default_attributes()
                .with_title("Three-RS: Spinning Cube");
            // ... 其余 renderer 初始化代码复制之前的 ...
             let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
            self.window = Some(window.clone());
            
            let instance = wgpu::Instance::default();
            let surface = instance.create_surface(window.clone()).unwrap();
            let size = window.inner_size();
            let renderer = Renderer::new(&instance, surface, size.width, size.height);
            self.renderer = Some(renderer);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(physical_size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(physical_size.width, physical_size.height);
                    self.camera.aspect = physical_size.width as f32 / physical_size.height as f32;
                    self.camera.update_projection_matrix();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(renderer) = &mut self.renderer {
                    
                    // === 核心：更新旋转逻辑 ===
                    if let Some(node_id) = self.cube_node_id {
                        if let Some(node) = self.scene.get_node_mut(node_id) {
                            // 每帧旋转一点点 (绕 Y 轴和 X 轴)
                            let rot_y = Quat::from_rotation_y(0.02);
                            let rot_x = Quat::from_rotation_x(0.01);
                            
                            // 累加旋转
                            node.rotation = node.rotation * rot_y * rot_x;
                            // 注意：现在的架构会自动检测 node.rotation 的变化并更新矩阵
                        }
                    }

                    // 渲染
                    renderer.render(&mut self.scene, &mut self.camera);
                    
                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    env_logger::init();
    let mut app = App::new();
    app.run();
}