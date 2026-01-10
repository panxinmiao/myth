use std::sync::{Arc};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};
use glam::{Vec3, Vec4};

// 引入你的引擎模块
use three::renderer::Renderer;
use three::core::scene::Scene;
use three::core::camera::Camera;
use three::core::geometry::{Geometry, Attribute};
use three::core::material::{MeshBasicMaterial};
use three::core::mesh::Mesh;
use three::core::texture::Texture;

/// 应用程序状态容器
pub struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    scene: Scene,
    camera: Camera,
}

impl App {
    pub fn new() -> Self {
        // 1. 初始化场景 (Scene)
        let mut scene = Scene::new();

        // --- 构建测试场景 (红色的三角形) ---
        let mut geometry = Geometry::new();
        geometry.set_attribute("position", Attribute::new_planar(&[
            [0.0f32, 0.5, 0.0],
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
        ], wgpu::VertexFormat::Float32x3));
        geometry.set_attribute("uv", Attribute::new_planar(&[
            [0.5f32, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ], wgpu::VertexFormat::Float32x2));

        
        let texture = Arc::new(
            Texture::create_solid_color("red_tex", [255, 0, 0, 255])
        );
        
        let mut basic_mat = MeshBasicMaterial::new(Vec4::new(1.0, 1.0, 1.0, 1.0));
        basic_mat.map = Some(texture.clone());
    
        // 创建 Mesh 并加入场景
        let mesh = Mesh::from_resource(
            geometry, 
            basic_mat.into()
        );

        scene.add_mesh(mesh, None);

        // 2. 初始化相机 (Camera)
        let mut camera = Camera::new_perspective(
            45.0, 
            1.0, // aspect ratio (稍后在 resize 中更新)
            0.1, 
            100.0
        );
        camera.transform.translation = glam::Vec3::new(0.0, 0.0, 3.0).into();

        camera.look_at(Vec3::ZERO, Vec3::Y);

        Self {
            window: None,
            renderer: None,
            scene,
            camera,
        }
    }

    /// 运行 App
    pub fn run(&mut self) {
        let event_loop = EventLoop::new().unwrap();
        event_loop.set_control_flow(ControlFlow::Poll); // 持续渲染循环
        event_loop.run_app(self).unwrap();
    }
}

// 实现 winit 0.30 的 ApplicationHandler
impl ApplicationHandler for App {
    // 1. 只有在这里才能创建窗口 (Application Resumed)
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("Three-RS First Light");
            
            let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
            self.window = Some(window.clone());

            // --- 初始化 Renderer ---
            // winit 0.30 的 resumed 是同步的，所以我们需要 block_on 来等待 async 的 adapter 创建
            // 这里使用了 pollster 库 (wgpu 常用依赖) 来阻塞执行 async
            let instance = wgpu::Instance::default();
            
            // 注意：Renderer::new 需要 Surface<'static>。
            // 使用 Arc<Window> 可以满足这一生命周期要求。
            let surface = instance.create_surface(window.clone()).unwrap();
            
            let size = window.inner_size();
            
            let renderer = Renderer::new(
                &instance,
                surface,
                size.width,
                size.height
            );

            self.renderer = Some(renderer);
            println!("Renderer initialized!");
        }
    }

    // 2. 窗口事件处理
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                println!("Goodbye!");
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(physical_size.width, physical_size.height);
                    // 更新相机长宽比
                    self.camera.aspect = physical_size.width as f32 / physical_size.height as f32;
                    self.camera.update_projection_matrix(); // 假设你有这个方法
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(renderer) = &mut self.renderer {
                    // --- 渲染一帧 ---
                    renderer.render(&mut self.scene, &mut self.camera);
                    
                    // 请求下一帧重绘
                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            _ => {}
        }
    }

    // 处理无事件时的逻辑 (用于驱动 continuous rendering)
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

// main 入口
fn main() {
    // 初始化日志
    env_logger::init();
    
    let mut app = App::new();
    app.run();
}