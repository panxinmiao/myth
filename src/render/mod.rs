//! 渲染器模块

pub mod resources;
pub mod pipeline;
pub mod data;
pub mod passes;
pub mod context; // 新增
pub mod settings; // 新增

use std::sync::Arc;
use winit::window::Window;

use crate::scene::Scene;
use crate::scene::camera::Camera;
use crate::assets::{AssetServer};

use self::resources::ResourceManager;
use self::pipeline::{PipelineCache};
use self::data::{ModelBufferManager};
use self::context::{RenderContext, RenderState};
use self::settings::RendererSettings;

/// 主渲染器
pub struct Renderer {
    settings: RendererSettings,
    context: Option<RenderContext>,
    _size: winit::dpi::PhysicalSize<u32>,
}

impl Renderer {
    /// 阶段 1: 创建配置 (无 GPU 资源)
    pub fn new(settings: RendererSettings) -> Self {
        Self {
            settings,
            context: None,
            _size: winit::dpi::PhysicalSize::new(0, 0),
        }
    }

    /// 阶段 2: 初始化 GPU 上下文 (需要 Window)
    pub async fn init(&mut self, window: Arc<Window>) {
        if self.context.is_some() { return; }

        let size = window.inner_size();
        self._size = size;

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();
        
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: self.settings.power_preference,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.unwrap();

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: self.settings.required_features,
                required_limits: self.settings.required_limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            },
        ).await.unwrap();

        let config = surface.get_default_config(&adapter, size.width, size.height).unwrap();
        surface.configure(&device, &config);

        // 初始化子系统
        let mut resource_manager = ResourceManager::new(device.clone(), queue.clone());
        let model_buffer_manager = ModelBufferManager::new(&mut resource_manager);
        
        // 创建 RenderState
        let render_state = RenderState::new();

        // 静态方法创建深度 Buffer
        let depth_texture_view = RenderContext::create_depth_texture(&device, &config, self.settings.depth_format);

        // 构建上下文
        self.context = Some(RenderContext {
            device,
            queue,
            surface,
            config,
            depth_format: self.settings.depth_format,
            depth_texture_view,
            clear_color: self.settings.clear_color,
            render_state,
            resource_manager,
            model_buffer_manager,
            pipeline_cache: PipelineCache::new(),
        });
        
        log::info!("Renderer Initialized");
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self._size = winit::dpi::PhysicalSize::new(width, height);
        if let Some(ctx) = &mut self.context {
            ctx.resize(width, height);
        }
    }

    /// 渲染代理
    pub fn render(&mut self, scene: &Scene, camera: &Camera, assets: &AssetServer) {
        if self._size.width == 0 || self._size.height == 0 {
            return;
        }
        if let Some(ctx) = &mut self.context {
            ctx.render_frame(scene, camera, assets);
        }
    }
}