//! 渲染器模块
//!
//! 负责将 Scene 绘制到屏幕，采用 Core - Graph - Pipeline 分层架构：
//! - core: WGPU 上下文封装（Device, Queue, Surface, ResourceManager）
//! - graph: 渲染管线组织（RenderFrame, Pass, Sort）
//! - pipeline: PSO 缓存与构建

pub mod core;
pub mod graph;
pub mod pipeline;
pub mod settings;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::assets::AssetServer;
use crate::errors::Result;
use crate::scene::Scene;
use crate::scene::camera::RenderCamera;

use self::core::{ResourceManager, WgpuContext};
use self::graph::RenderFrame;
use self::graph::RenderNode;
use self::pipeline::PipelineCache;
use self::settings::RenderSettings;

/// 主渲染器
pub struct Renderer {
    settings: RenderSettings,
    context: Option<RendererState>,
    _size: [u32; 2],
}

/// 渲染器内部状态
struct RendererState {
    wgpu_ctx: WgpuContext,
    resource_manager: ResourceManager,
    pipeline_cache: PipelineCache,
    render_frame: RenderFrame,
}

impl Renderer {
    /// 阶段 1: 创建配置 (无 GPU 资源)
    pub fn new(settings: RenderSettings) -> Self {
        Self {
            settings,
            context: None,
            _size: [0, 0],
        }
    }

    /// 阶段 2: 初始化 GPU 上下文 (接受任何窗口句柄)
    pub async fn init<W>(&mut self, window: W, width: u32, height: u32) -> Result<()>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        if self.context.is_some() {
            return Ok(());
        }

        self._size = [width, height];

        // 1. 创建 WGPU 上下文
        let wgpu_ctx = WgpuContext::new(window, &self.settings, width, height).await?;

        // 2. 初始化资源管理器
        let resource_manager =
            ResourceManager::new(wgpu_ctx.device.clone(), wgpu_ctx.queue.clone());

        // 3. 创建渲染帧管理器
        let render_frame = RenderFrame::new(wgpu_ctx.device.clone());

        // 4. 组装状态
        self.context = Some(RendererState {
            wgpu_ctx,
            resource_manager,
            pipeline_cache: PipelineCache::new(),
            render_frame,
        });

        log::info!("Renderer Initialized");
        Ok(())
    }

    pub fn resize(&mut self, width: u32, height: u32, _scale_factor: f32) {
        self._size = [width, height];
        if let Some(state) = &mut self.context {
            state.wgpu_ctx.resize(width, height);
        }
    }

    /// 渲染场景
    /// 
    /// # 参数
    /// - `extra_nodes`: 额外的渲染节点（如 UI Pass），将在内置 Pass 之后执行
    pub fn render(
        &mut self, 
        scene: &mut Scene, 
        camera: RenderCamera, 
        assets: &AssetServer, 
        time: f32,
        extra_nodes: &[&dyn RenderNode],
    ) {
        if self._size[0] == 0 || self._size[1] == 0 {
            return;
        }
        if let Some(state) = &mut self.context {
            state.render_frame.render(
                &mut state.wgpu_ctx,
                &mut state.resource_manager,
                &mut state.pipeline_cache,
                scene,
                &camera,
                assets,
                time,
                extra_nodes,
            );
        }
    }

    // === 公开方法：用于外部插件 (如 UI Pass) ===

    /// 获取 wgpu Device 引用
    /// 
    /// 用于外部插件初始化 GPU 资源
    pub fn device(&self) -> Option<&wgpu::Device> {
        self.context.as_ref().map(|s| &s.wgpu_ctx.device)
    }

    /// 获取 wgpu Queue 引用
    /// 
    /// 用于外部插件提交命令
    pub fn queue(&self) -> Option<&wgpu::Queue> {
        self.context.as_ref().map(|s| &s.wgpu_ctx.queue)
    }

    /// 获取 Surface 纹理格式
    /// 
    /// 用于外部插件配置渲染管线
    pub fn surface_format(&self) -> Option<wgpu::TextureFormat> {
        self.context.as_ref().map(|s| s.wgpu_ctx.config.format)
    }

    /// 获取 WgpuContext 的引用
    /// 
    /// 用于外部插件访问底层 GPU 资源（Device, Queue, Surface 等）
    /// 注意：仅在 Renderer 初始化后可用
    pub fn wgpu_ctx(&self) -> Option<&WgpuContext> {
        self.context.as_ref().map(|s| &s.wgpu_ctx)
    }
}
