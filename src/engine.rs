//! 引擎核心模块
//!
//! `ThreeEngine` 是纯粹的引擎实例，不包含窗口管理逻辑。
//! 它可以被不同的前端（Winit、Python、WebAssembly 等）驱动。

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::assets::AssetServer;
use crate::renderer::graph::RenderNode;
use crate::renderer::settings::RenderSettings;
use crate::renderer::Renderer;
use crate::resources::input::Input;
use crate::scene::manager::SceneManager;

/// 纯粹的引擎实例，不包含窗口管理逻辑
pub struct ThreeEngine {
    pub renderer: Renderer,
    pub scene_manager: SceneManager,
    pub assets: AssetServer,
    pub input: Input,

    pub(crate) time: f32,
    pub(crate) frame_count: u64,
}

impl ThreeEngine {
    /// 创建引擎（轻量级，无 GPU 资源）
    #[must_use]
    pub fn new(settings: RenderSettings) -> Self {
        Self {
            renderer: Renderer::new(settings),
            scene_manager: SceneManager::new(),
            assets: AssetServer::new(),
            input: Input::new(),
            time: 0.0,
            frame_count: 0,
        }
    }

    /// 初始化 GPU 资源（接受任何实现了窗口句柄 Trait 的对象）
    ///
    /// # Errors
    /// 如果 GPU 初始化失败（无可用适配器、设备请求失败等）则返回错误。
    pub async fn init<W>(&mut self, window: W, width: u32, height: u32) -> crate::errors::Result<()>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        self.renderer.init(window, width, height).await?;

        // if self.scene_manager.active_handle().is_none() {
        //     self.scene_manager.create_active();
        // }

        Ok(())
    }

    /// 处理窗口尺寸变化
    pub fn resize(&mut self, width: u32, height: u32, scale_factor: f32) {

        self.renderer.resize(width, height, scale_factor);
        self.input.inject_resize(width, height);

        if width > 0 && height > 0 {
            self.update_camera_aspect(width as f32 / height as f32);
        }

    }

    pub fn update(&mut self, dt: f32) {
        self.time += dt;
        self.frame_count += 1;

        if let Some(scene) = self.scene_manager.active_scene_mut() {
            scene.update(&self.input, dt);
        }

        self.input.start_frame();
    }

    // /// 渲染当前帧
    pub fn render(&mut self, extra_nodes: &[&dyn RenderNode]) {

        let Some(scene_handle) = self.scene_manager.active_handle() else {
            return;
        };
        let Some(scene) = self.scene_manager.get_scene_mut(scene_handle) else {
            return;
        };
        let Some(cam_node) = scene.active_camera else {
            return;
        };
        let Some(cam) = scene.cameras.get(cam_node) else {
            return;
        };

        let camera = cam.extract_render_camera();
        let time = self.time;

        self.renderer
            .render(scene, camera, &self.assets, time, extra_nodes);
    }

    fn update_camera_aspect(&mut self, aspect: f32) {
        let Some(scene) = self.scene_manager.active_scene_mut() else {
            return;
        };
        let Some(cam_handle) = scene.active_camera else {
            return;
        };
        if let Some(cam) = scene.cameras.get_mut(cam_handle) {
            cam.aspect = aspect;
            cam.update_projection_matrix();
        }
    }
}

impl Default for ThreeEngine {
    fn default() -> Self {
        Self::new(RenderSettings::default())
    }
}



#[derive(Debug, Clone, Copy)]
pub struct FrameState {
    pub time: f32,       // 游戏运行总时长
    pub dt: f32,         // 上一帧的间隔
    pub frame_count: u64,
}