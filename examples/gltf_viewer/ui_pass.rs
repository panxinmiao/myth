//! UI Pass 插件模块
//!
//! 这是一个外部插件示例，演示如何将 egui 集成到 three-rs 渲染引擎中。
//! 该模块暂时**不属于**引擎核心，是"用户代码"(User Land Code)。
//! 未来，它可能会被移入引擎核心，作为官方 UI 解决方案提供。
//!
//! # 设计说明
//! - `UiPass` 实现了 `RenderNode` trait，可以被注入到引擎的 RenderGraph 中
//! - 使用 `RefCell` 处理 `RenderNode::run(&self)` 的不可变借用限制
//! - CPU 侧逻辑（输入处理、帧开始/结束）由应用显式调用
//! - GPU 侧逻辑（绘制）由 RenderGraph 自动调用

use std::cell::RefCell;

use rustc_hash::FxHashMap;
use winit::window::Window;
use winit::event::WindowEvent;
use wgpu::{Device, TextureFormat};

use three::{assets::TextureHandle, renderer::graph::{RenderContext, RenderNode}};

/// UI 渲染 Pass
/// 
/// 封装 egui 的完整生命周期，作为 RenderNode 注入到引擎中。
/// 
/// # 使用流程
/// 1. 创建: `UiPass::new(device, format, window)`
/// 2. 每帧循环:
///    - 输入处理: `handle_input(window, event)` (在事件处理中)
///    - 开始: `begin_frame(window)` (在 update 开始)
///    - 构建 UI: 使用 `context()` 访问 egui::Context
///    - 结束: `end_frame(window)` (在 update 结束)
///    - 渲染: 通过 `renderer.render(..., &[&ui_pass])` 自动调用
pub struct UiPass {
    /// egui 上下文
    egui_ctx: egui::Context,
    /// winit 状态管理器
    state: RefCell<egui_winit::State>,
    /// wgpu 渲染器
    renderer: RefCell<egui_wgpu::Renderer>,
    
    /// 每帧的绘制数据
    clipped_primitives: RefCell<Vec<egui::ClippedPrimitive>>,
    /// 纹理变化
    textures_delta: RefCell<egui::TexturesDelta>,
    /// 屏幕描述符
    screen_descriptor: RefCell<egui_wgpu::ScreenDescriptor>,


    // === 延迟注册系统， 提供给外部App访问内部纹理的方法 ===
    /// 待处理的纹理请求队列
    texture_requests: RefCell<Vec<TextureHandle>>,
    /// 已注册的纹理映射表 (Handle -> egui::TextureId)
    texture_map: RefCell<FxHashMap<TextureHandle, egui::TextureId>>,
    /// 记录 GPU 资源 ID 用于失效检测 (Handle -> GpuImageId)
    gpu_resource_ids: RefCell<FxHashMap<TextureHandle, u64>>,

}

impl UiPass {
    /// 创建新的 UI Pass
    /// 
    /// # 参数
    /// - `device`: wgpu 设备
    /// - `output_format`: 输出纹理格式
    /// - `window`: 窗口引用
    pub fn new(
        device: &Device,
        output_format: TextureFormat,
        window: &Window,
    ) -> Self {
        let size = window.inner_size();
        let egui_ctx = egui::Context::default();
        
        // egui_winit 初始化
        let id = egui_ctx.viewport_id();
        let state = egui_winit::State::new(
            egui_ctx.clone(),
            id,
            window,
            None,
            None,
            None,
        );

        // 类型转换 (wgpu 版本兼容)
        let output_format_egui: egui_wgpu::wgpu::TextureFormat = unsafe {
            std::mem::transmute(output_format)
        };
        
        let renderer = egui_wgpu::Renderer::new(
            unsafe { std::mem::transmute::<&Device, &egui_wgpu::wgpu::Device>(device) },
            output_format_egui,
            egui_wgpu::RendererOptions::default(),
        );

        Self {
            egui_ctx,
            state: RefCell::new(state),
            renderer: RefCell::new(renderer),
            clipped_primitives: RefCell::new(Vec::new()),
            textures_delta: RefCell::new(egui::TexturesDelta::default()),
            screen_descriptor: RefCell::new(egui_wgpu::ScreenDescriptor {
                size_in_pixels: [size.width, size.height],
                pixels_per_point: window.scale_factor() as f32,
            }),

            // 初始化延迟注册系统
            texture_requests: RefCell::new(Vec::new()),
            texture_map: RefCell::new(FxHashMap::default()),
            gpu_resource_ids: RefCell::new(FxHashMap::default()),
        }
    }

    // === CPU 侧公开 API ===

    /// 处理输入事件
    /// 
    /// 返回 `true` 表示事件被 UI 消耗，应用不应再处理
    pub fn handle_input(&self, window: &Window, event: &WindowEvent) -> bool {
        let response = self.state.borrow_mut().on_window_event(window, event);

        if let WindowEvent::MouseInput { state: winit::event::ElementState::Released, .. } = event {
            return false;
        }

        // let is_mouse_released = matches!(event, 
        //     WindowEvent::MouseInput { state: winit::event::ElementState::Released, .. }
        // );

        // // 1. 优先检查鼠标释放事件，防止“点击穿透”问题
        // if is_mouse_released {
        //     return false;
        // }

        response.consumed
    }

    #[allow(dead_code)]
    pub fn request_texture(&self, handle: TextureHandle) -> Option<egui::TextureId> {
        // 1. 如果已存在，直接返回
        if let Some(&id) = self.texture_map.borrow().get(&handle) {
            return Some(id);
        }

        // 2. 如果不存在，检查是否已在请求队列中
        let mut requests = self.texture_requests.borrow_mut();
        if !requests.contains(&handle) {
            requests.push(handle);
        }

        None
    }

    #[allow(dead_code)]
    pub fn free_texture(&self, handle: TextureHandle) {
        if let Some(id) = self.texture_map.borrow_mut().remove(&handle) {
            self.gpu_resource_ids.borrow_mut().remove(&handle);
            self.renderer.borrow_mut().free_texture(&id);
        }
    }


    pub fn register_native_texture(
        &self, 
        device: &wgpu::Device, 
        view: &wgpu::TextureView, 
        filter: wgpu::FilterMode
    ) -> egui::TextureId {

        let device_egui: &egui_wgpu::wgpu::Device = unsafe { std::mem::transmute(device) };
        let view_egui: &egui_wgpu::wgpu::TextureView = unsafe { std::mem::transmute(view) };

        self.renderer.borrow_mut().register_native_texture(
            device_egui,
            view_egui,
            filter
        )
    }


    /// 每帧开始时调用
    pub fn begin_frame(&self, window: &Window) {
        let raw_input = self.state.borrow_mut().take_egui_input(window);
        self.egui_ctx.begin_pass(raw_input);
    }

    /// 用户 UI 构建完成后调用
    /// 
    /// 会生成绘制数据供 GPU 阶段使用
    pub fn end_frame(&self, window: &Window) {
        let egui::FullOutput {
            shapes,
            textures_delta,
            platform_output,
            ..
        } = self.egui_ctx.end_pass();

        // 处理平台输出（鼠标指针、剪贴板等）
        self.state.borrow_mut().handle_platform_output(window, platform_output);

        *self.textures_delta.borrow_mut() = textures_delta;
        *self.clipped_primitives.borrow_mut() = self.egui_ctx.tessellate(
            shapes, 
            self.egui_ctx.pixels_per_point()
        );
    }

    /// 获取 egui 上下文
    /// 
    /// 用于构建 UI
    pub fn context(&self) -> &egui::Context {
        &self.egui_ctx
    }

    /// 窗口大小调整
    pub fn resize(&self, width: u32, height: u32, scale_factor: f32) {
        let mut desc = self.screen_descriptor.borrow_mut();
        desc.size_in_pixels = [width, height];
        desc.pixels_per_point = scale_factor;
    }

    /// 检查 UI 是否想要捕获键盘输入
    #[allow(dead_code)]
    pub fn wants_keyboard_input(&self) -> bool {
        self.egui_ctx.egui_wants_keyboard_input()
    }

    /// 检查 UI 是否想要捕获鼠标输入
    #[allow(dead_code)]
    pub fn wants_pointer_input(&self) -> bool {
        self.egui_ctx.egui_wants_pointer_input()
    }
}

/// 实现 RenderNode trait
/// 
/// 这使得 UiPass 可以被注入到 RenderGraph 中，由引擎统一调度
impl RenderNode for UiPass {
    fn name(&self) -> &str {
        "UI Pass (egui)"
    }

    fn run(&self, ctx: &mut RenderContext, encoder: &mut wgpu::CommandEncoder) {

        // === 1. 处理延迟纹理注册 ===
        {
            let mut requests = self.texture_requests.borrow_mut();
            
            // 如果没有请求，直接跳过，连闭包都不用创建
            if !requests.is_empty() {
                let mut map = self.texture_map.borrow_mut();
                let mut gpu_ids = self.gpu_resource_ids.borrow_mut();
                
                // 获取资源管理器
                let resources = &ctx.resource_manager;
                
                // retain: 仅保留那些“注册失败（未就绪）”的请求
                // 返回 true = 保留 (未就绪)
                // 返回 false = 移除 (已注册)
                requests.retain(|&handle| {
                    // 1. 尝试查找绑定信息
                    if let Some(binding) = resources.get_texture_binding(handle) {
                        let image_id = binding.cpu_image_id;
                        
                        // 2. 尝试获取 GPU 图像资源
                        if let Some(gpu_image) = resources.get_image(image_id) {
                            // 3. 执行注册
                            // let device_egui: &egui_wgpu::wgpu::Device = unsafe { std::mem::transmute(&ctx.wgpu_ctx.device) };
                            // let view_egui: &egui_wgpu::wgpu::TextureView = unsafe { std::mem::transmute(&gpu_image.default_view) };
                            
                            // let id = renderer.register_native_texture(
                            //     device_egui,
                            //     view_egui,
                            //     egui_wgpu::wgpu::FilterMode::Linear,
                            // );
                            let id = self.register_native_texture(&ctx.wgpu_ctx.device, &gpu_image.default_view, egui_wgpu::wgpu::FilterMode::Linear);
                            
                            map.insert(handle, id);
                            gpu_ids.insert(handle, gpu_image.id);
                            
                            return false; // 已处理，从队列中移除
                        }
                    }
                    
                    true // 还没准备好，保留在队列中，下帧再试
                });
            }
        }


        let device = &ctx.wgpu_ctx.device;
        let queue = &ctx.wgpu_ctx.queue;
        let view = ctx.surface_view;

        // 转换为 egui_wgpu 版本的类型
        let device_egui: &egui_wgpu::wgpu::Device = unsafe { std::mem::transmute(device) };
        let queue_egui: &egui_wgpu::wgpu::Queue = unsafe { std::mem::transmute(queue) };
        let encoder_egui: &mut egui_wgpu::wgpu::CommandEncoder = unsafe { std::mem::transmute(encoder) };
        let view_egui: &egui_wgpu::wgpu::TextureView = unsafe { std::mem::transmute(view) };

        let mut renderer = self.renderer.borrow_mut();
        let paint_jobs = self.clipped_primitives.borrow();
        let mut textures_delta = self.textures_delta.borrow_mut();
        let screen_desc = self.screen_descriptor.borrow();

        // 1. 更新纹理
        for (id, delta) in &textures_delta.set {
            renderer.update_texture(device_egui, queue_egui, *id, delta);
        }

        // 2. 更新几何缓冲
        renderer.update_buffers(
            device_egui, 
            queue_egui, 
            encoder_egui, 
            &paint_jobs, 
            &screen_desc
        );

        // 3. 录制绘制命令
        {
            let mut rpass = encoder_egui.begin_render_pass(&egui_wgpu::wgpu::RenderPassDescriptor {
                label: Some("egui Pass"),
                color_attachments: &[Some(egui_wgpu::wgpu::RenderPassColorAttachment {
                    view: view_egui,
                    resolve_target: None,
                    ops: egui_wgpu::wgpu::Operations {
                        load: egui_wgpu::wgpu::LoadOp::Load, // 覆盖在原画面上
                        store: egui_wgpu::wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            renderer.render(&mut rpass, &paint_jobs, &screen_desc);
        }

        // 4. 释放已删除的纹理
        for id in &textures_delta.free {
            renderer.free_texture(id);
        }
        
        // 清空 delta 防止重复处理
        textures_delta.set.clear();
        textures_delta.free.clear();
    }
}
