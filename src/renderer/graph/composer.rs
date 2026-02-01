//! 帧合成器
//!
//! `FrameComposer` 用于链式构建渲染管线并最终执行。
//! 它是连接 Prepare 阶段和 Execute 阶段的"胶水"对象。
//!
//! # 三阶段渲染架构
//!
//! 1. **Prepare (准备)**：提取数据 (Extract) 和准备资源 (Prepare)
//! 2. **Compose (组装)**：通过链式 API 添加 `RenderNode`
//! 3. **Execute (执行)**：获取 Surface，构建 RenderGraph 并提交 GPU 命令
//!
//! # 示例
//!
//! ```ignore
//! // 优雅的链式调用
//! renderer.begin_frame(scene, &camera, assets, time)?
//!     .add_node(RenderStage::UI, &ui_pass)
//!     .add_node(RenderStage::PostProcess, &bloom_pass)
//!     .render();
//! ```

use crate::assets::AssetServer;
use crate::renderer::core::{ResourceManager, WgpuContext};
use crate::renderer::graph::builder::FrameBuilder;
use crate::renderer::graph::context::RenderContext;
use crate::renderer::graph::frame::RenderFrame;
use crate::renderer::graph::node::RenderNode;
use crate::renderer::graph::stage::RenderStage;
use crate::renderer::pipeline::PipelineCache;
use crate::scene::Scene;
use crate::scene::camera::RenderCamera;

/// 帧合成器
///
/// 持有一帧渲染所需的所有上下文引用，提供链式 API 来添加渲染节点。
///
/// # 设计说明
///
/// - **权责分明**：`FrameComposer` 只负责上下文和流程控制
/// - **生命周期安全**：生命周期 `'a` 锁定 `Renderer` 的可变借用
/// - **Surface 延迟获取**：在 `.render()` 时才获取 Surface，减少持有时间
///
/// # 性能考虑
///
/// - 内部 `FrameBuilder` 预分配 16 个节点容量
/// - 排序使用 `FrameBuilder` 的高效排序机制
/// - 所有字段都是引用，无堆分配开销
pub struct FrameComposer<'a> {
    // GPU 上下文
    wgpu_ctx: &'a mut WgpuContext,
    resource_manager: &'a mut ResourceManager,
    pipeline_cache: &'a mut PipelineCache,

    // 场景数据（暂时需要可变引用，因为内置 Pass 可能修改 environment）
    // Todo: IBLComputePass 和 BRDFLutComputePass 需要修改 scene.environment。这是设计上的问题 
    // 理想情况下，这些 Pass 不应该直接修改 Scene，而应该通过返回值或写入 ResourceManager 的方式来更新。
    // Fix it !
    scene: &'a mut Scene,
    camera: &'a RenderCamera,
    assets: &'a AssetServer,
    time: f32,

    // 渲染帧数据（持有内置 Pass 和提取的场景数据）
    render_frame: &'a RenderFrame,

    // 构建器（收集渲染节点）
    builder: FrameBuilder<'a>,
}

impl<'a> FrameComposer<'a> {
    /// 创建新的帧合成器
    ///
    /// 内部会自动注入内置 Pass（BRDF LUT、IBL、Forward）。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        wgpu_ctx: &'a mut WgpuContext,
        resource_manager: &'a mut ResourceManager,
        pipeline_cache: &'a mut PipelineCache,
        render_frame: &'a mut RenderFrame,
        scene: &'a mut Scene,
        camera: &'a RenderCamera,
        assets: &'a AssetServer,
        time: f32,
    ) -> Self {
        // 创建 Builder 并注入内置 Pass
        let mut builder = FrameBuilder::new();
        builder
            .add_node(RenderStage::PreProcess, render_frame.brdf_pass())
            .add_node(RenderStage::PreProcess, render_frame.ibl_pass())
            .add_node(RenderStage::Opaque, render_frame.forward_pass());

        Self {
            wgpu_ctx,
            resource_manager,
            pipeline_cache,
            render_frame,
            scene,
            camera,
            assets,
            time,
            builder,
        }
    }

    /// 在指定阶段添加自定义渲染节点
    ///
    /// 支持链式调用。
    ///
    /// # 参数
    ///
    /// - `stage`: 渲染阶段（决定执行顺序）
    /// - `node`: 渲染节点引用
    ///
    /// # 示例
    ///
    /// ```ignore
    /// composer
    ///     .add_node(RenderStage::UI, &ui_pass)
    ///     .add_node(RenderStage::PostProcess, &bloom_pass)
    ///     .render();
    /// ```
    #[inline]
    pub fn add_node(mut self, stage: RenderStage, node: &'a dyn RenderNode) -> Self {
        self.builder.add_node(stage, node);
        self
    }

    /// 批量添加多个节点到同一阶段
    ///
    /// # 示例
    ///
    /// ```ignore
    /// composer
    ///     .add_nodes(RenderStage::PostProcess, &[&bloom, &fxaa, &tone_mapping])
    ///     .render();
    /// ```
    #[inline]
    pub fn add_nodes(mut self, stage: RenderStage, nodes: &[&'a dyn RenderNode]) -> Self {
        self.builder.add_nodes(stage, nodes);
        self
    }

    /// 执行渲染并呈现到屏幕
    ///
    /// 这是渲染流程的最后一步：
    /// 1. 获取 Surface 纹理
    /// 2. 构建 `RenderContext`
    /// 3. 将 Builder 转换为 `RenderGraph`（包含排序）
    /// 4. 执行渲染图
    /// 5. Present
    ///
    /// # 注意
    ///
    /// 此方法消费 `self`，调用后 `FrameComposer` 不可再使用。
    pub fn render(self) {
        // 1. 获取 Surface Texture（延迟到最后一刻，减少持有时间）
        let output = match self.wgpu_ctx.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost) => return,
            Err(e) => {
                log::error!("Render error: {:?}", e);
                return;
            }
        };

        let view_format = self.wgpu_ctx.view_format;

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor{
                format: Some(view_format),
                ..Default::default()
            });

        // 2. 构建 RenderContext
        let mut ctx = RenderContext {
            wgpu_ctx: self.wgpu_ctx,
            resource_manager: self.resource_manager,
            pipeline_cache: self.pipeline_cache,
            assets: self.assets,
            scene: self.scene,
            camera: self.camera,
            surface_view: &view,
            render_state: self.render_frame.render_state(),
            extracted_scene: self.render_frame.extracted_scene(),
            time: self.time,
        };

        // 3. Builder 转换为排序后的 RenderGraph
        let graph = self.builder.build();

        // 4. 执行渲染图
        graph.execute(&mut ctx);

        // 5. Present
        output.present();
    }
}
