//! 帧合成器
//!
//! `FrameComposer` 用于链式构建渲染管线并最终执行。
//! 它是连接 Prepare 阶段和 Execute 阶段的"胶水"对象。
//!
//! # 三阶段渲染架构
//!
//! 1. **Prepare (准备)**：提取数据 (Extract) 和准备资源 (Prepare)
//! 2. **Compose (组装)**：通过链式 API 添加 `RenderNode`
//! 3. **Execute (执行)**：获取 Surface，构建 `RenderGraph` 并提交 GPU 命令
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

use super::frame::RenderLists;
use crate::assets::AssetServer;
use crate::render::RenderState;
use crate::renderer::core::binding::GlobalBindGroupCache;
use crate::renderer::core::{ResourceManager, WgpuContext};
use crate::renderer::graph::ExtractedScene;
use crate::renderer::graph::builder::FrameBuilder;
use crate::renderer::graph::context::{FrameResources, RenderContext};
use crate::renderer::graph::node::RenderNode;
use crate::renderer::graph::stage::RenderStage;
use crate::renderer::pipeline::PipelineCache;
use crate::scene::Scene;
use crate::scene::camera::RenderCamera;

pub struct ComposerContext<'a> {
    pub wgpu_ctx: &'a mut WgpuContext,
    pub resource_manager: &'a mut ResourceManager,
    pub pipeline_cache: &'a mut PipelineCache,

    pub extracted_scene: &'a ExtractedScene,
    pub render_state: &'a RenderState,

    pub frame_resources: &'a FrameResources,
    pub global_bind_group_cache: &'a mut GlobalBindGroupCache,

    /// 渲染列表（由 `SceneCullPass` 填充）
    pub render_lists: &'a mut RenderLists,

    // 外部场景数据 todo: refactor
    pub scene: &'a mut Scene,
    pub camera: &'a RenderCamera,
    pub assets: &'a AssetServer,
    pub time: f32,
}

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
    ctx: ComposerContext<'a>,

    // 构建器（收集渲染节点）
    builder: FrameBuilder<'a>,
}

impl<'a> FrameComposer<'a> {
    /// 创建新的帧合成器
    ///
    /// 内部会自动注入内置 Pass（BRDF LUT、IBL、Forward）。
    pub(crate) fn new(builder: FrameBuilder<'a>, ctx: ComposerContext<'a>) -> Self {
        Self { ctx, builder }
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
    #[must_use]
    pub fn add_node(mut self, stage: RenderStage, node: &'a mut dyn RenderNode) -> Self {
        self.builder.add_node(stage, node);
        self
    }

    /// 批量添加多个节点到同一阶段
    ///
    /// # 示例
    ///
    /// ```ignore
    /// composer
    ///     .add_nodes(RenderStage::PostProcess, &[&mut bloom, &mut fxaa, &mut tone_mapping])
    ///     .render();
    /// ```
    #[inline]
    #[must_use]
    pub fn add_nodes<I>(mut self, stage: RenderStage, nodes: I) -> Self
    where
        I: IntoIterator<Item = &'a mut dyn RenderNode>,
    {
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
        let output = match self.ctx.wgpu_ctx.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost) => return,
            Err(e) => {
                log::error!("Render error: {e:?}");
                return;
            }
        };

        let view_format = self.ctx.wgpu_ctx.surface_view_format;

        let surface_view = output.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(view_format),
            ..Default::default()
        });

        // 2. 构建 RenderContext
        let mut render_ctx = RenderContext {
            wgpu_ctx: self.ctx.wgpu_ctx,
            resource_manager: self.ctx.resource_manager,
            pipeline_cache: self.ctx.pipeline_cache,
            assets: self.ctx.assets,
            scene: self.ctx.scene,
            camera: self.ctx.camera,
            surface_view: &surface_view,
            render_state: self.ctx.render_state,
            extracted_scene: self.ctx.extracted_scene,
            render_lists: self.ctx.render_lists,
            frame_resources: self.ctx.frame_resources,
            time: self.ctx.time,

            global_bind_group_cache: self.ctx.global_bind_group_cache,
            // current_color_texture_view: &self.ctx.frame_resources.scene_color_view[0],
            color_view_flip_flop: 0,
        };

        // 3. Builder 转换为排序后的 RenderGraph
        let mut graph = self.builder.build();

        graph.prepare(&mut render_ctx);

        // 4. 执行渲染图
        graph.execute(&mut render_ctx);

        // 5. Present
        output.present();
    }
}
