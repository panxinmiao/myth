//! 渲染图执行器
//!
//! `RenderGraph` 管理渲染节点的执行顺序。
//! 采用瞬态图设计，每帧创建新的图实例，只存储节点引用。

use smallvec::SmallVec;

use super::context::{ExecuteContext, PrepareContext};
use super::node::RenderNode;

/// 渲染图（瞬态引用容器）
///
/// 管理和执行渲染节点列表。采用瞬态设计，每帧创建新实例。
///
/// # 设计说明
/// - 使用生命周期参数 `'a` 存储节点引用，避免所有权转移
/// - 每帧创建新的 Graph 实例，开销极低（仅 Vec 指针操作）
/// - 节点本身持久化存储在 `RenderFrame` 中，复用内存
///
/// # 性能考虑
/// - 瞬态图避免了复杂的缓存失效逻辑
/// - 每帧重建图的开销约等于几次指针 push，可忽略不计
/// - 后续可扩展为 DAG 结构以支持并行编码
pub struct RenderGraph<'a> {
    nodes: SmallVec<[&'a mut dyn RenderNode; 8]>,
}

impl Default for RenderGraph<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> RenderGraph<'a> {
    /// 创建空的渲染图
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: SmallVec::new(),
        }
    }

    /// 预分配节点容量
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: SmallVec::with_capacity(capacity),
        }
    }

    /// 添加渲染节点引用
    ///
    /// 节点按添加顺序执行。
    #[inline]
    pub fn add_node(&mut self, node: &'a mut dyn RenderNode) {
        self.nodes.push(node);
    }

    pub fn prepare(&mut self, ctx: &mut PrepareContext) {
        for node in &mut self.nodes {
            node.prepare(ctx);
        }
    }

    /// 执行渲染图
    ///
    /// 创建 CommandEncoder，按顺序执行所有节点，最后提交命令。
    ///
    /// # 性能注意
    /// - 所有节点共享同一个 CommandEncoder，减少提交次数
    /// - Debug Group 用于 GPU 调试
    pub fn execute(&self, ctx: &ExecuteContext) {
        let mut encoder =
            ctx.wgpu_ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Graph Encoder"),
                });

        for node in &self.nodes {
            #[cfg(debug_assertions)]
            encoder.push_debug_group(node.name());
            node.run(ctx, &mut encoder);
            #[cfg(debug_assertions)]
            encoder.pop_debug_group();
        }

        ctx.wgpu_ctx.queue.submit(Some(encoder.finish()));
    }

    /// 获取节点数量
    #[inline]
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}
