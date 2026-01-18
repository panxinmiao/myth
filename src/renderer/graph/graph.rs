//! 渲染图执行器
//!
//! `RenderGraph` 管理渲染节点的执行顺序。
//! 当前采用简单的线性执行模型，后续可扩展为 DAG 结构以支持并行执行。

use super::node::RenderNode;
use super::context::RenderContext;

/// 渲染图
/// 
/// 管理和执行渲染节点列表。
/// 
/// # 当前实现
/// - 线性顺序执行所有节点
/// - 单个 CommandEncoder 贯穿整个图
/// - 支持 Debug Group 用于 GPU 调试
/// 
/// # 后续优化方向
/// - **图缓存**: 当渲染配置不变时，可缓存整个 RenderGraph 避免每帧重建
/// - **并行执行**: 分析节点依赖关系，将无依赖的节点并行编码
/// - **资源屏障**: 自动插入必要的资源屏障
/// - **条件执行**: 支持根据条件跳过某些节点
pub struct RenderGraph {
    nodes: Vec<Box<dyn RenderNode>>,
}

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderGraph {
    /// 创建空的渲染图
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// 预分配节点容量
    pub fn with_capacity(capacity: usize) -> Self {
        Self { nodes: Vec::with_capacity(capacity) }
    }

    /// 添加渲染节点
    /// 
    /// 节点按添加顺序执行。
    #[inline]
    pub fn add_node(&mut self, node: Box<dyn RenderNode>) {
        self.nodes.push(node);
    }

    /// 添加渲染节点（链式调用）
    #[inline]
    pub fn with_node(mut self, node: Box<dyn RenderNode>) -> Self {
        self.nodes.push(node);
        self
    }

    /// 执行渲染图
    /// 
    /// 创建 CommandEncoder，按顺序执行所有节点，最后提交命令。
    /// 
    /// # 性能注意
    /// - 所有节点共享同一个 CommandEncoder，减少提交次数
    /// - Debug Group 仅在调试模式下有开销
    pub fn execute(&self, ctx: &mut RenderContext) {
        let mut encoder = ctx.wgpu_ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Graph Encoder"),
        });

        for node in &self.nodes {
            encoder.push_debug_group(node.name());
            node.run(ctx, &mut encoder);
            encoder.pop_debug_group();
        }

        ctx.wgpu_ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// 获取节点数量
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// 清空所有节点
    #[inline]
    pub fn clear(&mut self) {
        self.nodes.clear();
    }
}
