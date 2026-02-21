//! 帧构建器
//!
//! `FrameBuilder` 提供流畅的 API 来构建每帧的渲染管线。
//! 允许用户在指定阶段插入、替换或移除渲染节点。

use super::graph::RenderGraph;
use super::node::RenderNode;
use super::stage::RenderStage;

/// 渲染节点条目
///
/// 存储节点引用及其所属阶段，用于排序和执行。
struct NodeEntry<'a> {
    /// 渲染阶段
    stage: RenderStage,
    /// 阶段内的插入顺序（用于稳定排序）
    order: u16,
    /// 节点引用
    node: &'a mut dyn RenderNode,
}

/// 帧构建器
///
/// 提供构建器模式来组织每帧的渲染管线。
///
/// # 设计原则
///
/// - **阶段化渲染**：通过 `RenderStage` 定义渲染顺序
/// - **灵活插入**：可在任意阶段插入自定义节点
/// - **零开销抽象**：编译时确定的阶段排序，无运行时查找开销
/// - **不持有节点**：仅存储节点引用，节点由调用者管理生命周期
///
/// # 用法
///
/// ```ignore
/// let mut builder = FrameBuilder::new();
///
/// // 添加内置 Pass
/// builder.add_node(RenderStage::PreProcess, &brdf_pass);
/// builder.add_node(RenderStage::Opaque, &forward_pass);
///
/// // 添加自定义 Pass
/// builder.add_node(RenderStage::UI, &ui_pass);
///
/// // 执行渲染
/// builder.execute(&mut render_context);
/// ```
///
/// # 性能考虑
///
/// - 内部 smallvec 预分配 16 个条目，覆盖大部分场景
/// - 排序使用标准库的 `sort_unstable_by_key`，高效且无额外内存开销
/// - 节点存储为引用，无堆分配开销
pub struct FrameBuilder<'a> {
    /// 节点列表（未排序）
    nodes: smallvec::SmallVec<[NodeEntry<'a>; 16]>,
    /// 下一个插入顺序号
    next_order: u16,
}

impl Default for FrameBuilder<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> FrameBuilder<'a> {
    /// 创建新的帧构建器
    ///
    /// 预分配 16 个节点的空间，覆盖典型渲染管线。
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: smallvec::SmallVec::with_capacity(16),
            next_order: 0,
        }
    }

    /// 创建指定容量的帧构建器
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: smallvec::SmallVec::with_capacity(capacity),
            next_order: 0,
        }
    }

    /// 在指定阶段添加渲染节点
    ///
    /// 同阶段内的节点按添加顺序执行。
    ///
    /// # 参数
    ///
    /// - `stage`: 渲染阶段
    /// - `node`: 渲染节点引用
    ///
    /// # 返回
    ///
    /// 返回 `&mut Self` 以支持链式调用。
    #[inline]
    pub fn add_node(&mut self, stage: RenderStage, node: &'a mut dyn RenderNode) -> &mut Self {
        self.nodes.push(NodeEntry {
            stage,
            order: self.next_order,
            node,
        });
        self.next_order = self.next_order.wrapping_add(1);
        self
    }

    /// 批量添加多个节点到同一阶段
    ///
    /// 适用于添加多个后处理效果或多个 UI 层。
    #[inline]
    pub fn add_nodes<I>(&mut self, stage: RenderStage, nodes: I) -> &mut Self
    where
        I: IntoIterator<Item = &'a mut dyn RenderNode>,
    {
        for node in nodes {
            // 这里 node 已经是 &'a mut dyn RenderNode 了，直接移动进去
            self.add_node(stage, node);
        }
        self
    }

    /// 获取当前节点数量
    #[inline]
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// 检查指定阶段是否有节点
    #[inline]
    #[must_use]
    pub fn has_stage(&self, stage: RenderStage) -> bool {
        self.nodes.iter().any(|e| e.stage == stage)
    }

    /// 清空所有节点（保留容量）
    #[inline]
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.next_order = 0;
    }

    /// 构建 `RenderGraph` 但不执行（用于调试或延迟执行）
    ///
    /// # 注意
    ///
    /// 返回的 `RenderGraph` 的生命周期与 `FrameBuilder` 相同。
    #[must_use]
    pub fn build(mut self) -> RenderGraph<'a> {
        self.nodes
            .sort_unstable_by_key(|e| (e.stage.order(), e.order));

        let mut graph = RenderGraph::with_capacity(self.nodes.len());

        for entry in self.nodes {
            // 现在我们可以把 entry.node (即 &'a mut dyn RenderNode)
            // 移动(Move) 进 graph 中了
            graph.add_node(entry.node);
        }

        graph
    }
}