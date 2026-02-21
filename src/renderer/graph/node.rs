//! 渲染节点 Trait
//!
//! 定义渲染图中节点的抽象接口。
//! 每个节点代表一个渲染 Pass 或计算任务。

use super::context::{ExecuteContext, PrepareContext};

/// 渲染节点 Trait
///
/// 所有渲染 Pass 必须实现此接口。
///
/// # 设计原则
/// - `prepare` 接收 `PrepareContext`（可变），用于资源分配和管线创建
/// - `run` 接收 `ExecuteContext`（只读）+ `CommandEncoder`，用于录制 GPU 命令
/// - 节点应该在 `prepare` 中完成所有可变操作，`run` 仅做只读渲染
///
/// # 性能考虑
/// - 避免在 `run` 中进行内存分配
/// - 利用 `encoder.push_debug_group` 进行 GPU 调试
///
pub trait RenderNode {
    /// 返回节点名称，用于调试和性能分析
    fn name(&self) -> &str;

    /// 准备阶段：分配资源、编译管线、构建 BindGroup
    ///
    /// 拥有对引擎子系统的可变访问权限。
    fn prepare(&mut self, _ctx: &mut PrepareContext) {}

    /// 执行阶段：录制 GPU 渲染命令
    ///
    /// # 参数
    /// - `ctx`: 只读执行上下文，包含所有共享资源
    /// - `encoder`: GPU 命令编码器
    fn run(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder);
}
