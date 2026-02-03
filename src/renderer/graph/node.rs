//! 渲染节点 Trait
//!
//! 定义渲染图中节点的抽象接口。
//! 每个节点代表一个渲染 Pass 或计算任务。

use super::context::RenderContext;

/// 渲染节点 Trait
/// 
/// 所有渲染 Pass 必须实现此接口。
/// 
/// # 设计原则
/// - `run` 方法接收 `RenderContext` 和 `CommandEncoder`
/// - 节点应该是无状态的或仅持有配置数据
/// - 实际的 GPU 资源通过 `RenderContext` 访问
/// 
/// # 性能考虑
/// - 避免在 `run` 中进行内存分配
/// - 利用 `encoder.push_debug_group` 进行 GPU 调试
/// 

pub trait RenderNode {
    /// 返回节点名称，用于调试和性能分析
    fn name(&self) -> &str;

    #[inline]
    fn output_to_screen(&self) -> bool{
        false
    }

    fn prepare(&mut self, _ctx: &mut RenderContext) {}
    
    /// 执行渲染逻辑
    /// 
    /// # 参数
    /// - `ctx`: 渲染上下文，包含所有共享资源
    /// - `encoder`: GPU 命令编码器
    fn run(
        &self, 
        ctx: &mut RenderContext, 
        encoder: &mut wgpu::CommandEncoder
    );
}
