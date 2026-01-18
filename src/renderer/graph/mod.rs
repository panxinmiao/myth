//! 渲染管线组织
//!
//! 提供：
//! - RenderFrame: 每帧渲染逻辑管理
//! - RenderState: 渲染状态（相机、时间等）
//! - TrackedRenderPass: 带状态追踪的渲染通道
//! - RenderItem/RenderCommand: 渲染数据结构
//! - ExtractedScene: 提取阶段的场景数据
//! - RenderGraph: 渲染图执行器
//! - RenderNode: 渲染节点 Trait
//! - RenderContext: 渲染上下文

pub mod frame;
pub mod pass;
pub mod render_state;
pub mod extracted;
pub mod context;
pub mod node;
pub mod graph;
pub mod passes;

pub use frame::RenderFrame;
pub use pass::TrackedRenderPass;
pub use render_state::RenderState;
pub use extracted::{ExtractedScene, ExtractedRenderItem, ExtractedSkeleton};
pub use context::RenderContext;
pub use node::RenderNode;
pub use graph::RenderGraph;
pub use passes::ForwardRenderPass;
