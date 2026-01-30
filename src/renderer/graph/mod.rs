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
//! - RenderStage: 渲染阶段定义
//! - FrameBuilder: 帧构建器

pub mod frame;
pub mod pass;
pub mod render_state;
pub mod extracted;
pub mod context;
pub mod node;
pub mod graph;
pub mod passes;
pub mod stage;
pub mod builder;

pub use frame::{RenderFrame, PreparedFrame};
pub use pass::TrackedRenderPass;
pub use render_state::RenderState;
pub use extracted::{ExtractedScene, ExtractedRenderItem, ExtractedSkeleton};
pub use context::RenderContext;
pub use node::RenderNode;
pub use graph::RenderGraph;
pub use passes::ForwardRenderPass;
pub use stage::RenderStage;
pub use builder::FrameBuilder;
