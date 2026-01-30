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
//! - FrameComposer: 帧合成器（链式 API）

pub mod builder;
pub mod composer;
pub mod context;
pub mod extracted;
pub mod frame;
pub mod graph;
pub mod node;
pub mod pass;
pub mod passes;
pub mod render_state;
pub mod stage;

pub use builder::FrameBuilder;
pub use composer::FrameComposer;
pub use context::RenderContext;
pub use extracted::{ExtractedRenderItem, ExtractedScene, ExtractedSkeleton};
pub use frame::RenderFrame;
pub use graph::RenderGraph;
pub use node::RenderNode;
pub use pass::TrackedRenderPass;
pub use passes::ForwardRenderPass;
pub use render_state::RenderState;
pub use stage::RenderStage;
