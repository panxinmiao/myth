//! 渲染管线组织
//!
//! 提供：
//! - RenderFrame: 每帧渲染逻辑管理
//! - RenderState: 渲染状态（相机、时间等）
//! - TrackedRenderPass: 带状态追踪的渲染通道
//! - RenderItem/RenderCommand: 渲染数据结构

pub mod frame;
pub mod pass;
pub mod render_state;

pub use frame::RenderFrame;
pub use pass::TrackedRenderPass;
pub use render_state::RenderState;
