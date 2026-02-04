//! Render Graph and Frame Management
//!
//! This module handles per-frame rendering organization:
//!
//! - [`RenderFrame`]: Per-frame render logic management
//! - [`RenderState`]: Render state (camera, time, etc.)
//! - [`TrackedRenderPass`]: Render pass with state tracking
//! - [`ExtractedScene`]: Scene data extracted for GPU rendering
//! - [`RenderGraph`]: Render graph executor
//! - [`RenderNode`]: Render node trait for custom passes
//! - [`RenderStage`]: Render stage definitions (Opaque, Transparent, UI, etc.)
//! - [`FrameBuilder`]: Frame construction utilities
//! - [`FrameComposer`]: Chainable API for frame composition
//! - [`RenderLists`]: Sorted render command lists
//!
//! # Frame Lifecycle
//!
//! 1. **Extract**: Scene data is copied into GPU-friendly format
//! 2. **Prepare**: Resources are uploaded and bind groups created
//! 3. **Queue**: Render items are sorted by material/distance
//! 4. **Render**: Render nodes execute their passes

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
pub use frame::{RenderFrame, RenderLists, RenderCommand, RenderKey};
pub use graph::RenderGraph;
pub use node::RenderNode;
pub use pass::TrackedRenderPass;
pub use passes::ForwardRenderPass;
pub use render_state::RenderState;
pub use stage::RenderStage;
