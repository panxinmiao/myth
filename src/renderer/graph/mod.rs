//! Render Graph and Frame Management
//!
//! This module handles per-frame rendering organization:
//!
//! - [`RenderFrame`]: Per-frame render logic management
//! - [`RenderState`]: Render state (camera, time, etc.)
//! - [`ExtractedScene`]: Scene data extracted for GPU rendering
//! - [`FrameComposer`]: Chainable API for frame composition (hook-based RDG)
//! - [`RenderLists`]: Sorted render command lists
//!
//! # Frame Lifecycle
//!
//! 1. **Extract**: Scene data is copied into GPU-friendly format
//! 2. **Prepare**: Resources are uploaded and bind groups created
//! 3. **Queue**: Render items are sorted by material/distance
//! 4. **Render**: RDG passes execute their passes

pub mod bake;
pub mod composer;
pub mod culling;
pub mod extracted;
pub mod frame;
pub mod rdg;
pub mod render_state;
pub mod shadow_utils;

pub use composer::FrameComposer;
pub use extracted::{ExtractedRenderItem, ExtractedScene, ExtractedSkeleton};
pub use frame::{
    BakedRenderLists, DrawCommand, RenderCommand, RenderFrame, RenderKey, RenderLists,
};
pub use render_state::RenderState;
