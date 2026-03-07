//! Graph Blackboard & Hook System
//!
//! The [`GraphBlackboard`] exposes a mutable resource cursor from the current
//! frame's render graph to external code (e.g. UI plugins, custom filters).
//! It carries the "current" colour buffer, scene depth, and final surface
//! target so hooks can read, modify, or override the main colour pipeline.
//!
//! [`HookStage`] determines *when* hooks run relative to the built-in
//! post-processing chain.

use super::types::TextureNodeId;

/// Mutable resource cursor published by the engine's graph builder.
///
/// The Composer updates `current_color` as rendering progresses through the
/// pipeline. When a hook stage is reached, the engine pauses the imperative
/// chain, stores the current state into the blackboard, and hands it to each
/// registered hook. Hooks may insert passes that read/write `current_color`
/// (and update it to their output). After all hooks complete, the engine
/// reclaims `current_color` and continues the pipeline.
///
/// # Fields
///
/// | Slot | Semantic | Mutability |
/// |------|----------|------------|
/// | `current_color` | Active colour buffer cursor | Read + Write |
/// | `scene_depth` | Main depth buffer (reverse-Z) | Read-only |
/// | `surface_out` | Final swap-chain output | Read-only |
pub struct GraphBlackboard {
    /// Active colour buffer cursor. Hooks may read this and replace it with
    /// the output of their injected passes to chain into the main pipeline.
    pub current_color: TextureNodeId,
    /// Main depth buffer (reverse-Z, written by scene passes). Typically
    /// read-only for hooks.
    pub scene_depth: TextureNodeId,
    /// Final swap-chain output target. UI and overlays should write here.
    pub surface_out: TextureNodeId,
}

/// Injection stage for custom pass hooks.
///
/// Determines **when** in the pipeline the hook's passes are wired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookStage {
    /// After scene rendering, before post-processing (Bloom, ToneMap, FXAA).
    BeforePostProcess,
    /// After all post-processing, before surface presentation.
    /// This is the typical stage for UI overlays.
    AfterPostProcess,
}
