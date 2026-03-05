//! Graph Blackboard & Custom Pass Hooks
//!
//! The [`GraphBlackboard`] exposes well-known resource slots from the current
//! frame's render graph to external code (e.g. UI plugins). Instead of
//! guessing internal resource IDs, users receive a typed structure with the
//! exact [`TextureNodeId`]s they need for correct wiring.
//!
//! [`CustomPassHook`] is a builder-time callback that lets external code
//! inject arbitrary [`PassNode`]s into the graph at a chosen stage.

use super::types::TextureNodeId;

/// Well-known resource slots published by the engine's graph builder.
///
/// The Composer populates this structure after registering the frame's core
/// resources. External hooks read it to wire their passes into the correct
/// attachment points.
///
/// # Fields
///
/// | Slot | Semantic | Typical consumer |
/// |------|----------|------------------|
/// | `scene_color` | HDR scene colour buffer | Custom post-FX |
/// | `scene_depth` | Main depth buffer (reverse-Z) | Depth-aware FX |
/// | `surface_out` | Final swap-chain output | UI overlay |
pub struct GraphBlackboard {
    /// HDR scene colour render target (written by Opaque / Skybox / Transparent).
    pub scene_color: TextureNodeId,
    /// Main depth buffer (reverse-Z, written by scene passes).
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

/// A builder-time callback that injects passes into the render graph.
///
/// The closure receives a mutable reference to the [`RenderGraph`] and a
/// read-only [`GraphBlackboard`] so it can register resources and wire
/// passes using the published slots.
///
/// # Example
///
/// ```rust,ignore
/// renderer.add_custom_pass_hook(HookStage::AfterPostProcess, |rdg, bb| {
///     let mut ui = UiPass { target_tex: bb.surface_out, .. };
///     rdg.add_pass(&mut ui);
/// });
/// ```
pub type CustomPassHook<'a> = Box<dyn FnMut(&mut super::graph::RenderGraph, &GraphBlackboard) + 'a>;
