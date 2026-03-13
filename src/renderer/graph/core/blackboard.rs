//! Graph Blackboard & Custom Pass Hooks
//!
//! The [`GraphBlackboard`] is a lightweight, pass-by-value **cursor** over
//! well-known resource slots from the current frame's render graph.  External
//! code (e.g. UI plugins) receives a copy, wires its passes, and returns an
//! updated copy — there is no shared mutable state.
//!
//! [`CustomPassHook`] is a builder-time callback that lets external code
//! inject arbitrary [`PassNode`]s into the graph at a chosen stage.

use super::types::TextureNodeId;

/// Well-known resource slots published by the engine's graph builder.
///
/// This is a **value type** (`Copy + Clone`).  Hooks receive the blackboard
/// by value, modify the slots they care about, and return a fresh copy.
/// The Rust compiler enforces that every hook path returns a valid
/// blackboard — forgetting to do so is a compile-time error.
///
/// # Fields
///
/// | Slot | Semantic | Typical consumer |
/// |------|----------|------------------|
/// | `scene_color` | HDR scene colour buffer | Custom post-FX |
/// | `scene_depth` | Main depth buffer (reverse-Z) | Depth-aware FX |
/// | `surface_out` | Final swap-chain output | UI overlay |
#[derive(Clone, Copy)]
pub struct GraphBlackboard {
    /// HDR scene colour render target (written by Opaque / Skybox / Transparent).
    pub scene_color: TextureNodeId,
    /// Main depth buffer (reverse-Z, written by scene passes).
    pub scene_depth: TextureNodeId,
    /// Final swap-chain output target.  UI and overlays should write here.
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
/// The closure **consumes** the current [`GraphBlackboard`] and **returns**
/// a potentially updated copy.  This pure-functional signature ensures that
/// the caller cannot silently forget to propagate modified resource slots
/// — the Rust type system makes omission a compile-time error.
///
/// # Example
///
/// ```rust,ignore
/// renderer.add_custom_pass_hook(HookStage::AfterPostProcess, |graph, cursor| {
///     let new_surface = graph.add_pass("UI_Pass", |builder| {
///         let exported = builder.mutate_and_export(cursor.surface_out, "Surface_With_UI");
///         let ui_node = UiPassNode { target_tex: exported, .. };
///         (ui_node, exported)
///     });
///     GraphBlackboard { surface_out: new_surface, ..cursor }
/// });
/// ```
pub type CustomPassHook<'a> = Box<
    dyn for<'g> FnMut(&mut super::graph::RenderGraph<'g>, GraphBlackboard) -> GraphBlackboard + 'a,
>;
