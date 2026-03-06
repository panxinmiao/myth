use super::graph::{FrameConfig, RenderGraph};
use super::types::TextureNodeId;

/// Builder for declaring a pass's resource dependencies.
///
/// Provides ergonomic APIs for creating, reading, and writing texture
/// resources within the graph, as well as looking up shared resources
/// by name via the graph's resource registry.
///
/// # Blackboard API
///
/// The "blackboard" is the graph's named resource registry.  Well-known
/// resources (e.g. `"Scene_Color_HDR"`, `"Scene_Depth"`) are registered
/// there by either the Composer or by producer passes.
///
/// - [`read_blackboard`](Self::read_blackboard) — read a required resource.
/// - [`write_blackboard`](Self::write_blackboard) — write a required resource.
/// - [`try_read_blackboard`](Self::try_read_blackboard) — read an optional resource.
/// - [`create_and_export`](Self::create_and_export) — create a resource and
///   publish it to the blackboard for downstream consumers.
///
/// # Frame Configuration
///
/// The builder exposes the current frame's resolution and device format
/// information via [`frame_config`](Self::frame_config), so passes can
/// derive their own `RdgTextureDesc`s in `setup()` without external push.
pub struct PassBuilder<'a> {
    pub(crate) graph: &'a mut RenderGraph,
    pub(crate) pass_index: usize,
}

impl<'a> PassBuilder<'a> {
    // ─── Frame Configuration ─────────────────────────────────────────

    /// Returns the current frame's rendering configuration (resolution,
    /// depth format, MSAA samples, surface/HDR formats).
    #[inline]
    pub fn frame_config(&self) -> &FrameConfig {
        self.graph.frame_config()
    }

    /// Shorthand for `(config.width, config.height)`.
    #[inline]
    pub fn global_resolution(&self) -> (u32, u32) {
        let c = self.graph.frame_config();
        (c.width, c.height)
    }

    // ─── Low-Level Resource API ──────────────────────────────────────

    /// Creates a new transient texture resource owned by this pass.
    pub fn create_texture(
        &mut self,
        name: &'static str,
        desc: super::types::RdgTextureDesc,
    ) -> TextureNodeId {
        let id = self.graph.register_resource(name, desc, false);
        self.graph.passes[self.pass_index].creates.push(id);
        self.write_texture(id)
    }

    /// Declares that this pass reads from the given texture resource.
    pub fn read_texture(&mut self, id: TextureNodeId) {
        self.graph.passes[self.pass_index].reads.push(id);
        self.graph.resources[id.0 as usize]
            .consumers
            .push(self.pass_index);
    }

    /// Declares that this pass writes to the given texture resource.
    pub fn write_texture(&mut self, id: TextureNodeId) -> TextureNodeId {
        self.graph.passes[self.pass_index].writes.push(id);
        self.graph.resources[id.0 as usize]
            .producers
            .push(self.pass_index);
        id
    }

    /// Marks this pass as having an externally-visible side effect.
    ///
    /// Side-effect passes are never culled, even if they don't write to
    /// any external resource.
    pub fn mark_side_effect(&mut self) {
        self.graph.passes[self.pass_index].has_side_effect = true;
    }

    // ─── Blackboard API (Semantic Resource Wiring) ───────────────────

    /// Reads a **required** resource from the blackboard.
    ///
    /// Convenience for `find_resource(name).unwrap()` + `read_texture()`.
    /// Panics if the resource has not been registered.
    #[inline]
    pub fn read_blackboard(&mut self, name: &str) -> TextureNodeId {
        let id = self
            .graph
            .find_resource(name)
            .unwrap_or_else(|| panic!("{name} must be registered before this pass"));
        self.read_texture(id);
        id
    }

    /// Writes a **required** resource from the blackboard.
    ///
    /// Convenience for `find_resource(name).unwrap()` + `write_texture()`.
    /// Panics if the resource has not been registered.
    #[inline]
    pub fn write_blackboard(&mut self, name: &str) -> TextureNodeId {
        let id = self
            .graph
            .find_resource(name)
            .unwrap_or_else(|| panic!("{name} must be registered before this pass"));
        self.write_texture(id);
        id
    }

    /// Reads an **optional** resource from the blackboard.
    ///
    /// Returns `None` if the resource has not been registered.
    /// If present, registers a read-dependency automatically.
    #[inline]
    pub fn try_read_blackboard(&mut self, name: &str) -> Option<TextureNodeId> {
        let id = self.graph.find_resource(name)?;
        self.read_texture(id);
        Some(id)
    }
}
