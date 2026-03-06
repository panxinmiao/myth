use super::graph::RenderGraph;
use super::types::TextureNodeId;

/// Builder for declaring a pass's resource dependencies.
///
/// Provides ergonomic APIs for creating, reading, and writing texture
/// resources within the graph, as well as looking up shared resources
/// by name via the graph's resource registry.
pub struct PassBuilder<'a> {
    pub(crate) graph: &'a mut RenderGraph,
    pub(crate) pass_index: usize,
}

impl<'a> PassBuilder<'a> {
    /// Creates a new transient texture resource owned by this pass.
    pub fn create_texture(
        &mut self,
        name: &'static str,
        desc: super::types::RdgTextureDesc,
    ) -> TextureNodeId {
        let id = self.graph.register_resource(name, desc, false);
        self.graph.passes[self.pass_index].creates.push(id);
        id
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

    /// Looks up a resource by name in the graph's resource registry.
    ///
    /// Returns `None` if no resource with that name has been registered
    /// in the current frame. Useful for passes that want to self-wire
    /// to well-known resources (e.g. "Scene_Color_HDR", "Scene_Depth").
    #[inline]
    pub fn find_resource(&self, name: &str) -> Option<TextureNodeId> {
        self.graph.find_resource(name)
    }
}
