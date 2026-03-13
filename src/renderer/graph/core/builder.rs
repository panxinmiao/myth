use super::graph::RenderGraph;
use super::types::{TextureDesc, TextureNodeId};

/// Builder for declaring a pass's resource dependencies during eager graph
/// construction.
///
/// Obtained exclusively inside the closure passed to
/// [`RenderGraph::add_pass`].  All topology wiring — resource creation, read
/// / write declarations, and alias production — happens **immediately** and
/// is captured before the closure returns.
pub struct PassBuilder<'graph, 'a> {
    pub(crate) graph: &'graph mut RenderGraph<'a>,
    pub(crate) pass_index: usize,
}

impl PassBuilder<'_, '_> {
    pub fn create_texture(&mut self, name: &'static str, desc: TextureDesc) -> TextureNodeId {
        let id = self.graph.register_resource(name, desc, false);
        self.graph.storage.passes[self.pass_index].creates.push(id);
        self.write_texture(id)
    }

    #[inline]
    pub fn create_and_export(&mut self, name: &'static str, desc: TextureDesc) -> TextureNodeId {
        self.create_texture(name, desc)
    }

    pub fn read_texture(&mut self, id: TextureNodeId) {
        self.graph.storage.passes[self.pass_index].reads.push(id);
        self.graph.storage.resources[id.0 as usize]
            .consumers
            .push(self.pass_index);
    }

    pub fn write_texture(&mut self, id: TextureNodeId) -> TextureNodeId {
        let res = &mut self.graph.storage.resources[id.0 as usize];

        if let Some(existing_producer) = res.producer {
            panic!(
                "SSA Violation in Pass '{}': Texture '{}' already has a producer (Pass '{}'). \
                 Use `builder.mutate_and_export()` to create a new version (alias).",
                self.graph.storage.passes[self.pass_index].name,
                res.name,
                self.graph.storage.passes[existing_producer].name
            );
        }

        self.graph.storage.passes[self.pass_index].writes.push(id);
        res.producer = Some(self.pass_index);
        id
    }

    #[must_use = "The returned TextureNodeId must be used for downstream wiring"]
    pub fn mutate_and_export(
        &mut self,
        input_id: TextureNodeId,
        new_name: &'static str,
    ) -> TextureNodeId {
        self.read_texture(input_id);
        let new_id = self.graph.create_alias(input_id, new_name);
        self.write_texture(new_id)
    }

    pub fn mark_side_effect(&mut self) {
        self.graph.storage.passes[self.pass_index].has_side_effect = true;
    }
}
