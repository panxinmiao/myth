use super::graph::RenderGraph;
use super::types::TextureNodeId;

pub struct PassBuilder<'a> {
    pub(crate) graph: &'a mut RenderGraph,
    pub(crate) pass_index: usize,
}

impl<'a> PassBuilder<'a> {
    pub fn create_texture(&mut self, name: &'static str) -> TextureNodeId {
        let id = self.graph.register_resource(name, false);
        self.graph.passes[self.pass_index].creates.push(id);
        id
    }

    pub fn read_texture(&mut self, id: TextureNodeId) {
        self.graph.passes[self.pass_index].reads.push(id);
        self.graph.resources[id.0 as usize]
            .consumers
            .push(self.pass_index);
    }

    pub fn write_texture(&mut self, id: TextureNodeId) -> TextureNodeId {
        self.graph.passes[self.pass_index].writes.push(id);
        self.graph.resources[id.0 as usize]
            .producers
            .push(self.pass_index);
        id
    }

    pub fn mark_side_effect(&mut self) {
        self.graph.passes[self.pass_index].has_side_effect = true;
    }
}
