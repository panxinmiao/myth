use crate::core::gpu::Tracked;

use super::graph::RenderGraph;
use super::types::{
    BufferDesc, BufferNodeId, GraphResourceType, ResourceClass, ResourceNodeId, TextureDesc,
    TextureNodeId,
};

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
        let id = self.graph.register_texture(name, desc, false);
        self.graph.storage.passes[self.pass_index]
            .creates
            .push(id.erase());
        self.write(id)
    }

    pub fn create_buffer(&mut self, name: &'static str, desc: BufferDesc) -> BufferNodeId {
        let id = self.graph.register_buffer(name, desc, false);
        self.graph.storage.passes[self.pass_index]
            .creates
            .push(id.erase());
        self.write(id)
    }

    pub fn read_external_texture(
        &mut self,
        name: &'static str,
        desc: TextureDesc,
        view: &Tracked<wgpu::TextureView>,
    ) -> TextureNodeId {
        let id = self.graph.import_external_texture(name, desc, view);
        self.read(id)
    }

    pub fn write_external_texture(
        &mut self,
        name: &'static str,
        desc: TextureDesc,
        view: &Tracked<wgpu::TextureView>,
    ) -> TextureNodeId {
        let id = self.graph.import_external_texture(name, desc, view);
        self.write(id)
    }

    pub fn read_external_buffer(
        &mut self,
        name: &'static str,
        desc: BufferDesc,
        buffer: &Tracked<wgpu::Buffer>,
    ) -> BufferNodeId {
        let id = self.graph.import_external_buffer(name, desc, buffer);
        self.read(id)
    }

    pub fn write_external_buffer(
        &mut self,
        name: &'static str,
        desc: BufferDesc,
        buffer: &Tracked<wgpu::Buffer>,
    ) -> BufferNodeId {
        let id = self.graph.import_external_buffer(name, desc, buffer);
        self.write(id)
    }

    pub fn read<T: GraphResourceType>(&mut self, id: ResourceNodeId<T>) -> ResourceNodeId<T> {
        let raw = id.erase();
        self.graph.storage.passes[self.pass_index].reads.push(raw);
        self.graph.storage.resources[raw.index() as usize]
            .consumers
            .push(self.pass_index);
        id
    }

    pub fn write<T: GraphResourceType>(&mut self, id: ResourceNodeId<T>) -> ResourceNodeId<T> {
        let raw = id.erase();
        let res = &mut self.graph.storage.resources[raw.index() as usize];
        let resource_class = match res.class() {
            ResourceClass::Texture => "Texture",
            ResourceClass::Buffer => "Buffer",
        };

        if let Some(existing_producer) = res.producer {
            panic!(
                "SSA Violation in Pass '{}': {} '{}' already has a producer (Pass '{}'). \
                 Use `builder.mutate()` to create a new version (alias).",
                self.graph.storage.passes[self.pass_index].name,
                resource_class,
                res.name,
                self.graph.storage.passes[existing_producer].name
            );
        }

        self.graph.storage.passes[self.pass_index].writes.push(raw);
        res.producer = Some(self.pass_index);
        id
    }

    #[inline]
    pub fn read_texture(&mut self, id: TextureNodeId) -> TextureNodeId {
        self.read(id)
    }

    #[inline]
    pub fn write_texture(&mut self, id: TextureNodeId) -> TextureNodeId {
        self.write(id)
    }

    #[inline]
    pub fn read_buffer(&mut self, id: BufferNodeId) -> BufferNodeId {
        self.read(id)
    }

    #[inline]
    pub fn write_buffer(&mut self, id: BufferNodeId) -> BufferNodeId {
        self.write(id)
    }

    #[must_use = "The returned resource handle must be used for downstream wiring"]
    pub fn mutate<T: GraphResourceType>(
        &mut self,
        input_id: ResourceNodeId<T>,
        new_name: &'static str,
    ) -> ResourceNodeId<T> {
        self.read(input_id);
        let new_id = self.graph.create_alias_typed(input_id, new_name);
        self.write(new_id)
    }

    #[must_use = "The returned TextureNodeId must be used for downstream wiring"]
    pub fn mutate_texture(
        &mut self,
        input_id: TextureNodeId,
        new_name: &'static str,
    ) -> TextureNodeId {
        self.mutate(input_id, new_name)
    }

    #[must_use = "The returned BufferNodeId must be used for downstream wiring"]
    pub fn mutate_buffer(
        &mut self,
        input_id: BufferNodeId,
        new_name: &'static str,
    ) -> BufferNodeId {
        self.mutate(input_id, new_name)
    }

    pub fn mark_side_effect(&mut self) {
        self.graph.storage.passes[self.pass_index].has_side_effect = true;
    }
}
