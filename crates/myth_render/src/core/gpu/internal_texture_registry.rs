//! Registry for internally generated texture views.

use rustc_hash::FxHashMap;

use super::generate_gpu_resource_id;

/// Internally generated texture views such as render targets and utility LUTs.
#[derive(Default)]
pub(crate) struct InternalTextureRegistry {
    /// Key: Resource ID (u64), value: `wgpu::TextureView`.
    resources: FxHashMap<u64, wgpu::TextureView>,
    /// Mapping from stable internal names to resource IDs.
    name_lookup: FxHashMap<String, u64>,
}

impl InternalTextureRegistry {
    pub(super) fn register_direct(&mut self, id: u64, view: wgpu::TextureView) {
        self.resources.insert(id, view);
    }

    pub(super) fn register_by_name(&mut self, name: &str, view: wgpu::TextureView) -> u64 {
        let id = *self
            .name_lookup
            .entry(name.to_string())
            .or_insert_with(generate_gpu_resource_id);
        self.register_direct(id, view);
        id
    }

    pub(super) fn register(&mut self, view: wgpu::TextureView) -> u64 {
        let id = generate_gpu_resource_id();
        self.resources.insert(id, view);
        id
    }

    #[inline]
    #[must_use]
    pub(super) fn get(&self, id: u64) -> Option<&wgpu::TextureView> {
        self.resources.get(&id)
    }

    pub(super) fn remove(&mut self, id: u64) {
        self.resources.remove(&id);
    }
}
