//! Persistent bind-group, layout, and shader-interface caches.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::pipeline::vertex::VertexLayoutSignature;

use super::{ResourceIdSet, generate_gpu_resource_id};

/// GPU global state (Group 0)
///
/// Contains Camera Uniforms, Light Storage Buffer, Environment Maps, etc.
///
/// Uses an "Ensure -> Collect IDs -> Check Fingerprint -> Rebind" pattern
pub struct GpuGlobalState {
    pub id: u32,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_id: u64,
    pub layout: wgpu::BindGroupLayout,
    pub layout_id: u64,
    pub binding_wgsl: String,
    /// Set of physical IDs of all dependent resources (used for automatic change detection)
    pub resource_ids: ResourceIdSet,
    pub last_used_frame: u64,
}

// Object BindGroup cache key (using the hash value of ResourceIdSet)
pub(crate) type ObjectBindGroupKey = u64;

#[derive(Clone)]
pub struct BindGroupContext {
    pub layout: wgpu::BindGroupLayout,
    pub layout_id: u64,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_id: u64,
    pub binding_wgsl: Arc<str>,
}

/// Persistent bind-group, layout, and shader-interface caches.
#[derive(Default)]
pub(crate) struct BindGroupStore {
    global_states: FxHashMap<u64, GpuGlobalState>,
    layout_cache: FxHashMap<Vec<wgpu::BindGroupLayoutEntry>, (wgpu::BindGroupLayout, u64)>,
    vertex_layout_cache: FxHashMap<VertexLayoutSignature, u64>,
    object_bind_group_cache: FxHashMap<ObjectBindGroupKey, BindGroupContext>,
    bind_group_id_lookup: FxHashMap<u64, BindGroupContext>,
}

impl BindGroupStore {
    pub(super) fn prune(&mut self, cutoff: u64) {
        self.global_states
            .retain(|_, v| v.last_used_frame >= cutoff);
    }

    #[inline]
    pub(super) fn clear_object_bind_group_caches(&mut self) {
        self.object_bind_group_cache.clear();
        self.bind_group_id_lookup.clear();
    }

    #[inline]
    #[must_use]
    pub(super) fn cached_object_bind_group(
        &self,
        key: ObjectBindGroupKey,
    ) -> Option<&BindGroupContext> {
        self.object_bind_group_cache.get(&key)
    }

    #[inline]
    pub(super) fn cache_object_bind_group(
        &mut self,
        key: ObjectBindGroupKey,
        bind_group_id: u64,
        context: BindGroupContext,
    ) {
        self.object_bind_group_cache.insert(key, context.clone());
        self.bind_group_id_lookup.insert(bind_group_id, context);
    }

    #[inline]
    #[must_use]
    pub(super) fn cached_bind_group(&self, bind_group_id: u64) -> Option<&BindGroupContext> {
        self.bind_group_id_lookup.get(&bind_group_id)
    }

    pub(super) fn get_or_create_layout(
        &mut self,
        device: &wgpu::Device,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> (wgpu::BindGroupLayout, u64) {
        if let Some(layout) = self.layout_cache.get(entries) {
            return layout.clone();
        }

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cached BindGroupLayout"),
            entries,
        });

        let id = generate_gpu_resource_id();
        self.layout_cache
            .insert(entries.to_vec(), (layout.clone(), id));
        (layout, id)
    }

    pub(super) fn get_or_create_vertex_layout_id(
        &mut self,
        signature: VertexLayoutSignature,
    ) -> u64 {
        if let Some(&id) = self.vertex_layout_cache.get(&signature) {
            return id;
        }

        let id = generate_gpu_resource_id();
        self.vertex_layout_cache.insert(signature, id);
        id
    }

    #[inline]
    #[must_use]
    pub(super) fn global_state(&self, state_id: u64) -> Option<&GpuGlobalState> {
        self.global_states.get(&state_id)
    }

    #[inline]
    pub(super) fn global_state_mut(&mut self, state_id: u64) -> Option<&mut GpuGlobalState> {
        self.global_states.get_mut(&state_id)
    }

    #[inline]
    pub(super) fn insert_global_state(&mut self, state_id: u64, gpu_state: GpuGlobalState) {
        self.global_states.insert(state_id, gpu_state);
    }
}
