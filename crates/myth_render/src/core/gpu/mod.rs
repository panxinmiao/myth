//! GPU Resource Manager
//!
//! Responsible for creating, updating, and managing GPU-side resources.
//!
//! Uses a modular design with different responsibilities split into separate files:
//! - buffer.rs: Buffer operations
//! - texture.rs: Texture and Image operations
//! - geometry.rs: Geometry operations
//! - material.rs: Material operations
//! - binding.rs: `BindGroup` operations
//! - resource_store.rs: GPU asset arenas and reverse indices
//! - bind_group_store.rs: `BindGroup`, layout, and shader-interface caches
//! - environment_store.rs: Scene environment and IBL utility resources
//! - internal_texture_registry.rs: Internally generated texture views
//! - allocator.rs: `ModelBufferAllocator`
//! - `resource_ids.rs`: Resource ID tracking and change detection
//!
//! # Resource Management Architecture
//!
//! Uses an "Ensure -> Check -> Rebuild" pattern:
//!
//! 1. **Ensure phase**: Ensure GPU resources exist with up-to-date data, returning physical resource IDs
//! 2. **Check phase**: Compare resource IDs for changes, deciding whether to rebuild `BindGroup`
//! 3. **Rebuild phase**: If rebuild is needed, collect `LayoutEntries` and check if a new Layout is required

mod allocator;
mod bind_group_store;
mod binding;
mod buffer;
mod environment;
mod environment_store;
mod geometry;
mod internal_texture_registry;
mod material;
mod mipmap;
mod resource_ids;
mod resource_store;
mod sampler_registry;
mod system_textures;
mod texture;
mod tracked;

use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) use crate::core::gpu::buffer::GpuBuffer;
pub use crate::core::gpu::buffer::GpuBufferHandle;
pub(crate) use crate::core::gpu::environment::EnvironmentComputeState;
pub(crate) use crate::core::gpu::environment::GpuEnvironment;
pub(crate) use crate::core::gpu::environment::{BRDF_LUT_SIZE, CubeSourceType};
pub(crate) use crate::core::gpu::geometry::GpuGeometry;
pub(crate) use crate::core::gpu::material::GpuMaterial;
pub(crate) use crate::core::gpu::texture::{GpuImage, ResourceState, TextureBinding};
pub use bind_group_store::{BindGroupContext, GpuGlobalState};
pub(crate) use bind_group_store::{BindGroupStore, ObjectBindGroupKey};
pub(crate) use environment_store::EnvironmentStore;
pub(crate) use internal_texture_registry::InternalTextureRegistry;

pub use crate::core::gpu::mipmap::MipmapGenerator;
pub(crate) use crate::core::gpu::mipmap::MipmapRequest;
pub use allocator::ModelBufferAllocator;
use myth_resources::buffer::{CpuBuffer, GpuData};
pub use resource_ids::{
    BindGroupFingerprint, EnsureResult, ResourceId, ResourceIdSet, hash_layout_entries,
};
pub(crate) use resource_store::GpuResourceStore;
pub use sampler_registry::{CommonSampler, SamplerRegistry};
pub use system_textures::SystemTextures;
pub use tracked::Tracked;

static NEXT_GPU_RESOURCE_ID: AtomicU64 = AtomicU64::new(1);

pub fn generate_gpu_resource_id() -> u64 {
    NEXT_GPU_RESOURCE_ID.fetch_add(1, Ordering::Relaxed)
}

// ============================================================================
// Resource Manager main structure
// ============================================================================

pub struct ResourceManager {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) frame_index: u64,

    pub(crate) resources: GpuResourceStore,
    pub(crate) bind_groups: BindGroupStore,
    pub(crate) environments: EnvironmentStore,
    pub(crate) internal_textures: InternalTextureRegistry,

    pub(crate) sampler_registry: SamplerRegistry,

    // === Model Buffer Allocator ===
    pub(crate) model_allocator: ModelBufferAllocator,

    /// Global system fallback textures and Group 3 bind-group infrastructure.
    ///
    /// See [`SystemTextures`] for the full list of data-semantic fallback
    /// textures and the screen bind-group layout / samplers.
    pub system_textures: SystemTextures,
}

impl ResourceManager {
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, anisotropy_clamp: u16) -> Self {
        let model_allocator = ModelBufferAllocator::new();

        let mut gpu_buffers = slotmap::SlotMap::with_key();
        let mut buffer_index = rustc_hash::FxHashMap::default();

        // Force initial allocation of the model buffer so that it has a stable GPU handle and ID from the start.
        model_allocator.flush_to_buffer(&device, &queue, &mut gpu_buffers, &mut buffer_index, 0);

        let system_textures = SystemTextures::new(&device, &queue);

        let sampler_registry = SamplerRegistry::new(&device, anisotropy_clamp);

        Self {
            device,
            queue,
            frame_index: 0,
            resources: GpuResourceStore::new(gpu_buffers, buffer_index),
            bind_groups: BindGroupStore::default(),
            environments: EnvironmentStore::default(),
            internal_textures: InternalTextureRegistry::default(),
            sampler_registry,
            model_allocator,
            system_textures,
        }
    }

    pub fn next_frame(&mut self) {
        self.frame_index += 1;
        self.model_allocator.reset();
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    #[inline]
    #[must_use]
    pub fn sampler_registry(&self) -> &SamplerRegistry {
        &self.sampler_registry
    }

    #[inline]
    pub fn set_global_anisotropy(&mut self, anisotropy_clamp: u16) {
        self.sampler_registry
            .set_global_anisotropy(anisotropy_clamp);
    }

    #[inline]
    #[must_use]
    pub fn system_textures(&self) -> &SystemTextures {
        &self.system_textures
    }

    #[must_use]
    pub(crate) fn take_mipmap_requests(&mut self) -> Vec<MipmapRequest> {
        self.resources.take_mipmap_requests()
    }

    pub fn flush_model_buffers(&mut self) {
        let (gpu_buffers, buffer_index) = self.resources.buffer_storage_mut();
        let resized = self.model_allocator.flush_to_buffer(
            &self.device,
            &self.queue,
            gpu_buffers,
            buffer_index,
            self.frame_index,
        );

        if resized {
            self.bind_groups.clear_object_bind_group_caches();
            log::info!("Model buffer resized. Object BindGroup caches cleared.");
        }
    }

    /// Allocate a Model Uniform slot, returning the byte offset
    #[inline]
    pub fn allocate_model_uniform(
        &mut self,
        data: myth_resources::uniforms::DynamicModelUniforms,
    ) -> u32 {
        self.model_allocator.allocate(data)
    }

    /// Get the current Model Buffer ID for cache validation
    #[inline]
    pub fn model_buffer_id(&self) -> u64 {
        self.model_allocator.buffer_handle().id()
    }

    /// Quickly retrieve `BindGroup` data by cached ID
    #[inline]
    pub fn get_cached_bind_group(&self, cached_bind_group_id: u64) -> Option<&BindGroupContext> {
        self.bind_groups.cached_bind_group(cached_bind_group_id)
    }

    pub fn prune(&mut self, ttl_frames: u64) {
        if self.frame_index < ttl_frames {
            return;
        }
        let cutoff = self.frame_index - ttl_frames;

        self.environments.prune(cutoff, &mut self.internal_textures);
        self.resources.prune(cutoff);
        self.bind_groups.prune(cutoff);
    }
}
