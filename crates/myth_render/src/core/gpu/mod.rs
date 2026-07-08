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
mod binding;
mod buffer;
mod environment;
mod geometry;
mod material;
mod mipmap;
mod resource_ids;
mod sampler_registry;
mod system_textures;
mod texture;
mod tracked;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use rustc_hash::{FxHashMap, FxHashSet};
use slotmap::SecondaryMap;

use myth_assets::{GeometryHandle, ImageHandle, MaterialHandle, TextureHandle};

pub(crate) use crate::core::gpu::buffer::GpuBuffer;
pub use crate::core::gpu::buffer::GpuBufferHandle;
pub(crate) use crate::core::gpu::environment::EnvironmentComputeState;
pub(crate) use crate::core::gpu::environment::GpuEnvironment;
pub(crate) use crate::core::gpu::environment::{BRDF_LUT_SIZE, CubeSourceType};
pub(crate) use crate::core::gpu::geometry::GpuGeometry;
pub(crate) use crate::core::gpu::material::GpuMaterial;
pub(crate) use crate::core::gpu::texture::{GpuImage, ResourceState, TextureBinding};
use crate::pipeline::vertex::VertexLayoutSignature;

pub use crate::core::gpu::mipmap::MipmapGenerator;
pub(crate) use crate::core::gpu::mipmap::MipmapRequest;
pub use allocator::ModelBufferAllocator;
use myth_resources::buffer::{CpuBuffer, GpuData};
pub use resource_ids::{
    BindGroupFingerprint, EnsureResult, ResourceId, ResourceIdSet, hash_layout_entries,
};
pub use sampler_registry::{CommonSampler, SamplerRegistry};
pub use system_textures::SystemTextures;
pub use tracked::Tracked;

static NEXT_GPU_RESOURCE_ID: AtomicU64 = AtomicU64::new(1);

pub fn generate_gpu_resource_id() -> u64 {
    NEXT_GPU_RESOURCE_ID.fetch_add(1, Ordering::Relaxed)
}

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

// ============================================================================
// Compact BindGroup cache key
// ============================================================================

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

// ============================================================================
// ResourceManager state components
// ============================================================================

/// GPU-side asset arenas and reverse indices.
///
/// This is the part of [`ResourceManager`] that mirrors CPU asset handles to
/// persistent GPU allocations. Keeping it separate lets callers borrow asset
/// storage independently from bind-group and environment state.
pub(crate) struct GpuResourceStore {
    pub gpu_geometries: SecondaryMap<GeometryHandle, GpuGeometry>,
    pub gpu_materials: SecondaryMap<MaterialHandle, GpuMaterial>,
    pub gpu_images: SecondaryMap<ImageHandle, GpuImage>,

    /// Mapping from `TextureHandle` to (`ImageId`, `SamplerId`)
    pub texture_bindings: SecondaryMap<TextureHandle, TextureBinding>,

    /// All GPU buffers stored in a contiguous arena for O(1) handle-based access.
    pub gpu_buffers: slotmap::SlotMap<GpuBufferHandle, GpuBuffer>,
    /// Reverse index: CPU-side buffer ID -> SlotMap handle.
    pub buffer_index: FxHashMap<u64, GpuBufferHandle>,

    pending_mipmap_requests: Vec<MipmapRequest>,
    pending_mipmap_keys: FxHashSet<(u64, u64)>,
}

impl GpuResourceStore {
    #[must_use]
    fn new(
        gpu_buffers: slotmap::SlotMap<GpuBufferHandle, GpuBuffer>,
        buffer_index: FxHashMap<u64, GpuBufferHandle>,
    ) -> Self {
        Self {
            gpu_geometries: SecondaryMap::new(),
            gpu_materials: SecondaryMap::new(),
            gpu_images: SecondaryMap::new(),
            texture_bindings: SecondaryMap::new(),
            gpu_buffers,
            buffer_index,
            pending_mipmap_requests: Vec::new(),
            pending_mipmap_keys: FxHashSet::default(),
        }
    }

    pub fn queue_mipmap_request(&mut self, request: MipmapRequest) {
        if self.pending_mipmap_keys.insert(request.key()) {
            self.pending_mipmap_requests.push(request);
        }
    }

    #[must_use]
    pub fn take_mipmap_requests(&mut self) -> Vec<MipmapRequest> {
        self.pending_mipmap_keys.clear();
        std::mem::take(&mut self.pending_mipmap_requests)
    }
}

/// Persistent bind-group, layout, and shader-interface caches.
#[derive(Default)]
pub(crate) struct BindGroupStore {
    pub global_states: FxHashMap<u64, GpuGlobalState>,
    pub layout_cache: FxHashMap<Vec<wgpu::BindGroupLayoutEntry>, (wgpu::BindGroupLayout, u64)>,
    pub vertex_layout_cache: FxHashMap<VertexLayoutSignature, u64>,
    pub object_bind_group_cache: FxHashMap<ObjectBindGroupKey, BindGroupContext>,
    pub bind_group_id_lookup: FxHashMap<u64, BindGroupContext>,
}

/// Internally generated texture views such as render targets and utility LUTs.
#[derive(Default)]
pub(crate) struct InternalTextureRegistry {
    /// Key: Resource ID (u64), value: `wgpu::TextureView`.
    pub resources: FxHashMap<u64, wgpu::TextureView>,
    /// Mapping from stable internal names to resource IDs.
    pub name_lookup: FxHashMap<String, u64>,
}

impl InternalTextureRegistry {
    pub fn register_direct(&mut self, id: u64, view: wgpu::TextureView) {
        self.resources.insert(id, view);
    }

    pub fn register_by_name(&mut self, name: &str, view: wgpu::TextureView) -> u64 {
        let id = *self
            .name_lookup
            .entry(name.to_string())
            .or_insert_with(generate_gpu_resource_id);
        self.register_direct(id, view);
        id
    }

    pub fn register(&mut self, view: wgpu::TextureView) -> u64 {
        let id = generate_gpu_resource_id();
        self.resources.insert(id, view);
        id
    }

    #[inline]
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&wgpu::TextureView> {
        self.resources.get(&id)
    }

    pub fn remove(&mut self, id: u64) {
        self.resources.remove(&id);
    }
}

/// Scene environment and global IBL utility resources.
#[derive(Default)]
pub(crate) struct EnvironmentStore {
    pub scene_gpu_environments: FxHashMap<u32, GpuEnvironment>,
    pub brdf_lut_texture: Option<wgpu::Texture>,
    pub brdf_lut_view_id: Option<u64>,
    pub needs_brdf_compute: bool,
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
        let resized = self.model_allocator.flush_to_buffer(
            &self.device,
            &self.queue,
            &mut self.resources.gpu_buffers,
            &mut self.resources.buffer_index,
            self.frame_index,
        );

        if resized {
            self.bind_groups.object_bind_group_cache.clear();
            self.bind_groups.bind_group_id_lookup.clear();
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
        self.bind_groups
            .bind_group_id_lookup
            .get(&cached_bind_group_id)
    }

    pub fn prune(&mut self, ttl_frames: u64) {
        if self.frame_index < ttl_frames {
            return;
        }
        let cutoff = self.frame_index - ttl_frames;

        let stale_scene_envs: Vec<u32> = self
            .environments
            .scene_gpu_environments
            .iter()
            .filter_map(|(scene_id, gpu_env)| {
                (gpu_env.last_used_frame < cutoff).then_some(*scene_id)
            })
            .collect();

        for scene_id in stale_scene_envs {
            if let Some(gpu_env) = self.environments.scene_gpu_environments.remove(&scene_id) {
                self.internal_textures.remove(gpu_env.base_cube_view.id());
                self.internal_textures.remove(gpu_env.pmrem_view.id());
            }
        }

        self.resources
            .gpu_geometries
            .retain(|_, v| v.last_used_frame >= cutoff);
        self.resources
            .gpu_materials
            .retain(|_, v| v.last_used_frame >= cutoff);
        // Sampler cache uses a global cache; no per-Texture cleanup needed
        self.resources
            .gpu_buffers
            .retain(|_, v| v.last_used_frame >= cutoff);
        // Keep buffer_index in sync with the arena.
        self.resources
            .buffer_index
            .retain(|_, h| self.resources.gpu_buffers.contains_key(*h));
        self.resources
            .gpu_images
            .retain(|_, v| v.last_used_frame >= cutoff);
        self.bind_groups
            .global_states
            .retain(|_, v| v.last_used_frame >= cutoff);
        // texture_bindings are cleaned up following gpu_images
        self.resources
            .texture_bindings
            .retain(|_, b| self.resources.gpu_images.contains_key(b.image_handle));
    }
}
