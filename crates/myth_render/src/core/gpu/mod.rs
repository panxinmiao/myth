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

use rustc_hash::FxHashMap;
use slotmap::SecondaryMap;

use myth_assets::{GeometryHandle, ImageHandle, MaterialHandle, TextureHandle};

pub(crate) use crate::core::gpu::buffer::GpuBuffer;
pub use crate::core::gpu::buffer::GpuBufferHandle;
pub(crate) use crate::core::gpu::environment::GpuEnvironment;
pub(crate) use crate::core::gpu::environment::{BRDF_LUT_SIZE, CubeSourceType};
pub(crate) use crate::core::gpu::geometry::GpuGeometry;
pub(crate) use crate::core::gpu::material::GpuMaterial;
pub(crate) use crate::core::gpu::texture::{GpuImage, TextureBinding};
use crate::pipeline::vertex::VertexLayoutSignature;

use myth_resources::buffer::{CpuBuffer, GpuData};
use myth_resources::texture::TextureSource;

pub use crate::core::gpu::mipmap::MipmapGenerator;
pub use allocator::ModelBufferAllocator;
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
// Resource Manager main structure
// ============================================================================

pub struct ResourceManager {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) frame_index: u64,

    pub(crate) gpu_geometries: SecondaryMap<GeometryHandle, GpuGeometry>,
    pub(crate) gpu_materials: SecondaryMap<MaterialHandle, GpuMaterial>,
    pub(crate) gpu_images: SecondaryMap<ImageHandle, GpuImage>,

    pub(crate) global_states: FxHashMap<u64, GpuGlobalState>,

    /// Mapping from `TextureHandle` to (`ImageId`, `SamplerId`)
    pub(crate) texture_bindings: SecondaryMap<TextureHandle, TextureBinding>,

    /// All GPU buffers stored in a contiguous arena for O(1) handle-based access.
    pub(crate) gpu_buffers: slotmap::SlotMap<GpuBufferHandle, GpuBuffer>,
    /// Reverse index: CPU-side buffer ID → SlotMap handle.
    pub(crate) buffer_index: FxHashMap<u64, GpuBufferHandle>,

    pub(crate) sampler_registry: SamplerRegistry,

    pub(crate) layout_cache:
        FxHashMap<Vec<wgpu::BindGroupLayoutEntry>, (wgpu::BindGroupLayout, u64)>,

    /// Vertex layout cache: Signature -> ID
    pub vertex_layout_cache: FxHashMap<VertexLayoutSignature, u64>,

    // pub(crate) dummy_image: GpuImage,
    // pub(crate) dummy_env_image: GpuImage,
    pub(crate) mipmap_generator: MipmapGenerator,

    // === Model Buffer Allocator ===
    pub(crate) model_allocator: ModelBufferAllocator,

    // === Object BindGroup cache ===
    pub(crate) object_bind_group_cache: FxHashMap<ObjectBindGroupKey, BindGroupContext>,
    pub(crate) bind_group_id_lookup: FxHashMap<u64, BindGroupContext>,

    // === Environment Map Cache ===
    pub(crate) environment_map_cache: FxHashMap<TextureSource, GpuEnvironment>,
    pub(crate) brdf_lut_texture: Option<wgpu::Texture>,
    pub(crate) brdf_lut_view_id: Option<u64>,
    pub(crate) needs_brdf_compute: bool,
    /// Source that needs IBL compute this frame (set by `resolve_gpu_environment`)
    pub(crate) pending_ibl_source: Option<TextureSource>,

    /// Stores internally generated texture views (Render Targets / Attachments)
    /// Key: Resource ID (u64)
    /// Value: `wgpu::TextureView`
    pub(crate) internal_resources: FxHashMap<u64, wgpu::TextureView>,

    /// Mapping from internal texture names to IDs, ensuring ID stability across frames
    pub(crate) internal_name_lookup: FxHashMap<String, u64>,

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
        let mipmap_generator = MipmapGenerator::new(&device);
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
            gpu_geometries: SecondaryMap::new(),
            gpu_materials: SecondaryMap::new(),
            gpu_images: SecondaryMap::new(),
            sampler_registry,
            texture_bindings: SecondaryMap::new(),
            global_states: FxHashMap::default(),
            gpu_buffers,
            buffer_index,
            layout_cache: FxHashMap::default(),
            vertex_layout_cache: FxHashMap::default(),
            mipmap_generator,
            model_allocator,
            object_bind_group_cache: FxHashMap::default(),
            bind_group_id_lookup: FxHashMap::default(),
            environment_map_cache: FxHashMap::default(),
            brdf_lut_texture: None,
            brdf_lut_view_id: None,
            needs_brdf_compute: false,
            pending_ibl_source: None,
            internal_resources: FxHashMap::default(),
            internal_name_lookup: FxHashMap::default(),
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

    pub fn flush_model_buffers(&mut self) {
        let resized = self.model_allocator.flush_to_buffer(
            &self.device,
            &self.queue,
            &mut self.gpu_buffers,
            &mut self.buffer_index,
            self.frame_index,
        );

        if resized {
            self.object_bind_group_cache.clear();
            self.bind_group_id_lookup.clear();
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
        self.bind_group_id_lookup.get(&cached_bind_group_id)
    }

    pub fn prune(&mut self, ttl_frames: u64) {
        if self.frame_index < ttl_frames {
            return;
        }
        let cutoff = self.frame_index - ttl_frames;

        self.gpu_geometries
            .retain(|_, v| v.last_used_frame >= cutoff);
        self.gpu_materials
            .retain(|_, v| v.last_used_frame >= cutoff);
        // Sampler cache uses a global cache; no per-Texture cleanup needed
        self.gpu_buffers.retain(|_, v| v.last_used_frame >= cutoff);
        // Keep buffer_index in sync with the arena.
        self.buffer_index
            .retain(|_, h| self.gpu_buffers.contains_key(*h));
        self.gpu_images.retain(|_, v| v.last_used_frame >= cutoff);
        self.global_states
            .retain(|_, v| v.last_used_frame >= cutoff);
        // texture_bindings are cleaned up following gpu_images
        self.texture_bindings
            .retain(|_, b| self.gpu_images.contains_key(b.image_handle));
    }
}
