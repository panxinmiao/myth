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
mod texture;
mod tracked;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use rustc_hash::FxHashMap;
use slotmap::SecondaryMap;

use crate::assets::server::SamplerHandle;

pub(crate) use crate::renderer::core::resources::buffer::GpuBuffer;
pub(crate) use crate::renderer::core::resources::environment::GpuEnvironment;
pub(crate) use crate::renderer::core::resources::environment::{BRDF_LUT_SIZE, CubeSourceType};
pub(crate) use crate::renderer::core::resources::geometry::GpuGeometry;
pub(crate) use crate::renderer::core::resources::material::GpuMaterial;
pub(crate) use crate::renderer::core::resources::texture::{
    GpuImage, GpuSampler, TextureBinding, TextureViewKey,
};
use crate::renderer::pipeline::vertex::VertexLayoutSignature;
pub(crate) use crate::resources::texture::TextureSampler;

use crate::assets::{GeometryHandle, MaterialHandle, TextureHandle};
use crate::resources::buffer::{CpuBuffer, GpuData};
use crate::resources::texture::TextureSource;

pub use crate::renderer::core::resources::mipmap::MipmapGenerator;
pub use allocator::ModelBufferAllocator;
pub use resource_ids::{
    BindGroupFingerprint, EnsureResult, ResourceId, ResourceIdSet, hash_layout_entries,
};
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
    pub(crate) global_states: FxHashMap<u64, GpuGlobalState>,

    /// Mapping from `TextureHandle` to (`ImageId`, `SamplerId`)
    pub(crate) texture_bindings: SecondaryMap<TextureHandle, TextureBinding>,
    /// Mapping from `SamplerHandle` to `SamplerId`
    pub(crate) sampler_bindings: SecondaryMap<SamplerHandle, u64>,

    /// All GpuBuffers, keyed by CPU Buffer ID
    pub(crate) gpu_buffers: FxHashMap<u64, GpuBuffer>,
    /// All GpuImages, keyed by CPU Image ID
    pub(crate) gpu_images: FxHashMap<u64, GpuImage>,

    pub(crate) sampler_cache: FxHashMap<TextureSampler, GpuSampler>,
    pub(crate) sampler_id_lookup: FxHashMap<u64, wgpu::Sampler>,
    pub(crate) view_cache: FxHashMap<TextureViewKey, (wgpu::TextureView, u64)>,
    pub(crate) layout_cache:
        FxHashMap<Vec<wgpu::BindGroupLayoutEntry>, (wgpu::BindGroupLayout, u64)>,

    /// Vertex layout cache: Signature -> ID
    pub vertex_layout_cache: FxHashMap<VertexLayoutSignature, u64>,

    pub(crate) dummy_image: GpuImage,
    pub(crate) dummy_env_image: GpuImage,
    pub(crate) dummy_sampler: GpuSampler,
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

    pub(crate) shadow_2d_texture: Option<wgpu::Texture>,
    pub shadow_2d_array: Option<wgpu::TextureView>,
    pub(crate) shadow_2d_array_id: Option<u64>,
    pub(crate) shadow_2d_capacity: u32,
    pub(crate) shadow_map_size: u32,
    pub(crate) dummy_shadow_map: GpuImage,
    pub(crate) shadow_compare_sampler: GpuSampler,
}

impl ResourceManager {
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        // Create dummy 2D image
        let dummy_image = {
            let size = wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Dummy Image"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &[255u8, 255, 255, 255],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: Some(1),
                },
                size,
            );

            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

            GpuImage {
                id: generate_gpu_resource_id(),
                texture,
                default_view: view,
                default_view_dimension: wgpu::TextureViewDimension::D2,
                size,
                format: wgpu::TextureFormat::Rgba8Unorm,
                mip_level_count: 1,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                version: 0,
                generation_id: 0,
                mipmaps_generated: true,
                last_used_frame: u64::MAX,
            }
        };

        // Create dummy env image (cube map)
        let dummy_env_image = {
            let size = wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Dummy EnvMap Black"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            // Fill with black data (1x1 pixel * 4 bytes * 6 layers = 24 bytes)
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &[0u8; 24],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: Some(1),
                },
                size,
            );

            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::Cube),
                ..Default::default()
            });

            GpuImage {
                id: generate_gpu_resource_id(),
                texture,
                default_view: view,
                default_view_dimension: wgpu::TextureViewDimension::Cube,
                size,
                format: wgpu::TextureFormat::Rgba8Unorm,
                mip_level_count: 1,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                version: 0,
                generation_id: 0,
                mipmaps_generated: true,
                last_used_frame: u64::MAX,
            }
        };

        let dummy_sampler = GpuSampler {
            id: generate_gpu_resource_id(),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Dummy Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            }),
        };

        let dummy_shadow_map = {
            let size = wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            };

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Dummy Shadow 2D Array"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });

            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });

            GpuImage {
                id: generate_gpu_resource_id(),
                texture,
                default_view: view,
                default_view_dimension: wgpu::TextureViewDimension::D2Array,
                size,
                format: wgpu::TextureFormat::Depth32Float,
                mip_level_count: 1,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                version: 0,
                generation_id: 0,
                mipmaps_generated: true,
                last_used_frame: u64::MAX,
            }
        };

        let shadow_compare_sampler = GpuSampler {
            id: generate_gpu_resource_id(),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Shadow Comparison Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                compare: Some(wgpu::CompareFunction::LessEqual),
                ..Default::default()
            }),
        };

        let mipmap_generator = MipmapGenerator::new(&device);
        let model_allocator = ModelBufferAllocator::new();

        // Initialize Model GPU Buffer mapping
        let gpu_buffers = {
            let cpu_buf = model_allocator.cpu_buffer();
            let buffer_guard = cpu_buf.read();
            let gpu_buf = GpuBuffer::new(
                &device,
                buffer_guard.as_bytes(),
                cpu_buf.usage(),
                cpu_buf.label(),
            );

            let mut map = FxHashMap::default();
            map.insert(cpu_buf.id(), gpu_buf);
            map
        };

        // Initialize sampler_id_lookup and add dummy_sampler
        let mut sampler_id_lookup = FxHashMap::default();
        sampler_id_lookup.insert(dummy_sampler.id, dummy_sampler.sampler.clone());
        sampler_id_lookup.insert(
            shadow_compare_sampler.id,
            shadow_compare_sampler.sampler.clone(),
        );

        Self {
            device,
            queue,
            frame_index: 0,
            gpu_geometries: SecondaryMap::new(),
            gpu_materials: SecondaryMap::new(),
            texture_bindings: SecondaryMap::new(),
            sampler_bindings: SecondaryMap::new(),
            global_states: FxHashMap::default(),
            gpu_buffers,
            gpu_images: FxHashMap::default(),
            layout_cache: FxHashMap::default(),
            vertex_layout_cache: FxHashMap::default(),
            sampler_cache: FxHashMap::default(),
            sampler_id_lookup,
            view_cache: FxHashMap::default(),
            dummy_image,
            dummy_env_image,
            dummy_sampler,
            dummy_shadow_map,
            shadow_compare_sampler,
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
            shadow_2d_texture: None,
            shadow_2d_array: None,
            shadow_2d_array_id: None,
            shadow_2d_capacity: 0,
            shadow_map_size: 1,
        }
    }

    pub fn next_frame(&mut self) {
        self.frame_index += 1;
        self.model_allocator.reset();
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    pub fn ensure_shadow_maps(&mut self, required_2d_count: u32, required_map_size: u32) {
        if required_2d_count == 0 {
            return;
        }

        let mut target_capacity = self.shadow_2d_capacity.max(1);
        while target_capacity < required_2d_count {
            target_capacity = target_capacity.saturating_mul(2);
        }

        let target_size = required_map_size.max(1);
        let need_recreate = self.shadow_2d_array.is_none()
            || target_capacity > self.shadow_2d_capacity
            || target_size > self.shadow_map_size;

        if !need_recreate {
            return;
        }

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow 2D Array"),
            size: wgpu::Extent3d {
                width: target_size,
                height: target_size,
                depth_or_array_layers: target_capacity,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Shadow 2D Array View"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        if let Some(old_id) = self.shadow_2d_array_id {
            self.internal_resources.remove(&old_id);
        }

        self.shadow_2d_texture = Some(texture);
        self.shadow_2d_array = Some(array_view);
        let view_id = generate_gpu_resource_id();
        if let Some(view) = &self.shadow_2d_array {
            self.internal_resources.insert(view_id, view.clone());
        }
        self.shadow_2d_array_id = Some(view_id);
        self.shadow_2d_capacity = target_capacity;
        self.shadow_map_size = target_size;
    }

    pub fn create_shadow_2d_layer_view(&self, layer_index: u32) -> Option<wgpu::TextureView> {
        let texture = self.shadow_2d_texture.as_ref()?;
        Some(texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Shadow Layer View"),
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: layer_index,
            array_layer_count: Some(1),
            ..Default::default()
        }))
    }

    // Ensure Model Buffer capacity and synchronize GPU resources
    pub fn ensure_model_buffer_capacity(&mut self, count: usize) {
        let old_id = self.model_allocator.buffer_id();

        self.model_allocator.ensure_capacity(count);

        // If expansion occurred (ID changed), we need to immediately create the
        // corresponding GPU Buffer. Otherwise subsequent prepare_mesh calls will
        // fail to find the Buffer or bind to the old one.
        let new_id = self.model_allocator.buffer_id();
        if new_id != old_id {
            // Clear old caches (important to prevent BindGroups pointing to the old Buffer)
            self.object_bind_group_cache.clear();
            self.bind_group_id_lookup.clear();

            let cpu_buf = self.model_allocator.cpu_buffer();
            // Immediately create the new GpuBuffer (data is not filled yet, but
            // the wgpu::Buffer object must exist). For safety, we initialize it
            // with the CpuBuffer's default data.
            let buffer_guard = cpu_buf.read();
            let gpu_buf = GpuBuffer::new(
                &self.device,
                buffer_guard.as_bytes(),
                cpu_buf.usage(),
                cpu_buf.label(),
            );

            self.gpu_buffers.insert(new_id, gpu_buf);
        }
    }

    /// Allocate a Model Uniform slot, returning the byte offset
    #[inline]
    pub fn allocate_model_uniform(
        &mut self,
        data: crate::resources::uniforms::DynamicModelUniforms,
    ) -> u32 {
        self.model_allocator.allocate(data)
    }

    /// Upload Model Buffer to GPU before the end of each frame
    pub fn upload_model_buffer(&mut self) {
        if self.model_allocator.is_empty() {
            return;
        }

        self.model_allocator.flush_to_buffer();

        let allocator = &self.model_allocator;
        let buffer_ref = allocator.buffer_handle();

        let buffer_guard = allocator.cpu_buffer().read();

        let full_slice = buffer_guard.as_bytes();

        let stride = std::mem::size_of::<crate::resources::uniforms::DynamicModelUniforms>();

        // Only upload the valid data portion (cursor * stride)
        let used_bytes = allocator.len() * stride;

        let data_to_upload = &full_slice[0..used_bytes];

        Self::write_buffer_internal(
            &self.device,
            &self.queue,
            &mut self.gpu_buffers,
            self.frame_index,
            &buffer_ref,
            data_to_upload,
        );
    }

    /// Get the current Model Buffer ID for cache validation
    #[inline]
    pub fn model_buffer_id(&self) -> u64 {
        self.model_allocator.buffer_id()
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
        self.gpu_images.retain(|_, v| v.last_used_frame >= cutoff);
        self.global_states
            .retain(|_, v| v.last_used_frame >= cutoff);
        // texture_bindings are cleaned up following gpu_images
        self.texture_bindings
            .retain(|_, b| self.gpu_images.contains_key(&b.cpu_image_id));
    }
}
