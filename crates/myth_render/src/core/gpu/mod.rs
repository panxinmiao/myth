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
mod texture;
mod tracked;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use rustc_hash::FxHashMap;
use slotmap::SecondaryMap;

use myth_assets::{GeometryHandle, MaterialHandle, TextureHandle};

pub(crate) use crate::core::gpu::buffer::GpuBuffer;
pub use crate::core::gpu::buffer::GpuBufferHandle;
pub(crate) use crate::core::gpu::environment::GpuEnvironment;
pub(crate) use crate::core::gpu::environment::{BRDF_LUT_SIZE, CubeSourceType};
pub(crate) use crate::core::gpu::geometry::GpuGeometry;
pub(crate) use crate::core::gpu::material::GpuMaterial;
pub(crate) use crate::core::gpu::texture::{GpuImage, TextureBinding, TextureViewKey};
use crate::pipeline::vertex::VertexLayoutSignature;

use myth_resources::buffer::{CpuBuffer, GpuData};
use myth_resources::texture::TextureSource;

pub use crate::core::gpu::mipmap::MipmapGenerator;
pub use allocator::ModelBufferAllocator;
pub use resource_ids::{
    BindGroupFingerprint, EnsureResult, ResourceId, ResourceIdSet, hash_layout_entries,
};
pub use sampler_registry::{CommonSampler, SamplerRegistry};
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

    /// All GPU buffers stored in a contiguous arena for O(1) handle-based access.
    pub(crate) gpu_buffers: slotmap::SlotMap<GpuBufferHandle, GpuBuffer>,
    /// Reverse index: CPU-side buffer ID → SlotMap handle.
    pub(crate) buffer_index: FxHashMap<u64, GpuBufferHandle>,
    /// All GpuImages, keyed by CPU Image ID
    pub(crate) gpu_images: FxHashMap<u64, GpuImage>,
    pub(crate) sampler_registry: SamplerRegistry,

    pub(crate) view_cache: FxHashMap<TextureViewKey, (wgpu::TextureView, u64)>,
    pub(crate) layout_cache:
        FxHashMap<Vec<wgpu::BindGroupLayoutEntry>, (wgpu::BindGroupLayout, u64)>,

    /// Vertex layout cache: Signature -> ID
    pub vertex_layout_cache: FxHashMap<VertexLayoutSignature, u64>,

    pub(crate) dummy_image: GpuImage,
    pub(crate) dummy_env_image: GpuImage,
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

    // ─── Screen BindGroup Infrastructure (Group 3) ────────────────
    //
    // Static, create-once resources used by scene draw passes to bind the
    // "screen" descriptor set.  Group 3 contains transmission, SSAO, and
    // shadow map textures — all RDG transient resources that are swapped
    // to harmless dummies when their features are inactive.
    //
    /// BindGroupLayout for Group 3.
    pub screen_bind_group_layout: Tracked<wgpu::BindGroupLayout>,

    /// Shared linear-clamp sampler for transmission / SSAO sampling.
    pub screen_sampler: Tracked<wgpu::Sampler>,

    /// 1×1 white (R8Unorm = 255) dummy texture for SSAO-disabled fallback.
    pub ssao_dummy_view: Tracked<wgpu::TextureView>,

    /// 1×1 placeholder texture (Rgba16Float) for transmission-disabled fallback.
    pub dummy_transmission_view: Tracked<wgpu::TextureView>,

    /// 1×1 Depth32Float D2Array dummy for no-shadow-casters fallback.
    pub dummy_shadow_view: Tracked<wgpu::TextureView>,

    /// `LessEqual` comparison sampler for PCF shadow sampling.
    pub shadow_compare_sampler: Tracked<wgpu::Sampler>,
}

/// Lightweight bundle of persistent screen bind-group resources.
///
/// Cloned from [`ResourceManager`] and passed to RDG PassNodes so that
/// `build_screen_bind_group` can work without a reference to the full
/// `ResourceManager`, keeping it out of [`RdgPrepareContext`].
///
/// Group 3 layout:
///
/// | Binding | Resource                       |
/// |---------|--------------------------------|
/// | 0       | Transmission texture (Float)   |
/// | 1       | Screen sampler (Filtering)     |
/// | 2       | SSAO texture (Float)           |
/// | 3       | Shadow depth 2D-array          |
/// | 4       | Shadow comparison sampler      |
#[derive(Clone)]
pub struct ScreenBindGroupInfo {
    pub layout: Tracked<wgpu::BindGroupLayout>,
    pub sampler: Tracked<wgpu::Sampler>,
    pub ssao_dummy_view: Tracked<wgpu::TextureView>,
    pub dummy_transmission_view: Tracked<wgpu::TextureView>,
    /// 1×1 Depth32Float D2Array fallback for no-shadow frames.
    pub dummy_shadow_view: Tracked<wgpu::TextureView>,
    /// `LessEqual` comparison sampler for PCF shadow sampling.
    pub shadow_compare_sampler: Tracked<wgpu::Sampler>,
}

impl ScreenBindGroupInfo {
    /// Creates a `ScreenBindGroupInfo` from the given [`ResourceManager`].
    pub fn from_resource_manager(rm: &ResourceManager) -> Self {
        Self {
            layout: rm.screen_bind_group_layout.clone(),
            sampler: rm.screen_sampler.clone(),
            ssao_dummy_view: rm.ssao_dummy_view.clone(),
            dummy_transmission_view: rm.dummy_transmission_view.clone(),
            dummy_shadow_view: rm.dummy_shadow_view.clone(),
            shadow_compare_sampler: rm.shadow_compare_sampler.clone(),
        }
    }

    /// Build the screen / transient bind group (Group 3).
    ///
    /// Returns `(BindGroup, composite_id)` where `composite_id` is a hash
    /// of the resource IDs for `TrackedRenderPass` state tracking.
    pub fn build_screen_bind_group(
        &self,
        device: &wgpu::Device,
        cache: &mut crate::core::binding::GlobalBindGroupCache,
        transmission_view: &Tracked<wgpu::TextureView>,
        ssao_view: &Tracked<wgpu::TextureView>,
        shadow_view: &Tracked<wgpu::TextureView>,
    ) -> (wgpu::BindGroup, u64) {
        use crate::core::binding::BindGroupKey;

        let layout_id = self.layout.id();
        let sampler_id = self.sampler.id();
        let shadow_sampler_id = self.shadow_compare_sampler.id();

        let key = BindGroupKey::new(layout_id)
            .with_resource(transmission_view.id())
            .with_resource(sampler_id)
            .with_resource(ssao_view.id())
            .with_resource(shadow_view.id())
            .with_resource(shadow_sampler_id);

        let bind_group_id = transmission_view
            .id()
            .wrapping_mul(6_364_136_223_846_793_005)
            ^ ssao_view.id().wrapping_mul(1_442_695_040_888_963_407)
            ^ shadow_view.id().wrapping_mul(2_862_933_555_777_941_757)
            ^ sampler_id
            ^ shadow_sampler_id;

        let layout = &*self.layout;
        let sampler = &*self.sampler;
        let tv = &**transmission_view;
        let sv = &**ssao_view;
        let shv = &**shadow_view;
        let shs = &*self.shadow_compare_sampler;

        let bg = cache
            .get_or_create(key, || {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Screen BindGroup (Group 3)"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(tv),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(sv),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(shv),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(shs),
                        },
                    ],
                })
            })
            .clone();

        (bg, bind_group_id)
    }
}

impl ResourceManager {
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, anisotropy_clamp: u16) -> Self {
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

        // Shadow dummy: 1×1 Depth32Float D2Array (no-shadow fallback for Group 3)
        let dummy_shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy Shadow D2Array"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let dummy_shadow_view =
            Tracked::new(dummy_shadow_tex.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            }));

        let shadow_compare_sampler =
            Tracked::new(device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Shadow Comparison Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                compare: Some(wgpu::CompareFunction::LessEqual),
                ..Default::default()
            }));

        let mipmap_generator = MipmapGenerator::new(&device);
        let model_allocator = ModelBufferAllocator::new();

        let mut gpu_buffers = slotmap::SlotMap::with_key();
        let mut buffer_index = rustc_hash::FxHashMap::default();

        // Force initial allocation of the model buffer so that it has a stable GPU handle and ID from the start.
        model_allocator.flush_to_buffer(&device, &queue, &mut gpu_buffers, &mut buffer_index, 0);

        // ── Screen BindGroup (Group 3) static resources ────────────────
        let screen_bind_group_layout = Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Screen/Transient Layout (Group 3)"),
                entries: &[
                    // Binding 0: Transmission texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Binding 1: Screen sampler (filtering)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Binding 2: SSAO texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Binding 3: Shadow map (Depth 2D-array, RDG transient)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Binding 4: Shadow comparison sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                ],
            },
        ));

        let screen_sampler = Tracked::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Screen Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

        // SSAO dummy: 1×1 white R8Unorm texture (AO = 1.0 = fully lit)
        let ssao_dummy_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSAO White Dummy"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let ssao_dummy_view =
            Tracked::new(ssao_dummy_tex.create_view(&wgpu::TextureViewDescriptor::default()));

        // Transmission dummy: 1×1 HDR placeholder (black)
        let dummy_tx_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Transmission Dummy Texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_transmission_view =
            Tracked::new(dummy_tx_tex.create_view(&wgpu::TextureViewDescriptor::default()));

        let sampler_registry = SamplerRegistry::new(&device, anisotropy_clamp);

        Self {
            device,
            queue,
            frame_index: 0,
            gpu_geometries: SecondaryMap::new(),
            gpu_materials: SecondaryMap::new(),
            sampler_registry,
            texture_bindings: SecondaryMap::new(),
            global_states: FxHashMap::default(),
            gpu_buffers,
            buffer_index,
            gpu_images: FxHashMap::default(),
            layout_cache: FxHashMap::default(),
            vertex_layout_cache: FxHashMap::default(),
            view_cache: FxHashMap::default(),
            dummy_image,
            dummy_env_image,
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
            screen_bind_group_layout,
            screen_sampler,
            ssao_dummy_view,
            dummy_transmission_view,
            dummy_shadow_view,
            shadow_compare_sampler,
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
            .retain(|_, b| self.gpu_images.contains_key(&b.cpu_image_id));
    }
}
