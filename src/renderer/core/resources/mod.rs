//! GPU 资源管理器
//!
//! 负责 GPU 端资源的创建、更新和管理
//!
//! 采用模块化设计，将不同职责拆分到独立文件中：
//! - buffer.rs: Buffer 相关操作
//! - texture.rs: Texture 和 Image 相关操作  
//! - geometry.rs: Geometry 相关操作
//! - material.rs: Material 相关操作
//! - binding.rs: `BindGroup` 相关操作
//! - allocator.rs: `ModelBufferAllocator`
//! - `resource_ids.rs`: 资源 ID 追踪和变化检测
//!
//! # 资源管理架构
//!
//! 采用 "Ensure -> Check -> Rebuild" 模式：
//!
//! 1. **Ensure 阶段**: 确保 GPU 资源存在且数据最新，返回物理资源 ID
//! 2. **Check 阶段**: 比较资源 ID 是否变化，决定是否需要重建 `BindGroup`
//! 3. **Rebuild 阶段**: 如需重建，收集 `LayoutEntries` 并比较是否需要新 Layout

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

/// GPU 全局状态 (Group 0)
///
/// 包含 Camera Uniforms、Light Storage Buffer、Environment Maps 等
///
/// 采用 "Ensure -> Collect IDs -> Check Fingerprint -> Rebind" 模式
pub struct GpuGlobalState {
    pub id: u32,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_id: u64,
    pub layout: wgpu::BindGroupLayout,
    pub layout_id: u64,
    pub binding_wgsl: String,
    /// 所有依赖资源的物理 ID 集合（用于自动检测变化）
    pub resource_ids: ResourceIdSet,
    pub last_used_frame: u64,
}

// ============================================================================
// 紧凑的 BindGroup 缓存键
// ============================================================================

// Object BindGroup 缓存键（使用 ResourceIdSet 的哈希值）
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
// Resource Manager 主结构
// ============================================================================

pub struct ResourceManager {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) frame_index: u64,

    pub(crate) gpu_geometries: SecondaryMap<GeometryHandle, GpuGeometry>,
    pub(crate) gpu_materials: SecondaryMap<MaterialHandle, GpuMaterial>,
    pub(crate) global_states: FxHashMap<u64, GpuGlobalState>,

    /// `TextureHandle` 到 (`ImageId`, `SamplerId`) 的映射
    pub(crate) texture_bindings: SecondaryMap<TextureHandle, TextureBinding>,
    /// `SamplerHandle` 到 `SamplerId` 的映射
    pub(crate) sampler_bindings: SecondaryMap<SamplerHandle, u64>,

    /// 所有 GpuBuffer，Key 是 CPU Buffer 的 ID
    pub(crate) gpu_buffers: FxHashMap<u64, GpuBuffer>,
    /// 所有 GpuImage，Key 是 CPU Image 的 ID
    pub(crate) gpu_images: FxHashMap<u64, GpuImage>,

    pub(crate) sampler_cache: FxHashMap<TextureSampler, GpuSampler>,
    pub(crate) sampler_id_lookup: FxHashMap<u64, wgpu::Sampler>,
    pub(crate) view_cache: FxHashMap<TextureViewKey, (wgpu::TextureView, u64)>,
    pub(crate) layout_cache:
        FxHashMap<Vec<wgpu::BindGroupLayoutEntry>, (wgpu::BindGroupLayout, u64)>,

    /// 顶点布局缓存：Signature -> ID
    pub vertex_layout_cache: FxHashMap<VertexLayoutSignature, u64>,

    pub(crate) dummy_image: GpuImage,
    pub(crate) dummy_env_image: GpuImage,
    pub(crate) dummy_sampler: GpuSampler,
    pub(crate) mipmap_generator: MipmapGenerator,

    // === Model Buffer Allocator ===
    pub(crate) model_allocator: ModelBufferAllocator,

    // === Object BindGroup 缓存 ===
    pub(crate) object_bind_group_cache: FxHashMap<ObjectBindGroupKey, BindGroupContext>,
    pub(crate) bind_group_id_lookup: FxHashMap<u64, BindGroupContext>,

    // === Environment Map Cache ===
    pub(crate) environment_map_cache: FxHashMap<TextureSource, GpuEnvironment>,
    pub(crate) brdf_lut_texture: Option<wgpu::Texture>,
    pub(crate) brdf_lut_view_id: Option<u64>,
    pub(crate) needs_brdf_compute: bool,
    /// Source that needs IBL compute this frame (set by `resolve_gpu_environment`)
    pub(crate) pending_ibl_source: Option<TextureSource>,

    /// 存储内部生成的纹理视图 (Render Targets / Attachments)
    /// Key: Resource ID (u64)
    /// Value: `wgpu::TextureView`
    pub(crate) internal_resources: FxHashMap<u64, wgpu::TextureView>,

    /// 内部纹理的名称到 ID 的映射，保证 ID 跨帧稳定
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
        // 创建 dummy 2D image
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

        // 创建 dummy env image (cube map)
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

            // 填充黑色数据 (1x1 pixel * 4 bytes * 6 layers = 24 bytes)
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

        // 初始化 Model GPU Buffer 映射
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

        // 初始化 sampler_id_lookup 并添加 dummy_sampler
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

    // 确保 Model Buffer 容量并同步 GPU 资源
    pub fn ensure_model_buffer_capacity(&mut self, count: usize) {
        let old_id = self.model_allocator.buffer_id();

        self.model_allocator.ensure_capacity(count);

        // 如果发生了扩容 (ID 变了)，我们需要立即创建对应的 GPU Buffer
        // 否则后续的 prepare_mesh 会找不到 Buffer 或者绑定到旧 Buffer
        let new_id = self.model_allocator.buffer_id();
        if new_id != old_id {
            // 清理旧的缓存（这一步很重要，防止 BindGroup 指向旧 Buffer）
            self.object_bind_group_cache.clear();
            self.bind_group_id_lookup.clear();

            let cpu_buf = self.model_allocator.cpu_buffer();
            // 立即创建新的 GpuBuffer (虽然数据还没填，但 wgpu::Buffer 对象需要存在)
            // 注意：这里我们创建一个未初始化的 Buffer 即可，因为后面 upload_model_buffer 会填充数据
            // 但为了安全，我们可以用 CpuBuffer 的默认数据初始化
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

    /// 分配一个 Model Uniform，返回字节偏移量
    #[inline]
    pub fn allocate_model_uniform(
        &mut self,
        data: crate::resources::uniforms::DynamicModelUniforms,
    ) -> u32 {
        self.model_allocator.allocate(data)
    }

    /// 每帧结束前上传 Model Buffer 到 GPU
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

        // 只上传有效的数据部分 (cursor * stride)
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

    /// 获取当前 Model Buffer 的 ID，用于缓存验证
    #[inline]
    pub fn model_buffer_id(&self) -> u64 {
        self.model_allocator.buffer_id()
    }

    /// 通过缓存的 ID 快速获取 `BindGroup` 数据
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
        // Sampler 缓存使用全局缓存，不需要按 Texture 清理
        self.gpu_buffers.retain(|_, v| v.last_used_frame >= cutoff);
        self.gpu_images.retain(|_, v| v.last_used_frame >= cutoff);
        self.global_states
            .retain(|_, v| v.last_used_frame >= cutoff);
        // texture_bindings 跟随 gpu_images 清理
        self.texture_bindings
            .retain(|_, b| self.gpu_images.contains_key(&b.cpu_image_id));
    }
}
