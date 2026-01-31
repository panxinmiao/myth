//! GPU 资源管理器
//!
//! 负责 GPU 端资源的创建、更新和管理
//! 
//! 采用模块化设计，将不同职责拆分到独立文件中：
//! - buffer.rs: Buffer 相关操作
//! - texture.rs: Texture 和 Image 相关操作  
//! - geometry.rs: Geometry 相关操作
//! - material.rs: Material 相关操作
//! - binding.rs: BindGroup 相关操作
//! - allocator.rs: ModelBufferAllocator
//! - resource_ids.rs: 资源 ID 追踪和变化检测
//!
//! # 资源管理架构
//!
//! 采用 "Ensure -> Check -> Rebuild" 模式：
//! 
//! 1. **Ensure 阶段**: 确保 GPU 资源存在且数据最新，返回物理资源 ID
//! 2. **Check 阶段**: 比较资源 ID 是否变化，决定是否需要重建 BindGroup
//! 3. **Rebuild 阶段**: 如需重建，收集 LayoutEntries 并比较是否需要新 Layout

mod allocator;
mod buffer;
mod texture;
mod geometry;
mod material;
mod binding;
mod mipmap;
mod resource_ids;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use glam::Mat4;
use slotmap::SecondaryMap;
use core::ops::Range;
use rustc_hash::FxHashMap;

use crate::assets::server::SamplerHandle;
use crate::renderer::core::resources::mipmap::MipmapGenerator;
use crate::resources::texture::TextureSampler;

use crate::scene::SkeletonKey;
use crate::resources::buffer::{CpuBuffer, GpuData};
use crate::assets::{GeometryHandle, MaterialHandle, TextureHandle};

use crate::renderer::pipeline::vertex::GeneratedVertexLayout;

pub use allocator::ModelBufferAllocator;
pub use resource_ids::{EnsureResult, ResourceIdSet, BindGroupFingerprint, hash_layout_entries, ResourceId};

static NEXT_GPU_RESOURCE_ID: AtomicU64 = AtomicU64::new(1);

pub fn generate_gpu_resource_id() -> u64 {
    NEXT_GPU_RESOURCE_ID.fetch_add(1, Ordering::Relaxed)
}

// ============================================================================
// GPU 资源包装器
// ============================================================================

pub struct GpuBuffer {
    pub id: u64,
    pub buffer: wgpu::Buffer,
    pub size: u64,
    pub usage: wgpu::BufferUsages,
    pub label: String,
    pub last_used_frame: u64,
    pub version: u64,
    pub last_uploaded_version: u64,
    shadow_data: Option<Vec<u8>>,
}

impl GpuBuffer {
    pub fn new(device: &wgpu::Device, data: &[u8], usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        use wgpu::util::DeviceExt;
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label,
            contents: data,
            usage,
        });

        Self {
            id: generate_gpu_resource_id(),
            buffer,
            size: data.len() as u64,
            usage,
            label: label.unwrap_or("Buffer").to_string(),
            last_used_frame: 0,
            version: 0,
            last_uploaded_version: 0,
            shadow_data: None,
        }
    }

    pub fn enable_shadow_copy(&mut self) {
        if self.shadow_data.is_none() {
            self.shadow_data = Some(Vec::new());
        }
    }

    pub fn update_with_data(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) -> bool {
        if let Some(prev) = &mut self.shadow_data {
            if prev == data {
                return false;
            }
            if prev.len() != data.len() {
                *prev = vec![0u8; data.len()];
            }
            prev.copy_from_slice(data);
        }
        self.write_to_gpu(device, queue, data)
    }

    pub fn update_with_version(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8], new_version: u64) -> bool {
        if new_version <= self.version {
            return false;
        }
        self.version = new_version;
        self.write_to_gpu(device, queue, data)
    }

    fn write_to_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) -> bool {
        let new_size = data.len() as u64;
        if new_size > self.size {
            self.resize(device, new_size);
            queue.write_buffer(&self.buffer, 0, data);
            return true;
        }
        queue.write_buffer(&self.buffer, 0, data);
        false
    }

    fn resize(&mut self, device: &wgpu::Device, new_size: u64) {
        self.buffer.destroy();
        self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&self.label),
            size: new_size,
            usage: self.usage,
            mapped_at_creation: false,
        });
        self.size = new_size;
        self.id = generate_gpu_resource_id();
    }
}

/// 纹理资源映射
///
/// 将 TextureHandle 映射到对应的 GpuImage ID、View ID 和 GpuSampler ID
#[derive(Debug, Clone, Copy)]
pub struct TextureBinding {
    /// GPU 端图像视图 ID
    pub view_id: u64,
    /// CPU 端图像 ID
    pub cpu_image_id: u64,
    pub sampler_id: u64,
    /// CPU 端 Texture 版本（用于检测采样参数变化）
    pub texture_version: u64,
}

/// 纹理视图缓存键
///
/// 用于按需创建和缓存不同配置的 TextureView。
/// Key 包含 view_id，确保底层 Image 重建时自动失效。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureViewKey {
    pub view_id: u64,
    pub format: Option<wgpu::TextureFormat>,
    pub dimension: Option<wgpu::TextureViewDimension>,
    pub base_mip_level: u32,
    pub mip_level_count: Option<u32>,
    pub base_array_layer: u32,
    pub array_layer_count: Option<u32>,
    pub aspect: wgpu::TextureAspect,
}

impl TextureViewKey {
    #[inline]
    pub fn new(view_id: u64, desc: &wgpu::TextureViewDescriptor) -> Self {
        Self {
            view_id,
            format: desc.format,
            dimension: desc.dimension,
            base_mip_level: desc.base_mip_level,
            mip_level_count: desc.mip_level_count,
            base_array_layer: desc.base_array_layer,
            array_layer_count: desc.array_layer_count,
            aspect: desc.aspect,
        }
    }

    #[inline]
    pub fn default_for_view(view_id: u64) -> Self {
        Self {
            view_id,
            format: None,
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: wgpu::TextureAspect::All,
        }
    }
}

/// GPU 端图像资源
///
/// 包含物理纹理和默认视图，不包含采样器
pub struct GpuImage {
    pub id: u64,
    pub texture: wgpu::Texture,
    pub default_view: wgpu::TextureView,
    pub default_view_dimension: wgpu::TextureViewDimension,
    pub size: wgpu::Extent3d,
    pub format: wgpu::TextureFormat,
    pub mip_level_count: u32,
    pub usage: wgpu::TextureUsages,
    pub version: u64,
    pub generation_id: u64,
    pub mipmaps_generated: bool,
    pub last_used_frame: u64,
}

/// GPU 端采样器资源
/// 
/// 与 GpuImage 分离，实现全局缓存和复用
pub struct GpuSampler {
    pub id: u64,
    pub sampler: wgpu::Sampler,
}

/// GPU 端几何体资源
/// 
/// Vertex Buffer IDs 用于 Pipeline 缓存验证，不影响 BindGroup
pub struct GpuGeometry {
    pub layout_info: GeneratedVertexLayout,
    pub vertex_buffers: Vec<wgpu::Buffer>,
    pub vertex_buffer_ids: Vec<u64>,
    pub index_buffer: Option<(wgpu::Buffer, wgpu::IndexFormat, u32, u64)>,
    pub draw_range: Range<u32>,
    pub instance_range: Range<u32>,
    pub version: u64,
    pub last_data_version: u64,
    pub last_used_frame: u64,
}

/// GPU 端材质资源
/// 
/// 使用资源 ID 追踪机制自动检测变化
/// 
/// # 版本追踪三维分离
/// 
/// 1. **资源拓扑 (BindGroup)**: 由 `resource_ids` 追踪
///    - 纹理/采样器/Buffer ID 变化 -> 重建 BindGroup
/// 
/// 2. **资源内容 (Buffer Data)**: 由 `BufferRef` 追踪（外部）
///    - Atomic 版本号变化 -> 上传 Buffer
/// 
/// 3. **管线状态 (RenderPipeline)**: 由 `version` 追踪
///    - 深度写入/透明度/双面渲染等变化 -> 切换 Pipeline
pub struct GpuMaterial {
    pub bind_group: wgpu::BindGroup,
    pub bind_group_id: u64,
    pub layout: wgpu::BindGroupLayout,
    pub layout_id: u64,
    /// Layout entries 的哈希值（用于快速比较是否需要重建 Layout）
    pub layout_hash: u64,
    pub binding_wgsl: String,
    /// 所有依赖资源的物理 ID 集合（守卫 BindGroup 的有效性）
    pub resource_ids: ResourceIdSet,
    /// 记录生成此 GpuMaterial 时的 Material 版本（用于 Pipeline 缓存）
    pub version: u64,
    pub last_used_frame: u64,
    pub last_verified_frame: u64,
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

    /// TextureHandle 到 (ImageId, SamplerId) 的映射
    pub(crate) texture_bindings: SecondaryMap<TextureHandle, TextureBinding>,
    /// SamplerHandle 到 SamplerId 的映射
    pub(crate) sampler_bindings: SecondaryMap<SamplerHandle, u64>,
    
    /// 所有 GpuBuffer，Key 是 CPU Buffer 的 ID
    pub(crate) gpu_buffers: FxHashMap<u64, GpuBuffer>,
    /// 所有 GpuImage，Key 是 CPU Image 的 ID
    pub(crate) gpu_images: FxHashMap<u64, GpuImage>,

    pub(crate) sampler_cache: FxHashMap<TextureSampler, GpuSampler>,
    pub(crate) sampler_id_lookup: FxHashMap<u64, wgpu::Sampler>,
    pub(crate) view_cache: FxHashMap<TextureViewKey, (wgpu::TextureView, u64)>,
    pub(crate) layout_cache: FxHashMap<Vec<wgpu::BindGroupLayoutEntry>, (wgpu::BindGroupLayout, u64)>,

    pub(crate) dummy_image: GpuImage,
    pub(crate) dummy_env_image: GpuImage,
    pub(crate) dummy_sampler: GpuSampler,
    pub(crate) mipmap_generator: MipmapGenerator,

    // === Model Buffer Allocator ===
    pub(crate) model_allocator: ModelBufferAllocator,

    // === Object BindGroup 缓存 ===
    pub(crate) object_bind_group_cache: FxHashMap<ObjectBindGroupKey, BindGroupContext>,
    pub(crate) bind_group_id_lookup: FxHashMap<u64, BindGroupContext>,


    /// 存储内部生成的纹理视图 (Render Targets / Attachments)
    /// Key: Resource ID (u64)
    /// Value: wgpu::TextureView
    pub(crate) internal_resources: FxHashMap<u64, wgpu::TextureView>,

    /// 内部纹理的名称到 ID 的映射，保证 ID 跨帧稳定
    pub(crate) internal_name_lookup: FxHashMap<String, u64>,

    // === 骨骼 Buffer ===
    pub(crate) skeleton_buffers: FxHashMap<SkeletonKey, CpuBuffer<Vec<Mat4>>>,
}

impl ResourceManager {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        // 创建 dummy 2D image
        let dummy_image = {
            let size = wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 };
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
            let size = wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 };
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

        let mipmap_generator = MipmapGenerator::new(&device);
        let model_allocator = ModelBufferAllocator::new();

        // 初始化 Model GPU Buffer 映射
        let gpu_buffers = {
            let model_cpu_buffer = model_allocator.cpu_buffer();
            let buffer_guard  = model_cpu_buffer.read();
            let model_gpu_buffer = GpuBuffer::new(&device, buffer_guard.as_bytes(), model_cpu_buffer.usage(), model_cpu_buffer.label());
            
            let mut map = FxHashMap::default();
            map.insert(model_cpu_buffer.id(), model_gpu_buffer);
            map
        };

        // 初始化 sampler_id_lookup 并添加 dummy_sampler
        let mut sampler_id_lookup = FxHashMap::default();
        sampler_id_lookup.insert(dummy_sampler.id, dummy_sampler.sampler.clone());

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
            sampler_cache: FxHashMap::default(),
            sampler_id_lookup,
            view_cache: FxHashMap::default(),
            dummy_image,
            dummy_env_image,
            dummy_sampler,
            mipmap_generator,
            model_allocator,
            object_bind_group_cache: FxHashMap::default(),
            bind_group_id_lookup: FxHashMap::default(),
            internal_resources: FxHashMap::default(),
            internal_name_lookup: FxHashMap::default(),
            skeleton_buffers: FxHashMap::default(),
        }
    }

    pub fn next_frame(&mut self) {
        self.frame_index += 1;
        self.model_allocator.reset();
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
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
            let buffer_guard  = cpu_buf.read();
            let gpu_buf = GpuBuffer::new(
                &self.device, 
                buffer_guard.as_bytes(), 
                cpu_buf.usage(), 
                cpu_buf.label()
            );
            
            self.gpu_buffers.insert(new_id, gpu_buf);
        }
    }

    /// 分配一个 Model Uniform，返回字节偏移量
    #[inline]
    pub fn allocate_model_uniform(&mut self, data: crate::resources::uniforms::DynamicModelUniforms) -> u32 {
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

        let buffer_guard  = allocator.cpu_buffer().read();

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
            data_to_upload
        );
    }

    /// 获取当前 Model Buffer 的 ID，用于缓存验证
    #[inline]
    pub fn model_buffer_id(&self) -> u64 {
        self.model_allocator.buffer_id()
    }

    /// 通过缓存的 ID 快速获取 BindGroup 数据
    #[inline]
    pub fn get_cached_bind_group(&self, cached_bind_group_id: u64) -> Option<&BindGroupContext> {
        self.bind_group_id_lookup.get(&cached_bind_group_id)
    }

    pub fn prune(&mut self, ttl_frames: u64) {
        if self.frame_index < ttl_frames { return; }
        let cutoff = self.frame_index - ttl_frames;

        self.gpu_geometries.retain(|_, v| v.last_used_frame >= cutoff);
        self.gpu_materials.retain(|_, v| v.last_used_frame >= cutoff);
        // Sampler 缓存使用全局缓存，不需要按 Texture 清理
        self.gpu_buffers.retain(|_, v| v.last_used_frame >= cutoff);
        self.gpu_images.retain(|_, v| v.last_used_frame >= cutoff);
        self.global_states.retain(|_, v| v.last_used_frame >= cutoff);
        // texture_bindings 跟随 gpu_images 清理
        self.texture_bindings.retain(|_, b| self.gpu_images.contains_key(&b.cpu_image_id));
    }
}
