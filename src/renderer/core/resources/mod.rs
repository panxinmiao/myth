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
//! - global.rs: 全局渲染资源 (Light/Environment Buffer)

mod allocator;
mod buffer;
mod texture;
mod geometry;
mod material;
mod binding;
mod mipmap;
mod global;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use glam::Mat4;
use slotmap::SecondaryMap;
use core::ops::Range;
use rustc_hash::FxHashMap;

use crate::renderer::core::resources::mipmap::MipmapGenerator;
use crate::resources::texture::{Texture, TextureSampler};

use crate::scene::SkeletonKey;
use crate::resources::buffer::CpuBuffer;
use crate::assets::{GeometryHandle, MaterialHandle, TextureHandle};

use crate::renderer::pipeline::vertex::GeneratedVertexLayout;

pub use allocator::ModelBufferAllocator;
pub use global::GlobalResources;

static NEXT_RESOURCE_ID: AtomicU64 = AtomicU64::new(0);

pub fn generate_resource_id() -> u64 {
    NEXT_RESOURCE_ID.fetch_add(1, Ordering::Relaxed)
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
        let id = generate_resource_id();

        Self {
            id,
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
        self.id = generate_resource_id();
    }
}

pub struct GpuTexture {
    pub view: wgpu::TextureView,
    pub image_id: u64,
    pub image_generation_id: u64,
    pub version: u64,
    pub image_data_version: u64,
    pub last_used_frame: u64,
}

pub struct GpuImage {
    pub texture: wgpu::Texture,
    pub id: u64,
    pub version: u64,
    pub generation_id: u64,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: wgpu::TextureFormat,
    pub mip_level_count: u32,
    pub usage: wgpu::TextureUsages,
    pub mipmaps_generated: bool,
    pub last_used_frame: u64,
}

pub struct GpuGeometry {
    pub layout_info: Arc<GeneratedVertexLayout>,
    pub vertex_buffers: Vec<wgpu::Buffer>,
    pub vertex_buffer_ids: Vec<u64>,
    pub index_buffer: Option<(wgpu::Buffer, wgpu::IndexFormat, u32, u64)>,
    pub draw_range: Range<u32>,
    pub instance_range: Range<u32>,
    pub version: u64,
    pub last_data_version: u64,
    pub last_used_frame: u64,
}

pub struct GpuMaterial {
    pub bind_group: wgpu::BindGroup,
    pub bind_group_id: u64,
    pub layout: wgpu::BindGroupLayout,
    pub layout_id: u64,
    pub binding_wgsl: String,
    pub uniform_buffers: Vec<u64>,
    pub last_data_version: u64,
    pub last_binding_version: u64,
    pub last_layout_version: u64,
    pub last_used_frame: u64,
}

/// GPU 全局状态 (Group 0)
/// 
/// 包含 Camera Uniforms、Light Storage Buffer、Environment Maps 等
/// 
/// # 缓存策略
/// 
/// BindGroup 重建条件（结构指纹变化）：
/// - `render_state_buffer_id` 变化（Camera Buffer 重建）
/// - `global_structure_version` 变化（Light Buffer 扩容）
/// - `env_map` 变化（环境贴图切换）
/// 
/// 仅数据上传条件（数据版本变化）：
/// - `render_state_data_version` 变化 → 上传 Camera 数据
/// - `global_data_version` 变化 → 上传 Light/Env 数据
pub struct GpuGlobalState {
    pub id: u32,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_id: u64,
    pub layout: wgpu::BindGroupLayout,
    pub layout_id: u64,
    pub binding_wgsl: String,
    
    // === 结构指纹（变化时需重建 BindGroup）===
    /// RenderState 的 Buffer ID
    pub render_state_buffer_id: u64,
    /// GlobalResources 的结构版本
    pub global_structure_version: u64,
    /// 环境贴图 Handle
    pub env_map: Option<crate::assets::TextureHandle>,
    /// Environment Buffer ID
    pub env_buffer_id: u64,
    /// Light Buffer ID
    pub light_buffer_id: u64,
    
    // === 数据版本（变化时只需 write_buffer）===
    /// RenderState 数据版本
    pub last_render_state_data_version: u64,
    /// GlobalResources 数据版本
    pub last_global_data_version: u64,
    
    pub last_used_frame: u64,
}


// ============================================================================
// 紧凑的 BindGroup 缓存键
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ObjectBindGroupKey {
    pub model_buffer_id: u64,
    pub skeleton_buffer_id: Option<u64>,
    pub morph_buffer_id: Option<u64>,
}

#[derive(Clone)]
pub struct ObjectBindingData {
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
    pub(crate) gpu_textures: SecondaryMap<TextureHandle, GpuTexture>,
    pub(crate) gpu_samplers: SecondaryMap<TextureHandle, wgpu::Sampler>,

    pub(crate) global_states: FxHashMap<u64, GpuGlobalState>,
    pub(crate) gpu_buffers: FxHashMap<u64, GpuBuffer>,
    pub(crate) gpu_images: FxHashMap<u64, GpuImage>,

    pub(crate) sampler_cache: FxHashMap<TextureSampler, wgpu::Sampler>,
    pub(crate) layout_cache: FxHashMap<Vec<wgpu::BindGroupLayoutEntry>, (wgpu::BindGroupLayout, u64)>,

    pub(crate) dummy_texture: GpuTexture,
    pub(crate) dummy_env_texture: GpuTexture,
    pub(crate) dummy_sampler: wgpu::Sampler,
    pub(crate) mipmap_generator: MipmapGenerator,

    // === Model Buffer Allocator ===
    pub(crate) model_allocator: ModelBufferAllocator,

    // === Object BindGroup 缓存 ===
    pub(crate) object_bind_group_cache: FxHashMap<ObjectBindGroupKey, ObjectBindingData>,
    pub(crate) bind_group_id_lookup: FxHashMap<u64, ObjectBindingData>,

    // === 骨骼 Buffer ===
    pub(crate) skeleton_buffers: FxHashMap<SkeletonKey, CpuBuffer<Vec<Mat4>>>,

    // === 全局渲染资源 (Light/Environment) ===
    pub(crate) global_resources: GlobalResources,
}

impl ResourceManager {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {

        let dummy_tex = Texture::new_2d(Some("dummy"), 1, 1, Some(vec![255, 255, 255, 255]), wgpu::TextureFormat::Rgba8Unorm);
        let dummy_gpu_image = GpuImage::new(&device, &queue, &dummy_tex.image, 1, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        let dummy_gpu_tex = GpuTexture::new(&dummy_tex, &dummy_gpu_image);

        // 创建 dummy env texture (cube map)
        let dummy_env_tex = {
            // 手动创建 WGPU Texture
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
            let black_pixel = [0u8; 24];
   
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                &black_pixel,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
            );
            

            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::Cube),
                ..Default::default()
            });

            GpuTexture {
                view,
                image_id: Default::default(),
                version: 0,
                image_generation_id: 0,
                image_data_version: 0,
                last_used_frame: u64::MAX,
            }
        };

        let dummy_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Dummy Sampler"),
            ..Default::default()
        });

        let mipmap_generator = MipmapGenerator::new(&device);
        let model_allocator = ModelBufferAllocator::new();

        // 初始化 Model GPU Buffer 映射
        let gpu_buffers = {
            let model_cpu_buffer = model_allocator.cpu_buffer();
            let model_gpu_buffer = GpuBuffer::new(&device, model_cpu_buffer.as_bytes(), model_cpu_buffer.usage(), model_cpu_buffer.label());
            
            let mut map = FxHashMap::default();
            map.insert(model_cpu_buffer.id(), model_gpu_buffer);
            map
        };

        Self {
            device,
            queue,
            frame_index: 0,
            gpu_geometries: SecondaryMap::new(),
            gpu_materials: SecondaryMap::new(),
            gpu_textures: SecondaryMap::new(),
            gpu_samplers: SecondaryMap::new(),
            global_states: FxHashMap::default(),
            gpu_buffers,
            gpu_images: FxHashMap::default(),
            layout_cache: FxHashMap::default(),
            sampler_cache: FxHashMap::default(),
            dummy_texture: dummy_gpu_tex,
            dummy_env_texture: dummy_env_tex,
            dummy_sampler,
            mipmap_generator,
            model_allocator,
            object_bind_group_cache: FxHashMap::default(),
            bind_group_id_lookup: FxHashMap::default(),
            skeleton_buffers: FxHashMap::default(),
            global_resources: GlobalResources::new(),
        }
    }

    pub fn next_frame(&mut self) {
        self.frame_index += 1;
        self.model_allocator.reset();
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// 每帧开始时重置 Model Buffer Allocator
    // pub fn reset_model_buffer(&mut self) {
    //     self.model_allocator.reset();
    // }

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
        let data = allocator.cpu_buffer().as_bytes();

        if allocator.need_recreate_buffer() {
            // Buffer 重建后，所有缓存都失效
            self.object_bind_group_cache.clear();
            self.bind_group_id_lookup.clear();
        }

        Self::write_buffer_internal(
            &self.device,
            &self.queue,
            &mut self.gpu_buffers,
            self.frame_index,
            buffer_ref,
            data
        );
    }

    /// 获取当前 Model Buffer 的 ID，用于缓存验证
    #[inline]
    pub fn model_buffer_id(&self) -> u64 {
        self.model_allocator.buffer_id()
    }

    /// 通过缓存的 ID 快速获取 BindGroup 数据
    #[inline]
    pub fn get_cached_bind_group(&self, cached_bind_group_id: u64) -> Option<&ObjectBindingData> {
        self.bind_group_id_lookup.get(&cached_bind_group_id)
    }

    pub fn prune(&mut self, ttl_frames: u64) {
        if self.frame_index < ttl_frames { return; }
        let cutoff = self.frame_index - ttl_frames;

        self.gpu_geometries.retain(|_, v| v.last_used_frame >= cutoff);
        self.gpu_materials.retain(|_, v| v.last_used_frame >= cutoff);
        self.gpu_textures.retain(|_, v| v.last_used_frame >= cutoff);
        self.gpu_samplers.retain(|k, _| self.gpu_textures.contains_key(k));
        self.gpu_buffers.retain(|_, v| v.last_used_frame >= cutoff);
        self.gpu_images.retain(|_, v| v.last_used_frame >= cutoff);
        self.global_states.retain(|_, v| v.last_used_frame >= cutoff);
    }
}
