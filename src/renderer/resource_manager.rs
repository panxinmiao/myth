// src/renderer/resource_manager.rs

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::vec;

use uuid::Uuid;
use wgpu::PrimitiveTopology;
use core::ops::Range;

use crate::core::geometry::{Geometry};
use crate::core::material::Material;
use crate::core::texture::Texture;
use crate::core::binding::{Bindable, BindingType, BindingResource, BindingDescriptor};

use super::vertex_layout::{self, GeneratedVertexLayout};

// 全局资源 ID 生成器
static NEXT_RESOURCE_ID: AtomicU64 = AtomicU64::new(1);

// 引入新抽象
use super::gpu_buffer::GpuBuffer;
use super::gpu_texture::GpuTexture;

pub fn generate_resource_id() -> u64 {
    NEXT_RESOURCE_ID.fetch_add(1, Ordering::Relaxed)
}

// ============================================================================
// GPU 资源包装器
// ============================================================================

pub struct GPUGeometry {
    pub layout_info: Arc<GeneratedVertexLayout>,

    // 我们仍然需要持有 wgpu::Buffer 的引用以便 Rendering 时绑定
    // 但这些引用必须在 prepare_geometry 时刷新，以防底层 buffer resize
    pub vertex_buffers: Vec<wgpu::Buffer>,
    pub vertex_buffer_ids: Vec<u64>,
    pub index_buffer: Option<(wgpu::Buffer, wgpu::IndexFormat, u32, u64)>,

    pub topology: PrimitiveTopology,
    pub draw_range: Range<u32>,
    pub instance_range: Range<u32>, 
    pub version: u64,
    pub last_used_frame: u64,
}

/// 材质的 GPU 资源
pub struct GPUMaterial {
    /// 主 Uniform Buffer (对应 Binding 0)
    pub uniform_buffer: GpuBuffer,
    
    pub bind_group: wgpu::BindGroup,
    pub bind_group_id: u64,
    pub layout: Arc<wgpu::BindGroupLayout>,

    
    // 布局签名 (Hash of Vec<BindingDescriptor>)
    // 用于判断是否需要切换 Pipeline Layout
    pub layout_hash: u64, 
    // 资源签名列表
    // 用于判断是否需要重建 BindGroup
    // 对应每个 Slot 的资源指纹 (例如 Texture ID + GenerationID)
    pub resource_signatures: Vec<Option<u64>>, 
    pub last_used_frame: u64,
}

// ============================================================================
// Resource Manager
// ============================================================================

pub struct ResourceManager {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    frame_index: u64,

    buffers: HashMap<Uuid, GpuBuffer>,
    textures: HashMap<Uuid, GpuTexture>,

    geometries: HashMap<Uuid, GPUGeometry>,
    materials: HashMap<Uuid, GPUMaterial>,
    
    // Layout 缓存: Hash(BindingDescriptors) -> Layout
    // 这样不同材质如果结构相同，可以复用 Layout
    layout_cache: HashMap<u64, Arc<wgpu::BindGroupLayout>>,

    dummy_texture: GpuTexture, 
}

impl ResourceManager {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {

        let dummy_tex_data = Texture::new_2d("dummy", 1, 1, Some(vec![255, 255, 255, 255]), wgpu::TextureFormat::Rgba8Unorm); 
        let dummy_texture = GpuTexture::new(&device, &queue, &dummy_tex_data);
        
        Self {
            device,
            queue,
            frame_index: 0,
            buffers: HashMap::new(),
            geometries: HashMap::new(),
            materials: HashMap::new(),
            textures: HashMap::new(),
            layout_cache: HashMap::new(),
            dummy_texture,
        }
    }

    pub fn next_frame(&mut self) {
        self.frame_index += 1;
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    // ========================================================================
    // Geometry Logic
    // ========================================================================

    pub fn prepare_geometry(&mut self, geometry: &Geometry) {
        let geometry_id = geometry.id;

        // 1. 确保所有依赖的 Buffer 都已创建并更新
        // 我们遍历 Geometry 的所有属性 Buffer，确保它们在 GPU 上是最新的
        // 这一步也处理了 Buffer Resize 的情况
        let mut resized_buffers = HashSet::new();
        
        for attr in geometry.attributes.values() {
            let buffer_arc = &attr.buffer;
            let cpu_buf = buffer_arc.read().unwrap();
            let id = cpu_buf.id;
            
            // 获取或创建
            let gpu_buf = self.buffers.entry(id).or_insert_with(|| {
                GpuBuffer::new(
                    &self.device, 
                    &cpu_buf.data, 
                    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, 
                    Some(&format!("GeoBuf-{:?}", id))
                )
            });
            
            // 标记活跃
            gpu_buf.last_used_frame = self.frame_index;

            // 检查版本并更新 (GpuBuffer 内部处理 Version 检查)
            let resized = gpu_buf.update_with_version(
                &self.device, 
                &self.queue, 
                &cpu_buf.data, 
                cpu_buf.version
            );
            
            if resized {
                resized_buffers.insert(id);
            }
        }

        // Index Buffer 处理
        if let Some(indices) = &geometry.index_attribute {
            let cpu_buf = indices.buffer.read().unwrap();
            let id = cpu_buf.id;
            let gpu_buf = self.buffers.entry(id).or_insert_with(|| {
                 GpuBuffer::new(
                    &self.device, 
                    &cpu_buf.data, 
                    wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST, 
                    Some("IndexBuf")
                )
            });
            gpu_buf.last_used_frame = self.frame_index;
            let resized = gpu_buf.update_with_version(&self.device, &self.queue, &cpu_buf.data, cpu_buf.version);
            if resized { resized_buffers.insert(id); }
        }

        // 2. 检查 Geometry 是否需要重建结构
        let needs_rebuild = if let Some(gpu_geo) = self.geometries.get(&geometry_id) {
            geometry.version > gpu_geo.version || !resized_buffers.is_empty()
        } else {
            true
        };

        if needs_rebuild {
            self.create_gpu_geometry(geometry);
        }

        if let Some(gpu_geo) = self.geometries.get_mut(&geometry.id) {
            gpu_geo.draw_range = geometry.draw_range.clone();
            gpu_geo.topology = geometry.topology;
            gpu_geo.last_used_frame = self.frame_index;
        }
    }

    pub fn get_geometry(&self, geometry_id: uuid::Uuid) -> Option<&GPUGeometry> {
        self.geometries.get(&geometry_id)
    }

    fn create_gpu_geometry(&mut self, geometry: &Geometry) {
        let layout_info = Arc::new(vertex_layout::generate_vertex_layout(geometry));

        let mut vertex_buffers = Vec::new();
        let mut vertex_buffer_ids = Vec::new();
        
        for layout_desc in &layout_info.buffers {
            // layout_desc.buffer 是 CPU 端的 Arc<RwLock<GeometryBuffer>>
            // 我们需要它的 ID 来查找 GPU Buffer
            let cpu_buf = layout_desc.buffer.read().unwrap();
            let gpu_buf = self.buffers.get(&cpu_buf.id).expect("Buffer should represent by prepare_geometry");
            vertex_buffers.push(gpu_buf.buffer.clone()); // 这是一个 cheap clone (Handle)
            vertex_buffer_ids.push(gpu_buf.id);
        }

        let index_buffer = if let Some(indices) = &geometry.index_attribute {
            let cpu_buf = indices.buffer.read().unwrap();
            let gpu_buf = self.buffers.get(&cpu_buf.id).unwrap();
            let format = match indices.format {
                wgpu::VertexFormat::Uint16 => wgpu::IndexFormat::Uint16,
                wgpu::VertexFormat::Uint32 => wgpu::IndexFormat::Uint32,
                _ => wgpu::IndexFormat::Uint16,
            };
            Some((gpu_buf.buffer.clone(), format, indices.count, gpu_buf.id))
        } else {
            None
        };

        let gpu_geo = GPUGeometry {
            layout_info,
            vertex_buffers,
            vertex_buffer_ids,
            index_buffer,
            topology: geometry.topology,
            draw_range: geometry.draw_range.clone(),
            instance_range: 0..1,
            version: geometry.version,
            last_used_frame: self.frame_index,
        };

        self.geometries.insert(geometry.id, gpu_geo);
    }


    // ========================================================================
    // Material Logic (完全重构)
    // ========================================================================

    /// 准备材质：检查更新、创建 BindGroup、上传 Uniform
    pub fn prepare_material(&mut self, material: &Material, texture_assets: &HashMap<Uuid, Arc<RwLock<Texture>>>) {

        // let id = material.id;
        // let (descriptors ,resources) = material.get_bindings();


        let id = material.id;
        
        // 1. 获取 Binding 描述和资源
        let (descriptors ,resources) = material.get_bindings();

        // 计算 Layout Hash (用于 Pipeline 兼容性检查)
        let layout_hash = self.compute_layout_hash(&descriptors);


        // 1. 自动上传依赖的纹理
        for res in &resources {
            if let BindingResource::Texture(Some(tex_id)) = res {
                if let Some(tex_arc) = texture_assets.get(tex_id) {
                    let texture = tex_arc.read().unwrap();
                    self.add_or_update_texture(&texture);
                }
            }
        }
        
        // 计算 资源签名 (用于 BindGroup 重建检查)
        // 我们需要把 BindingResource 转换为可比较的 ID/Signature

        let current_signatures: Vec<Option<u64>> = resources.iter().map(|res| {
            match res {
                BindingResource::Buffer(_) => None, // Buffer 内容更新不影响 BindGroup 结构，只影响 write_buffer
                BindingResource::BufferId(uuid) => Some(uuid_to_u64(uuid)), // 外部 Buffer ID 变了需要重建
                BindingResource::Texture(Some(tid)) => {
                    // 如果引用了纹理，签名不仅包含 ID，还包含纹理的 GenerationID (Resized?)
                    if let Some(gpu_tex) = self.textures.get(tid) {
                        // 组合 UUID hash 和 gen_id，这里简化处理，只用 gen_id 配合 tid
                        // todo: 实际应该: hash(tid, gpu_tex.generation_id)
                        Some(gpu_tex.generation_id.wrapping_add(uuid_to_u64(tid))) 
                    } else {
                        // 纹理未就绪，视为 Dummy (Signature=0 或特定值)
                        Some(0) 
                    }
                },
                BindingResource::Texture(None) => Some(0), // No Texture
                BindingResource::Sampler(Some(tid)) => Some(uuid_to_u64(tid)), // Sampler ID
                BindingResource::Sampler(None) => Some(0),
            }
        }).collect();

        // 2. 检查是否存在或需要重建
        let needs_rebuild = if let Some(gpu_mat) = self.materials.get(&id) {
             // Layout 变了 (Define 变了) OR 资源引用变了 (换图了)
             gpu_mat.layout_hash != layout_hash || gpu_mat.resource_signatures != current_signatures
        } else {
            true // 新材质
        };

        if needs_rebuild {
            // A. 获取/创建 Layout
            let layout = self.get_or_create_layout(&descriptors, layout_hash);

            // B. Uniform Buffer (使用 GpuBuffer 管理)
            let uniform_buffer = if let Some(old_mat) = self.materials.remove(&id) {
                // 如果是重建 BindGroup，我们继承旧的 Buffer，防止数据丢失
                // 注意：这里使用了 remove 再 insert，为了拿所有权
                old_mat.uniform_buffer
            } else {

                 // 新材质：创建并启用 Diff 模式
                 let initial_data = resources.iter().find_map(|r| match r {
                     BindingResource::Buffer(data) => Some(data),
                     _ => None
                 }).expect("material unifrom buffer not exist");

                 let mut buf = GpuBuffer::new(
                    &self.device, 
                    *initial_data, 
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, 
                    Some(&format!("MatBuf-{:?}", id))
                 );
                 buf.enable_shadow_copy(); // 关键！启用 Diff
                 buf

            };

            // C. 创建 BindGroup
            let (bind_group, bg_id) = self.create_bind_group(
                &layout, 
                &descriptors, 
                &resources, 
                &uniform_buffer.buffer
            );

            // D. 存入/更新 Map
            let gpu_mat = GPUMaterial {
                uniform_buffer,
                bind_group,
                bind_group_id: bg_id,
                layout,
                layout_hash,
                resource_signatures: current_signatures,
                last_used_frame: self.frame_index,
            };
            self.materials.insert(id, gpu_mat);
        }

        // 3. 数据更新 (Uniform Buffer Upload)
        let gpu_mat = self.materials.get_mut(&id).unwrap();
        gpu_mat.last_used_frame = self.frame_index;

        if let Some(BindingResource::Buffer(data)) = resources.iter().find(|r| matches!(r, BindingResource::Buffer(_))) {
            gpu_mat.uniform_buffer.update_with_data(&self.device, &self.queue, data);
        }

    }

    pub fn get_material(&self, material_id: uuid::Uuid) -> Option<&GPUMaterial> {
        self.materials.get(&material_id)
    }

    // --- 内部辅助 ---

    fn compute_layout_hash(&self, descriptors: &[BindingDescriptor]) -> u64 {
        let mut hasher = DefaultHasher::new();
        descriptors.hash(&mut hasher);
        hasher.finish()
    }

    fn get_or_create_layout(&mut self, descriptors: &[BindingDescriptor], hash: u64) -> Arc<wgpu::BindGroupLayout> {
        if let Some(layout) = self.layout_cache.get(&hash) {
            return layout.clone();
        }

        let entries: Vec<wgpu::BindGroupLayoutEntry> = descriptors.iter().map(|desc| {
            wgpu::BindGroupLayoutEntry {
                binding: desc.index,
                visibility: desc.visibility,
                ty: match desc.bind_type {
                    BindingType::UniformBuffer => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    BindingType::StorageBuffer { read_only } => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    BindingType::Texture { sample_type, view_dimension, multisampled } => wgpu::BindingType::Texture {
                        sample_type,
                        view_dimension,
                        multisampled,
                    },
                    BindingType::Sampler { type_ } => wgpu::BindingType::Sampler(type_),
                },
                count: None,
            }
        }).collect();

        let layout = Arc::new(self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Auto Generated Material Layout"),
            entries: &entries,
        }));

        self.layout_cache.insert(hash, layout.clone());
        layout
    }

    fn create_bind_group(
        &self, 
        layout: &wgpu::BindGroupLayout, 
        descriptors: &[BindingDescriptor],
        resources: &[BindingResource],
        uniform_buffer: &wgpu::Buffer, // 传入当前材质管理的 Uniform Buffer
    ) -> (wgpu::BindGroup, u64) {
        
        let mut entries = Vec::new();

        for (i, desc) in descriptors.iter().enumerate() {
            let resource_data = &resources[i];
            
            let binding_resource = match resource_data {
                // 1. Uniform Buffer: 使用传入的 buffer
                BindingResource::Buffer(_) => {
                    uniform_buffer.as_entire_binding()
                },
                
                // 2. Texture: 查表
                BindingResource::Texture(tid_opt) => {
                    let gpu_tex = if let Some(tid) = tid_opt {
                        self.textures.get(tid).unwrap_or(&self.dummy_texture)
                    } else {
                        &self.dummy_texture
                    };
                    wgpu::BindingResource::TextureView(&gpu_tex.view)
                },
                
                // 3. Sampler: 查表 (通常跟 Texture ID 一样，或者是分离的)
                BindingResource::Sampler(tid_opt) => {
                     let gpu_tex = if let Some(tid) = tid_opt {
                        self.textures.get(tid).unwrap_or(&self.dummy_texture)
                    } else {
                        &self.dummy_texture
                    };
                    wgpu::BindingResource::Sampler(&gpu_tex.sampler)
                },

                // 4. External Buffer ID (未实现，预留)
                BindingResource::BufferId(_) => {
                    panic!("External Buffer binding not implemented yet in ResourceManager");
                }
            };

            entries.push(wgpu::BindGroupEntry {
                binding: desc.index,
                resource: binding_resource,
            });
        }

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Auto BindGroup"),
            layout,
            entries: &entries,
        });

        (bind_group, generate_resource_id())
    }

    // ========================================================================
    // Texture Logic (保持不变)
    // ========================================================================

    pub fn add_or_update_texture(&mut self, texture: &Texture) {
        let gpu_tex = self.textures.entry(texture.id).or_insert_with(|| {
             GpuTexture::new(&self.device, &self.queue, texture)
        });
        
        gpu_tex.last_used_frame = self.frame_index;
        gpu_tex.update(&self.device, &self.queue, texture);
    }


    // ========================================================================
    // Garbage Collection
    // ========================================================================

    pub fn prune(&mut self, ttl_frames: u64) {
        if self.frame_index < ttl_frames { return; }
        let cutoff = self.frame_index - ttl_frames;

        self.geometries.retain(|_, v| v.last_used_frame >= cutoff);
        self.buffers.retain(|_, v| v.last_used_frame >= cutoff);
        self.materials.retain(|_, v| v.last_used_frame >= cutoff);
        self.textures.retain(|_, v| v.last_used_frame >= cutoff);
        // Layout 缓存也可以清理，但通常 Layout 占内存极小，保留无妨
    }
}

// Helper
fn uuid_to_u64(uuid: &uuid::Uuid) -> u64 {
    let bytes = uuid.as_bytes();
    u64::from_le_bytes(bytes[0..8].try_into().unwrap())
}