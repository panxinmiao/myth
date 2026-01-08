use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::vec;

use uuid::Uuid;
use core::ops::Range;

use crate::core::geometry::{Geometry};
use crate::core::material::Material;
use crate::core::texture::Texture;
use crate::core::binding::{BindingType, BindingResource, BindingDescriptor};
use crate::core::buffer::DataBuffer;

use super::vertex_layout::{self, GeneratedVertexLayout};
use super::gpu_buffer::GpuBuffer;
use super::gpu_texture::GpuTexture;

// 全局资源 ID 生成器
static NEXT_RESOURCE_ID: AtomicU64 = AtomicU64::new(1);

pub fn generate_resource_id() -> u64 {
    NEXT_RESOURCE_ID.fetch_add(1, Ordering::Relaxed)
}

// ============================================================================
// GPU 资源包装器
// ============================================================================

pub struct GPUGeometry {
    pub layout_info: Arc<GeneratedVertexLayout>,

    pub vertex_buffers: Vec<wgpu::Buffer>,
    pub vertex_buffer_ids: Vec<u64>,
    pub index_buffer: Option<(wgpu::Buffer, wgpu::IndexFormat, u32, u64)>,

    pub draw_range: Range<u32>,
    pub instance_range: Range<u32>, 
    pub version: u64,
    pub last_used_frame: u64,
}

/// 材质的 GPU 资源
pub struct GPUMaterial {    
    pub bind_group: wgpu::BindGroup,
    pub bind_group_id: u64,
    pub layout: Arc<wgpu::BindGroupLayout>,


    // 缓存字段
    pub last_version: u64,
    pub last_resource_hash: u64,
    // layout 相关的 defines 实际上影响的是 Pipeline，但这里我们保存它以防万一
    // 或者仅仅用于判断是否需要重建 Layout (通常 Layout 只在 Defines 变的时候变)
    pub last_defines_hash: u64,


    pub last_used_frame: u64,
    // 布局签名 (Hash of Vec<BindingDescriptor>)
    // 用于判断是否需要切换 Pipeline Layout
    // pub layout_hash: u64, 
    // // 资源签名列表
    // // 用于判断是否需要重建 BindGroup
    // // 对应每个 Slot 的资源指纹 (例如 Texture ID + GenerationID)
    // pub resource_signatures: Vec<Option<u64>>, 
    // pub material_version: u64,
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

    // 辅助函数：计算 Hash
    fn calculate_hash<T: Hash>(t: &T) -> u64 {
        let mut s = DefaultHasher::new();
        t.hash(&mut s);
        s.finish()
    }

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
    // 通用 Buffer Logic
    // ========================================================================

    /// 准备 GPU Buffer：如果不存在则创建，如果版本过期则更新
    /// 返回当前 GPU 资源的物理 ID (用于检测 Resize)
    pub fn prepare_buffer(&mut self, cpu_buffer: &DataBuffer) -> u64 {
        let id = cpu_buffer.id;

        // 1. 获取或创建
        let gpu_buf = self.buffers.entry(id).or_insert_with(|| {
            let mut buf = GpuBuffer::new(
                &self.device, 
                &cpu_buffer.data, 
                cpu_buffer.usage, 
                Some(&cpu_buffer.label)
            );
            // 对于 Uniform Buffer，默认开启 Shadow Copy 以优化频繁的小数据更新
            if cpu_buffer.usage.contains(wgpu::BufferUsages::UNIFORM) {
                buf.enable_shadow_copy();
            }
            buf
        });

        gpu_buf.last_used_frame = self.frame_index;

        // 2. 更新 (GpuBuffer 内部会比较 version)
        if cpu_buffer.usage.contains(wgpu::BufferUsages::UNIFORM) {
            // Uniform 走 Diff 逻辑 (忽略 version)
            gpu_buf.update_with_data(&self.device, &self.queue, &cpu_buffer.data);
        } else {
            // 其他 (Vertex/Index/Storage) 走 Version 逻辑
            gpu_buf.update_with_version(
                &self.device, 
                &self.queue, 
                &cpu_buffer.data, 
                cpu_buffer.version
            );
        }

        gpu_buf.id
    }

    // ========================================================================
    // Geometry Logic
    // ========================================================================

    pub fn prepare_geometry(&mut self, geometry: &Geometry) {
        // let geometry_id = geometry.id;

        let mut resized_buffers = HashSet::new();

        // 1. 统一处理 Vertex Buffers
        for attr in geometry.attributes.values() {
            let cpu_buf = attr.buffer.read();
            let old_id = self.buffers.get(&cpu_buf.id).map(|b| b.id);
            
            let new_id = self.prepare_buffer(&cpu_buf);
            
            // 检测到底层 Buffer 是否重建了 (Resize)
            if let Some(oid) = old_id {
                if oid != new_id { resized_buffers.insert(cpu_buf.id); }
            }
        }

        // 2. 统一处理 Index Buffer
        if let Some(indices) = &geometry.index_attribute {
            let cpu_buf = indices.buffer.read();
            let old_id = self.buffers.get(&cpu_buf.id).map(|b| b.id);
            let new_id = self.prepare_buffer(&cpu_buf);
            if let Some(oid) = old_id {
                if oid != new_id { resized_buffers.insert(cpu_buf.id); }
            }
        }

        // 3. 检查重建 Geometry 绑定
        let needs_rebuild = if let Some(gpu_geo) = self.geometries.get(&geometry.id) {
            geometry.version > gpu_geo.version || !resized_buffers.is_empty()
        } else {
            true
        };

        if needs_rebuild {
            self.create_gpu_geometry(geometry);
        }

        if let Some(gpu_geo) = self.geometries.get_mut(&geometry.id) {
            // gpu_geo.draw_range = geometry.draw_range.clone();
            // gpu_geo.topology = geometry.topology;
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
            let cpu_buf = layout_desc.buffer.read();
            let gpu_buf = self.buffers.get(&cpu_buf.id).expect("Buffer prepared");
            vertex_buffers.push(gpu_buf.buffer.clone()); 
            vertex_buffer_ids.push(gpu_buf.id);
        }

        let index_buffer = if let Some(indices) = &geometry.index_attribute {
            let cpu_buf = indices.buffer.read();
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
            draw_range: geometry.draw_range.clone(),
            instance_range: 0..1,
            version: geometry.version,
            last_used_frame: self.frame_index,
        };

        self.geometries.insert(geometry.id, gpu_geo);
    }


    // ========================================================================
    // Material Logic
    // ========================================================================

    pub fn prepare_material(&mut self, material: &Material, texture_assets: &HashMap<Uuid, Arc<RwLock<Texture>>>) {

        // [L0] 粗粒度检查
        if let Some(gpu_mat) = self.materials.get_mut(&material.id) {
            if gpu_mat.last_version == material.version {
                gpu_mat.last_used_frame = self.frame_index;
                return; // 完全没变，直接返回
            }
        }

        // --- 开始更新流程 ---

        // 1. 总是刷新 Uniforms (开销极小)
        material.flush_uniforms();

        // 2. 获取当前状态
        let resources = material.get_resources();
        let defines = material.get_defines();

        let resource_hash = Self::calculate_hash(&resources);
        let defines_hash = Self::calculate_hash(&defines);

        // 3. 检查是否可以复用 BindGroup
        // 我们需要先拿到 old_cache (如果存在)
        // 注意：这里需要处理 ownership，为了代码清晰，我们采用 remove-then-insert 或者 get_mut
        
        let needs_rebuild_layout = if let Some(gpu_mat) = self.materials.get(&material.id) {
             gpu_mat.last_defines_hash != defines_hash
        } else {
            true
        };

        let needs_rebuild_bindgroup = if let Some(gpu_mat) = self.materials.get(&material.id) {
            needs_rebuild_layout || gpu_mat.last_resource_hash != resource_hash
        } else {
            true
        };

        if !needs_rebuild_bindgroup {
            // 只更新版本号，不需要重建 GPU 资源
            if let Some(gpu_mat) = self.materials.get_mut(&material.id) {
                gpu_mat.last_version = material.version;
                gpu_mat.last_used_frame = self.frame_index;
            }
            return;
        }


        // let id = material.id;

        // let (descriptors ,resources) = material.get_bindings();

        // 4.预处理所有资源 (Buffers & Textures)
        for res in &resources {
            match res {
                BindingResource::Buffer { buffer, offset: _, size: _ } => {
                    let cpu_buf = buffer.read();
                    self.prepare_buffer(&cpu_buf); // 自动处理所有 Buffer 的创建和更新！
                },
                BindingResource::Texture(Some(tex_id)) => {
                    if let Some(tex_arc) = texture_assets.get(tex_id) {
                        self.add_or_update_texture(&tex_arc.read().unwrap());
                    }
                },
                _ => {}
            }
        }

        // 5.重建 Layout (如果 Defines 变了)
        // 注意：如果只是换贴图，Defines 不变，Layout 也不变
        let layout = if needs_rebuild_layout {
            // 调用 material.get_layout() 并创建 wgpu::BindGroupLayout
            let descriptors = material.get_layout();
            self.get_or_create_layout(&descriptors).0
        } else {
            // 复用旧的
            self.materials.get(&material.id).unwrap().layout.clone()
        };

        // 6. 重建 BindGroup
        let bind_group = self.create_bind_group(&layout, &resources).0; // 需要实现 create_bind_group

        // 7. 更新缓存
        let new_gpu_mat = GPUMaterial {
            bind_group,
            bind_group_id: generate_resource_id(), // 生成新的 ID 触发 RenderPass set_bind_group
            layout,
            last_version: material.version,
            last_resource_hash: resource_hash,
            last_defines_hash: defines_hash,
            last_used_frame: self.frame_index,
        };

        self.materials.insert(material.id, new_gpu_mat);

    }

    pub fn get_material(&self, material_id: uuid::Uuid) -> Option<&GPUMaterial> {
        self.materials.get(&material_id)
    }

    // --- 内部辅助 ---

    pub fn compute_layout_hash(&self, descriptors: &[BindingDescriptor]) -> u64 {
        let mut hasher = DefaultHasher::new();
        descriptors.hash(&mut hasher);
        hasher.finish()
    }

    pub fn get_or_create_layout(&mut self, descriptors: &[BindingDescriptor]) -> (Arc<wgpu::BindGroupLayout>, u64) {

        let hash = self.compute_layout_hash(descriptors);

        if let Some(layout) = self.layout_cache.get(&hash) {
            return (layout.clone(), hash);
        }

        let entries: Vec<wgpu::BindGroupLayoutEntry> = descriptors.iter().map(|desc| {
            wgpu::BindGroupLayoutEntry {
                binding: desc.index,
                visibility: desc.visibility,
                ty: match desc.bind_type {
                    BindingType::UniformBuffer { dynamic, min_size }=> wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: dynamic,
                        min_binding_size: min_size.and_then(wgpu::BufferSize::new),
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
        (layout, hash)
    }

    pub fn create_bind_group(
        &self, 
        layout: &wgpu::BindGroupLayout, 
        // descriptors: &[BindingDescriptor],
        resources: &[BindingResource],
    ) -> (wgpu::BindGroup, u64) {
        
        let mut entries = Vec::new();

        for (i, resource_data) in resources.iter().enumerate() {

            let binding_resource = match resource_data {
                BindingResource::Buffer { buffer, offset, size } => {
                    let cpu_id = buffer.id;
                    let gpu_buf = self.buffers.get(&cpu_id).expect("Buffer should be prepared before creating bindgroup");
                    // gpu_buf.buffer.as_entire_binding()
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &gpu_buf.buffer,
                        offset: *offset,
                        size: size.and_then(wgpu::BufferSize::new), // 处理 Option<u64> -> Option<NonZeroU64>
                    })
                },
                BindingResource::Texture(tid_opt) => {
                    let gpu_tex = if let Some(tid) = tid_opt {
                        self.textures.get(tid).unwrap_or(&self.dummy_texture)
                    } else { &self.dummy_texture };
                    wgpu::BindingResource::TextureView(&gpu_tex.view)
                },
                BindingResource::Sampler(tid_opt) => {
                     let gpu_tex = if let Some(tid) = tid_opt {
                        self.textures.get(tid).unwrap_or(&self.dummy_texture)
                    } else { &self.dummy_texture };
                    wgpu::BindingResource::Sampler(&gpu_tex.sampler)
                },
                _ => panic!("Unsupported binding resource"),
            };

            entries.push(wgpu::BindGroupEntry {
                binding: i as u32,   // 假设顺序对应, todo: 是否需要更严谨的 mapping？ descriptor.index
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