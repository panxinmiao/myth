use std::collections::{HashMap, HashSet};
use std::sync::{Arc};
use std::sync::atomic::{AtomicU64, Ordering};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::vec;

use uuid::Uuid;
use core::ops::Range;

use crate::core::geometry::{Geometry};
use crate::core::material::Material;
use crate::core::texture::Texture;
use crate::core::world::WorldEnvironment;
use crate::core::buffer::DataBuffer;

use crate::renderer::binding::{BindingResource, Bindings};
use crate::renderer::resource_builder::{ResourceBuilder};
use crate::renderer::vertex_layout::{self, GeneratedVertexLayout};
use crate::renderer::gpu_buffer::GpuBuffer;
use crate::renderer::gpu_texture::GpuTexture;

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

    pub binding_wgsl: String,

    // 缓存字段
    pub last_version: u64,
    pub last_resource_hash: u64,

    pub last_used_frame: u64,
}

// 场景全局环境的 GPU 资源
pub struct GPUWorld {
    pub bind_group: wgpu::BindGroup,
    pub bind_group_id: u64,
    pub layout: Arc<wgpu::BindGroupLayout>,
    pub binding_wgsl: String, 
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
    worlds: HashMap<Uuid, GPUWorld>,
    
    // Layout 缓存: Hash(BindingDescriptors) -> Layout
    // 这样不同材质如果结构相同，可以复用 Layout
    // layout_cache: HashMap<u64, Arc<wgpu::BindGroupLayout>>,
    layout_cache: HashMap<Vec<wgpu::BindGroupLayoutEntry>, Arc<wgpu::BindGroupLayout>>,

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
            worlds: HashMap::new(),
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

    pub fn prepare_geometry(&mut self, geometry: &Geometry) -> &GPUGeometry {
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
            geometry.version() > gpu_geo.version || !resized_buffers.is_empty()
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

        self.geometries.get(&geometry.id).unwrap()
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
            let gpu_buf = self.buffers.get(&cpu_buf.id).expect("Buffer shoould prepared");
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


        let mut draw_range = geometry.draw_range.clone();
        if draw_range == (0..u32::MAX) {
            if let Some(attr) = geometry.attributes.get("position") {
                // 优先使用 position 的数量
                draw_range = draw_range.start..std::cmp::min(attr.count, draw_range.end);
            } else if let Some(attr) = geometry.attributes.values().next() {
                // 否则使用任意一个属性的数量
                draw_range = draw_range.start..std::cmp::min(attr.count, draw_range.end);
            } else {
                // 没有任何属性，绘制 0 个点
                draw_range = 0..0;
            }
        }

        let gpu_geo = GPUGeometry {
            layout_info,
            vertex_buffers,
            vertex_buffer_ids,
            index_buffer,
            draw_range: draw_range,
            instance_range: 0..1,
            version: geometry.version(),
            last_used_frame: self.frame_index,
        };

        self.geometries.insert(geometry.id, gpu_geo);
    }


    // ========================================================================
    // Material Logic
    // ========================================================================

    pub fn prepare_material(&mut self, material: &Material) -> &GPUMaterial{

        // [L0] 粗粒度检查

        // 先进行一次不可变查找，确认是否命中且版本一致
        let cache_hit = if let Some(gpu_mat) = self.materials.get(&material.id) {
            gpu_mat.last_version == material.version()
        } else {
            false
        };

        // 如果命中，重新获取可变引用，更新时间戳并返回
        if cache_hit {
            let gpu_mat = self.materials.get_mut(&material.id).unwrap();
            gpu_mat.last_used_frame = self.frame_index;
            return gpu_mat;
        }

        // --- 开始更新流程 ---

        // 1. 总是刷新 Uniforms (开销极小)
        material.flush_uniforms();

        // 2. 获取当前状态
        let mut builder = ResourceBuilder::new();
        material.define_bindings(&mut builder);

        // 3. 预处理资源 & 计算资源 Hash
        let mut resource_hasher = DefaultHasher::new();

        for res in &builder.resources {
            match res {
                BindingResource::Buffer { buffer, .. } => {
                    let cpu_buf = buffer.read();
                    self.prepare_buffer(&cpu_buf); // 确保 Buffer 已上传
                    // Hash ID
                    buffer.id.hash(&mut resource_hasher);
                },
                BindingResource::Texture(tex_opt) => {
                    if let Some(tex_arc) = tex_opt {
                        {
                            let tex = tex_arc;
                            self.add_or_update_texture(&tex); // 确保 GPU 端存在
                            // 2. Hash：使用 Texture ID 参与 Hash
                            tex.id.hash(&mut resource_hasher);
                        }
                    } else {
                        0.hash(&mut resource_hasher); // 空纹理
                    }
                },
                BindingResource::Sampler(tex_opt) => {
                    // Sampler 同理，也要确保 Texture 资源已上传(因为 Sampler 存在 GpuTexture 里)
                     if let Some(tex_arc) = tex_opt {
                        {
                            let tex = tex_arc;
                            self.add_or_update_texture(&tex);
                            tex.id.hash(&mut resource_hasher); // 这里的 Hash 最好加上 "Sampler" 混淆，不过分开 BindingIndex 已经够了
                        }
                    } else {
                        0.hash(&mut resource_hasher);
                    }
                }
            }
        }

        let resource_hash = resource_hasher.finish();

        // 4. 获取 Layout (利用缓存去重)
        let layout = self.get_or_create_layout(&builder.layout_entries);

        // 5. 检查是否需要重建 BindGroup
        // 仅当 Layout 指针变化 OR 资源 Hash 变化 OR 是新材质时才重建

        let mut can_reuse_bindgroup = false;

        if let Some(gpu_mat) = self.materials.get(&material.id) {
            let layout_unchanged = Arc::ptr_eq(&gpu_mat.layout, &layout);
            let resources_unchanged = gpu_mat.last_resource_hash == resource_hash;
            if layout_unchanged && resources_unchanged {
                can_reuse_bindgroup = true;
            }
        }

        if can_reuse_bindgroup {
             let gpu_mat = self.materials.get_mut(&material.id).unwrap();
             gpu_mat.last_version = material.version();
             gpu_mat.last_used_frame = self.frame_index;
             return gpu_mat;
        }


        // 6. 创建 BindGroup
        let (bind_group, bg_id) = self.create_bind_group(&layout, &builder.resources);

        // 如果需要重建，顺便生成 WGSL
        let binding_wgsl = builder.generate_wgsl(1); // Group 1

        // 7. 更新 Material 缓存
        let gpu_mat = GPUMaterial {
            bind_group,
            bind_group_id: bg_id,
            layout,
            binding_wgsl,
            last_version: material.version(),
            last_resource_hash: resource_hash,
            last_used_frame: self.frame_index,
        };
        self.materials.insert(material.id, gpu_mat);

        self.materials.get(&material.id).unwrap()

    }

    pub fn get_material(&self, material_id: uuid::Uuid) -> Option<&GPUMaterial> {
        self.materials.get(&material_id)
    }

    // --- 内部辅助 ---

    pub fn get_or_create_layout(&mut self, entries: &[wgpu::BindGroupLayoutEntry]) -> Arc<wgpu::BindGroupLayout> {

        // 1. 查缓存
        // HashMap 允许我们用 slice (&[T]) 去查询 Vec<T> 的 Key，非常高效
        if let Some(layout) = self.layout_cache.get(entries) {
            return layout.clone();
        }

        // 2. 创建 Layout
        // label 可以设为 None，或者根据 entries 生成一个简短的签名
        let layout = Arc::new(self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cached BindGroupLayout"),
            entries,
        }));

        // 3. 存入缓存
        // 这里需要 clone 一份 entries 作为 Key 存起来
        self.layout_cache.insert(entries.to_vec(), layout.clone());
        
        // 返回
        layout
    }

    pub fn create_bind_group(
        &self, 
        layout: &wgpu::BindGroupLayout, 
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
                BindingResource::Texture(tex_opt) => {
                    let gpu_tex = if let Some(tex_arc) = tex_opt {
                        let id = tex_arc.id;
                        self.textures.get(&id).unwrap_or(&self.dummy_texture)
                    } else { 
                        &self.dummy_texture 
                    };
                    wgpu::BindingResource::TextureView(&gpu_tex.view)
                },
                BindingResource::Sampler(tex_opt) => {
                     let gpu_tex = if let Some(tex_arc) = tex_opt {
                        let id = tex_arc.id;
                        self.textures.get(&id).unwrap_or(&self.dummy_texture)
                    } else { 
                        &self.dummy_texture 
                    };
                    wgpu::BindingResource::Sampler(&gpu_tex.sampler)
                },
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

    pub fn prepare_global(&mut self, env: &WorldEnvironment) -> &GPUWorld {
        
        // [A] 上传 Buffer 数据 (CPU -> GPU)
        // 注意：prepare_buffer 接受 &DataBuffer，我们需要通过 read() 获取
        {
            let frame_buf = env.frame_uniforms.read();
            self.prepare_buffer(&frame_buf);
        }
        {
            let light_buf = env.light_uniforms.read();
            self.prepare_buffer(&light_buf);
        }

        // [B] 检查或创建 BindGroup
        // 简化策略：只要 id 存在就不重建 (Global Layout 很少变)
        if !self.worlds.contains_key(&env.id) {
            
            // 1. 定义绑定
            let mut builder = ResourceBuilder::new();
            env.define_bindings(&mut builder); // 自动收集 frame 和 light buffer

            // 2. 获取 Layout (复用缓存机制)
            let layout = self.get_or_create_layout(&builder.layout_entries);

            // 3. 创建 BindGroup
            let (bind_group, bg_id) = self.create_bind_group(&layout, &builder.resources);

            // 4. 生成 WGSL (Group 0)
            let binding_wgsl = builder.generate_wgsl(0); 

            let gpu_world = GPUWorld {
                bind_group,
                bind_group_id: bg_id,
                layout,
                binding_wgsl,
                last_used_frame: self.frame_index,
            };

            self.worlds.insert(env.id, gpu_world);
        }

        let gpu_world = self.worlds.get_mut(&env.id).unwrap();
        gpu_world.last_used_frame = self.frame_index;
        gpu_world
    }

    // 获取 World 资源的辅助方法
    pub fn get_world(&self, id: Uuid) -> Option<&GPUWorld> {
        self.worlds.get(&id)
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