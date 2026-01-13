use std::collections::{HashMap};
use std::sync::{Arc};
use std::sync::atomic::{AtomicU64, Ordering};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::vec;

use slotmap::{SecondaryMap};
use core::ops::Range;

use crate::resources::geometry::Geometry;
use crate::resources::texture::{Texture, TextureSampler};
use crate::scene::environment::Environment;
use crate::resources::buffer::BufferRef;
use crate::resources::uniform_slot::UniformSlot;
use crate::resources::image::Image;
use crate::assets::{AssetServer, GeometryHandle, MaterialHandle, TextureHandle};

use crate::render::resources::binding::{BindingResource, Bindings};
use crate::render::resources::builder::ResourceBuilder;
use crate::render::pipeline::vertex::{GeneratedVertexLayout};
use crate::render::resources::buffer::GpuBuffer;
use crate::render::resources::texture::GpuTexture;
use crate::render::resources::image::GpuImage;
use crate::render::RenderState;

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

/// 材质的 GPU 资源（三级版本控制）
pub struct GpuMaterial {    
    pub bind_group: wgpu::BindGroup,
    pub bind_group_id: u64,

    pub layout: wgpu::BindGroupLayout,
    pub layout_id: u64,

    pub binding_wgsl: String,
    
    // GPU Buffer 引用（用于快速更新 uniform 数据）
    pub uniform_buffers: Vec<u64>, // Buffer IDs

    // 三级版本控制
    pub last_data_version: u64,      // Uniform 数据版本
    pub last_binding_version: u64,   // 绑定资源版本（纹理等）
    pub last_layout_version: u64,    // 管线布局版本（shader 特性）

    pub last_used_frame: u64,
}

/// 环境/全局资源的 GPU 缓存
pub struct GpuEnvironment {
    pub bind_group: wgpu::BindGroup,
    pub bind_group_id: u64,
    pub layout: wgpu::BindGroupLayout,
    pub layout_id: u64,
    pub binding_wgsl: String,
    pub last_version: u64,
    pub last_resource_hash: u64,
    pub last_used_frame: u64,
}

// ============================================================================
// Resource Manager
// ============================================================================

pub struct ResourceManager {
    device: wgpu::Device,
    queue: wgpu::Queue,
    frame_index: u64,

    gpu_geometries: SecondaryMap<GeometryHandle, GPUGeometry>,
    gpu_materials: SecondaryMap<MaterialHandle, GpuMaterial>,
    gpu_textures: SecondaryMap<TextureHandle, GpuTexture>,
    // 采样器映射表：TextureHandle -> Sampler
    gpu_samplers: SecondaryMap<TextureHandle, wgpu::Sampler>,

    worlds: HashMap<u64, GpuEnvironment>, 
    // Image 使用 u64 ID (高性能)
    gpu_buffers: HashMap<u64, GpuBuffer>,
    gpu_images: HashMap<u64, GpuImage>,

    sampler_cache: HashMap<TextureSampler, wgpu::Sampler>,
    layout_cache: HashMap<Vec<wgpu::BindGroupLayoutEntry>, (wgpu::BindGroupLayout, u64)>,

    dummy_texture: GpuTexture, 
    //默认采样器
    dummy_sampler: wgpu::Sampler,
}

impl ResourceManager {

    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {

        let dummy_tex = Texture::new_2d("dummy", 1, 1, Some(vec![255, 255, 255, 255]), wgpu::TextureFormat::Rgba8Unorm); 
        let dummy_gpu_image = GpuImage::new(&device, &queue, &dummy_tex.image);
        let dummy_gpu_tex = GpuTexture::new(&dummy_tex, &dummy_gpu_image);

        let dummy_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Dummy Sampler"),
            ..Default::default()
        });
        
        Self {
            device,
            queue,
            frame_index: 0,
            
            // 初始化 SecondaryMap
            gpu_geometries: SecondaryMap::new(),
            gpu_materials: SecondaryMap::new(),
            gpu_textures: SecondaryMap::new(),
            gpu_samplers: SecondaryMap::new(),
            
            worlds: HashMap::new(),
            gpu_buffers: HashMap::new(),
            gpu_images: HashMap::new(),

            layout_cache: HashMap::new(),
            sampler_cache: HashMap::new(),

            dummy_texture: dummy_gpu_tex,
            dummy_sampler,
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
    pub fn prepare_buffer(&mut self, buffer_ref: &BufferRef) -> u64 {
        let id = buffer_ref.id;

        // 1. [Hot Path] 无锁检查版本！
        // 如果版本没变，直接返回。这里没有任何锁竞争。
        if let Some(gpu_buf) = self.gpu_buffers.get_mut(&id) {
            if buffer_ref.version() <= gpu_buf.version {
                gpu_buf.last_used_frame = self.frame_index;
                return gpu_buf.id;
            }
        }

        // 2. [Cold Path] 只有确实需要更新时，才获取锁并拷贝数据
        let gpu_buf = self.gpu_buffers.entry(id).or_insert_with(|| {
            // 创建逻辑... 需要先 read_data() 获取初始数据
            let data = buffer_ref.read_data();

            let mut buf = GpuBuffer::new(
                &self.device, 
                &data, 
                buffer_ref.usage, 
                Some(&buffer_ref.label)
            );
            // 对于 Uniform Buffer，默认开启 Shadow Copy 以优化频繁的小数据更新
            if buffer_ref.usage.contains(wgpu::BufferUsages::UNIFORM) {
                buf.enable_shadow_copy();
            }
            buf
        });

        // 更新逻辑
        {
            let data = buffer_ref.read_data(); // 这里才发生短暂的锁
            if buffer_ref.usage.contains(wgpu::BufferUsages::UNIFORM) {
                gpu_buf.update_with_data(&self.device, &self.queue, &data);
            } else {
                gpu_buf.update_with_version(&self.device, &self.queue, &data, buffer_ref.version());
            }
        }
   
        gpu_buf.last_used_frame = self.frame_index;
        gpu_buf.id
    }

    /// 【新】准备 UniformSlot（通用版本，接收数据引用）
    /// 无锁直接访问数据，性能更优
    pub fn prepare_uniform_slot_data(
        &mut self,
        slot_id: u64,
        data: &[u8],
        label: &str,
    ) -> u64 {
        // 1. 检查版本（暂时简化，总是更新）
        // TODO: 可以添加 version 参数来优化
        
        // 2. 创建或更新
        let gpu_buf = self.gpu_buffers.entry(slot_id).or_insert_with(|| {
            let mut buf = GpuBuffer::new(
                &self.device,
                data,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some(label),
            );
            buf.enable_shadow_copy(); // Uniform自动启用Shadow Copy
            buf
        });

        // 3. 更新数据（Diff模式）
        gpu_buf.update_with_data(&self.device, &self.queue, data);
        gpu_buf.last_used_frame = self.frame_index;
        gpu_buf.id
    }

    /// 【保留】准备 UniformSlot（从 UniformSlot<T> 直接准备）
    pub fn prepare_uniform_slot<T: bytemuck::Pod>(
        &mut self, 
        slot: &UniformSlot<T>
    ) -> u64 {
        let id = slot.id();

        // 1. 检查版本
        if let Some(gpu_buf) = self.gpu_buffers.get_mut(&id) {
            if slot.version() <= gpu_buf.version {
                gpu_buf.last_used_frame = self.frame_index;
                return gpu_buf.id;
            }
        }

        // 2. 创建或更新
        let gpu_buf = self.gpu_buffers.entry(id).or_insert_with(|| {
            let mut buf = GpuBuffer::new(
                &self.device,
                slot.as_bytes(),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some(slot.label()),
            );
            buf.enable_shadow_copy(); // Uniform自动启用Shadow Copy
            buf
        });

        // 3. 更新数据（Diff模式）
        gpu_buf.update_with_data(&self.device, &self.queue, slot.as_bytes());
        gpu_buf.version = slot.version();
        gpu_buf.last_used_frame = self.frame_index;
        gpu_buf.id
    }

    // ========================================================================
    // Geometry Logic
    // ========================================================================

    pub fn prepare_geometry(
        &mut self, 
        assets: &AssetServer, 
        handle: GeometryHandle
    ) -> Option<&GPUGeometry> {


        let geometry = assets.get_geometry(handle)?;

        let mut buffer_ids_changed = false;

        // 1. 统一处理 Vertex Buffers
        for attr in geometry.attributes.values() {
            let new_id = self.prepare_buffer(&attr.buffer);

            if let Some(gpu_geo) = self.gpu_geometries.get(handle) {
                if !gpu_geo.vertex_buffer_ids.contains(&new_id) {
                     // 只要有一个 Buffer ID 不在旧列表里，认为变了。
                     buffer_ids_changed = true;
                }
            }
        }

        // 2. 检查 Index Buffer
        if let Some(indices) = &geometry.index_attribute {
            let new_id = self.prepare_buffer(&indices.buffer);
             if let Some(gpu_geo) = self.gpu_geometries.get(handle) {
                 if let Some((_, _, _, old_id)) = gpu_geo.index_buffer {
                     if old_id != new_id { buffer_ids_changed = true; }
                 }
             }
        }

        let needs_rebuild = if let Some(gpu_geo) = self.gpu_geometries.get(handle) {
            geometry.structure_version() > gpu_geo.version || buffer_ids_changed
        } else {
            true
        };

        if needs_rebuild {
            self.create_gpu_geometry(geometry, handle);
        }

        if let Some(gpu_geo) = self.gpu_geometries.get_mut(handle) {
            gpu_geo.last_used_frame = self.frame_index;
            return Some(gpu_geo);
        }
        None
    }

    fn create_gpu_geometry(&mut self, geometry: &Geometry, handle: GeometryHandle) {
        let layout_info = Arc::new(crate::render::pipeline::vertex::generate_vertex_layout(geometry));

        let mut vertex_buffers = Vec::new();
        let mut vertex_buffer_ids = Vec::new();
        
        for layout_desc in &layout_info.buffers {
            let gpu_buf = self.gpu_buffers.get(&layout_desc.buffer.id).expect("Vertex buffer should be prepared");
            vertex_buffers.push(gpu_buf.buffer.clone()); 
            vertex_buffer_ids.push(gpu_buf.id);
        }

        let index_buffer = if let Some(indices) = &geometry.index_attribute {
            let gpu_buf = self.gpu_buffers.get(&indices.buffer.id).expect("Index buffer should be prepared");
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
            version: geometry.structure_version(),
            last_used_frame: self.frame_index,
        };

        self.gpu_geometries.insert(handle, gpu_geo);
    }

    pub fn get_geometry(&self, handle: GeometryHandle) -> Option<&GPUGeometry> {
        self.gpu_geometries.get(handle)
    }


    fn compute_resource_hash(&self, resources: &[BindingResource]) -> u64 {
        let mut hasher = DefaultHasher::new();
        for res in resources {
            match res {
                BindingResource::UniformSlot { slot_id, .. } => {
                    // 使用 slot_id 作为稳定标识
                    slot_id.hash(&mut hasher);
                    // 可选：包含版本号来检测数据变化
                    if let Some(gpu_buf) = self.gpu_buffers.get(slot_id) {
                        gpu_buf.version.hash(&mut hasher);
                    }
                },
                BindingResource::Buffer { buffer, .. } => {
                    let gpu_id = buffer.id;
                    gpu_id.hash(&mut hasher);
                    if let Some(gpu_buf) = self.gpu_buffers.get(&gpu_id) {
                        gpu_buf.version.hash(&mut hasher);
                    }
                },
                BindingResource::Texture(handle_opt) => {
                    if let Some(h) = handle_opt {
                        if let Some(t) = self.gpu_textures.get(*h) {
                            t.version.hash(&mut hasher);
                            t.image_id.hash(&mut hasher);
                            t.image_generation_id.hash(&mut hasher);
                        } else { 0.hash(&mut hasher); }
                    } else { 0.hash(&mut hasher); }
                },
                BindingResource::Sampler(handle_opt) => {
                    if let Some(h) = handle_opt {
                        if let Some(t) = self.gpu_textures.get(*h) {
                            t.version.hash(&mut hasher);
                        } else { 0.hash(&mut hasher); }
                    } else { 0.hash(&mut hasher); }
                },
                BindingResource::_Phantom(_) => {
                    // 忽略 PhantomData
                }
            }
        }
        hasher.finish()
    }

    // ========================================================================
    // Material Logic
    // ========================================================================
    pub fn prepare_material(
        &mut self, 
        assets: &AssetServer, 
        handle: MaterialHandle
    ) -> Option<&GpuMaterial> {
        let material = assets.get_material(handle)?;

        // === 三级版本控制检查 ===
        let uniform_ver = material.data.uniform_version();
        let binding_ver = material.data.binding_version();
        let layout_ver = material.data.layout_version();

        // 情况 A: 全新材质，走完整构建流程
        if !self.gpu_materials.contains_key(handle) {
            return Some(self.build_full_material(assets, handle, &material));
        }

        let gpu_mat = self.gpu_materials.get(handle).unwrap();

        // 情况 B: 检查更新（热路径）
        
        // 1. 检查最严重的：Layout/Pipeline 变化
        if layout_ver != gpu_mat.last_layout_version {
            // 重建 Pipeline + BindGroup + 上传数据
            return Some(self.build_full_material(assets, handle, &material));
        }

        // 2. 检查次严重的：Binding 变化
        if binding_ver != gpu_mat.last_binding_version {
            // 只需重建 BindGroup（Pipeline 复用）
            return Some(self.rebuild_bind_group(handle, &material));
        }

        // 3. 检查最轻微的：数据变化
        if uniform_ver != gpu_mat.last_data_version {
            // 极快：直接写入 GPU Buffer
            self.update_material_uniforms(handle, &material);
        }

        // 更新帧计数
        let gpu_mat = self.gpu_materials.get_mut(handle).unwrap();
        gpu_mat.last_used_frame = self.frame_index;
        
        Some(gpu_mat)
    }
    
    /// 完整构建材质（冷路径）
    fn build_full_material(
        &mut self,
        assets: &AssetServer,
        handle: MaterialHandle,
        material: &crate::resources::material::Material,
    ) -> &GpuMaterial {
        let mut builder = ResourceBuilder::new();
        material.define_bindings(&mut builder);

        // 准备所有资源
        let mut uniform_buffers = Vec::new();
        
        for resource in &builder.resources {
            match resource {
                BindingResource::UniformSlot { slot_id, data, label } => {
                    let buf_id = self.prepare_uniform_slot_data(*slot_id, data, label);
                    uniform_buffers.push(buf_id);
                },
                BindingResource::Buffer { buffer, .. } => {
                    self.prepare_buffer(buffer);
                },
                BindingResource::Texture(handle_opt) => {
                    if let Some(handle) = handle_opt {
                        self.prepare_texture(assets, *handle);
                    }
                    // Texture 准备需要 AssetServer，这里暂时跳过
                    // 实际实现应该通过参数传递 AssetServer
                },
                BindingResource::Sampler(_) | BindingResource::_Phantom(_) => {},
            }
        }

        let (layout, layout_id) = self.get_or_create_layout(&builder.layout_entries);
        let (bind_group, bg_id) = self.create_bind_group(&layout, &builder.resources);
        let binding_wgsl = builder.generate_wgsl(1);

        let gpu_mat = GpuMaterial {
            bind_group,
            bind_group_id: bg_id,
            layout,
            layout_id,
            binding_wgsl,
            uniform_buffers,
            last_data_version: material.data.uniform_version(),
            last_binding_version: material.data.binding_version(),
            last_layout_version: material.data.layout_version(),
            last_used_frame: self.frame_index,
        };
        
        self.gpu_materials.insert(handle, gpu_mat);
        self.gpu_materials.get(handle).unwrap()
    }
    
    /// 重建 BindGroup（中路径）
    fn rebuild_bind_group(
        &mut self,
        handle: MaterialHandle,
        material: &crate::resources::material::Material,
    ) -> &GpuMaterial {
        let mut builder = ResourceBuilder::new();
        material.define_bindings(&mut builder);

        // 准备资源（纹理可能变了）- 这里需要 AssetServer，暂时跳过
        
        // 先获取 layout 的克隆（避免借用冲突）
        let layout = {
            let gpu_mat = self.gpu_materials.get(handle).unwrap();
            gpu_mat.layout.clone()
        };
        
        // 创建新的 bind_group
        let (bind_group, bg_id) = self.create_bind_group(&layout, &builder.resources);
        
        // 更新 gpu_mat
        {
            let gpu_mat = self.gpu_materials.get_mut(handle).unwrap();
            gpu_mat.bind_group = bind_group;
            gpu_mat.bind_group_id = bg_id;
            gpu_mat.last_binding_version = material.data.binding_version();
            gpu_mat.last_used_frame = self.frame_index;
        }
        
        self.gpu_materials.get(handle).unwrap()
    }
    
    /// 更新 Uniform 数据（热路径）
    fn update_material_uniforms(
        &mut self,
        handle: MaterialHandle,
        material: &crate::resources::material::Material,
    ) {
        use crate::resources::material::MaterialData;
        
        // 直接写入 GPU Buffer（最快路径）
        match &material.data {
            MaterialData::Basic(m) => {
                let data = bytemuck::bytes_of(m.uniforms());
                if let Some(gpu_mat) = self.gpu_materials.get(handle) {
                    if let Some(&buf_id) = gpu_mat.uniform_buffers.first() {
                        if let Some(gpu_buf) = self.gpu_buffers.get(&buf_id) {
                            self.queue.write_buffer(&gpu_buf.buffer, 0, data);
                        }
                    }
                }
            },
            MaterialData::Phong(m) => {
                let data = bytemuck::bytes_of(m.uniforms());
                if let Some(gpu_mat) = self.gpu_materials.get(handle) {
                    if let Some(&buf_id) = gpu_mat.uniform_buffers.first() {
                        if let Some(gpu_buf) = self.gpu_buffers.get(&buf_id) {
                            self.queue.write_buffer(&gpu_buf.buffer, 0, data);
                        }
                    }
                }
            },
            MaterialData::Standard(m) => {
                let data = bytemuck::bytes_of(m.uniforms());
                if let Some(gpu_mat) = self.gpu_materials.get(handle) {
                    if let Some(&buf_id) = gpu_mat.uniform_buffers.first() {
                        if let Some(gpu_buf) = self.gpu_buffers.get(&buf_id) {
                            self.queue.write_buffer(&gpu_buf.buffer, 0, data);
                        }
                    }
                }
            },
        }
        
        // 更新版本号
        if let Some(gpu_mat) = self.gpu_materials.get_mut(handle) {
            gpu_mat.last_data_version = material.data.uniform_version();
        }
    }

    pub fn get_material(&self, handle: MaterialHandle) -> Option<&GpuMaterial> {
        self.gpu_materials.get(handle)
    }

    // --- 内部辅助 ---

    pub fn get_or_create_layout(&mut self, entries: &[wgpu::BindGroupLayoutEntry]) -> (wgpu::BindGroupLayout, u64) {

        // 1. 查缓存
        // HashMap 允许我们用 slice (&[T]) 去查询 Vec<T> 的 Key，非常高效
        if let Some(layout) = self.layout_cache.get(entries) {
            return layout.clone();
        }

        // 2. 创建 Layout
        // label 可以设为 None，或者根据 entries 生成一个简短的签名
        let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cached BindGroupLayout"),
            entries,
        });

        // [新增] 生成唯一 ID
        let id = generate_resource_id();

        // 3. 存入缓存
        // 这里需要 clone 一份 entries 作为 Key 存起来
        self.layout_cache.insert(entries.to_vec(), (layout.clone(), id));
        
        // 返回
        (layout, id)
    }

    pub fn create_bind_group(
        &self, 
        layout: &wgpu::BindGroupLayout, 
        resources: &[BindingResource],
    ) -> (wgpu::BindGroup, u64) {
        
        let mut entries = Vec::new();

        for (i, resource_data) in resources.iter().enumerate() {

            let binding_resource = match resource_data {
                BindingResource::UniformSlot { slot_id, .. } => {
                    let gpu_buf = self.gpu_buffers.get(slot_id)
                        .expect("UniformSlot should be prepared before creating bindgroup");
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &gpu_buf.buffer,
                        offset: 0,
                        size: None,
                    })
                },
                BindingResource::Buffer { buffer, offset, size } => {
                    let cpu_id = buffer.id;
                    let gpu_buf = self.gpu_buffers.get(&cpu_id).expect("Buffer should be prepared before creating bindgroup");
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &gpu_buf.buffer,
                        offset: *offset,
                        size: size.and_then(wgpu::BufferSize::new),
                    })
                },
                BindingResource::Texture(handle_opt) => {
                    let gpu_tex = if let Some(handle) = handle_opt {
                        self.gpu_textures.get(*handle).unwrap_or(&self.dummy_texture)
                    } else { 
                        &self.dummy_texture 
                    };
                    wgpu::BindingResource::TextureView(&gpu_tex.view)

                },
                BindingResource::Sampler(handle_opt) => {
                    let sampler = if let Some(handle) = handle_opt {
                        self.gpu_samplers.get(*handle).unwrap_or(&self.dummy_sampler)
                    } else { 
                        &self.dummy_sampler 
                    };
                    wgpu::BindingResource::Sampler(sampler)
                },
                BindingResource::_Phantom(_) => {
                    unreachable!("_Phantom should never be used")
                },
            };

            entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
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
    // Texture Logic
    // ========================================================================

    // 1. 准备 Image (Internal)
    fn prepare_image(&mut self, image: &Image) -> &GpuImage {
        let id = image.id();
        if let Some(gpu_img) = self.gpu_images.get_mut(&id) {
            // 更新 (内部会检查 version 和 generation)
            gpu_img.update(&self.device, &self.queue, image);
            gpu_img.last_used_frame = self.frame_index;
        } else {
            // 创建
            let gpu_img = GpuImage::new(&self.device, &self.queue, image);
            self.gpu_images.insert(id, gpu_img);
        }
        self.gpu_images.get(&id).unwrap()
    }

    // 2. 准备 Texture (Public)
    pub fn prepare_texture(
        &mut self, 
        assets: &AssetServer, 
        handle: TextureHandle
    ) -> Option<&GpuTexture> {
        let texture_asset = assets.get_texture(handle)?;
        
        // [A] 先准备底层 Image
        self.prepare_image(&texture_asset.image);
        
        let image_id = texture_asset.image.id();
        let gpu_image = self.gpu_images.get(&image_id).unwrap();

        // [B] 准备/更新 GpuTexture
        let mut needs_update = false;
        if let Some(gpu_tex) = self.gpu_textures.get_mut(handle) {
            // 依赖检查
            let config_changed = gpu_tex.version != texture_asset.version();
            let image_recreated = gpu_tex.image_generation_id != gpu_image.generation_id;
            let image_swapped = gpu_tex.image_id != image_id;

            if config_changed || image_recreated || image_swapped {
                needs_update = true;
            }
            gpu_tex.last_used_frame = self.frame_index;
        } else {
            needs_update = true;
            // self.gpu_textures.insert(handle, gpu_tex);
        }

        if needs_update {
            // 1. GpuTexture (View Only)
            let gpu_tex = GpuTexture::new(texture_asset, gpu_image);
            self.gpu_textures.insert(handle, gpu_tex);

            // 2. Sampler (Cached & Deduped)
            // 使用 TextureSampler 配置去查找或创建唯一 Sampler
            let sampler = self.get_or_create_sampler(&texture_asset.sampler, &texture_asset.name);
            self.gpu_samplers.insert(handle, sampler);
        }

        if let Some(gpu_tex) = self.gpu_textures.get_mut(handle) {
            gpu_tex.last_used_frame = self.frame_index;
        }

        self.gpu_textures.get(handle)
    }


    fn get_or_create_sampler(&mut self, config: &TextureSampler, label: &str) -> wgpu::Sampler {
        if let Some(sampler) = self.sampler_cache.get(config) {
            return sampler.clone();
        }

        // Create new sampler
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&format!("{}_sampler", label)),
            address_mode_u: config.address_mode_u,
            address_mode_v: config.address_mode_v,
            address_mode_w: config.address_mode_w,
            mag_filter: config.mag_filter,
            min_filter: config.min_filter,
            mipmap_filter: config.mipmap_filter,
            compare: config.compare,
            anisotropy_clamp: config.anisotropy_clamp,
            ..Default::default()
        });

        self.sampler_cache.insert(*config, sampler.clone());
        sampler
    }

    pub fn prepare_global(&mut self, _assets: &AssetServer, env: &Environment, render_state: &RenderState) -> &GpuEnvironment {

        let mut builder = ResourceBuilder::new();
        render_state.define_bindings(&mut builder);
        env.define_bindings(&mut builder);

        // ✅ 通用地准备所有资源（从 BindingResource 中获取信息）
        for resource in &builder.resources {
            match resource {
                BindingResource::UniformSlot { slot_id, data, label } => {
                    self.prepare_uniform_slot_data(*slot_id, data, label);
                },
                BindingResource::Buffer { buffer, .. } => {
                    self.prepare_buffer(buffer);
                },
                _ => {} // Texture/Sampler 不需要在这里处理
            }
        }

        // ✅ 计算资源hash（资源已准备好）
        let resource_hash = self.compute_resource_hash(&builder.resources);

        let (layout, layout_id) = self.get_or_create_layout(&builder.layout_entries);
        let world_id = Self::compose_env_render_state_id(render_state.id, env.id);
        let is_valid = if let Some(gpu_env) = self.worlds.get(&world_id) {
            gpu_env.layout_id == layout_id && gpu_env.last_resource_hash == resource_hash
        } else {
            false
        };

        if is_valid {
            if let Some(gpu_env) = self.worlds.get_mut(&world_id) {
                gpu_env.last_used_frame = self.frame_index;
                gpu_env.last_version = world_id;
            }
            return self.worlds.get(&world_id).unwrap();
        }

        // 6. 重建 BindGroup (Create)
        log::debug!("Rebuilding Global BindGroup. Layout: {}, ResHash: {}", layout_id, resource_hash);

        let (bind_group, bg_id) = self.create_bind_group(&layout, &builder.resources);
        let binding_wgsl = builder.generate_wgsl(0); 

        let gpu_world = GpuEnvironment {
            bind_group,
            bind_group_id: bg_id,
            layout,
            layout_id,
            binding_wgsl,
            last_version: world_id,
            last_resource_hash: resource_hash,
            last_used_frame: self.frame_index,
        };
        self.worlds.insert(world_id, gpu_world);
        self.worlds.get(&world_id).unwrap()

    }

    fn compose_env_render_state_id(render_state_id: u32, env_id: u32) -> u64 {
        ((render_state_id as u64) << 32) | (env_id as u64)
    }

    // 获取 World 资源的辅助方法
    pub fn get_world(&self, render_state_id: u32, env_id: u32) -> Option<&GpuEnvironment> {
        let world_id = Self::compose_env_render_state_id(render_state_id, env_id);
        self.worlds.get(&world_id)
    }
    // ========================================================================
    // Garbage Collection
    // ========================================================================

    pub fn prune(&mut self, ttl_frames: u64) {
        if self.frame_index < ttl_frames { return; }
        let cutoff = self.frame_index - ttl_frames;

        self.gpu_geometries.retain(|_, v| v.last_used_frame >= cutoff);
        self.gpu_materials.retain(|_, v| v.last_used_frame >= cutoff);
        self.gpu_textures.retain(|_, v| v.last_used_frame >= cutoff);
    
        self.gpu_samplers.retain(|k, _| self.gpu_textures.contains_key(k));

        self.gpu_buffers.retain(|_, v| v.last_used_frame >= cutoff);
        self.gpu_images.retain(|_, v| v.last_used_frame >= cutoff);
        self.worlds.retain(|_, v| v.last_used_frame >= cutoff);
        // Layout 缓存也可以清理，但通常 Layout 占内存极小，保留无妨
    }
}