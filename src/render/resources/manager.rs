use std::collections::{HashMap};
use std::sync::{Arc};
use std::sync::atomic::{AtomicU64, Ordering};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::vec;

use slotmap::{SecondaryMap};
use core::ops::Range;

use crate::resources::geometry::Geometry;
use crate::resources::texture::Texture;
use crate::scene::environment::Environment;
use crate::resources::buffer::BufferRef;
use crate::resources::image::Image;
use crate::assets::{AssetServer, GeometryHandle, MaterialHandle, TextureHandle};

use crate::render::resources::binding::{BindingResource, Bindings};
use crate::render::resources::builder::ResourceBuilder;
use crate::render::pipeline::vertex::{GeneratedVertexLayout};
use crate::render::resources::buffer::GpuBuffer;
use crate::render::resources::texture::GpuTexture;
use crate::render::resources::image::GpuImage;

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

    pub layout: wgpu::BindGroupLayout,
    pub layout_id: u64,

    pub binding_wgsl: String,

    // 缓存字段
    pub last_version: u64,
    pub last_resource_hash: u64,

    pub last_used_frame: u64,
}

// 场景全局环境的 GPU 资源
pub struct GPUEnviroment {
    pub bind_group: wgpu::BindGroup,
    pub bind_group_id: u64,
    pub layout: wgpu::BindGroupLayout,
    pub binding_wgsl: String, 
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
    gpu_materials: SecondaryMap<MaterialHandle, GPUMaterial>,
    gpu_textures: SecondaryMap<TextureHandle, GpuTexture>,
    worlds: HashMap<u64, GPUEnviroment>, 
    // Image 使用 u64 ID (高性能)
    gpu_buffers: HashMap<u64, GpuBuffer>,
    gpu_images: HashMap<u64, GpuImage>,
    
    layout_cache: HashMap<Vec<wgpu::BindGroupLayoutEntry>, (wgpu::BindGroupLayout, u64)>,
    dummy_texture: GpuTexture, 
}

impl ResourceManager {

    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {

        let dummy_tex = Texture::new_2d("dummy", 1, 1, Some(vec![255, 255, 255, 255]), wgpu::TextureFormat::Rgba8Unorm); 
        let dummy_gpu_image = GpuImage::new(&device, &queue, &dummy_tex.image);
        let dummy_gpu_tex = GpuTexture::new(&device, &dummy_tex, &dummy_gpu_image);
        
        Self {
            device,
            queue,
            frame_index: 0,
            
            // 初始化 SecondaryMap
            gpu_textures: SecondaryMap::new(),
            gpu_geometries: SecondaryMap::new(),
            gpu_materials: SecondaryMap::new(),
            
            gpu_buffers: HashMap::new(),
            gpu_images: HashMap::new(),
            worlds: HashMap::new(),
            layout_cache: HashMap::new(),
            dummy_texture: dummy_gpu_tex,
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

        // 3. 检查重建 Geometry 绑定
        let needs_rebuild = if let Some(gpu_geo) = self.gpu_geometries.get(handle) {
            geometry.version() > gpu_geo.version || buffer_ids_changed
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
            version: geometry.version(),
            last_used_frame: self.frame_index,
        };

        self.gpu_geometries.insert(handle, gpu_geo);
    }

    pub fn get_geometry(&self, handle: GeometryHandle) -> Option<&GPUGeometry> {
        self.gpu_geometries.get(handle)
    }

    // ========================================================================
    // Material Logic
    // ========================================================================

    pub fn prepare_material(
        &mut self, 
        assets: &AssetServer, 
        handle: MaterialHandle
    ) -> Option<&GPUMaterial> {

        let material = assets.get_material(handle)?;

        // [L0] 粗粒度检查
        if let Some(gpu_mat) = self.gpu_materials.get_mut(handle) {
            if gpu_mat.last_version == material.version() {
                gpu_mat.last_used_frame = self.frame_index;
            } else {
                // 版本不同，需要更新
                self.update_gpu_material(assets, handle);
            }
        } else {
            // 不存在，需要创建
            self.update_gpu_material(assets, handle);
        }

        self.gpu_materials.get(handle)
    }

    fn update_gpu_material(&mut self, assets: &AssetServer, handle: MaterialHandle) {
        let material = assets.get_material(handle).unwrap(); // 安全，前面检查过
        
        // 1. Flush Uniforms
        material.flush_uniforms();

        // 2. Define Bindings
        let mut builder = ResourceBuilder::new();
        material.define_bindings(&mut builder);

        // 3. 处理资源 & Hash
        let mut resource_hasher = DefaultHasher::new();
        
        for res in &builder.resources {
            match res {
                BindingResource::Buffer { buffer, .. } => {
                    self.prepare_buffer(&buffer); // 确保 Buffer 已上传
                    // Hash ID
                    buffer.id.hash(&mut resource_hasher);
                },
                BindingResource::Texture(handle_opt) => {
                    if let Some(tex_handle) = handle_opt {
                        // 如果准备成功（说明资源存在且有效），则参与 Hash
                        if self.prepare_texture(assets, *tex_handle).is_some() {
                            tex_handle.hash(&mut resource_hasher);
                        } else {
                            // 资源准备失败（可能被删了），当做空处理
                            0.hash(&mut resource_hasher);
                        }
                    } else {
                        0.hash(&mut resource_hasher); // 空纹理
                    }
                },
                BindingResource::Sampler(handle_opt) => {
                    if let Some(tex_handle) = handle_opt {
                        if self.prepare_texture(assets, *tex_handle).is_some() {
                            tex_handle.hash(&mut resource_hasher);
                        } else {
                            0.hash(&mut resource_hasher);
                        }
                    } else {
                        0.hash(&mut resource_hasher);
                    }
                }
            }
        }
        let resource_hash = resource_hasher.finish();
        let (layout, layout_id) = self.get_or_create_layout(&builder.layout_entries);


        // 5. 检查是否需要重建 BindGroup
        // 仅当 Layout 指针变化 OR 资源 Hash 变化 OR 是新材质时才重建
        let mut can_reuse_bindgroup = false;

        if let Some(gpu_mat) = self.gpu_materials.get(handle) {
            let layout_unchanged = gpu_mat.layout_id == layout_id;
            let resources_unchanged = gpu_mat.last_resource_hash == resource_hash;
            if layout_unchanged && resources_unchanged {
                can_reuse_bindgroup = true;
            }
        }

        if can_reuse_bindgroup {
             let gpu_mat = self.gpu_materials.get_mut(handle).unwrap(); 
             gpu_mat.last_version = material.version();
             gpu_mat.last_used_frame = self.frame_index;
             return;
        }

        // 6. 创建 BindGroup
        let (bind_group, bg_id) = self.create_bind_group(&layout, &builder.resources);
        let binding_wgsl = builder.generate_wgsl(1);

        let gpu_mat = GPUMaterial {
            bind_group,
            bind_group_id: bg_id,
            layout,
            layout_id,
            binding_wgsl,
            last_version: material.version(),
            last_resource_hash: resource_hash,
            last_used_frame: self.frame_index,
        };
        
        self.gpu_materials.insert(handle, gpu_mat);
    }

    pub fn get_material(&self, handle: MaterialHandle) -> Option<&GPUMaterial> {
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
                BindingResource::Buffer { buffer, offset, size } => {
                    let cpu_id = buffer.id;
                    let gpu_buf = self.gpu_buffers.get(&cpu_id).expect("Buffer should be prepared before creating bindgroup");
                    // gpu_buf.buffer.as_entire_binding()
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &gpu_buf.buffer,
                        offset: *offset,
                        size: size.and_then(wgpu::BufferSize::new), // 处理 Option<u64> -> Option<NonZeroU64>
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
                    let gpu_tex = if let Some(handle) = handle_opt {
                        self.gpu_textures.get(*handle).unwrap_or(&self.dummy_texture)
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
    // Texture Logic
    // ========================================================================

    // 1. 准备 Image (Internal)
    // 这里传入 &Image 引用即可，类似 BufferRef
    fn prepare_image(&mut self, image: &Image) -> &GpuImage {
        let id = image.id();
        
        // 检查是否存在
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
        if let Some(gpu_tex) = self.gpu_textures.get_mut(handle) {
            // 依赖检查
            let config_changed = gpu_tex.version != texture_asset.version();
            let image_recreated = gpu_tex.image_generation_id != gpu_image.generation_id;
            let image_swapped = gpu_tex.image_id != image_id;

            if config_changed || image_recreated || image_swapped {
                *gpu_tex = GpuTexture::new(&self.device, texture_asset, gpu_image);
            }
            gpu_tex.last_used_frame = self.frame_index;
        } else {
            let gpu_tex = GpuTexture::new(&self.device, texture_asset, gpu_image);
            self.gpu_textures.insert(handle, gpu_tex);
        }

        self.gpu_textures.get(handle)
    }



    // pub fn add_or_update_texture(&mut self, handle: TextureHandle, texture: &Texture) {
    //     if let Some(gpu_tex) = self.gpu_textures.get_mut(handle) {
    //         gpu_tex.last_used_frame = self.frame_index;
    //         gpu_tex.update(&self.device, &self.queue, texture);
    //     } else {
    //         // 插入新资源
    //         let mut gpu_tex = GpuTexture::new(&self.device, &self.queue, texture);
    //         gpu_tex.last_used_frame = self.frame_index;
    //         self.gpu_textures.insert(handle, gpu_tex);
    //     }
    // }

    pub fn prepare_global(&mut self, env: &Environment) -> &GPUEnviroment {
        
        // [A] 上传 Buffer 数据 (CPU -> GPU)
        // 注意：prepare_buffer 接受 &DataBuffer，我们需要通过 read() 获取
        {
            let frame_buf = &env.frame_uniforms;
            self.prepare_buffer(&frame_buf);
        }
        {
            let light_buf = &env.light_uniforms;
            self.prepare_buffer(&light_buf);
        }

        // [B] 检查或创建 BindGroup
        // 简化策略：只要 id 存在就不重建 (Global Layout 很少变)
        if !self.worlds.contains_key(&env.id) {
            
            // 1. 定义绑定
            let mut builder = ResourceBuilder::new();
            env.define_bindings(&mut builder); // 自动收集 frame 和 light buffer

            // 2. 获取 Layout (复用缓存机制)
            let (layout, _layout_id) = self.get_or_create_layout(&builder.layout_entries);

            // 3. 创建 BindGroup
            let (bind_group, bg_id) = self.create_bind_group(&layout, &builder.resources);

            // 4. 生成 WGSL (Group 0)
            let binding_wgsl = builder.generate_wgsl(0); 

            let gpu_world = GPUEnviroment {
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
    pub fn get_world(&self, id: u64) -> Option<&GPUEnviroment> {
        self.worlds.get(&id)
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

        self.gpu_buffers.retain(|_, v| v.last_used_frame >= cutoff);
        self.gpu_images.retain(|_, v| v.last_used_frame >= cutoff);
        self.worlds.retain(|_, v| v.last_used_frame >= cutoff);
        // Layout 缓存也可以清理，但通常 Layout 占内存极小，保留无妨
    }
}