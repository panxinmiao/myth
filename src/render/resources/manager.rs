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

pub struct GpuGeometry {
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

    gpu_geometries: SecondaryMap<GeometryHandle, GpuGeometry>,
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

    mipmap_generator: crate::render::resources::mipmap_generator::MipmapGenerator,
}

impl ResourceManager {

    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {

        let dummy_tex = Texture::new_2d(Some("dummy"), 1, 1, Some(vec![255, 255, 255, 255]), wgpu::TextureFormat::Rgba8Unorm); 
        let dummy_gpu_image = GpuImage::new(&device, &queue, &dummy_tex.image, 1, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        let dummy_gpu_tex = GpuTexture::new(&dummy_tex, &dummy_gpu_image);

        let dummy_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Dummy Sampler"),
            ..Default::default()
        });

        let mipmap_generator = crate::render::resources::mipmap_generator::MipmapGenerator::new(&device);
        
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

            mipmap_generator,
        }
    }

    pub fn next_frame(&mut self) {
        self.frame_index += 1;
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    pub fn write_buffer(&mut self, buffer_ref: &BufferRef, data: &[u8]) -> u64 {
        let id = buffer_ref.id();
        
        if let Some(gpu_buf) = self.gpu_buffers.get_mut(&id) {
            // 直接写，无锁！
            self.queue.write_buffer(&gpu_buf.buffer, 0, data);
            gpu_buf.last_used_frame = self.frame_index;
            gpu_buf.id
        } else {
            let mut gpu_buf = GpuBuffer::new(
                &self.device,
                data, // 初始数据
                buffer_ref.usage(),
                buffer_ref.label()
            );
            gpu_buf.last_used_frame = self.frame_index;
            let buf_id = gpu_buf.id;
            self.gpu_buffers.insert(id, gpu_buf);
            buf_id
        }
    }

    /// [Modified] 准备静态 Buffer (Pull 模式变体)
    /// 从 Attribute 中提取数据并上传
    pub fn prepare_attribute_buffer(&mut self, attr: &crate::resources::geometry::Attribute) -> u64 {
        let id = attr.buffer.id();

        if let Some(gpu_buf) = self.gpu_buffers.get_mut(&id) {
            // 检查版本，如果数据更新了则重新上传
            if attr.version > gpu_buf.last_uploaded_version
                && let Some(data) = &attr.data {
                    let bytes: &[u8] = data.as_ref();
                    self.queue.write_buffer(&gpu_buf.buffer, 0, bytes);
                    gpu_buf.last_uploaded_version = attr.version;
                }
            gpu_buf.last_used_frame = self.frame_index;
            return gpu_buf.id;
        }

        if let Some(data) = &attr.data {
             let bytes: &[u8] = data.as_ref();
             let mut gpu_buf = GpuBuffer::new(
                &self.device,
                bytes,
                attr.buffer.usage(),
                attr.buffer.label()
            );
            gpu_buf.last_uploaded_version = attr.version;
            gpu_buf.last_used_frame = self.frame_index;
            let buf_id = gpu_buf.id;
            self.gpu_buffers.insert(id, gpu_buf);
            buf_id
        } else {
            log::error!("Geometry attribute buffer {:?} missing CPU data for upload!", attr.buffer.label());
            // 策略 A: 如果之前有缓存，复用旧的（如果有的话）
            if let Some(gpu_buf) = self.gpu_buffers.get_mut(&id) {
                return gpu_buf.id;
            }
            // 策略 B: 创建一个最小的空 Buffer 作为占位符
            let dummy_data = [0u8; 1];
            let gpu_buf = GpuBuffer::new(
                &self.device,
                &dummy_data,
                attr.buffer.usage(),
                Some("Dummy Fallback Buffer")
            );
            let buf_id = gpu_buf.id;
            self.gpu_buffers.insert(id, gpu_buf);
            buf_id
        }
    }

    pub fn prepare_uniform_slot_data(
        &mut self,
        slot_id: u64,
        data: &[u8],
        label: &str,
    ) -> u64 {
        let gpu_buf = self.gpu_buffers.entry(slot_id).or_insert_with(|| {
            let mut buf = GpuBuffer::new(
                &self.device,
                data,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some(label),
            );
            buf.enable_shadow_copy();
            buf
        });

        gpu_buf.update_with_data(&self.device, &self.queue, data);
        gpu_buf.last_used_frame = self.frame_index;
        gpu_buf.id
    }

    pub fn prepare_geometry(
        &mut self, 
        assets: &AssetServer, 
        handle: GeometryHandle
    ) -> Option<&GpuGeometry> {
        let geometry = assets.get_geometry(handle)?;

        let mut buffer_ids_changed = false;

        // 1. 统一处理 Vertex Buffers (使用新的 prepare_attribute_buffer)
        for attr in geometry.attributes.values() {
            let new_id = self.prepare_attribute_buffer(attr);

            if let Some(gpu_geo) = self.gpu_geometries.get(handle)
                && !gpu_geo.vertex_buffer_ids.contains(&new_id) {
                     // 只要有一个 Buffer ID 不在旧列表里，认为变了。
                     buffer_ids_changed = true;
                }
        }

        // 2. 检查 Index Buffer
        if let Some(indices) = &geometry.index_attribute {
            let new_id = self.prepare_attribute_buffer(indices);
             if let Some(gpu_geo) = self.gpu_geometries.get(handle)
                 && let Some((_, _, _, old_id)) = gpu_geo.index_buffer
                     && old_id != new_id { buffer_ids_changed = true; }
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
            let gpu_buf = self.gpu_buffers.get(&layout_desc.buffer.id()).expect("Vertex buffer should be prepared");
            vertex_buffers.push(gpu_buf.buffer.clone()); 
            vertex_buffer_ids.push(gpu_buf.id);
        }

        let index_buffer = if let Some(indices) = &geometry.index_attribute {
            let gpu_buf = self.gpu_buffers.get(&indices.buffer.id()).expect("Index buffer should be prepared");
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

        let gpu_geo = GpuGeometry {
            layout_info,
            vertex_buffers,
            vertex_buffer_ids,
            index_buffer,
            draw_range,
            instance_range: 0..1,
            version: geometry.structure_version(),
            last_used_frame: self.frame_index,
        };

        self.gpu_geometries.insert(handle, gpu_geo);
    }

    pub fn get_geometry(&self, handle: GeometryHandle) -> Option<&GpuGeometry> {
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
                    let gpu_id = buffer.id();
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

// Material Logic
    pub fn prepare_material(
        &mut self, 
        assets: &AssetServer, 
        handle: MaterialHandle
    ) {
        let Some(material) = assets.get_material(handle) else{
            log::warn!("Material {:?} not found in AssetServer.", handle);
            return;
        };

        let uniform_ver = material.data.uniform_version();
        let binding_ver = material.data.binding_version();
        let layout_ver = material.data.layout_version();

        if !self.gpu_materials.contains_key(handle) {
            self.build_full_material(assets, handle, material);
        }

        let gpu_mat = self.gpu_materials.get(handle).expect("gpu material should exist.");
        
        if layout_ver != gpu_mat.last_layout_version {
            // 重建 Pipeline + BindGroup + 上传数据
            self.build_full_material(assets, handle, material);
            return;
        }

        if binding_ver != gpu_mat.last_binding_version {
            // 只需重建 BindGroup（Pipeline 复用）
            self.rebuild_bind_group(assets, handle, material);
            return;
        }

        if uniform_ver != gpu_mat.last_data_version {
            // 极快：直接写入 GPU Buffer
            self.update_material_uniforms(handle, material);
        }

        // 更新帧计数
        let gpu_mat = self.gpu_materials.get_mut(handle).expect("gpu material should exist.");
        gpu_mat.last_used_frame = self.frame_index;
        
        // Some(gpu_mat)
    }

    fn prepare_binding_resources(
        &mut self,
        assets: &AssetServer,
        resources: &[BindingResource],
    ) -> Vec<u64> {
        // 准备所有资源
        let mut uniform_buffers = Vec::new();
        
        for resource in resources {
            match resource {
                BindingResource::UniformSlot { slot_id, data, label } => {
                    let buf_id = self.prepare_uniform_slot_data(*slot_id, data, label);
                    uniform_buffers.push(buf_id);
                },
                BindingResource::Buffer { buffer, .. } => {
                    let id = buffer.id();
                    if !self.gpu_buffers.contains_key(&id) {
                        // 创建一个空的 GPU buffer 占位（大小参考 buffer.size）
                        let empty_data = vec![0u8; buffer.size()];
                        let mut gpu_buf = GpuBuffer::new(
                            &self.device,
                            &empty_data,
                            buffer.usage(),
                            buffer.label(),
                        );
                        gpu_buf.last_used_frame = self.frame_index;
                        self.gpu_buffers.insert(id, gpu_buf);
                    }
                },
                BindingResource::Texture(handle_opt) => {
                    if let Some(handle) = handle_opt {
                        self.prepare_texture(assets, *handle);
                    }
                },
                _ => {}
            }
        }
        uniform_buffers
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

        let uniform_buffers = self.prepare_binding_resources(assets, &builder.resources);

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
        self.gpu_materials.get(handle).expect("Just inserted")
    }
    
    /// 重建 BindGroup（中路径）
    fn rebuild_bind_group(
        &mut self,
        assets: &AssetServer,
        handle: MaterialHandle,
        material: &crate::resources::material::Material,
    ) -> &GpuMaterial {
        let mut builder = ResourceBuilder::new();
        material.define_bindings(&mut builder);

        let uniform_buffers = self.prepare_binding_resources(assets, &builder.resources);
        
        // 先获取 layout 的克隆（避免借用冲突）
        let layout = {
            let gpu_mat = self.gpu_materials.get(handle).expect("gpu material should exist.");
            gpu_mat.layout.clone()
        };
        
        // 创建新的 bind_group
        let (bind_group, bg_id) = self.create_bind_group(&layout, &builder.resources);
        
        // 更新 gpu_mat
        {
            let gpu_mat = self.gpu_materials.get_mut(handle).expect("gpu material should exist.");
            gpu_mat.bind_group = bind_group;
            gpu_mat.bind_group_id = bg_id;
            gpu_mat.uniform_buffers = uniform_buffers;
            gpu_mat.last_binding_version = material.data.binding_version();
            gpu_mat.last_used_frame = self.frame_index;
        }
        
        self.gpu_materials.get(handle).expect("gpu material should exist.")
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
                if let Some(gpu_mat) = self.gpu_materials.get(handle)
                    && let Some(&buf_id) = gpu_mat.uniform_buffers.first()
                        && let Some(gpu_buf) = self.gpu_buffers.get(&buf_id) {
                            self.queue.write_buffer(&gpu_buf.buffer, 0, data);
                        }
            },
            MaterialData::Phong(m) => {
                let data = bytemuck::bytes_of(m.uniforms());
                if let Some(gpu_mat) = self.gpu_materials.get(handle)
                    && let Some(&buf_id) = gpu_mat.uniform_buffers.first()
                        && let Some(gpu_buf) = self.gpu_buffers.get(&buf_id) {
                            self.queue.write_buffer(&gpu_buf.buffer, 0, data);
                        }
            },
            MaterialData::Standard(m) => {
                let data = bytemuck::bytes_of(m.uniforms());
                if let Some(gpu_mat) = self.gpu_materials.get(handle)
                    && let Some(&buf_id) = gpu_mat.uniform_buffers.first()
                        && let Some(gpu_buf) = self.gpu_buffers.get(&buf_id) {
                            self.queue.write_buffer(&gpu_buf.buffer, 0, data);
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

        if let Some(layout) = self.layout_cache.get(entries) {
            return layout.clone();
        }

        let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cached BindGroupLayout"),
            entries,
        });

        let id = generate_resource_id();

        self.layout_cache.insert(entries.to_vec(), (layout.clone(), id));
        
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
                    let cpu_id = buffer.id();
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
    fn prepare_image(&mut self, image: &Image, required_mip_count: u32, required_usage: wgpu::TextureUsages) {
        let id = image.id();

        let mut needs_recreate = false;

        if let Some(gpu_img) = self.gpu_images.get(&id) {
            if gpu_img.mip_level_count < required_mip_count {
                needs_recreate = true;
            }
            else if !gpu_img.usage.contains(required_usage) {
                needs_recreate = true;
            }
        } else {
            needs_recreate = true; // 不存在，直接创建
        }

        if needs_recreate {
            self.gpu_images.remove(&id);

            let mut gpu_img = GpuImage::new(
                &self.device, 
                &self.queue, 
                image, 
                required_mip_count,
                required_usage
            );
            gpu_img.last_used_frame = self.frame_index;
            self.gpu_images.insert(id, gpu_img);
        } else {
            if let Some(gpu_img) = self.gpu_images.get_mut(&id) {
                gpu_img.update(&self.device, &self.queue, image);
                gpu_img.last_used_frame = self.frame_index;
            }
        }
    }

    // 2. 准备 Texture (Public)
    pub fn prepare_texture(
        &mut self, 
        assets: &AssetServer, 
        handle: TextureHandle
    ) {
        let Some(texture_asset) = assets.get_texture(handle) else {
            log::warn!("Texture asset not found for handle: {:?}", handle);
            return;
        };

        // 极速热路径 (Hot Path) - 版本完全匹配时直接返回
        // 检查所有版本号：如果 Texture 配置、Image 结构、Image 数据都没变，直接返回！
        if let Some(gpu_tex) = self.gpu_textures.get_mut(handle) {
            let tex_ver_match = gpu_tex.version == texture_asset.version();
            let img_id_match = gpu_tex.image_id == texture_asset.image.id();
            let img_gen_match = gpu_tex.image_generation_id == texture_asset.image.generation_id();
            let img_data_match = gpu_tex.image_data_version == texture_asset.image.version();

            if tex_ver_match && img_id_match && img_gen_match && img_data_match {
                // 完全命中缓存，无需计算 mip_count，无需检查 mipmaps_generated
                // 因为如果版本没变，上次 prepare 一定已经处理好了 Mipmap
                gpu_tex.last_used_frame = self.frame_index;
                // 更新 Image 的活跃帧 (防止被过早回收)
                if let Some(gpu_img) = self.gpu_images.get_mut(&gpu_tex.image_id) {
                    gpu_img.last_used_frame = self.frame_index;
                }
                return; 
            }
        }

        // =========================================================
        // 冷路径 (Cold Path) - 只有版本不匹配时才执行复杂的意图计算
        // =========================================================

        let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;

        let image_mips = 1; // 这里简化处理，假设 ImageInner 只有 1 层 (如果支持 DDS/KTX 需读取 descriptor)
        let generated_mips = if texture_asset.generate_mipmaps {
             texture_asset.mip_level_count()
        } else {
             1
        };

        let final_mip_count = std::cmp::max(image_mips, generated_mips);

        if final_mip_count > 1 {
            usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
        }

        self.prepare_image(
            &texture_asset.image, 
            final_mip_count, 
            usage
        );

        let image_id = texture_asset.image.id();
        let gpu_image = self.gpu_images.get(&image_id).expect("GpuImage should be ready");


        // === 3. Mipmap 生成 (Mipmap Generation) ===
        // 仅当需要生成 且 尚未生成时 才执行
        // gpu_image.mipmaps_generated 标记会在 upload_data 时被置为 false
        if texture_asset.generate_mipmaps && !gpu_image.mipmaps_generated {
            let gpu_img_mut = self.gpu_images.get_mut(&image_id).unwrap();
            
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Mipmap Gen") });
            
            self.mipmap_generator.generate(
                &self.device, 
                &mut encoder, 
                &gpu_img_mut.texture, 
                gpu_img_mut.mip_level_count
            );
            
            self.queue.submit(Some(encoder.finish()));
            gpu_img_mut.mipmaps_generated = true;
        }

        let gpu_image = self.gpu_images.get(&image_id).unwrap();


        // [B] 准备/更新 GpuTexture
        let mut needs_update_texture = false;
        if let Some(gpu_tex) = self.gpu_textures.get_mut(handle) {
            // 依赖检查
            let config_changed = gpu_tex.version != texture_asset.version();
            let image_recreated = gpu_tex.image_generation_id != gpu_image.generation_id;
            let image_swapped = gpu_tex.image_id != image_id;

            if config_changed || image_recreated || image_swapped {
                needs_update_texture = true;
            }
            gpu_tex.last_used_frame = self.frame_index;
        } else {
            needs_update_texture = true;
            // self.gpu_textures.insert(handle, gpu_tex);
        }

        if needs_update_texture {
            // 1. GpuTexture (View Only)
            let gpu_tex = GpuTexture::new(texture_asset, gpu_image);
            self.gpu_textures.insert(handle, gpu_tex);

            // 2. Sampler (Cached & Deduped)
            // 使用 TextureSampler 配置去查找或创建唯一 Sampler
            let sampler = self.get_or_create_sampler(&texture_asset.sampler, texture_asset.name());
            self.gpu_samplers.insert(handle, sampler);
        }

        if let Some(gpu_tex) = self.gpu_textures.get_mut(handle) {
            gpu_tex.last_used_frame = self.frame_index;
        }

    }


    fn get_or_create_sampler(&mut self, config: &TextureSampler, label: Option<&str>) -> wgpu::Sampler {
        if let Some(sampler) = self.sampler_cache.get(config) {
            return sampler.clone();
        }

        // Create new sampler
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label,
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

    pub fn prepare_global(&mut self, _assets: &AssetServer, env: &Environment, render_state: &RenderState) {

        let mut builder = ResourceBuilder::new();
        render_state.define_bindings(&mut builder);
        env.define_bindings(&mut builder);

        // 通用地准备所有资源（从 BindingResource 中获取信息）
        for resource in &builder.resources {
            match resource {
                BindingResource::UniformSlot { slot_id, data, label } => {
                    self.prepare_uniform_slot_data(*slot_id, data, label);
                },
                BindingResource::Buffer { buffer, .. } => {
                    // 确保 GPU buffer 占位存在（对于纯句柄式 BufferRef）
            // 如果 Environment 有灯光数据，主动推送到 GPU
            if !env.gpu_light_data.is_empty() {
                let light_bytes = bytemuck::cast_slice(&env.gpu_light_data);
                let _ = self.write_buffer(&env.light_storage_buffer, light_bytes);
            }

                    let id = buffer.id();
                    if !self.gpu_buffers.contains_key(&id) {
                        let empty_data = vec![0u8; buffer.size()];
                        let mut gpu_buf = GpuBuffer::new(
                            &self.device,
                            &empty_data,
                            buffer.usage(),
                            buffer.label(),
                        );
                        gpu_buf.last_used_frame = self.frame_index;
                        self.gpu_buffers.insert(id, gpu_buf);
                    }
                },
                _ => {} // Texture/Sampler 不需要在这里处理
            }
        }

        // 计算资源 hash（资源已准备好）
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
            return;
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