use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use uuid::Uuid;
use wgpu::util::DeviceExt;
use crate::core::geometry::Geometry;
use crate::core::texture::Texture;
use crate::core::material::{Material, MaterialValue};
use super::layout_generator::{self, GeneratedLayout};
use super::shader_generator::{self, MaterialShaderInfo};
use super::uniform_packer;

// ============================================================================
// GPU 资源容器
// ============================================================================

pub struct GPUBuffer {
    pub buffer: wgpu::Buffer,
    pub size: u64,
    pub version: u64,
}

pub struct GPUGeometry {
    /// 核心 Layout 信息 (Pipeline 需要)
    pub layout_info: Arc<GeneratedLayout>,
    
    /// 实际的 Vertex Buffer 引用列表 (按 Slot 顺序)
    pub vertex_buffers: Vec<Arc<GPUBuffer>>,
    
    /// Index Buffer (Buffer, Format, Count)
    pub index_buffer: Option<(Arc<GPUBuffer>, wgpu::IndexFormat, u32)>,
    
    /// 实际绘制的顶点数 (如果没有 Index Buffer)
    pub vertex_count: u32,
    /// 实际绘制的实例数 (Instance Count) - 默认为 1，实际渲染时从 Mesh 获取
    pub instance_count: u32, 
}

pub struct GPUTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: Arc<wgpu::Sampler>, // Sampler 属于 Texture 实例
    pub version: u64,
}

pub struct GPUMaterial {
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
    pub shader_info: Arc<MaterialShaderInfo>,
    pub version: u64,
}

// ============================================================================
// Manager 实现
// ============================================================================

pub struct ResourceManager {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    buffers: HashMap<Uuid, Arc<GPUBuffer>>, // Buffer 缓存 (Key = Buffer.id)
    geometries: HashMap<Uuid, Arc<GPUGeometry>>, // Geometry 缓存 (Key = Geometry.id)

    // Material Cache
    textures: HashMap<Uuid, Arc<GPUTexture>>,
    materials: HashMap<Uuid, Arc<GPUMaterial>>,
    layouts: HashMap<String, Arc<wgpu::BindGroupLayout>>, // Key = shader_hash
    
    // Defaults
    default_texture: Arc<GPUTexture>, // 粉色 1x1
}

impl ResourceManager {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {

        // 创建默认 Missing Texture (粉色)
        let default_tex_cpu = Texture::create_solid_color("Default-Missing", [255, 0, 255, 255]);
        let default_texture = Self::upload_texture(&device, &queue, &default_tex_cpu);

        Self {
            device,
            queue,
            buffers: HashMap::new(),
            geometries: HashMap::new(),
            textures: HashMap::new(),
            materials: HashMap::new(),
            layouts: HashMap::new(),
            default_texture,
        }
    }

    /// 核心入口：获取或更新 GPU Geometry
    pub fn get_or_update_geometry(&mut self, geometry: &Geometry) -> Arc<GPUGeometry> {
        let mut needs_rebuild = false;

        // 1. 检查是否存在
        if !self.geometries.contains_key(&geometry.id) {
            needs_rebuild = true;
        } else {
            // 2. 深度脏检查：检查所有 Vertex Attribute 的 Buffer 版本
            for cpu_attr in geometry.attributes.values() {
                if !self.check_buffer_cache(cpu_attr.buffer.clone()) {
                    needs_rebuild = true; break;
                }
            }
            // 3. 检查 Index Buffer 版本
            if let Some(idx_attr) = &geometry.index_attribute {
                if !self.check_buffer_cache(idx_attr.buffer.clone()) {
                    needs_rebuild = true;
                }
            }
        }

        if needs_rebuild {
            let gpu_geo = self.create_gpu_geometry(geometry);
            self.geometries.insert(geometry.id, gpu_geo.clone());
            return gpu_geo;
        }

        self.geometries.get(&geometry.id).unwrap().clone()
    }

    /// 辅助：检查 Buffer 缓存是否有效 (true = valid, false = dirty/missing)
    fn check_buffer_cache(&self, buffer_ref: Arc<RwLock<crate::core::geometry::GeometryBuffer>>) -> bool {
        let cpu_buf = buffer_ref.read().unwrap();
        if let Some(gpu_buf) = self.buffers.get(&cpu_buf.id) {
            gpu_buf.version == cpu_buf.version
        } else {
            false
        }
    }

    fn create_gpu_geometry(&mut self, geometry: &Geometry) -> Arc<GPUGeometry> {
        // 1. 动态生成 Layout
        // 这一步会分析所有属性，决定 Buffer 的绑定顺序
        let layout_info = Arc::new(layout_generator::generate_vertex_layout(geometry));
        
        let mut vertex_buffers = Vec::new();
        
        // 2. 收集需要的 GPUBuffer
        // 我们利用生成器逻辑中相同的分组逻辑：Buffer ID
        // 为了确保顺序与 layout_info.buffers 一致，我们必须依赖 BTreeMap 的确定性
        let mut buffer_groups: std::collections::BTreeMap<uuid::Uuid, Arc<crate::core::geometry::Attribute>> = std::collections::BTreeMap::new();
        for attr in geometry.attributes.values() {
            let buf_id = attr.buffer.read().unwrap().id;
            buffer_groups.entry(buf_id).or_insert(Arc::new(attr.clone()));
        }

        for (_, attr) in buffer_groups {
            let buffer_ref = attr.buffer.clone();
            // 【重要】明确指定 VERTEX 用途
            let gpu_buf = self.get_or_create_buffer(buffer_ref, wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST);
            vertex_buffers.push(gpu_buf);
        }

        // 3. 处理 Index Buffer
        let index_buffer = if let Some(cpu_idx) = &geometry.index_attribute {
            let buffer_ref = cpu_idx.buffer.clone();
            // 【重要】明确指定 INDEX 用途
            let gpu_buf = self.get_or_create_buffer(buffer_ref, wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST);
            
            // 自动判断 Format
            let format = match cpu_idx.format {
                wgpu::VertexFormat::Uint16 => wgpu::IndexFormat::Uint16,
                wgpu::VertexFormat::Uint32 => wgpu::IndexFormat::Uint32,
                _ => {
                    log::error!("Unsupported index format: {:?}, defaulting to Uint16", cpu_idx.format);
                    wgpu::IndexFormat::Uint16
                }
            };
            Some((gpu_buf, format, cpu_idx.count))
        } else {
            None
        };

        // 4. 计算 Vertex Count
        // 逻辑：如果没有 Index Buffer，DrawCount 取所有 Vertex StepMode 属性中最小的 count
        // 忽略 Instance StepMode 的属性
        let vertex_count = if geometry.index_attribute.is_some() {
            0 // 使用 DrawIndexed，vertex_count 实际上不决定绘制次数
        } else {
             // 找到所有 Per-Vertex 的属性
            let min_count = geometry.attributes.values()
                .filter(|a| a.step_mode == wgpu::VertexStepMode::Vertex)
                .map(|a| a.count)
                .min()
                .unwrap_or(0);
            
            // 应用 draw_range
            let range_count = geometry.draw_range.1;
            min_count.min(range_count)
        };

        Arc::new(GPUGeometry {
            layout_info,
            vertex_buffers,
            index_buffer,
            vertex_count,
            instance_count: 1, // 默认 1，实际渲染时可能会被 Mesh 的 Instancing 逻辑覆盖
        })
    }

    /// 获取或创建 Buffer (底层)
    fn get_or_create_buffer(
        &mut self, 
        cpu_buffer_ref: Arc<RwLock<crate::core::geometry::GeometryBuffer>>,
        usage: wgpu::BufferUsages
    ) -> Arc<GPUBuffer> {
        let cpu_buffer = cpu_buffer_ref.read().unwrap();
        
        // Check cache
        if let Some(gpu_buf) = self.buffers.get(&cpu_buffer.id) {
            if gpu_buf.version == cpu_buffer.version {
                return gpu_buf.clone();
            }
        }

        // Create new
        let label = format!("Buffer-{}", cpu_buffer.id);
        let new_raw_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&label),
            contents: &cpu_buffer.data,
            usage,
        });

        let gpu_buffer = Arc::new(GPUBuffer {
            buffer: new_raw_buffer,
            size: cpu_buffer.data.len() as u64,
            version: cpu_buffer.version,
        });

        self.buffers.insert(cpu_buffer.id, gpu_buffer.clone());
        gpu_buffer
    }

    // ========================================================================
    // Texture Logic
    // ========================================================================

    pub fn get_or_update_texture(&mut self, texture: &Texture) -> Arc<GPUTexture> {
        if let Some(gpu_tex) = self.textures.get(&texture.id) {
            if gpu_tex.version == texture.version {
                return gpu_tex.clone();
            }
        }
        let gpu_tex = Self::upload_texture(&self.device, &self.queue, texture);
        self.textures.insert(texture.id, gpu_tex.clone());
        gpu_tex
    }

    fn upload_texture(device: &wgpu::Device, queue: &wgpu::Queue, texture: &Texture) -> Arc<GPUTexture> {
        // 1. Texture
        let size = wgpu::Extent3d { width: texture.source.width, height: texture.source.height, depth_or_array_layers: 1 };
        let raw_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&texture.name),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture.source.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // 2. Data
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &raw_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &texture.source.data,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * texture.source.width), rows_per_image: None },
            size,
        );

        // 3. View
        let view = raw_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 4. Sampler (Created from texture asset config)
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&format!("{}-Sampler", texture.name)),
            address_mode_u: texture.sampler.address_mode_u,
            address_mode_v: texture.sampler.address_mode_v,
            address_mode_w: texture.sampler.address_mode_w,
            mag_filter: texture.sampler.mag_filter,
            min_filter: texture.sampler.min_filter,
            mipmap_filter: texture.sampler.mipmap_filter,
            ..Default::default()
        });

        Arc::new(GPUTexture {
            texture: raw_texture,
            view,
            sampler: Arc::new(sampler),
            version: texture.version,
        })
    }

// ========================================================================
    // Material Logic
    // ========================================================================

    pub fn get_or_update_material(
        &mut self, 
        material: &Material
    ) -> Arc<GPUMaterial> {
        // 1. Cache Check
        if let Some(gpu_mat) = self.materials.get(&material.id) {
            if gpu_mat.version == material.version {
                return gpu_mat.clone();
            }
        }

        // 2. Generate Shader Info
        let shader_info = Arc::new(shader_generator::generate_material_layout(material));

        // 3. Get Layout
        let layout = self.get_or_create_layout(&shader_info);

        // 4. Create Uniform Buffer
        let uniform_data = uniform_packer::pack_material_uniforms(material);
        let uniform_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Uniform-{}", material.id)),
            contents: &uniform_data,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // 5. Build Bind Group Entries
        let mut entries = Vec::new();
        
        // Binding 0: Uniforms
        entries.push(wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() });

        // 定义一个临时结构来暂存解析好的数据
        struct ResolvedTextureInfo {
            tex_slot: u32,
            samp_slot: Option<u32>,
            gpu_tex: Arc<GPUTexture>,
        }

        let mut prepared_textures = Vec::new();

        // Bindings: Textures & Samplers
        for (prop_name, tex_slot) in &shader_info.texture_bindings {
            // A. Resolve Texture
            let gpu_tex = if let Some(MaterialValue::Texture(tex_ref)) = material.properties.get(prop_name) {
                let cpu_tex = tex_ref.read().unwrap();
                // 直接更新并获取
                self.get_or_update_texture(&cpu_tex)
            } else {
                // 如果 ShaderInfo 里有这个绑定，但 Material 里没有 (这通常不会发生，因为 Info 是基于 Material 生成的)
                // 唯一的例外是 Fallback 逻辑，或者 Shader 模板强制要求某些纹理
                self.default_texture.clone()
            };

            // 获取 Sampler slot
            let samp_slot = shader_info.sampler_bindings.get(prop_name).cloned();

            // 存入列表 (Move Ownership)
            prepared_textures.push(ResolvedTextureInfo {
                tex_slot: *tex_slot,
                samp_slot,
                gpu_tex, 
            });
        }

        for item in &prepared_textures {
            // 1. Bind Texture View
            entries.push(wgpu::BindGroupEntry {
                binding: item.tex_slot,
                resource: wgpu::BindingResource::TextureView(&item.gpu_tex.view),
            });

            // 2. Bind Texture Sampler (如果存在)
            if let Some(s_slot) = item.samp_slot {
                entries.push(wgpu::BindGroupEntry {
                    binding: s_slot,
                    resource: wgpu::BindingResource::Sampler(&item.gpu_tex.sampler),
                });
            }
        }

        // 6. Create Bind Group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("BG-{}", material.id)),
            layout: &layout,
            entries: &entries,
        });

        let gpu_mat = Arc::new(GPUMaterial {
            bind_group,
            uniform_buffer,
            shader_info,
            version: material.version,
        });

        self.materials.insert(material.id, gpu_mat.clone());
        gpu_mat
    }

    fn get_or_create_layout(&mut self, info: &MaterialShaderInfo) -> Arc<wgpu::BindGroupLayout> {
        if let Some(l) = self.layouts.get(&info.shader_hash) { return l.clone(); }
        let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("Layout-{}", info.shader_hash)),
            entries: &info.layout_entries,
        });
        let arc = Arc::new(layout);
        self.layouts.insert(info.shader_hash.clone(), arc.clone());
        arc
    }

}