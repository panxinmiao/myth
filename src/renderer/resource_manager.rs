use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use uuid::Uuid;
use wgpu::util::DeviceExt;

use crate::core::geometry::{Geometry, GeometryBuffer};
use super::layout_generator::{self, GeneratedLayout};
use crate::core::material::{Material, MaterialFeatures};
use crate::core::texture::Texture;

// ============================================================================
// GPU 资源包装器 (包含状态追踪)
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
    pub vertex_buffers: Vec<wgpu::Buffer>,
    
    /// Index Buffer (Buffer, Format, Count)
    pub index_buffer: Option<(wgpu::Buffer, wgpu::IndexFormat, u32)>,
    
    // 绘图参数
    pub vertex_count: u32,
    pub instance_count: u32, 

    pub version: u64,
}

pub struct GPUTexture {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub version: u64, // 记录创建此 GPU 资源时对应的 CPU Texture 版本
}

pub struct GPUMaterial {
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub layout: Arc<wgpu::BindGroupLayout>,

    // --- 状态追踪 ---
    pub last_sync_version: u64,           // 上次同步的 Material version
    pub active_features: MaterialFeatures,// 当前 BindGroup 对应的特性
    pub active_texture_ids: Vec<Option<Uuid>>, // 当前 BindGroup 绑定的纹理 ID 列表
}

// ============================================================================
// Resource Manager
// ============================================================================

pub struct ResourceManager {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    buffers: HashMap<Uuid, GPUBuffer>,
    geometries: HashMap<Uuid, GPUGeometry>,

    // 资源缓存
    materials: HashMap<Uuid, GPUMaterial>,
    textures: HashMap<Uuid, GPUTexture>,
    
    // Layout 缓存 (避免重复创建)
    material_layouts: HashMap<MaterialFeatures, Arc<wgpu::BindGroupLayout>>,

    // 默认资源
    dummy_texture: GPUTexture, 
}

impl ResourceManager {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let dummy_texture = Self::create_dummy_texture(&device, &queue);
        
        Self {
            device,
            queue,
            buffers: HashMap::new(),
            geometries: HashMap::new(),
            materials: HashMap::new(),
            textures: HashMap::new(),
            material_layouts: HashMap::new(),
            dummy_texture,
        }
    }


    pub fn get_or_update_geometry(&mut self, geometry: &Geometry) -> &GPUGeometry {
        // 1. 检查是否需要重新构建 GPUGeometry 结构
        // 如果 Geometry 本身 version 变了 (比如加减了属性)，需要重建布局
        // 如果 Geometry 不存在，也需要创建
        let needs_rebuild = if let Some(gpu_geo) = self.geometries.get(&geometry.id) {
            geometry.version > gpu_geo.version
        } else {
            true
        };

        if needs_rebuild {
            // 注意：这里会重新生成 layout 并检查所有 buffer 的数据版本
            self.create_gpu_geometry(geometry);
        } else {
            // 2. 即使 Geometry 结构没变，Buffer 的内容可能变了 (数据热更)
            // 我们需要快速遍历当前 GPUGeometry 持有的 Buffers 进行检查
            // 但为了简化逻辑且保证健壮性，我们可以复用 create_gpu_geometry 中的
            // update_buffer_data 逻辑。
            // 
            // 优化点：如果想极致性能，可以在这里只遍历 geometry.attributes
            // 调用 self.upload_buffer_if_needed(...)
            self.check_geometry_buffers_updates(geometry);
        }

        self.geometries.get(&geometry.id).unwrap()
    }

    fn create_gpu_geometry(&mut self, geometry: &Geometry) {
        // 1. 第一步：生成 Layout
        // 这一步会分析所有属性，决定 Buffer 的绑定顺序
        let layout_info = Arc::new(layout_generator::generate_vertex_layout(geometry));

        // 2. 第二步：根据 Layout 收集 Buffers
        let mut vertex_buffers = Vec::new();
        
        // layout_info.buffers 是排好序的 Slot 描述列表
        for layout_desc in &layout_info.buffers {
            let buffer_arc = &layout_desc.buffer;
            let wgpu_buf = self.get_or_create_buffer(buffer_arc, wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST);
            vertex_buffers.push(wgpu_buf);
        }

        // 3. 处理 Index Buffer (同前)
        let index_buffer = if let Some(indices) = &geometry.index_attribute {
             let buffer = self.get_or_create_buffer(&indices.buffer, wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST);
             let format = match indices.format {
                wgpu::VertexFormat::Uint16 => wgpu::IndexFormat::Uint16,
                wgpu::VertexFormat::Uint32 => wgpu::IndexFormat::Uint32,
                _ => {
                    log::error!("Unsupported index format: {:?}, defaulting to Uint16", indices.format);
                    wgpu::IndexFormat::Uint16
                }
            };
            // 假设 format 是 Uint32
            Some((buffer, format, indices.count))
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

        // 4. 组装 GPUGeometry
        let gpu_geo = GPUGeometry {
            layout_info,
            vertex_buffers, // 这里已经是按 Slot 排好序的了！
            index_buffer,
            vertex_count,
            instance_count: 1, // 初始默认为 1
            version: geometry.version,
        };

        self.geometries.insert(geometry.id, gpu_geo);
    }


    /// 检查几何体引用的所有 Buffer 是否需要上传新数据
    fn check_geometry_buffers_updates(&mut self, geometry: &Geometry) {
        for attr in geometry.attributes.values() {
            self.upload_buffer_if_needed(&attr.buffer);
        }
        if let Some(indices) = &geometry.index_attribute {
            self.upload_buffer_if_needed(&indices.buffer);
        }
    }

    // ========================================================================
    // Buffer 处理 (底层)
    // ========================================================================
    
    // 这是一个通用的帮助方法，用于上传任意 GeometryBuffer
    fn get_or_create_buffer(
        &mut self, 
        buffer_arc: &Arc<RwLock<GeometryBuffer>>,
        usage: wgpu::BufferUsages
    ) -> wgpu::Buffer {
        let cpu_buf = buffer_arc.read().unwrap();
        
        // 1. 检查缓存
        if let Some(gpu_buf) = self.buffers.get_mut(&cpu_buf.id) {
            // 如果版本旧了，更新
            if cpu_buf.version > gpu_buf.version { 
                self.queue.write_buffer(&gpu_buf.buffer, 0, &cpu_buf.data);
                // 更新版本号
                gpu_buf.version = cpu_buf.version;
             }
            return gpu_buf.buffer.clone();
        }

        // 2. 创建新 Buffer
        let wgpu_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Buffer {:?}", cpu_buf.id)),
            contents: &cpu_buf.data, // 假设 GeometryBuffer 有 data: Vec<u8>
            usage: usage,
        });

        self.buffers.insert(cpu_buf.id, GPUBuffer {
            buffer: wgpu_buffer.clone(),
            size: cpu_buf.data.len() as u64,
            version: cpu_buf.version,
        });

        wgpu_buffer
    }


    // 核心上传逻辑：检查版本 -> queue.write_buffer
    fn upload_buffer_if_needed(&mut self, buffer_ref: &Arc<RwLock<GeometryBuffer>>) {
        let cpu_buf = buffer_ref.read().unwrap();
        
        // 情况 A: Buffer 不存在 -> 创建
        if !self.buffers.contains_key(&cpu_buf.id) {
            // 释放读锁，因为我们需要创建 Buffer (可能不需要 &mut self，但逻辑清晰点)
            drop(cpu_buf); 
            self.create_new_buffer(buffer_ref, wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST);
            return;
        }

        // 情况 B: Buffer 存在 -> 检查版本
        let gpu_buf = self.buffers.get_mut(&cpu_buf.id).unwrap();
        
        if cpu_buf.version > gpu_buf.version {
            // 数据变了，上传！
            // 注意：如果数据大小变大了，write_buffer 可能会崩溃
            // 生产环境应该检查 size，如果变大则销毁重建
            if (cpu_buf.data.len() as u64) > gpu_buf.size {
                // 推荐实现：self.create_new_buffer(...) 覆盖旧的
                drop(cpu_buf);
                self.create_new_buffer(buffer_ref, wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST);
                return;
            }

            self.queue.write_buffer(&gpu_buf.buffer, 0, &cpu_buf.data);
            gpu_buf.version = cpu_buf.version;
        }
    }

    fn create_new_buffer(&mut self, buffer_ref: &Arc<RwLock<GeometryBuffer>>, usage: wgpu::BufferUsages) {
        let cpu_buf = buffer_ref.read().unwrap();
        
        let wgpu_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("GeometryBuffer {:?}", cpu_buf.id)),
            contents: &cpu_buf.data,
            usage,
        });

        self.buffers.insert(cpu_buf.id, GPUBuffer {
            buffer: wgpu_buffer,
            size: cpu_buf.data.len() as u64,
            version: cpu_buf.version,
        });
    }

    // ========================================================================
    // Material 处理 (生产级逻辑)
    // ========================================================================

    pub fn get_or_update_material(&mut self, material: &Material) -> &GPUMaterial {
        // 1. 检查是否存在
        if !self.materials.contains_key(&material.id) {
            self.create_gpu_material(material);
        }

        let (last_sync_version, active_features, active_texture_ids) = {
            let gpu_mat = self.materials.get(&material.id).unwrap();
            (
                gpu_mat.last_sync_version,
                gpu_mat.active_features,
                gpu_mat.active_texture_ids.clone(), 
            )
        };

        // // 2. 获取可变引用进行检查
        // let gpu_mat = self.materials.get_mut(&material.id).unwrap();

        // 3. 检查结构性变化 (Structural Change)
        // 结构变化包括：Feature 改变 (如开启/关闭贴图) OR 纹理引用 ID 改变 (如更换贴图)
        let current_features = material.features();
        let current_texture_ids = material.textures();

        // 这里的逻辑非常关键：
        // 即使 Feature 没变（都是 USE_MAP），但如果 map 的 UUID 变了，
        // 或者 map 指向的 Texture 版本变了（贴图内容热更了），都需要重建 BindGroup。
        
        let needs_bind_group_rebuild = 
            active_features != current_features || 
            active_texture_ids != current_texture_ids;
            // self.check_if_any_texture_updated(&current_texture_ids);

        if needs_bind_group_rebuild {
            // 重建 BindGroup (耗时操作，仅在必要时执行)
            let layout = self.get_or_create_material_layout(current_features);
            let bind_group = self.create_material_bind_group(material, &layout);
            
            // 获取可变借用写入新状态 (需要 &mut self)
            let gpu_mat = self.materials.get_mut(&material.id).unwrap();
            gpu_mat.bind_group = bind_group;
            gpu_mat.layout = layout;
            gpu_mat.active_features = current_features;
            gpu_mat.active_texture_ids = current_texture_ids;
            
            // 注意：Uniform Buffer 不需要重建，除非 Struct 大小变了（但在我们的架构中是固定的）
        }

        // 4. 检查数据更新 (Data Update)
        // 只有当 Material 的版本号比 GPU 缓存的版本新时，才上传数据
        let gpu_mat = self.materials.get_mut(&material.id).unwrap();

        if material.version > last_sync_version {
            self.queue.write_buffer(&gpu_mat.uniform_buffer, 0, material.as_bytes());
            gpu_mat.last_sync_version = material.version;
        }

        // 返回引用
        self.materials.get(&material.id).unwrap()
    }

    // ========================================================================
    // 内部辅助逻辑
    // ========================================================================

    fn create_gpu_material(&mut self, material: &Material) {
        let features = material.features();
        let layout = self.get_or_create_material_layout(features);
        
        // 创建 Buffer
        let uniform_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Material Uniforms {:?}", material.id)),
            contents: material.as_bytes(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // 创建 BindGroup
        let bind_group = self.create_material_bind_group(material, &layout);

        let gpu_mat = GPUMaterial {
            uniform_buffer,
            bind_group,
            layout,
            last_sync_version: material.version,
            active_features: features,
            active_texture_ids: material.textures(),
        };

        self.materials.insert(material.id, gpu_mat);
    }

    fn create_material_bind_group(
        &self, 
        material: &Material, 
        layout: &wgpu::BindGroupLayout
    ) -> wgpu::BindGroup {
        let uniform_buffer = &self.materials.get(&material.id).unwrap().uniform_buffer;
        
        let mut entries = Vec::new();
        
        // Binding 0: Uniforms
        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        });

        // 动态绑定纹理
        let mut binding_idx = 1;
        let features = material.features();
        let texture_ids = material.textures(); // [Map, Normal, Roughness, Emissive]

        // 必须与 ShaderGenerator 和 Material::textures 顺序严格一致
        let feature_list = [
            MaterialFeatures::USE_MAP,
            MaterialFeatures::USE_NORMAL_MAP,
            MaterialFeatures::USE_ROUGHNESS_MAP,
            MaterialFeatures::USE_EMISSIVE_MAP,
        ];

        for (i, feature) in feature_list.iter().enumerate() {
            if features.contains(*feature) {
                // 尝试获取纹理
                let tex_id_opt = texture_ids.get(i).cloned().flatten();
                
                // 获取 GPU 资源，如果纹理未就绪或不存在，使用占位图
                let gpu_tex = if let Some(id) = tex_id_opt {
                    self.textures.get(&id).unwrap_or(&self.dummy_texture)
                } else {
                    &self.dummy_texture
                };

                // Binding N
                entries.push(wgpu::BindGroupEntry {
                    binding: binding_idx,
                    resource: wgpu::BindingResource::TextureView(&gpu_tex.view),
                });
                // Binding N+1
                entries.push(wgpu::BindGroupEntry {
                    binding: binding_idx + 1,
                    resource: wgpu::BindingResource::Sampler(&gpu_tex.sampler),
                });

                binding_idx += 2;
            }
        }

        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Material BindGroup"),
            layout,
            entries: &entries,
        })
    }

    fn get_or_create_material_layout(&mut self, features: MaterialFeatures) -> Arc<wgpu::BindGroupLayout> {
        if let Some(layout) = self.material_layouts.get(&features) {
            return layout.clone();
        }

        // ... 创建 Layout 代码 (与之前相同，省略以节省空间) ...
        // 关键点：根据 features 动态 push Texture 和 Sampler 的 entry
        
        let mut entries = vec![
             wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        ];
        
        let mut binding_idx = 1;
        let feature_list = [
            MaterialFeatures::USE_MAP,
            MaterialFeatures::USE_NORMAL_MAP,
            MaterialFeatures::USE_ROUGHNESS_MAP,
            MaterialFeatures::USE_EMISSIVE_MAP,
        ];
        
        for feature in feature_list {
             if features.contains(feature) {
                 // Texture
                 entries.push(wgpu::BindGroupLayoutEntry {
                    binding: binding_idx,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                });
                // Sampler
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: binding_idx + 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                });
                binding_idx += 2;
             }
        }
        
        let layout = Arc::new(self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("Material Layout {:?}", features)),
            entries: &entries,
        }));

        self.material_layouts.insert(features, layout.clone());
        layout
    }
    
    /// 检查材质引用的纹理是否有更新（例如异步加载完成）
    // fn check_if_any_texture_updated(&self) -> bool {
    //     // 这里比较 tricky。理想情况下，BindGroup 引用的是 View，
    //     // 如果 Texture 内容变了但 View 没变（直接 write_texture），则不需要重建 BindGroup。
    //     // 但如果 Texture 被销毁并重新创建了（View 变了），则需要重建。
    //     // 简单起见，我们目前假设 Texture 的 update 只是 write_texture，不改变 View。
    //     // 
    //     // 除非：之前是 Placeholder，现在变成了真图。这种情况下 View 一定变了。
    //     // 我们通过比较 "Material 认为的 ID" 和 "Material 上次绑定时 ID 对应的 GPUTexture" 来判断吗？
    //     // 不，我们在 GPUTexture 里存一个 version 也没用，因为 BindGroup 已经锁死了 View。
        
    //     // 生产级做法：
    //     // 如果 texture_ids 列表里的某个 ID，在 self.textures 里的 version 
    //     // 大于该 Material 上次构建 BindGroup 时记录的该 Texture 的 version...
    //     // 这需要 GPUMaterial 记录每个 Texture 的 version。
    //     // 
    //     // 简化版生产做法：
    //     // 只要 Texture 的 View 变了（比如从 dummy 变成了 real），self.textures 里该 UUID 的 Entry 就会变。
    //     // 但是 BindGroup 内部持有的是旧 View 的引用。
    //     // 我们怎么知道旧 View 过期了？
    //     // 
    //     // 答案：我们不需要在这里做复杂的 version 比较。
    //     // 当 add_texture 被调用时，我们知道哪个 ID 变了。
    //     // 但在这里 (get_material)，我们只能被动检查。
        
    //     // 这种情况下，最稳健的做法是：
    //     // 如果 Material 引用的某个 Texture ID，在 self.textures 中存在，
    //     // 并且它不再是 Dummy Texture (或者我们有一个 generation id)，
    //     // 而我们上次绑定时它可能是 Dummy。
        
    //     // 这里暂时返回 false，依靠 "Data Update" 无法解决 "View 替换"。
    //     // 真正的引擎会有一个 Event Bus： TextureUpdated(Uuid) -> 遍历 Materials -> 标记 Dirty。
    //     false 
    // }

    // ========================================================================
    // Texture 处理
    // ========================================================================

    pub fn add_or_update_texture(&mut self, texture: &Texture) {
        // 如果 Texture 已存在，且版本没变，忽略
        if let Some(gpu_tex) = self.textures.get(&texture.id) {
            if gpu_tex.version >= texture.version {
                return;
            }
        }

        // 创建新 Texture 资源
        let size = wgpu::Extent3d {
            width: texture.source.width,
            height: texture.source.height,
            depth_or_array_layers: 1,
        };

        let gpu_texture_obj = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("Texture {:?}", texture.id)),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb, // 需从 Texture 获取格式
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        if let Some(data) = &texture.source.data {
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &gpu_texture_obj,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * texture.source.width),
                    rows_per_image: Some(texture.source.height),
                },
                size,
            );
        }

        let view = gpu_texture_obj.create_view(&wgpu::TextureViewDescriptor::default());
        
        // 创建 Sampler (根据 Texture 配置)
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat, 
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // 插入或更新
        self.textures.insert(texture.id, GPUTexture {
            view,
            sampler,
            version: texture.version,
        });
        
        // !!! 关键点 !!!
        // 这是一个生产环境的简单实现：
        // 当纹理更新时，我们必须通知所有使用该纹理的材质“变脏”。
        // 否则 Material 的 BindGroup 依然持有旧的 View (或者 Dummy View)。
        // 
        // 暴力做法：遍历所有 Material，如果使用了该 Texture，强制 active_texture_ids 失效。
        for gpu_mat in self.materials.values_mut() {
            if gpu_mat.active_texture_ids.contains(&Some(texture.id)) {
                // 强制让它认为结构变了，下次 get_material 时会自动重建 BindGroup
                gpu_mat.active_texture_ids = Vec::new(); // 清空，制造差异
            }
        }
    }

    fn create_dummy_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> GPUTexture {
        // ... (保持之前的实现，创建一个1x1白色纹理) ...
        let size = wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // ... write white pixel ...
        queue.write_texture(
             wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255, 255, 255, 255],
             wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            size,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        
        GPUTexture {
            view,
            sampler,
            version: 0,
        }
    }
}