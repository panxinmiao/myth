mod layout_generator;
mod resource_manager;
mod uniforms;
mod shader_generator;
mod dynamic_buffer;
mod pipeline;
mod tracked_render_pass;

use std::sync::Arc;
use std::collections::HashSet;
use wgpu::util::DeviceExt;

use crate::core::scene::Scene;
use crate::core::camera::Camera;
use crate::core::mesh::Mesh;

use self::resource_manager::ResourceManager;
use self::pipeline::PipelineCache;
use self::dynamic_buffer::DynamicBuffer;
use self::uniforms::{GlobalUniforms, DynamicModelUniforms, Mat3A};
use self::tracked_render_pass::TrackedRenderPass;


// Helper: Uuid -> u64
fn uuid_to_u64(uuid: &uuid::Uuid) -> u64 {
    let bytes = uuid.as_bytes();
    u64::from_le_bytes(bytes[0..8].try_into().unwrap())
}

pub struct Renderer {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub depth_format: wgpu::TextureFormat,
    
    // 子系统
    resource_manager: ResourceManager,
    pipeline_cache: PipelineCache,
    model_buffer_manager: DynamicBuffer, // Group 2 管理器

    // 全局资源 (Group 0)
    global_uniform_buffer: wgpu::Buffer,
    global_bind_group: wgpu::BindGroup,
    global_bind_group_layout: wgpu::BindGroupLayout, // 公开给 PipelineCache 使用

    // 深度缓冲
    depth_texture_view: wgpu::TextureView,
}

impl Renderer {
    pub fn new(
        instance: &wgpu::Instance,
        surface: wgpu::Surface<'static>,
        width: u32, 
        height: u32
    ) -> Self {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })).unwrap();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            },
        )).unwrap();

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let config = surface.get_default_config(&adapter, width, height).unwrap();
        surface.configure(&device, &config);

        // 1. 初始化 Global Uniforms (Group 0)
        let global_uniforms = GlobalUniforms::default();
        let global_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Global Uniforms"),
            contents: bytemuck::bytes_of(&global_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let global_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Global BindGroup Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let global_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Global BindGroup"),
            layout: &global_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: global_buffer.as_entire_binding(),
            }],
        });

        // 2. 初始化 Model Manager (Group 2)
        let model_buffer_manager = DynamicBuffer::new(&device, "Model");

        // 3. 深度缓冲
        let depth_texture_view = Self::create_depth_texture(&device, &config);

        Self {
            device: device.clone(),
            queue: queue.clone(),
            surface,
            config,
            depth_format: wgpu::TextureFormat::Depth32Float,
            resource_manager: ResourceManager::new(device, queue),
            pipeline_cache: PipelineCache::new(),
            model_buffer_manager,
            global_uniform_buffer: global_buffer,
            global_bind_group,
            global_bind_group_layout,
            depth_texture_view,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture_view = Self::create_depth_texture(&self.device, &self.config);
        }
    }

    fn create_depth_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    pub fn render(&mut self, scene: &mut Scene, camera: &Camera) {
        // 1. 更新场景矩阵
        scene.update_matrix_world();

        // 2. 更新相机 Uniform
        let output = match self.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost) => {
                self.resize(self.config.width, self.config.height);
                return;
            },
            Err(e) => {
                eprintln!("Render error: {:?}", e);
                return;
            }
        };

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let view_matrix = camera.get_view_matrix(Some(scene));
        let proj_matrix = camera.get_projection_matrix();

        let vp_matrix = proj_matrix * view_matrix;
        let vp_matrix_inverse = vp_matrix.inverse();

        let globals = GlobalUniforms {
            view_projection: vp_matrix,
            view_projection_inverse: vp_matrix_inverse,
            view_matrix: view_matrix,
        };
        self.queue.write_buffer(&self.global_uniform_buffer, 0, bytemuck::bytes_of(&globals));


        // =========================================================
        // 3. 收集 (Collect)
        // =========================================================
        
        struct RenderItem<'a> {
            mesh: &'a Mesh,
            model_matrix: glam::Mat4,
            // distance: f32, // 用于透明排序
            sort_key: u64, // 排序键 (Pipeline ID + Material ID)
        }

        let mut render_list = Vec::new();


        // 遍历场景收集 Mesh
        // 创建一个集合，用于记录本帧需要用到的纹理 ID
        let mut active_texture_ids = HashSet::new();

        for (_id, node) in scene.nodes.iter() {

            if let Some(mesh_idx) = node.mesh {
                if let Some(mesh) = scene.meshes.get(mesh_idx) {
                    // 提前检查可见性
                    if !node.visible || !mesh.visible { continue; }

                    // TODO: 视锥体剔除 (Frustum Culling)

                    let mat = mesh.material.read().unwrap();
                    // 获取该材质用到的所有纹理 ID
                    for tex_id_opt in mat.textures() {
                        if let Some(tex_id) = tex_id_opt {
                            active_texture_ids.insert(tex_id);
                        }
                    }

                    let model_matrix = node.world_matrix_as_mat4();
                    // let model_matrix_inverse = model_matrix.inverse();

                    // --- 计算 Sort Key ---
                    // 目标：将 PipelineID (高位) + MaterialID (中位) + Depth (低位) 压缩到一个 u64
                    // 这里简化实现：使用 Material ID 的 Hash 作为主要排序依据
                    // 生产环境建议：Renderer 维护一个 MaterialID -> SortKey 的映射缓存
                    let mat_id_hash = uuid_to_u64(&mesh.material.read().unwrap().id); // 注意：这里还是拿了锁，但只拿一次
                    
                    // 对于不透明物体，按材质排序以减少状态切换
                    // 对于透明物体，按距离排序 (从远到近)
                    // 这里暂且只处理不透明物体排序
                    let sort_key = mat_id_hash;
                    
                    // 准备数据
                    render_list.push(RenderItem {
                        mesh,
                        model_matrix,
                        // distance: 0.0, // TODO: 计算相机距离
                        sort_key: sort_key,
                    });

                }
            }
        }

        if render_list.is_empty() { return; }

        // 只上传收集到的纹理
        for tex_id in active_texture_ids {
            // 从 scene.textures 里找到数据
            if let Some(tex_arc) = scene.textures.get(&tex_id) {
                let texture = tex_arc.read().unwrap();
                // 上传！
                self.resource_manager.add_or_update_texture(&texture);
            }
}

        // =========================================================
        // 4. 排序 (Sort)
        // =========================================================
        
        render_list.sort_unstable_by_key(|item| item.sort_key);


        // =========================================================
        // 5. 准备 Uniform 数据 & 上传 (Prepare & Upload)
        // =========================================================
        
        let mut model_uniforms_data = Vec::with_capacity(render_list.len());
        
        for item in &render_list {
            let model_matrix_inverse = item.model_matrix.inverse();
            let normal_matrix = Mat3A::from_mat4(model_matrix_inverse.transpose());
            model_uniforms_data.push(DynamicModelUniforms {
                model_matrix: item.model_matrix,
                model_matrix_inverse,
                normal_matrix,
                _padding: [0.0; 20],
            });
        }

        // 上传所有动态物体的矩阵 (自动扩容)
        self.model_buffer_manager.write_and_expand(
            &self.device, 
            &self.queue, 
            &model_uniforms_data
        );

        // =========================================================
        // 5. 绘制循环 (Draw Loop)
        // =========================================================

        // =========================================================
        // Phase A: 资源准备 (Mutable Pass)
        // =========================================================
        for item in &render_list {
            let geometry = item.mesh.geometry.read().unwrap();
            let material = item.mesh.material.read().unwrap();

            // 1. 更新 Geometry
            self.resource_manager.prepare_geometry(&geometry);
            // 2. 更新 Material
            self.resource_manager.prepare_material(&material);

            // 3. 更新 Pipeline (需要先拿到刚才 prepare 好的 GPU 资源)
            // 为了拿到 GPUGeometry 的信息来构建 Pipeline Key，我们需要临时只读借用一下
            // 注意：因为 prepare 结束了，这里可以借用
            if let Some(gpu_geometry) = self.resource_manager.get_geometry(geometry.id) {
                if let Some(gpu_material) = self.resource_manager.get_material(material.id) {
                     self.pipeline_cache.get_or_create(
                        &self.device,
                        &material,
                        gpu_geometry, // 传入
                        geometry.id,
                        self.config.format,
                        wgpu::TextureFormat::Depth32Float,
                        &self.global_bind_group_layout,
                        &gpu_material.layout,
                        &self.model_buffer_manager.layout
                    );
                }
            }
        }

        // =========================================================
        // Phase B: 渲染录制 (Immutable Pass)
        // 此时 resource_manager 和 pipeline_cache 都是只读的
        // =========================================================
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let raw_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1, g: 0.2, b: 0.3, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            let mut tracked_pass = TrackedRenderPass::new(raw_pass);

            // 绑定永远不变的 Global (Group 0)
            tracked_pass.set_bind_group(0, 0, &self.global_bind_group, &[]);

            // 状态追踪 (去重)

            for (i, item) in render_list.iter().enumerate() {
                let geometry = item.mesh.geometry.read().unwrap();
                let material = item.mesh.material.read().unwrap();

                // 获取 GPU 资源
                // 使用 unwrap 是安全的，因为 Phase A 保证了它们存在
                let gpu_geometry = self.resource_manager.get_geometry(geometry.id).unwrap();
                let gpu_material = self.resource_manager.get_material(material.id).unwrap();

                //2. 获取 Pipeline (只读引用)

                if let Some((pipeline_ref, pipeline_id)) = self.pipeline_cache.get_pipeline(
                    &material,
                    gpu_geometry,
                    geometry.id
                    // ... 这里的参数构建 FastKey 用，开销极小
                ) {
                    tracked_pass.set_pipeline(pipeline_id, pipeline_ref);
                }

                // B. Material BindGroup (Group 1)
                tracked_pass.set_bind_group(1, gpu_material.bind_group_id, &gpu_material.bind_group, &[]);

                // C. Model BindGroup (Group 2) - 动态 Offset
                // 绑定的是同一个 BindGroup，只是 Offset 不同
                let offset = (i * 256) as u32; 
                let model_bg_id = 9999; // 给 Model BindGroup 一个固定的特殊 ID
                tracked_pass.set_bind_group(
                    2, 
                    model_bg_id, 
                    &self.model_buffer_manager.bind_group, 
                    &[offset]
                );

                // D. Vertex Buffers
                for (slot, buffer) in gpu_geometry.vertex_buffers.iter().enumerate() {
                    tracked_pass.set_vertex_buffer(slot as u32, gpu_geometry.vertex_buffer_ids[slot], buffer.slice(..));
                }

                if let Some((index_buffer, index_format, count, id)) = &gpu_geometry.index_buffer {
                    tracked_pass.set_index_buffer(*id, index_buffer.slice(..), *index_format);
                    tracked_pass.draw_indexed(0..*count, 0, gpu_geometry.instance_range.clone());
                }else{
                    tracked_pass.draw(gpu_geometry.draw_range.clone(), gpu_geometry.instance_range.clone());
                }

            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }


}