mod vertex_layout;
mod resource_manager;
mod shader_generator;
mod pipeline;
mod tracked_render_pass;
mod shader_manager;
mod gpu_buffer;
mod gpu_texture;
mod model_buffer_manager;
mod resource_builder;
mod binding;

use std::sync::Arc;
use glam::{Mat4, Mat3A};

use crate::core::scene::Scene;
use crate::core::camera::{Camera};
use crate::core::mesh::Mesh;
use crate::core::uniforms::{DynamicModelUniforms};

use self::resource_manager::ResourceManager;
use self::pipeline::PipelineCache;
use self::model_buffer_manager::{ModelBufferManager, ObjectBindingData};
use self::tracked_render_pass::TrackedRenderPass;

use crate::core::uuid_to_u64;
use crate::core::world::WorldEnvironment;

pub struct Renderer {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub depth_format: wgpu::TextureFormat,
    
    // 子系统
    resource_manager: ResourceManager,
    model_buffer_manager: ModelBufferManager,
    pipeline_cache: PipelineCache,
    // model_buffer_manager: DynamicBuffer, // Group 2 管理器

    // 全局资源 (Group 0)
    pub world: WorldEnvironment,

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

        let mut resource_manager = ResourceManager::new(device.clone(), queue.clone());

        // 2. 初始化 Model Manager (Group 2)
        // let model_buffer_manager = DynamicBuffer::new(&mut resource_manager, "Model");
        let model_buffer_manager = ModelBufferManager::new(&mut resource_manager);

        let world = WorldEnvironment::new();

        // 3. 深度缓冲
        let depth_texture_view = Self::create_depth_texture(&device, &config);

        Self {
            device: device.clone(),
            queue: queue.clone(),
            surface,
            config,
            depth_format: wgpu::TextureFormat::Depth32Float,
            resource_manager,
            pipeline_cache: PipelineCache::new(),
            model_buffer_manager,
            world,
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

    pub fn render(&mut self, scene: &mut Scene, camera: &mut Camera) {

        self.resource_manager.next_frame();

        // 1. 更新场景矩阵
        scene.update_matrix_world();

        self.world.update(camera, scene);

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
        
        // =========================================================
        // 3. 收集 (Collect)
        // =========================================================
        
        struct RenderItem<'a> {
            mesh: &'a Mesh,
            model_matrix: glam::Mat4,
            distance_sq: f32, // 用于透明排序
        }
        
        let mut render_list = Vec::new();
        //let mut transparent_list = Vec::new();
        
        let frustum = camera.frustum();

        let camera_pos = camera.world_matrix().transform_point3(glam::Vec3::ZERO);

        for (_id, node) in scene.nodes.iter() {

            if let Some(mesh_idx) = node.mesh {
                if let Some(mesh) = scene.meshes.get(mesh_idx) {
                    // 提前检查可见性
                    if !node.visible || !mesh.visible { continue; }

                    let node_world_matrix = *node.world_matrix();
                    // let mat = mesh.material.read().unwrap();
                    if let Some(bs) = &mesh.geometry.bounding_sphere{
                        // 注意：bounding sphere 需要变换到世界空间
                        // 这里简单处理：假设 scale 均匀，取 max scale * radius
                        let scale = node.scale.max_element(); 
                        let center = node_world_matrix.transform_point3(bs.center);
                        if !frustum.intersects_sphere(center, bs.radius * scale) {
                            continue;
                        }
                    }

                    // 计算距离 (平方即可，开方开销大)
                    let distance_sq = camera_pos.distance_squared(node_world_matrix.translation.into());
                    
                    // 准备数据
                    let item = RenderItem {
                        mesh,
                        model_matrix: Mat4::from(node_world_matrix),
                        distance_sq,
                    };

                    render_list.push(item);

                }
            }
        }

        // if render_list.is_empty() { return; }



        // =========================================================
        // 4 资源准备 (Mutable Pass)
        // =========================================================
        
        struct RenderCommand{
            object_data: ObjectBindingData,
            geometry_id: uuid::Uuid,
            material_id: uuid::Uuid,
            renderer_pipeline: Arc<wgpu::RenderPipeline>,
            renderer_pipeline_id: u64,
            model_matrix: Mat4,
            sort_key: RenderKey,
        }

        let mut opaque_cmd_list = Vec::new();
        let mut transparent_cmd_list = Vec::new();

        self.resource_manager.prepare_global(&self.world);

        for item in &render_list {
            let geometry = &item.mesh.geometry;
            let material = &item.mesh.material;

            // let model_matrix = item.model_matrix;

            // 1. 更新 Geometry
            self.resource_manager.prepare_geometry(&geometry);
            // 2. 更新 Material
            self.resource_manager.prepare_material(&material);

            // 2. 准备 Object BindGroup
            let object_data = self.model_buffer_manager.prepare_bind_group(&mut self.resource_manager, &geometry);
            let gpu_material = self.resource_manager.get_material(material.id).expect("gpu material not found");
            let gpu_geometry =  self.resource_manager.get_geometry(geometry.id).expect("gpu geometry not found");
            let gpu_world = self.resource_manager.get_world(self.world.id).expect("gpu world not found");
            
            // 3. 更新 Pipeline (需要先拿到刚才 prepare 好的 GPU 资源)
            let pipeline = self.pipeline_cache.get_or_create(
                &self.device,
                &material,
                &geometry,
                &scene,
                gpu_material,
                &object_data,
                &gpu_world,
                self.config.format,
                wgpu::TextureFormat::Depth32Float,
                &gpu_geometry.layout_info, // 传入 Vertex Layout
            );

            // --- 计算 Sort Key ---
            // 目标：将 PipelineID (高位) + MaterialID (中位) + Depth (低位) 压缩到一个 u64
            // Todo 重构 资源ID系统，不在渲染循环中使用UUID。目前 as u16肯定有问题。
            let sort_key = RenderKey::new(
                (pipeline.1) as u16,
                uuid_to_u64(&material.id) as u16,
                item.distance_sq
            );

            let render_cmd = RenderCommand {
                object_data: object_data,
                geometry_id: geometry.id,
                material_id: material.id,
                renderer_pipeline: pipeline.0,
                renderer_pipeline_id: pipeline.1,
                model_matrix: item.model_matrix,
                sort_key: sort_key,
            };

            if material.transparent {
                transparent_cmd_list.push(render_cmd);
            } else {
                opaque_cmd_list.push(render_cmd);
            }

        }


        // cmd list 排序
        opaque_cmd_list.sort_unstable_by(|a, b| {
            a.sort_key.cmp(&b.sort_key)
        });

        transparent_cmd_list.sort_unstable_by(|a, b| {
            b.sort_key.cmp(&a.sort_key) // 注意 b cmp a 是降序
        });

        let mut command_list = opaque_cmd_list;
        command_list.append(&mut transparent_cmd_list);

        // =========================================================
        // 5. 准备 model_uniforms 数据 & 上传 (Prepare & Upload)
        // =========================================================
        
        let mut model_uniforms_data = Vec::with_capacity(command_list.len());
        
        for item in &command_list {
            let model_matrix_inverse = item.model_matrix.inverse();
            let normal_matrix = Mat3A::from_mat4(model_matrix_inverse.transpose());
            model_uniforms_data.push(DynamicModelUniforms {
                model_matrix: item.model_matrix,
                model_matrix_inverse,
                normal_matrix,
                __padding_20: [0.0; 20].into(),
            });
        }

        // 上传所有动态物体的矩阵 (自动扩容)
        self.model_buffer_manager.write_uniforms(&mut self.resource_manager, &model_uniforms_data);

        // =========================================================
        // 6. 绘制循环 (Immutable Pass)
        // Phase B: 渲染录制 (Immutable Pass)
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

            if let Some(gpu_world) = self.resource_manager.get_world(self.world.id) {
                // 绑定 Group 0
                tracked_pass.set_bind_group(0, gpu_world.bind_group_id, &gpu_world.bind_group, &[]);
            }
            // tracked_pass.set_bind_group(0, 0, &gpu_world.bind_group, &[]);

            // 获取 stride，确保安全
            let dynamic_stride = std::mem::size_of::<DynamicModelUniforms>() as u32;

            // 状态追踪 (去重)

            for (i, cmd) in command_list.iter().enumerate() {
 
                // 获取 GPU 资源
                // 使用 unwrap 是安全的，因为 Phase A 保证了它们存在
                let gpu_geometry =  self.resource_manager.get_geometry(cmd.geometry_id).unwrap();   
                let gpu_material = self.resource_manager.get_material(cmd.material_id).unwrap();

                tracked_pass.set_pipeline(cmd.renderer_pipeline_id, &cmd.renderer_pipeline);


                // B. Material BindGroup (Group 1)
                tracked_pass.set_bind_group(1, gpu_material.bind_group_id, &gpu_material.bind_group, &[]);

                // C. Model BindGroup (Group 2) - 动态 Offset
                // 绑定的是同一个 BindGroup，只是 Offset 不同
                let offset = i as u32 * dynamic_stride; 
                tracked_pass.set_bind_group(
                    2, 
                    cmd.object_data.bind_group_id, 
                    &cmd.object_data.bind_group,
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

        // 提交绘制后，检查是否需要 GC
        // 每 60 帧检查一次，清理掉超过 600 帧 (10秒) 没用的资源
        if self.resource_manager.frame_index() % 60 == 0 {
            self.resource_manager.prune(600);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }


}


#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RenderKey(u64);

impl RenderKey {
    // 掩码常量
    const _MASK_PIPELINE: u64 = 0xFFFF_0000_0000_0000;
    const _MASK_MATERIAL: u64 = 0x0000_FFFF_0000_0000;
    const MASK_DEPTH: u64    = 0x0000_0000_FFFF_FFFF;

    // 偏移量
    const SHIFT_PIPELINE: u64 = 48;
    const SHIFT_MATERIAL: u64 = 32;

    pub fn new(pipeline_id: u16, material_id: u16, depth: f32) -> Self {
        // 1. 处理 Pipeline (放在最高位)
        let p_bits = (pipeline_id as u64) << Self::SHIFT_PIPELINE;

        // 2. 处理 Material (放在中间)
        let m_bits = (material_id as u64) << Self::SHIFT_MATERIAL;

        // 3. 处理 Depth (放在低位)
        // 假设是标准深度 0.0 (Near) -> 1.0 (Far)
        // 对于不透明物体，我们需要由近到远，深度越小 Key 越小，直接转换即可。
        // 注意：如果 depth 可能是负数，需要反转符号位，但一般 View Space Depth 都是正的。
        let d_bits = depth.to_bits() as u64; 
        
        // 如果是 Reversed-Z (1.0 Near, 0.0 Far)，且想由近到远：
        // let d_bits = (!depth.to_bits()) as u64; // 取反以保持排序顺序
        
        // 组合
        // 注意：d_bits 必须确保只占 32 位（f32 to_bits 本身就是 u32，转 u64 后高位为0，安全）
        Self(p_bits | m_bits | d_bits)
    }

    //用于调试解码
    pub fn decode(&self) -> (u16, u16, f32) {
        let p = (self.0 >> Self::SHIFT_PIPELINE) as u16;
        let m = (self.0 >> Self::SHIFT_MATERIAL) as u16;
        let d = f32::from_bits((self.0 & Self::MASK_DEPTH) as u32);
        (p, m, d)
    }
}