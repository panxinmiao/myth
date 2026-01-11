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
mod gpu_image;

use glam::{Mat4, Mat3A};
use slotmap::Key;

use crate::core::scene::Scene;
use crate::core::camera::{Camera};
use crate::core::uniforms::{DynamicModelUniforms};
use crate::core::assets::{GeometryHandle, MaterialHandle};

use self::resource_manager::ResourceManager;
use self::pipeline::PipelineCache;
use self::model_buffer_manager::{ModelBufferManager, ObjectBindingData};
use self::tracked_render_pass::TrackedRenderPass;

use crate::core::world::WorldEnvironment;

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
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
            device: device,
            queue: queue,
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
        }else{
            self.config.width = 1;
            self.config.height = 1;
        }
        self.surface.configure(&self.device, &self.config);
        self.depth_texture_view = Self::create_depth_texture(&self.device, &self.config);

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
        
        struct RenderItem {
            geo_handle: GeometryHandle,
            mat_handle: MaterialHandle,
            model_matrix: glam::Mat4,
            distance_sq: f32, // 用于透明排序
        }
        
        let mut render_list = Vec::new();
        let frustum = camera.frustum();
        let camera_pos = camera.world_matrix().transform_point3(glam::Vec3::ZERO);

        for (_id, node) in scene.nodes.iter() {

            if let Some(mesh_handle) = node.mesh {
                if let Some(mesh) = scene.meshes.get(mesh_handle) {
                    // 提前检查可见性
                    if !node.visible || !mesh.visible { continue; }

                    // 2. 从 AssetServer 获取 Geometry (用于包围盒计算)
                    // mesh.geometry 是 GeometryHandle
                    let geometry = scene.assets.get_geometry(mesh.geometry).expect("Geometry asset missing");

                    let node_world_matrix = *node.world_matrix();
                    // let mat = mesh.material.read().unwrap();
                    if let Some(bs) = &geometry.bounding_sphere{
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
                        geo_handle: mesh.geometry,
                        mat_handle: mesh.material,
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
            geometry_handle: GeometryHandle, 
            material_handle: MaterialHandle,
            renderer_pipeline: wgpu::RenderPipeline,
            renderer_pipeline_id: u16,
            model_matrix: Mat4,
            sort_key: RenderKey,
        }

        let mut opaque_cmd_list = Vec::new();
        let mut transparent_cmd_list = Vec::new();

        self.resource_manager.prepare_global(&self.world);

        for item in &render_list {
            // 拿到 CPU 资源引用
            let geometry = scene.assets.get_geometry(item.geo_handle).unwrap();
            let material = scene.assets.get_material(item.mat_handle).unwrap();

            // 1. 更新 
            self.resource_manager.prepare_geometry(&scene.assets, item.geo_handle);
            self.resource_manager.prepare_material(&scene.assets, item.mat_handle);

            let object_data = self.model_buffer_manager.prepare_bind_group(&mut self.resource_manager, item.geo_handle, &geometry);

            // 2. 获取 GPU 资源
            // 使用 Handle 获取
            let gpu_geometry = self.resource_manager.get_geometry(item.geo_handle).expect("gpu material not found");
            let gpu_material = self.resource_manager.get_material(item.mat_handle).expect("gpu material not found");
            let gpu_world = self.resource_manager.get_world(self.world.id).expect("gpu world not found");
            
            // 3. 更新 Pipeline (需要先拿到刚才 prepare 好的 GPU 资源)
            let pipeline = self.pipeline_cache.get_or_create(
                &self.device,
                item.mat_handle,
                &material,
                item.geo_handle,
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
            // 高 32 位是 version，低 32 位是 index
            let mat_id = item.mat_handle.data().as_ffi() as u32; 

            let pipeline_id = pipeline.1;

            let sort_key = RenderKey::new(
                pipeline_id,
                mat_id,
                item.distance_sq
            );

            let render_cmd = RenderCommand {
                object_data: object_data,
                geometry_handle: item.geo_handle,
                material_handle: item.mat_handle,
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
                let gpu_geometry =  self.resource_manager.get_geometry(cmd.geometry_handle).unwrap();   
                let gpu_material = self.resource_manager.get_material(cmd.material_handle).unwrap();

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
    // 63-50 (14 bits): Pipeline
    // 49-30 (20 bits): Material Index (SlotMap index)
    // 29-00 (30 bits): Depth (Reversed or Standard)
    
    // 掩码常量 (可选，用于 debug)
    const _MASK_PIPELINE: u64 = 0xFFFC_0000_0000_0000;
    const _MASK_MATERIAL: u64 = 0x0003_FFFF_C000_0000;
    const _MASK_DEPTH: u64    = 0x0000_0000_3FFF_FFFF;


    pub fn new(pipeline_id: u16, material_index: u32, depth: f32) -> Self {
        // 1. Pipeline (High 14 bits)
        // 限制在 14 bit (max 16383)
        let p_bits = ((pipeline_id & 0x3FFF) as u64) << 50;

        // 2. Material (Mid 20 bits)
        // SlotMap index 限制在 20 bit (max ~1,000,000)
        let m_bits = ((material_index & 0xFFFFF) as u64) << 30;

        // 3. Depth (Low 30 bits)
        // 假设标准深度 0.0 (近) -> 1.0 (远)
        // 将 f32 转为 u32 的 bit pattern。如果是正数，IEEE754 的顺序与整数顺序一致。
        // 右移 2 位以适应 30 bits。
        let d_u32 = if depth.is_sign_negative() { 
            0 
        } else { 
            depth.to_bits() >> 2
        };
        let d_bits = (d_u32 as u64) & 0x3FFF_FFFF;

        Self(p_bits | m_bits | d_bits)
    }

}