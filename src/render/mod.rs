//! 渲染器模块

pub mod resources;
pub mod pipeline;
pub mod data;
pub mod passes;

use glam::{Mat4, Mat3A};
use slotmap::Key;
use std::sync::Arc;
use winit::window::Window;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;

use crate::scene::Scene;
use crate::scene::camera::Camera;
use crate::resources::uniforms::DynamicModelUniforms;
use crate::assets::{GeometryHandle, MaterialHandle, AssetServer};
use crate::scene::environment::Environment;
use crate::resources::buffer::BufferRef;
use crate::resources::uniforms::{RenderStateUniforms};

use self::resources::ResourceManager;
use self::pipeline::{PipelineCache, FastPipelineKey};

use self::data::{ModelBufferManager, ObjectBindingData};
use self::passes::TrackedRenderPass;

/// 渲染排序键 (Pipeline ID + Material ID + Depth)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct RenderKey(u64);

impl RenderKey {
    const _MASK_PIPELINE: u64 = 0xFFFC_0000_0000_0000;
    const _MASK_MATERIAL: u64 = 0x0003_FFFF_C000_0000;
    const _MASK_DEPTH: u64    = 0x0000_0000_3FFF_FFFF;

    pub fn new(pipeline_id: u16, material_index: u32, depth: f32) -> Self {
        let p_bits = ((pipeline_id & 0x3FFF) as u64) << 50;
        let m_bits = ((material_index & 0xFFFFF) as u64) << 30;
        let d_u32 = if depth.is_sign_negative() { 
            0 
        } else { 
            depth.to_bits() >> 2
        };
        let d_bits = (d_u32 as u64) & 0x3FFF_FFFF;
        Self(p_bits | m_bits | d_bits)
    }
}

/// 内部使用的渲染项 (剔除后的结果)
struct RenderItem {
    geo_handle: GeometryHandle,
    mat_handle: MaterialHandle,
    model_matrix: Mat4,
    distance_sq: f32,
}

/// 准备好提交给 GPU 的指令
struct RenderCommand {
    object_data: ObjectBindingData,
    geometry_handle: GeometryHandle, 
    material_handle: MaterialHandle,

    pipeline_id: u16,
    pipeline: wgpu::RenderPipeline,

    model_matrix: Mat4, 

    sort_key: RenderKey,
    dynamic_offset: u32,
}

pub(crate) struct RenderState {
    pub id: u32,
    pub(crate) uniform_buffer: BufferRef,
}

static NEXT_RENDER_STATE_ID: AtomicU32 = AtomicU32::new(0);

impl RenderState {
    fn new() -> Self {
        Self {
            id: NEXT_RENDER_STATE_ID.fetch_add(0, Ordering::Relaxed),
            uniform_buffer: BufferRef::empty(wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, Some("RenderState Uniforms")),
        }
    }

    fn update(&mut self, camera: &Camera) {
        let view_matrix = camera.view_matrix;
        let vp_matrix =  camera.view_projection_matrix;

        let frame_uniform = RenderStateUniforms{
            view_projection: vp_matrix,
            view_projection_inverse: vp_matrix.inverse(),
            view_matrix,
        };

        self.uniform_buffer.update(&[frame_uniform]);
    }
}

/// 主渲染器
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,

    pub depth_format: wgpu::TextureFormat,
    depth_texture_view: wgpu::TextureView,  
    
    pub(crate) render_state: RenderState,
    // 子系统
    resource_manager: ResourceManager,
    model_buffer_manager: ModelBufferManager,
    pipeline_cache: PipelineCache,

    _size: winit::dpi::PhysicalSize<u32>,

}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();
        
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

        let config = surface.get_default_config(&adapter, size.width, size.height).unwrap();
        surface.configure(&device, &config);

        // 初始化子系统
        let mut resource_manager = ResourceManager::new(device.clone(), queue.clone());
        let model_buffer_manager = ModelBufferManager::new(&mut resource_manager);
        // 创建深度缓冲
        let depth_format = wgpu::TextureFormat::Depth32Float;
        let depth_texture_view = Self::create_depth_texture(&device, &config, depth_format);

        let render_state = RenderState::new();

        Self {
            device,
            queue,
            surface,
            config,
            depth_format,
            depth_texture_view,
            render_state,
            resource_manager,
            pipeline_cache: PipelineCache::new(),
            model_buffer_manager,
            _size: size,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self._size = winit::dpi::PhysicalSize::new(width, height);
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture_view = Self::create_depth_texture(&self.device, &self.config, self.depth_format);
        }
    }

    fn create_depth_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration, format: wgpu::TextureFormat) -> wgpu::TextureView {
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
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// 核心渲染入口
    pub fn render(&mut self, scene: &Scene, camera: &Camera, assets: &AssetServer) {

        if self._size.width == 0 || self._size.height == 0 {
            return;
        }

        // 0. 帧首准备 (清理上一帧的临时资源)
        self.resource_manager.next_frame();

        // swapchain 获取当前帧的输出纹理
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

        // 1. 准备全局资源 (Camera, Light, Environment)
        self.prepare_global_resources(&scene.environment, camera);

        // 2. 剔除与收集 (Culling)
        let render_items = self.cull_scene(scene, assets, camera);

        // 3. 准备 GPU 资源 & 生成指令 (Sorting & Batching preparation)
        let (mut opaque_cmds, mut transparent_cmds) = self.prepare_and_sort_commands(scene, assets, &render_items);

        // 4. 写入每帧动态数据 (Model Matrices)
        self.upload_dynamic_uniforms(&mut opaque_cmds, &mut transparent_cmds);
        
        // 5. 执行 RenderPass
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") });
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.1, b: 0.1, a: 1.0 }),
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

            // 使用 TrackedRenderPass 包装原始 Pass
            let mut tracked = TrackedRenderPass::new(pass);
            // 绘制不透明物体 (Front-to-Back)
            self.draw_list(&mut tracked, &opaque_cmds, &scene.environment);
            // 绘制透明物体 (Back-to-Front)
            self.draw_list(&mut tracked, &transparent_cmds, &scene.environment);
        }

        // 6. 提交与清理
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        
        if self.resource_manager.frame_index() % 60 == 0 {
            self.resource_manager.prune(600);
        }
    }

    // --- 内部辅助方法 (拆分逻辑) ---
    fn prepare_global_resources(&mut self, environment: &Environment, camera: &Camera) {
        // 全局资源来自 RenderState 和 Scene.Environment

        // 更新 RenderState 中的 Camera 数据
        self.render_state.update(camera);
        self.resource_manager.prepare_global(environment, &self.render_state);
    }

    fn cull_scene(
        &self, 
        scene: &Scene, 
        assets: &AssetServer, 
        camera: &Camera, 
    ) -> Vec<RenderItem> {
        let mut list = Vec::new();
        let frustum = camera.frustum;
        let camera_pos = camera.world_matrix.translation;

        for (_id, node) in scene.nodes.iter() {
            if let Some(mesh_idx) = node.mesh {
                if let Some(mesh) = scene.meshes.get(mesh_idx) {
                    if !node.visible || !mesh.visible { continue; }

                    let geo_handle = mesh.geometry;
                    let mat_handle = mesh.material;

                    // 视锥剔除
                    let geometry = assets.get_geometry(geo_handle).expect("Missing Geometry");
                    let node_world = *node.world_matrix();
                    
                    if let Some(bs) = &geometry.bounding_sphere {
                        let scale = node.scale.max_element();
                        let center = node_world.transform_point3(bs.center);
                        if !frustum.intersects_sphere(center, bs.radius * scale) {
                            continue;
                        }
                    }

                    // 距离排序用
                    let distance_sq = camera_pos.distance_squared(node_world.translation.into());

                    list.push(RenderItem {
                        geo_handle,
                        mat_handle,
                        model_matrix: Mat4::from(node_world),
                        distance_sq,
                    });
                }
            }
        }
        list
    }

    fn prepare_and_sort_commands(
        &mut self, 
        scene: &Scene, 
        assets: &AssetServer, 
        items: &[RenderItem]
    ) -> (Vec<RenderCommand>, Vec<RenderCommand>) {
        let mut opaque = Vec::new();
        let mut transparent = Vec::new();

        for item in items {
            // 1. 获取 CPU 端资源 (Assets)
            let geometry = assets.get_geometry(item.geo_handle).expect("Geometry asset missing");
            let material = assets.get_material(item.mat_handle).expect("Material asset missing");

            // 2. 确保 GPU 端资源就绪 (ResourceManager)
            self.resource_manager.prepare_geometry(assets, item.geo_handle);
            self.resource_manager.prepare_material(assets, item.mat_handle);

            // 3. 获取/创建 Model Uniform 的 BindGroup (Group 2)
            let object_data = self.model_buffer_manager.prepare_bind_group(
                &mut self.resource_manager, 
                item.geo_handle, 
                geometry
            );

            // 4. 获取 GPU 资源引用
            let gpu_geometry = self.resource_manager.get_geometry(item.geo_handle).unwrap();
            let gpu_material = self.resource_manager.get_material(item.mat_handle).unwrap();
            let gpu_world = self.resource_manager.get_world(self.render_state.id, scene.environment.id).unwrap();    
            
            // 5. 获取/编译 Pipeline (PSO)

            let fast_key = FastPipelineKey {
                material_handle: item.mat_handle,
                material_version: material.version(),
                geometry_handle: item.geo_handle,
                geometry_version: geometry.version(),
                render_state_id: self.render_state.id,
                scene_id: scene.environment.id,
            };


            let (pipeline, pipeline_id) = self.pipeline_cache.get_or_create(
                &self.device,
                fast_key,
 
                geometry,
                material,
                scene,

                &gpu_geometry.layout_info,
                gpu_material,
                &object_data,
                gpu_world,
                self.config.format,
                self.depth_format,
            );

            // 6. 生成排序键
            let mat_id = item.mat_handle.data().as_ffi() as u32; 
            let sort_key = RenderKey::new(pipeline_id, mat_id, item.distance_sq);

            let cmd = RenderCommand {
                object_data,
                geometry_handle: item.geo_handle,
                material_handle: item.mat_handle,
                pipeline_id,
                pipeline,
                model_matrix: item.model_matrix,
                sort_key,
                dynamic_offset: 0,
            };

            if material.transparent {
                transparent.push(cmd);
            } else {
                opaque.push(cmd);
            }
        }

        // 7. 排序
        opaque.sort_unstable_by(|a, b| a.sort_key.cmp(&b.sort_key));
        transparent.sort_unstable_by(|a, b| b.sort_key.cmp(&a.sort_key)); 

        (opaque, transparent)
    }

    fn upload_dynamic_uniforms(
        &mut self, 
        opaque: &mut [RenderCommand], 
        transparent: &mut [RenderCommand]
    ) {
        let total_count = opaque.len() + transparent.len();
        if total_count == 0 { return; }

        let mut data = Vec::with_capacity(total_count);

        let mut process_cmds = |cmds: &mut [RenderCommand], start_idx: usize| {
            for (i, cmd) in cmds.iter_mut().enumerate() {
                let global_idx = start_idx + i;
                
                let model_matrix_inverse = cmd.model_matrix.inverse();
                let normal_matrix = Mat3A::from_mat4(model_matrix_inverse.transpose());
                
                data.push(DynamicModelUniforms {
                    model_matrix: cmd.model_matrix,
                    model_matrix_inverse,
                    normal_matrix,
                    ..Default::default()
                });

                let dynamic_stride = std::mem::size_of::<DynamicModelUniforms>() as u32;
                cmd.dynamic_offset = global_idx as u32 * dynamic_stride;
            }
        };

        process_cmds(opaque, 0);
        process_cmds(transparent, opaque.len());

        self.model_buffer_manager.write_uniforms(&mut self.resource_manager, &data);
    }

    fn draw_list<'pass>(&'pass self, pass: &mut TrackedRenderPass<'pass>, cmds: &'pass [RenderCommand], environment: &Environment) {
        if cmds.is_empty() { return; }

        // 1. 设置全局 Group 0
        if let Some(gpu_global) = self.resource_manager.get_world(self.render_state.id, environment.id) {
            pass.set_bind_group(0, gpu_global.bind_group_id, &gpu_global.bind_group, &[]);
        } else {
            return;
        }

        // 2. 遍历指令
        for cmd in cmds {
            // A. 设置 Pipeline
            pass.set_pipeline(cmd.pipeline_id, &cmd.pipeline);

            // B. 设置 Material Group 1
            if let Some(gpu_material) = self.resource_manager.get_material(cmd.material_handle) {
                pass.set_bind_group(1, gpu_material.bind_group_id, &gpu_material.bind_group, &[]);
            }

            // C. 设置 Object Group 2
            pass.set_bind_group(
                2, 
                cmd.object_data.bind_group_id, 
                &cmd.object_data.bind_group, 
                &[cmd.dynamic_offset]
            );

            // D. 设置 Vertex & Index Buffers 并绘制
            if let Some(gpu_geometry) = self.resource_manager.get_geometry(cmd.geometry_handle) {
                for (slot, buffer) in gpu_geometry.vertex_buffers.iter().enumerate() {
                    pass.set_vertex_buffer(
                        slot as u32, 
                        gpu_geometry.vertex_buffer_ids[slot], 
                        buffer.slice(..)
                    );
                }

                if let Some((index_buffer, index_format, count, id)) = &gpu_geometry.index_buffer {
                    pass.set_index_buffer(*id, index_buffer.slice(..), *index_format);
                    pass.draw_indexed(0..*count, 0, gpu_geometry.instance_range.clone());
                } else {
                    pass.draw(gpu_geometry.draw_range.clone(), gpu_geometry.instance_range.clone());
                }
            }
        }
    }
}
