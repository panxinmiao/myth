use std::sync::atomic::{AtomicU32, Ordering};
use glam::{Mat4, Mat3A};
use slotmap::Key;
use log::{warn, error};

use crate::scene::{Scene};
use crate::scene::camera::Camera;
use crate::scene::environment::Environment;
use crate::assets::{AssetServer, GeometryHandle, MaterialHandle};
use crate::resources::uniforms::{DynamicModelUniforms, RenderStateUniforms};
use crate::resources::buffer::CpuBuffer;

use super::resources::ResourceManager;
use super::pipeline::{PipelineCache, FastPipelineKey};
use super::data::{ModelBufferManager, ObjectBindingData};
use super::passes::TrackedRenderPass;

/// 渲染排序键 (Pipeline ID + Material ID + Depth)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RenderKey(u64);

impl RenderKey {
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

/// 内部使用的渲染项
#[derive(Clone)]
pub struct RenderItem {
    pub geo_handle: GeometryHandle,
    pub mat_handle: MaterialHandle,
    pub model_matrix: Mat4,
    pub distance_sq: f32,
}

/// 准备好提交给 GPU 的指令
pub struct RenderCommand {
    pub object_data: ObjectBindingData,
    pub geometry_handle: GeometryHandle, 
    pub material_handle: MaterialHandle,

    pub render_state_id: u32,
    pub env_id: u32,

    pub pipeline_id: u16,
    pub pipeline: wgpu::RenderPipeline,

    pub model_matrix: Mat4, 

    pub sort_key: RenderKey,
    pub dynamic_offset: u32,
}

pub struct RenderState {
    pub id: u32,
    pub uniforms: CpuBuffer<RenderStateUniforms>,
}

static NEXT_RENDER_STATE_ID: AtomicU32 = AtomicU32::new(0);

impl Default for RenderState {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderState {
    pub fn new() -> Self {
        Self {
            id: NEXT_RENDER_STATE_ID.fetch_add(1, Ordering::Relaxed),
            uniforms: CpuBuffer::new(
                RenderStateUniforms::default(),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("RenderState Uniforms")
            ),
        }
    }

    fn update(&mut self, camera: &Camera, time: f32) {
        let view_matrix = camera.view_matrix;
        let vp_matrix = camera.view_projection_matrix;
        let camera_position = camera.world_matrix.translation.to_vec3();

        let mut u = self.uniforms.write();
        u.view_projection = vp_matrix;
        u.view_projection_inverse = vp_matrix.inverse();
        u.view_matrix = view_matrix;
        u.camera_position = camera_position;
        u.time = time;
    }
}

// ============================================================================
//  Render Context
// ============================================================================

pub struct RenderContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    
    pub depth_format: wgpu::TextureFormat,
    pub depth_texture_view: wgpu::TextureView,  
    pub clear_color: wgpu::Color, // from settings

    pub render_state: RenderState,
    
    // 子系统
    pub resource_manager: ResourceManager,
    pub model_buffer_manager: ModelBufferManager,
    pub pipeline_cache: PipelineCache,
}

impl RenderContext {
    pub fn render_frame(&mut self, scene: &Scene, camera: &Camera, assets: &AssetServer, time: f32) {
        self.resource_manager.next_frame();

        let output = match self.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost) => return, // Resize should be handled by event loop
            Err(e) => {
                eprintln!("Render error: {:?}", e);
                return;
            }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.prepare_global_resources(assets, &scene.environment, camera, time);
        let render_items = self.cull_scene(scene, assets, camera);
        let (mut opaque_cmds, mut transparent_cmds) = self.prepare_and_sort_commands(scene, assets, &render_items);
        self.upload_dynamic_uniforms(&mut opaque_cmds, &mut transparent_cmds);

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") });

        {
            let pass_desc = wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view, // <--- 借用发生在这里
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };

            let pass = encoder.begin_render_pass(&pass_desc);
            let mut tracked = TrackedRenderPass::new(pass);

            Self::draw_list(&self.resource_manager, &mut tracked, &opaque_cmds, scene.environment.id, self.render_state.id);
            Self::draw_list(&self.resource_manager, &mut tracked, &transparent_cmds, scene.environment.id, self.render_state.id);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        if self.resource_manager.frame_index().is_multiple_of(60) {
            self.resource_manager.prune(600);
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture_view = Self::create_depth_texture(&self.device, &self.config, self.depth_format);
        }
    }

    pub(crate) fn create_depth_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration, format: wgpu::TextureFormat) -> wgpu::TextureView {
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

    // --- Internal Logic ---

    fn prepare_global_resources(&mut self, assets: &AssetServer, environment: &Environment, camera: &Camera, time: f32) {
        self.render_state.update(camera, time);
        self.resource_manager.prepare_global(assets, environment, &self.render_state);
    }

    fn cull_scene(&self, scene: &Scene, assets: &AssetServer, camera: &Camera) -> Vec<RenderItem> {
        let mut list = Vec::new();
        let frustum = camera.frustum;
        let camera_pos = camera.world_matrix.translation;

        for (node_id, node) in scene.nodes.iter() {
            if let Some(mesh_idx) = node.mesh
                && let Some(mesh) = scene.meshes.get(mesh_idx) {
                    if !node.visible || !mesh.visible { continue; }

                    let geo_handle = mesh.geometry;
                    let mat_handle = mesh.material;

                    let geometry = match assets.get_geometry(geo_handle) {
                        Some(geo) => geo,
                        None => {
                            // 频率限制日志：实际项目中可以用 lazy_static 或 atomic 限制每秒打印次数，防止刷屏
                            warn!("Node {:?} refers to missing Geometry {:?}", node_id, geo_handle);
                            continue;
                        }
                    };
                    let node_world = *node.world_matrix();
                    
                    if let Some(bs) = &geometry.bounding_sphere {
                        let scale = node.scale.max_element();
                        let center = node_world.transform_point3(bs.center);
                        if !frustum.intersects_sphere(center, bs.radius * scale) {
                            continue;
                        }
                    }

                    let distance_sq = camera_pos.distance_squared(node_world.translation);

                    list.push(RenderItem {
                        geo_handle,
                        mat_handle,
                        model_matrix: Mat4::from(node_world),
                        distance_sq,
                    });
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
        if let Some(bg_color) = scene.background {

            self.clear_color = wgpu::Color {
                r: bg_color.x as f64,
                g: bg_color.y as f64,
                b: bg_color.z as f64,
                a: bg_color.w as f64,
            };
        }

        let mut opaque = Vec::new();
        let mut transparent = Vec::new();

        for item in items {
            let geometry = assets.get_geometry(item.geo_handle).expect("Geometry asset missing");
            let material = assets.get_material(item.mat_handle).expect("Material asset missing");

            self.resource_manager.prepare_geometry(assets, item.geo_handle);
            self.resource_manager.prepare_material(assets, item.mat_handle);

            let object_data = self.model_buffer_manager.prepare_bind_group(
                &mut self.resource_manager, 
                item.geo_handle, 
                geometry
            );

            let gpu_geometry = if let Some(g) = self.resource_manager.get_geometry(item.geo_handle){
                g
            } else {
                error!("CRITICAL: GpuGeometry missing immediately after prepare! Handle: {:?}", item.geo_handle);
                continue;
            };
            let gpu_material = if let Some(m) = self.resource_manager.get_material(item.mat_handle) {
                m
            } else {
                error!("CRITICAL: GPU Material missing immediately after prepare! Handle: {:?}", item.mat_handle);
                continue;
            };
            let gpu_world = if let Some(w) = self.resource_manager.get_world(self.render_state.id, scene.environment.id) {
                w
            } else {
                error!("Render Environment missing for render state {:?} and scene {:?}", self.render_state.id, scene.environment.id);
                continue;
            };
            
            let fast_key = FastPipelineKey {
                material_handle: item.mat_handle,
                material_version: material.layout_version(),
                geometry_handle: item.geo_handle,
                geometry_version: geometry.layout_version(),
                scene_id: scene.environment.id,
                scene_version: scene.environment.layout_version,
                render_state_id: self.render_state.id,
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

            let mat_id = item.mat_handle.data().as_ffi() as u32; 
            let sort_key = RenderKey::new(pipeline_id, mat_id, item.distance_sq);

            let cmd = RenderCommand {
                object_data,
                geometry_handle: item.geo_handle,
                material_handle: item.mat_handle,
                render_state_id: self.render_state.id,
                env_id: scene.environment.id,

                pipeline_id,
                pipeline,
                model_matrix: item.model_matrix,
                sort_key,
                dynamic_offset: 0,
            };

            if material.transparent() {
                transparent.push(cmd);
            } else {
                opaque.push(cmd);
            }
        }

        opaque.sort_unstable_by(|a, b| a.sort_key.cmp(&b.sort_key));
        transparent.sort_unstable_by(|a, b| b.sort_key.cmp(&a.sort_key)); 

        (opaque, transparent)
    }

    fn upload_dynamic_uniforms(&mut self, opaque: &mut [RenderCommand], transparent: &mut [RenderCommand]) {
        let total_count = opaque.len() + transparent.len();
        if total_count == 0 { return; }

        let mut data = Vec::with_capacity(total_count);
        let mut process_cmds = |cmds: &mut [RenderCommand], start_idx: usize| {
            for (i, cmd) in cmds.iter_mut().enumerate() {
                let global_idx = start_idx + i;
                let world_matrix_inverse = cmd.model_matrix.inverse();
                let normal_matrix = Mat3A::from_mat4(world_matrix_inverse.transpose());
                
                data.push(DynamicModelUniforms {
                    world_matrix: cmd.model_matrix,
                    world_matrix_inverse,
                    normal_matrix,
                    ..Default::default()
                });

                let dynamic_stride = std::mem::size_of::<DynamicModelUniforms>() as u32;
                cmd.dynamic_offset = global_idx as u32 * dynamic_stride;
            }
        };

        process_cmds(opaque, 0);
        process_cmds(transparent, opaque.len());

        self.model_buffer_manager.write_uniforms(&mut self.resource_manager, data);
    }

    // 静态方法：不依赖 &self，只依赖明确传入的 resource_manager
    fn draw_list<'pass>(
        resource_manager: &'pass ResourceManager,
        pass: &mut TrackedRenderPass<'pass>, 
        cmds: &'pass [RenderCommand], 
        env_id: u32,
        render_state_id: u32
    ) {
        if cmds.is_empty() { return; }

        if let Some(gpu_global) = resource_manager.get_world(render_state_id, env_id) {
            pass.set_bind_group(0, gpu_global.bind_group_id, &gpu_global.bind_group, &[]);
        } else {
            return;
        }

        for cmd in cmds {
            pass.set_pipeline(cmd.pipeline_id, &cmd.pipeline);

            if let Some(gpu_material) = resource_manager.get_material(cmd.material_handle) {
                pass.set_bind_group(1, gpu_material.bind_group_id, &gpu_material.bind_group, &[]);
            }

            pass.set_bind_group(
                2, 
                cmd.object_data.bind_group_id, 
                &cmd.object_data.bind_group, 
                &[cmd.dynamic_offset]
            );

            if let Some(gpu_geometry) = resource_manager.get_geometry(cmd.geometry_handle) {
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