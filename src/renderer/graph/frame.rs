//! 渲染帧管理
//!
//! RenderFrame 负责：extract, prepare_commands, 执行渲染
//! 每一帧重置它

use glam::{Mat3A, Mat4};
use slotmap::Key;
use log::{warn, error};

use crate::scene::{NodeIndex, Scene};
use crate::scene::camera::Camera;
use crate::scene::environment::Environment;
use crate::assets::{AssetServer, GeometryHandle, MaterialHandle};
use crate::resources::GeometryFeatures;
use crate::resources::uniforms::DynamicModelUniforms;

use crate::renderer::core::{WgpuContext, ResourceManager};
use crate::renderer::graph::{TrackedRenderPass, RenderState};
use crate::renderer::graph::extracted::ExtractedScene;
use crate::renderer::managers::{ModelManager, SkeletonManager, ObjectBindingData};
use crate::renderer::pipeline::{PipelineCache, PipelineKey, FastPipelineKey};

/// 渲染排序键 (Pipeline ID + Material ID + Depth)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RenderKey(u64);

impl RenderKey {
    pub fn new(pipeline_id: u16, material_index: u32, depth: f32) -> Self {
        let p_bits = ((pipeline_id & 0x3FFF) as u64) << 50;
        let m_bits = ((material_index & 0xFFFFF) as u64) << 30;
        let d_u32 = if depth.is_sign_negative() { 0 } else { depth.to_bits() >> 2 };
        let d_bits = (d_u32 as u64) & 0x3FFF_FFFF;
        Self(p_bits | m_bits | d_bits)
    }
}

/// 内部使用的渲染项
#[derive(Clone)]
pub struct RenderItem {
    pub geo_handle: GeometryHandle,
    pub mat_handle: MaterialHandle,
    pub node_index: NodeIndex,
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

/// 渲染帧管理器
pub struct RenderFrame {
    render_state: RenderState,
    clear_color: wgpu::Color,
}

impl Default for RenderFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderFrame {
    pub fn new() -> Self {
        Self {
            render_state: RenderState::new(),
            clear_color: wgpu::Color::BLACK,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        wgpu_ctx: &mut WgpuContext,
        resource_manager: &mut ResourceManager,
        model_manager: &mut ModelManager,
        skeleton_manager: &mut SkeletonManager,
        pipeline_cache: &mut PipelineCache,
        scene: &Scene,
        camera: &Camera,
        assets: &AssetServer,
        time: f32,
    ) {
        resource_manager.next_frame();

        let output = match wgpu_ctx.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost) => return,
            Err(e) => {
                eprintln!("Render error: {:?}", e);
                return;
            }
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // ========================================================================
        // Extract 阶段：从 Scene 中提取当前帧所需的精简数据
        // ========================================================================
        let extracted = ExtractedScene::extract(scene, camera, assets);
        
        // 设置背景颜色
        if let Some(bg_color) = extracted.background {
            self.clear_color = wgpu::Color {
                r: bg_color.x as f64,
                g: bg_color.y as f64,
                b: bg_color.z as f64,
                a: bg_color.w as f64,
            };
        }

        // ========================================================================
        // Prepare 阶段：准备 GPU 资源（此时不再需要 Scene 的借用）
        // ========================================================================
        self.prepare_global_resources(resource_manager, assets, &scene.environment, camera, time);
        self.upload_skeletons_extracted(resource_manager, skeleton_manager, &extracted);

        let (mut opaque_cmds, mut transparent_cmds) = self.prepare_and_sort_commands_extracted(
            wgpu_ctx,
            resource_manager,
            model_manager,
            skeleton_manager,
            pipeline_cache,
            assets,
            &extracted,
        );
        self.upload_dynamic_uniforms(resource_manager, model_manager, &mut opaque_cmds, &mut transparent_cmds);

        // ========================================================================
        // Render 阶段：执行实际的渲染命令
        // ========================================================================
        let mut encoder = wgpu_ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

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
                    view: &wgpu_ctx.depth_texture_view,
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

            Self::draw_list(resource_manager, &mut tracked, &opaque_cmds, extracted.environment_id, self.render_state.id);
            Self::draw_list(resource_manager, &mut tracked, &transparent_cmds, extracted.environment_id, self.render_state.id);
        }

        wgpu_ctx.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        if resource_manager.frame_index().is_multiple_of(60) {
            resource_manager.prune(600);
        }
    }

    fn prepare_global_resources(
        &mut self,
        resource_manager: &mut ResourceManager,
        assets: &AssetServer,
        environment: &Environment,
        camera: &Camera,
        time: f32,
    ) {
        self.render_state.update(camera, time);
        resource_manager.prepare_global(assets, environment, &self.render_state);
    }

    #[allow(dead_code)]
    fn upload_skeletons(
        &self,
        resource_manager: &mut ResourceManager,
        skeleton_manager: &mut SkeletonManager,
        scene: &Scene,
    ) {
        for (skel_key, skeleton) in &scene.skins {
            skeleton_manager.update(resource_manager, skel_key, &skeleton.joint_matrices);
        }
    }

    /// 使用提取的骨骼数据上传到 GPU
    fn upload_skeletons_extracted(
        &self,
        resource_manager: &mut ResourceManager,
        skeleton_manager: &mut SkeletonManager,
        extracted: &ExtractedScene,
    ) {
        for skel in &extracted.skeletons {
            skeleton_manager.update(resource_manager, skel.skeleton_key, &skel.joint_matrices);
        }
    }

    /// 基于 ExtractedScene 准备和排序渲染命令
    #[allow(clippy::too_many_arguments)]
    fn prepare_and_sort_commands_extracted(
        &mut self,
        wgpu_ctx: &WgpuContext,
        resource_manager: &mut ResourceManager,
        model_manager: &mut ModelManager,
        skeleton_manager: &SkeletonManager,
        pipeline_cache: &mut PipelineCache,
        assets: &AssetServer,
        extracted: &ExtractedScene,
    ) -> (Vec<RenderCommand>, Vec<RenderCommand>) {
        let mut opaque = Vec::new();
        let mut transparent = Vec::new();

        for item in &extracted.render_items {
            // 安全获取资源，如果缺失则跳过
            let Some(geometry) = assets.get_geometry(item.geometry) else {
                warn!("Geometry {:?} missing during render prepare", item.geometry);
                continue;
            };
            let Some(material) = assets.get_material(item.material) else {
                warn!("Material {:?} missing during render prepare", item.material);
                continue;
            };

            resource_manager.prepare_geometry(assets, item.geometry);
            resource_manager.prepare_material(assets, item.material);

            let skin_binding = item.skin_binding.as_ref();

            let object_data = model_manager.prepare_bind_group(
                resource_manager,
                skeleton_manager,
                item.geometry,
                geometry,
                skin_binding,
            );

            let Some(gpu_geometry) = resource_manager.get_geometry(item.geometry) else {
                error!("CRITICAL: GpuGeometry missing for {:?}", item.geometry);
                continue;
            };
            let Some(gpu_material) = resource_manager.get_material(item.material) else {
                error!("CRITICAL: GpuMaterial missing for {:?}", item.material);
                continue;
            };
            let Some(gpu_world) = resource_manager.get_world(self.render_state.id, extracted.environment_id) else {
                error!("Render Environment missing for render state {:?} and env {:?}", self.render_state.id, extracted.environment_id);
                continue;
            };

            let mut geo_features = geometry.get_features();
            let mut instance_variants = 0;

            if skin_binding.is_none() {
                geo_features.remove(GeometryFeatures::USE_SKINNING);
                instance_variants |= 1 << 0;
            }

            let fast_key = FastPipelineKey {
                material_handle: item.material,
                material_version: material.layout_version(),
                geometry_handle: item.geometry,
                geometry_version: geometry.layout_version(),
                instance_variants,
                scene_id: extracted.environment_id,
                scene_version: extracted.environment_layout_version,
                render_state_id: self.render_state.id,
            };

            let (pipeline, pipeline_id) = if let Some(p) = pipeline_cache.get_pipeline_fast(fast_key) {
                p.clone()
            } else {
                let canonical_key = PipelineKey {
                    mat_features: material.get_features(),
                    geo_features,
                    scene_features: extracted.scene_features,
                    topology: geometry.topology,
                    cull_mode: material.cull_mode(),
                    depth_write: material.depth_write(),
                    depth_compare: if material.depth_test() { wgpu::CompareFunction::Less } else { wgpu::CompareFunction::Always },
                    blend_state: if material.transparent() { Some(wgpu::BlendState::ALPHA_BLENDING) } else { None },
                    color_format: wgpu_ctx.config.format,
                    depth_format: wgpu_ctx.depth_format,
                    sample_count: 1,
                };

                let (pipeline, pipeline_id) = pipeline_cache.get_pipeline(
                    &wgpu_ctx.device,
                    material.shader_name(),
                    canonical_key,
                    &gpu_geometry.layout_info,
                    gpu_material,
                    &object_data,
                    gpu_world,
                );

                pipeline_cache.insert_pipeline_fast(fast_key, (pipeline.clone(), pipeline_id));
                (pipeline, pipeline_id)
            };

            let mat_id = item.material.data().as_ffi() as u32;
            let sort_key = RenderKey::new(pipeline_id, mat_id, item.distance_sq);

            let cmd = RenderCommand {
                object_data,
                geometry_handle: item.geometry,
                material_handle: item.material,
                render_state_id: self.render_state.id,
                env_id: extracted.environment_id,
                pipeline_id,
                pipeline,
                model_matrix: item.world_matrix,
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

    #[allow(dead_code)]
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
                            warn!("Node {:?} refers to missing Geometry {:?}", node_id, geo_handle);
                            continue;
                        }
                    };
                    let node_world = node.transform.world_matrix;

                    if let Some(bs) = &geometry.bounding_sphere {
                        let scale = node.transform.scale.max_element();
                        let center = node_world.transform_point3(bs.center);
                        if !frustum.intersects_sphere(center, bs.radius * scale) {
                            continue;
                        }
                    }

                    let distance_sq = camera_pos.distance_squared(node_world.translation);

                    list.push(RenderItem {
                        geo_handle,
                        mat_handle,
                        node_index: node_id,
                        model_matrix: Mat4::from(node.transform.world_matrix),
                        distance_sq,
                    });
                }
        }
        list
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    fn prepare_and_sort_commands(
        &mut self,
        wgpu_ctx: &WgpuContext,
        resource_manager: &mut ResourceManager,
        model_manager: &mut ModelManager,
        skeleton_manager: &SkeletonManager,
        pipeline_cache: &mut PipelineCache,
        scene: &Scene,
        assets: &AssetServer,
        items: &[RenderItem],
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
            // 安全获取资源，如果缺失则跳过
            let Some(geometry) = assets.get_geometry(item.geo_handle) else {
                warn!("Geometry {:?} missing during render prepare", item.geo_handle);
                continue;
            };
            let Some(material) = assets.get_material(item.mat_handle) else {
                warn!("Material {:?} missing during render prepare", item.mat_handle);
                continue;
            };

            resource_manager.prepare_geometry(assets, item.geo_handle);
            resource_manager.prepare_material(assets, item.mat_handle);

            let skin_binding = if let Some(node) = scene.get_node(item.node_index) {
                node.skin.as_ref()
            } else {
                None
            };

            let object_data = model_manager.prepare_bind_group(
                resource_manager,
                skeleton_manager,
                item.geo_handle,
                geometry,
                skin_binding,
            );

            let Some(gpu_geometry) = resource_manager.get_geometry(item.geo_handle) else {
                error!("CRITICAL: GpuGeometry missing for {:?}", item.geo_handle);
                continue;
            };
            let Some(gpu_material) = resource_manager.get_material(item.mat_handle) else {
                error!("CRITICAL: GpuMaterial missing for {:?}", item.mat_handle);
                continue;
            };
            let Some(gpu_world) = resource_manager.get_world(self.render_state.id, scene.environment.id) else {
                error!("Render Environment missing for render state {:?} and scene {:?}", self.render_state.id, scene.environment.id);
                continue;
            };

            let mut geo_features = geometry.get_features();
            let mut instance_variants = 0;

            if skin_binding.is_none() {
                geo_features.remove(GeometryFeatures::USE_SKINNING);
                instance_variants |= 1 << 0;
            }

            let fast_key = FastPipelineKey {
                material_handle: item.mat_handle,
                material_version: material.layout_version(),
                geometry_handle: item.geo_handle,
                geometry_version: geometry.layout_version(),
                instance_variants,
                scene_id: scene.environment.id,
                scene_version: scene.environment.layout_version(),
                render_state_id: self.render_state.id,
            };

            let (pipeline, pipeline_id) = if let Some(p) = pipeline_cache.get_pipeline_fast(fast_key) {
                p.clone()
            } else {
                let canonical_key = PipelineKey {
                    mat_features: material.get_features(),
                    geo_features,
                    scene_features: scene.get_features(),
                    topology: geometry.topology,
                    cull_mode: material.cull_mode(),
                    depth_write: material.depth_write(),
                    depth_compare: if material.depth_test() { wgpu::CompareFunction::Less } else { wgpu::CompareFunction::Always },
                    blend_state: if material.transparent() { Some(wgpu::BlendState::ALPHA_BLENDING) } else { None },
                    color_format: wgpu_ctx.config.format,
                    depth_format: wgpu_ctx.depth_format,
                    sample_count: 1,
                };

                let (pipeline, pipeline_id) = pipeline_cache.get_pipeline(
                    &wgpu_ctx.device,
                    material.shader_name(),
                    canonical_key,
                    &gpu_geometry.layout_info,
                    gpu_material,
                    &object_data,
                    gpu_world,
                );

                pipeline_cache.insert_pipeline_fast(fast_key, (pipeline.clone(), pipeline_id));
                (pipeline, pipeline_id)
            };

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

    fn upload_dynamic_uniforms(
        &self,
        resource_manager: &mut ResourceManager,
        model_manager: &mut ModelManager,
        opaque: &mut [RenderCommand],
        transparent: &mut [RenderCommand],
    ) {
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

        model_manager.write_uniforms(resource_manager, data);
    }

    fn draw_list<'pass>(
        resource_manager: &'pass ResourceManager,
        pass: &mut TrackedRenderPass<'pass>,
        cmds: &'pass [RenderCommand],
        env_id: u32,
        render_state_id: u32,
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
                &[cmd.dynamic_offset],
            );

            if let Some(gpu_geometry) = resource_manager.get_geometry(cmd.geometry_handle) {
                for (slot, buffer) in gpu_geometry.vertex_buffers.iter().enumerate() {
                    pass.set_vertex_buffer(
                        slot as u32,
                        gpu_geometry.vertex_buffer_ids[slot],
                        buffer.slice(..),
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
