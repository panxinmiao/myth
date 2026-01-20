//! Forward 渲染 Pass
//!
//! 实现前向渲染管线的主要绘制逻辑。

use std::cell::RefCell;
use glam::Mat3A;
use log::{warn, error};
use slotmap::Key;

use crate::renderer::graph::{RenderNode, RenderContext, TrackedRenderPass};
use crate::renderer::graph::frame::{RenderKey, RenderCommand};
use crate::renderer::pipeline::{PipelineKey, FastPipelineKey};
use crate::resources::material::Side;
use crate::resources::GeometryFeatures;
use crate::resources::uniforms::DynamicModelUniforms;

/// Forward 渲染 Pass
/// 
/// 执行标准的前向渲染流程：
/// 1. 准备并排序渲染命令
/// 2. 开始渲染通道
/// 3. 绘制不透明物体（前向后排序）
/// 4. 绘制透明物体（后向前排序）
/// 
/// # 设计说明
/// 使用 `RefCell` 提供内部可变性，使得 `RenderNode::run(&self, ...)` 
/// 能够修改内部命令列表。这是 Rust 中常见的内部可变性模式。
/// 
/// # 性能考虑
/// - 命令列表预分配并复用内存，避免每帧分配
/// - `RefCell` 的运行时借用检查开销极小（单线程场景下约等于一次原子操作）
/// - 使用 `TrackedRenderPass` 避免冗余状态切换
/// - Pipeline 和 BindGroup 缓存减少 GPU 状态变更
pub struct ForwardRenderPass {
    /// 清屏颜色
    pub clear_color: wgpu::Color,
    /// 复用的不透明命令列表（使用 RefCell 提供内部可变性）
    opaque_commands: RefCell<Vec<RenderCommand>>,
    /// 复用的透明命令列表（使用 RefCell 提供内部可变性）
    transparent_commands: RefCell<Vec<RenderCommand>>,
}

impl ForwardRenderPass {
    pub fn new(clear_color: wgpu::Color) -> Self {
        Self {
            clear_color,
            opaque_commands: RefCell::new(Vec::with_capacity(512)),
            transparent_commands: RefCell::new(Vec::with_capacity(128)),
        }
    }

    /// 准备并排序渲染命令
    fn prepare_and_sort_commands(&self, ctx: &mut RenderContext) {
        let mut opaque = self.opaque_commands.borrow_mut();
        let mut transparent = self.transparent_commands.borrow_mut();
        
        opaque.clear();
        transparent.clear();
        
        for item_idx in 0..ctx.extracted_scene.render_items.len() {
            let item = &ctx.extracted_scene.render_items[item_idx];
            
            let Some(geometry) = ctx.assets.get_geometry(item.geometry) else {
                warn!("Geometry {:?} missing during render prepare", item.geometry);
                continue;
            };
            let Some(material) = ctx.assets.get_material(item.material) else {
                warn!("Material {:?} missing during render prepare", item.material);
                continue;
            };

            let Some(mesh) = ctx.scene.meshes.get_mut(item.mesh_key) else {
                warn!("Mesh {:?} missing during render prepare", item.mesh_key);
                continue;
            };

            let skeleton = if let Some(skel_key) = item.skeleton {
                ctx.scene.skins.get(skel_key)
            } else {
                None
            };
    
            let object_data = ctx.resource_manager.prepare_mesh(ctx.assets, mesh, skeleton);

            let Some(object_data) = object_data else {
                warn!("Failed to prepare ObjectBindingData for Mesh {:?}", item.mesh_key);
                continue;
            };

            let Some(gpu_geometry) = ctx.resource_manager.get_geometry(item.geometry) else {
                error!("CRITICAL: GpuGeometry missing for {:?}", item.geometry);
                continue;
            };
            let Some(gpu_material) = ctx.resource_manager.get_material(item.material) else {
                error!("CRITICAL: GpuMaterial missing for {:?}", item.material);
                continue;
            };
            let Some(gpu_world) = ctx.resource_manager.get_global_state(ctx.render_state.id, ctx.extracted_scene.scene_id) else {
                error!("Render Environment missing for render_state_id {}, scene_id {}", ctx.render_state.id, ctx.extracted_scene.scene_id);
                continue;
            };

            let mut geo_features = geometry.get_features();
            let mut instance_variants = 0;

            if skeleton.is_none() {
                geo_features.remove(GeometryFeatures::USE_SKINNING);
                instance_variants |= 1 << 0;
            }

            // 使用 GPU 端的 layout_id 构建快速缓存 Key
            // 这比 CPU 端的 version 更精确地反映 Pipeline 兼容性
            let fast_key = FastPipelineKey {
                material_handle: item.material,
                material_layout_id: gpu_material.layout_id,
                geometry_handle: item.geometry,
                geometry_layout_version: geometry.layout_version(),
                instance_variants,
                scene_id: ctx.extracted_scene.scene_features.bits(),
                render_state_id: ctx.render_state.id,
            };

            let (pipeline, pipeline_id) = if let Some(p) = ctx.pipeline_cache.get_pipeline_fast(fast_key) {
                p.clone()
            } else {
                let canonical_key = PipelineKey {
                    mat_features: material.get_features(),
                    geo_features,
                    scene_features: ctx.extracted_scene.scene_features,
                    topology: geometry.topology,
                    cull_mode: match material.side() {
                        Side::Front => Some(wgpu::Face::Back),
                        Side::Back => Some(wgpu::Face::Front),
                        Side::Double => None,
                    },
                    depth_write: material.depth_write(),
                    depth_compare: if material.depth_test() { wgpu::CompareFunction::Less } else { wgpu::CompareFunction::Always },
                    blend_state: if material.transparent() { Some(wgpu::BlendState::ALPHA_BLENDING) } else { None },
                    color_format: ctx.wgpu_ctx.config.format,
                    depth_format: ctx.wgpu_ctx.depth_format,
                    sample_count: 1,
                };

                let (pipeline, pipeline_id) = ctx.pipeline_cache.get_pipeline(
                    &ctx.wgpu_ctx.device,
                    material.shader_name(),
                    canonical_key,
                    &gpu_geometry.layout_info,
                    gpu_material,
                    &object_data,
                    &gpu_world.binding_wgsl,
                    &gpu_world.layout,
                );

                ctx.pipeline_cache.insert_pipeline_fast(fast_key, (pipeline.clone(), pipeline_id));
                (pipeline, pipeline_id)
            };

            if let Some(mesh) = ctx.scene.meshes.get_mut(item.mesh_key) {
                mesh.render_cache.pipeline_id = Some(pipeline_id);
            }

            let mat_id = item.material.data().as_ffi() as u32;
            let sort_key = RenderKey::new(pipeline_id, mat_id, item.distance_sq);

            let cmd = RenderCommand {
                object_data,
                geometry_handle: item.geometry,
                material_handle: item.material,
                render_state_id: ctx.render_state.id,
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
    }

    /// 上传动态 Uniform 数据
    fn upload_dynamic_uniforms(&self, ctx: &mut RenderContext) {
        let mut opaque = self.opaque_commands.borrow_mut();
        let mut transparent = self.transparent_commands.borrow_mut();
        
        let total_count = opaque.len() + transparent.len();
        if total_count == 0 { return; }

        for cmd in opaque.iter_mut() {
            let world_matrix_inverse = cmd.model_matrix.inverse();
            let normal_matrix = Mat3A::from_mat4(world_matrix_inverse.transpose());

            let offset = ctx.resource_manager.allocate_model_uniform(DynamicModelUniforms {
                world_matrix: cmd.model_matrix,
                world_matrix_inverse,
                normal_matrix,
                ..Default::default()
            });

            cmd.dynamic_offset = offset;
        }

        for cmd in transparent.iter_mut() {
            let world_matrix_inverse = cmd.model_matrix.inverse();
            let normal_matrix = Mat3A::from_mat4(world_matrix_inverse.transpose());

            let offset = ctx.resource_manager.allocate_model_uniform(DynamicModelUniforms {
                world_matrix: cmd.model_matrix,
                world_matrix_inverse,
                normal_matrix,
                ..Default::default()
            });

            cmd.dynamic_offset = offset;
        }

        ctx.resource_manager.upload_model_buffer();
    }

    /// 执行绘制列表
    fn draw_list<'pass>(
        ctx: &'pass RenderContext,
        pass: &mut TrackedRenderPass<'pass>,
        cmds: &'pass [RenderCommand],
        // render_state_id: u32,
        // scene_id: u64,
    ) {
        if cmds.is_empty() { return; }

        if let Some(gpu_global) = ctx.resource_manager.get_global_state(ctx.render_state.id, ctx.extracted_scene.scene_id) {
            pass.set_bind_group(0, gpu_global.bind_group_id, &gpu_global.bind_group, &[]);
        } else {
            return;
        }

        for cmd in cmds {
            pass.set_pipeline(cmd.pipeline_id, &cmd.pipeline);

            if let Some(gpu_material) = ctx.resource_manager.get_material(cmd.material_handle) {
                pass.set_bind_group(1, gpu_material.bind_group_id, &gpu_material.bind_group, &[]);
            }

            pass.set_bind_group(
                2,
                cmd.object_data.bind_group_id,
                &cmd.object_data.bind_group,
                &[cmd.dynamic_offset],
            );

            if let Some(gpu_geometry) = ctx.resource_manager.get_geometry(cmd.geometry_handle) {
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

impl RenderNode for ForwardRenderPass {
    fn name(&self) -> &str {
        "Forward Pass"
    }

    fn run(&self, ctx: &mut RenderContext, encoder: &mut wgpu::CommandEncoder) {
        // 1. 准备渲染命令（通过 RefCell 获取内部可变性）
        self.prepare_and_sort_commands(ctx);
        
        // 2. 上传动态 Uniform
        self.upload_dynamic_uniforms(ctx);

        // 3. 获取深度视图
        let depth_view = ctx.wgpu_ctx.get_depth_view();

        // 4. 开始渲染通道并执行绘制
        {
            // 获取不可变借用（必须在 TrackedRenderPass 之前声明以保证生命周期）
            let opaque = self.opaque_commands.borrow();
            let transparent = self.transparent_commands.borrow();

            let pass_desc = wgpu::RenderPassDescriptor {
                label: Some("Forward Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ctx.surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
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


            Self::draw_list(
                ctx,
                &mut tracked,
                &opaque,
            );
            Self::draw_list(
                ctx,
                &mut tracked,
                &transparent,
            );
        }
    }
}
