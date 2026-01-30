//! Forward 渲染 Pass
//!
//! 实现前向渲染管线的主要绘制逻辑。

use std::cell::RefCell;
use glam::Mat3A;
use log::{warn, error};
use slotmap::Key;

use crate::renderer::graph::{RenderNode, RenderContext, TrackedRenderPass};
use crate::renderer::graph::frame::{RenderKey};
use crate::renderer::pipeline::{PipelineKey, FastPipelineKey};
use crate::renderer::pipeline::shader_gen::ShaderCompilationOptions;
use crate::resources::material::{AlphaMode, Side};
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

    commands : RefCell<CommandsBundle>,
}

impl ForwardRenderPass {
    pub fn new(clear_color: wgpu::Color) -> Self {
        Self {
            clear_color,
            commands: RefCell::new(CommandsBundle::new()),
        }
    }

    /// 准备并排序渲染命令
    fn prepare_and_sort_commands(&self, ctx: &mut RenderContext) {
        // let mut opaque = self.commands.borrow_mut().opaque_commands;
        // let mut transparent = self.commands.borrow_mut().transparent_commands;

        let mut commands_bundle = self.commands.borrow_mut();

        commands_bundle.clear();


        let Some(gpu_world) = ctx.resource_manager.get_global_state(ctx.render_state.id, ctx.extracted_scene.scene_id) else {
            error!("Render Environment missing for render_state_id {}, scene_id {}", ctx.render_state.id, ctx.extracted_scene.scene_id);
            return;
        };

        commands_bundle.gpu_global_bind_group_id =gpu_world.bind_group_id;
        commands_bundle.gpu_global_bind_group = Some(gpu_world.bind_group.clone());
        
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


            let object_bind_group = &item.object_bind_group;

            let Some(gpu_geometry) = ctx.resource_manager.get_geometry(item.geometry) else {
                error!("CRITICAL: GpuGeometry missing for {:?}", item.geometry);
                continue;
            };
            let Some(gpu_material) = ctx.resource_manager.get_material(item.material) else {
                error!("CRITICAL: GpuMaterial missing for {:?}", item.material);
                continue;
            };


            // 使用版本号构建快速缓存 Key
            // 注意：scene_id 的哈希计算已被缓存优化，成本较低
            let fast_key = FastPipelineKey {
                material_handle: item.material,
                material_version: gpu_material.version,
                geometry_handle: item.geometry,
                geometry_version: geometry.layout_version(),
                instance_variants: item.item_variant_flags,
                global_state_id: gpu_world.id,
            };

            // ========== 热路径优化：先检查 L1 缓存 ==========
            let (pipeline, pipeline_id) = if let Some(p) = ctx.pipeline_cache.get_pipeline_fast(fast_key) {
                // L1 缓存命中：直接使用已缓存的 Pipeline，无需计算 shader_defines
                p.clone()
            } else {
                // L1 缓存未命中：需要完整计算 shader_defines 以构建/查找 Pipeline
                let geo_defines = geometry.shader_defines();

                let mat_defines = material.shader_defines();

                // let object_defines = item.item_shader_defines;

                let options = ShaderCompilationOptions::from_merged(
                    &mat_defines,
                    &geo_defines,
                    &ctx.extracted_scene.scene_defines,
                    &item.item_shader_defines,
                );
                let shader_hash = options.compute_hash();

                let canonical_key = PipelineKey {
                    shader_hash,
                    topology: geometry.topology,
                    cull_mode: match material.side() {
                        Side::Front => Some(wgpu::Face::Back),
                        Side::Back => Some(wgpu::Face::Front),
                        Side::Double => None,
                    },
                    depth_write: material.depth_write(),
                    // Reverse Z: Greater for depth test
                    depth_compare: if material.depth_test() { wgpu::CompareFunction::Greater } else { wgpu::CompareFunction::Always },
                    blend_state: if material.alpha_mode() == AlphaMode::Blend { Some(wgpu::BlendState::ALPHA_BLENDING) } else { None },
                    color_format: ctx.wgpu_ctx.config.format,
                    depth_format: ctx.wgpu_ctx.depth_format,
                    sample_count: 1,
                };

                let (pipeline, pipeline_id) = ctx.pipeline_cache.get_pipeline(
                    &ctx.wgpu_ctx.device,
                    material.shader_name(),
                    canonical_key,
                    &options,
                    &gpu_geometry.layout_info,
                    gpu_material,
                    object_bind_group,
                    &gpu_world,
                );

                ctx.pipeline_cache.insert_pipeline_fast(fast_key, (pipeline.clone(), pipeline_id));
                (pipeline, pipeline_id)
            };

            // 回写 pipeline 缓存到 Mesh
            if let Some(mesh) = ctx.scene.meshes.get_mut(item.node_handle) {
                mesh.render_cache.pipeline_id = Some(pipeline_id);
            }

            let mat_id = item.material.data().as_ffi() as u32;

            let is_transparent = material.alpha_mode() == AlphaMode::Blend;

            let sort_key = RenderKey::new(pipeline_id, mat_id, item.distance_sq, is_transparent);

            let cmd = RenderCommand {
                object_bind_group: object_bind_group.clone(),
                geometry_handle: item.geometry,
                material_handle: item.material,
                pipeline_id,
                pipeline,
                model_matrix: item.world_matrix,
                sort_key,
                dynamic_offset: 0,
            };

            if is_transparent {
                commands_bundle.insert_transparent(cmd);
            } else {
                commands_bundle.insert_opaque(cmd);
            }
        }

        commands_bundle.sort_commands();
    }

    /// 上传动态 Uniform 数据
    fn upload_dynamic_uniforms(&self, ctx: &mut RenderContext) {
        let mut cmds = self.commands.borrow_mut();
        if cmds.is_empty() {
            return;
        }

        let opaque = &mut cmds.opaque_commands;

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

        let transparent = &mut cmds.transparent_commands;

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

        for cmd in cmds {
            pass.set_pipeline(cmd.pipeline_id, &cmd.pipeline);

            if let Some(gpu_material) = ctx.resource_manager.get_material(cmd.material_handle) {
                pass.set_bind_group(1, gpu_material.bind_group_id, &gpu_material.bind_group, &[]);
            }

            pass.set_bind_group(
                2,
                cmd.object_bind_group.bind_group_id,
                &cmd.object_bind_group.bind_group,
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
            let commands_bundle = self.commands.borrow();

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
                        // Reverse Z 清除为 0.0（远裁剪面）
                        load: wgpu::LoadOp::Clear(0.0),
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

            if let Some(gpu_global_bind_group) = &commands_bundle.gpu_global_bind_group {
                tracked.set_bind_group(0, commands_bundle.gpu_global_bind_group_id, &gpu_global_bind_group, &[]);
            } else {
                return;
            }

            Self::draw_list(
                ctx,
                &mut tracked,
                &commands_bundle.opaque_commands,
            );
            Self::draw_list(
                ctx,
                &mut tracked,
                &commands_bundle.transparent_commands,
            );
        }
    }
}


struct RenderCommand {
    object_bind_group: crate::renderer::core::BindGroupContext,
    geometry_handle: crate::assets::GeometryHandle,
    material_handle: crate::assets::MaterialHandle,
    pipeline_id: u16,
    pipeline: wgpu::RenderPipeline,
    model_matrix: glam::Mat4,
    sort_key: RenderKey,
    dynamic_offset: u32,
}

struct CommandsBundle {
    opaque_commands: Vec<RenderCommand>,
    transparent_commands: Vec<RenderCommand>,
    gpu_global_bind_group_id: u64,
    gpu_global_bind_group: Option<wgpu::BindGroup>,
}

impl CommandsBundle {
    pub fn new() -> Self {
        Self {
            opaque_commands: Vec::with_capacity(512),
            transparent_commands: Vec::with_capacity(128),
            gpu_global_bind_group_id: 0,
            gpu_global_bind_group: None,
        }
    }

    pub fn clear(&mut self) {
        self.opaque_commands.clear();
        self.transparent_commands.clear();
    }

    pub fn insert_opaque(&mut self, cmd: RenderCommand) {
        self.opaque_commands.push(cmd);
    }
    pub fn insert_transparent(&mut self, cmd: RenderCommand) {
        self.transparent_commands.push(cmd);
    }

    pub fn sort_commands(&mut self) {
        self.opaque_commands.sort_unstable_by(|a, b| a.sort_key.cmp(&b.sort_key));
        self.transparent_commands.sort_unstable_by(|a, b| a.sort_key.cmp(&b.sort_key));
    }

    pub fn is_empty(&self) -> bool {
        self.opaque_commands.is_empty() && self.transparent_commands.is_empty()
    }
}