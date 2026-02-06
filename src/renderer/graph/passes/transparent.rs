//! Transparent Render Pass
//!
//! 绘制透明物体的 Pass，用于 PBR/HDR 渲染路径。
//!
//! # 数据流
//! ```text
//! RenderLists.transparent → TransparentPass → HDR Scene Color
//! ```
//!
//! # `RenderPass` 配置
//! - `LoadOp`: Load (继承 `OpaquePass` 的结果)
//! - `StoreOp`: Store (保留结果供后处理使用)
//!
//! # 注意
//! 此 Pass 在 `OpaquePass` 和可选的 `TransmissionCopyPass` 之后执行。

use crate::renderer::graph::frame::RenderCommand;
use crate::renderer::graph::{RenderContext, RenderNode, TrackedRenderPass};

/// Transparent Render Pass
///
/// 仅绘制 `render_lists.transparent` 中的物体。
/// 继承不透明渲染的结果，结果存储供后处理使用。
///
/// # 性能考虑
/// - 命令列表按 Depth (Back-to-Front) 排序，确保正确的 Alpha 混合
/// - 如果有 Transmission 效果，此 Pass 在 `TransmissionCopyPass` 之后执行
pub struct TransparentPass;

impl TransparentPass {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// 获取渲染目标
    fn get_render_target<'a>(
        ctx: &'a RenderContext,
    ) -> (&'a wgpu::TextureView, Option<&'a wgpu::TextureView>) {
        let target_view = ctx.get_scene_render_target_view();
        let is_msaa = ctx.wgpu_ctx.msaa_samples > 1;

        if is_msaa {
            let msaa_view = ctx
                .frame_resources
                .scene_msaa_view
                .as_ref()
                .expect("MSAA view missing");
            (msaa_view, Some(target_view))
        } else {
            (target_view, None)
        }
    }

    /// 执行绘制列表
    fn draw_list<'pass>(
        ctx: &'pass RenderContext,
        pass: &mut TrackedRenderPass<'pass>,
        cmds: &'pass [RenderCommand],
    ) {
        if cmds.is_empty() {
            return;
        }

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

            let screen_bind_group = &ctx.frame_resources.screen_bind_group;
            pass.set_bind_group(3, screen_bind_group.id(), screen_bind_group, &[]);

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
                    pass.draw(
                        gpu_geometry.draw_range.clone(),
                        gpu_geometry.instance_range.clone(),
                    );
                }
            }
        }
    }
}

impl Default for TransparentPass {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderNode for TransparentPass {
    fn name(&self) -> &'static str {
        "Transparent Pass"
    }

    fn run(&self, ctx: &mut RenderContext, encoder: &mut wgpu::CommandEncoder) {
        let render_lists = &ctx.render_frame.render_lists;

        // 获取全局 BindGroup（即使没有透明物体也需要，因为可能需要 resolve）
        let Some(gpu_global_bind_group) = &render_lists.gpu_global_bind_group else {
            log::warn!("TransparentPass: gpu_global_bind_group missing, skipping");
            return;
        };

        let (color_view, resolve_target) = Self::get_render_target(ctx);
        let depth_view = &ctx.frame_resources.depth_view;

        // 计算最终的 store/resolve 配置
        let (store_op, final_resolve_target) = if resolve_target.is_some() {
            // MSAA: 最后 resolve，不保存 MSAA buffer
            (wgpu::StoreOp::Discard, resolve_target)
        } else {
            (wgpu::StoreOp::Store, None)
        };

        let pass_desc = wgpu::RenderPassDescriptor {
            label: Some("Transparent Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: final_resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // 继承 OpaquePass 的结果
                    store: store_op,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load, // 继承深度
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };

        let raw_pass = encoder.begin_render_pass(&pass_desc);
        let mut tracked_pass = TrackedRenderPass::new(raw_pass);

        tracked_pass.set_bind_group(
            0,
            render_lists.gpu_global_bind_group_id,
            gpu_global_bind_group,
            &[],
        );

        // 只有当有透明物体时才绘制
        if !render_lists.transparent.is_empty() {
            Self::draw_list(ctx, &mut tracked_pass, &render_lists.transparent);
        }
        // 即使没有透明物体，也需要这个 Pass 来完成 MSAA resolve
    }
}
