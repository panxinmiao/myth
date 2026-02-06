//! Simple Forward Render Pass
//!
//! 简化版 Forward Pass，用于 LDR/非 HDR 渲染路径。
//! 在单个 `RenderPass` 中依次绘制不透明和透明物体。
//!
//! # 数据流
//! ```text
//! RenderLists (from SceneCullPass) → SimpleForwardPass → Surface/LDR Target
//! ```
//!
//! # 使用场景
//! - 低配模式
//! - UI 场景
//! - 非 PBR 场景
//! - 不需要 Transmission 效果的场景

use crate::renderer::graph::frame::RenderCommand;
use crate::renderer::graph::{RenderContext, RenderNode, TrackedRenderPass};

/// Simple Forward Render Pass
///
/// 在单个 `RenderPass` 中完成所有绘制：
/// 1. Clear 颜色和深度缓冲
/// 2. 绘制不透明物体（Front-to-Back）
/// 3. 绘制透明物体（Back-to-Front）
///
/// # 性能考虑
/// - 使用 `TrackedRenderPass` 避免冗余状态切换
/// - 命令列表已预排序，无需额外排序开销
pub struct SimpleForwardPass {
    /// Clear color
    pub clear_color: wgpu::Color,
}

impl SimpleForwardPass {
    #[must_use]
    pub fn new(clear_color: wgpu::Color) -> Self {
        Self { clear_color }
    }

    /// 获取渲染目标
    ///
    /// 返回 (`color_view`, `resolve_view`)
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

impl Default for SimpleForwardPass {
    fn default() -> Self {
        Self::new(wgpu::Color::BLACK)
    }
}

impl RenderNode for SimpleForwardPass {
    fn name(&self) -> &'static str {
        "Simple Forward Pass"
    }

    fn run(&self, ctx: &mut RenderContext, encoder: &mut wgpu::CommandEncoder) {
        let render_lists = &ctx.render_frame.render_lists;

        // 获取全局 BindGroup
        let Some(gpu_global_bind_group) = &render_lists.gpu_global_bind_group else {
            log::warn!("SimpleForwardPass: gpu_global_bind_group missing, skipping");
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

        // 单个 RenderPass：Clear → Opaque → Transparent → Resolve
        let pass_desc = wgpu::RenderPassDescriptor {
            label: Some("Simple Forward Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: final_resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(self.clear_color),
                    store: store_op,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    // Reverse Z: Clear to 0.0 (far clipping plane)
                    load: wgpu::LoadOp::Clear(0.0),
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

        // 设置全局 BindGroup
        tracked_pass.set_bind_group(
            0,
            render_lists.gpu_global_bind_group_id,
            gpu_global_bind_group,
            &[],
        );

        // 先绘制不透明物体（Front-to-Back）
        Self::draw_list(ctx, &mut tracked_pass, &render_lists.opaque);

        // 再绘制透明物体（Back-to-Front）
        Self::draw_list(ctx, &mut tracked_pass, &render_lists.transparent);

        // RenderPass 结束时自动 resolve（如果配置了 resolve_target）
    }
}
