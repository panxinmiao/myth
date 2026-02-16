//! Opaque Render Pass
//!
//! 仅绘制不透明物体的 Pass，用于 PBR/HDR 渲染路径。
//!
//! # 数据流
//! ```text
//! RenderLists.opaque → OpaquePass → HDR Scene Color
//! ```
//!
//! # `RenderPass` 配置
//! - `LoadOp`: Clear (清空颜色和深度)
//! - `StoreOp`: Store (保留结果供后续 Pass 使用)

use crate::renderer::graph::frame::RenderCommand;
use crate::renderer::graph::{RenderContext, RenderNode, TrackedRenderPass};

/// Opaque Render Pass
///
/// 仅绘制 `render_lists.opaque` 中的物体。
/// 清空颜色缓冲和深度缓冲，结果存储供后续 Pass 使用。
///
/// # 性能考虑
/// - 命令列表按 Pipeline > Material > Depth 排序，最小化状态切换
/// - Front-to-Back 排序利用 Early-Z 剔除
pub struct OpaquePass {
    /// Clear color
    pub clear_color: wgpu::Color,
}

impl OpaquePass {
    #[must_use]
    pub fn new(clear_color: wgpu::Color) -> Self {
        Self { clear_color }
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

impl Default for OpaquePass {
    fn default() -> Self {
        Self::new(wgpu::Color::BLACK)
    }
}

impl RenderNode for OpaquePass {
    fn name(&self) -> &'static str {
        "Opaque Pass"
    }

    fn run(&self, ctx: &mut RenderContext, encoder: &mut wgpu::CommandEncoder) {
        let render_lists = &ctx.render_lists;

        // 获取全局 BindGroup
        let Some(gpu_global_bind_group) = &render_lists.gpu_global_bind_group else {
            log::warn!("OpaquePass: gpu_global_bind_group missing, skipping");
            return;
        };

        let (color_view, _resolve_target) = Self::get_render_target(ctx);
        let depth_view = &ctx.frame_resources.depth_view;

        // Use scene background color for clearing.
        // When a skybox pass follows, the clear color only shows through
        // debug visualization; otherwise it is the final background.
        let clear_color = ctx.scene.background.clear_color();

        let pass_desc = wgpu::RenderPassDescriptor {
            label: Some("Opaque Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None, // Opaque Pass 不 resolve，等 Transparent Pass 完成后再 resolve
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear_color),
                    store: wgpu::StoreOp::Store,
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

        tracked_pass.set_bind_group(
            0,
            render_lists.gpu_global_bind_group_id,
            gpu_global_bind_group,
            &[],
        );

        Self::draw_list(ctx, &mut tracked_pass, &render_lists.opaque);
    }
}
