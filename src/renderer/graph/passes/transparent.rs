//! Transparent Render Pass
//!
//! draw transparent objects, used in PBR/HDR rendering path.
//!
//! # Data Flow
//! ```text
//! RenderLists.transparent → TransparentPass → HDR Scene Color
//! ```
//!
//! # `RenderPass` Configuration
//! - `LoadOp`: Load (inherit results from `OpaquePass`)
//! - `StoreOp`: Store (retain results for post-processing)
//!
//! # Note
//! This pass runs after `OpaquePass` and an optional `TransmissionCopyPass`.

use crate::renderer::graph::context::{ExecuteContext, GraphResource};
use crate::renderer::graph::frame::RenderCommand;
use crate::renderer::graph::{RenderNode, TrackedRenderPass};

/// Transparent Render Pass
///
/// Only draws objects in `render_lists.transparent`.
/// Inherits results from the opaque pass, results are stored for post-processing.
///
/// # Performance Considerations
/// - Command lists are sorted by Depth (Back-to-Front) to ensure correct Alpha blending
/// - If there are Transmission effects, this Pass runs after `TransmissionCopyPass`
pub struct TransparentPass;

impl TransparentPass {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Determine render target views based on MSAA settings.
    fn get_render_target<'a>(
        ctx: &'a ExecuteContext,
    ) -> (&'a wgpu::TextureView, Option<&'a wgpu::TextureView>) {
        let target_view = ctx.get_scene_render_target_view();

        if let Some(msaa_view) = ctx.try_get_resource_view(GraphResource::SceneMsaa) {
            (msaa_view, Some(target_view))
        } else {
            (target_view, None)
        }
    }

    /// Execute the draw list
    fn draw_list<'pass>(
        ctx: &'pass ExecuteContext,
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

    fn run(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let render_lists = &ctx.render_lists;

        // Get global BindGroup (needed even if there are no transparent objects, for potential resolve)
        let Some(gpu_global_bind_group) = &render_lists.gpu_global_bind_group else {
            log::warn!("TransparentPass: gpu_global_bind_group missing, skipping");
            return;
        };

        let (color_view, resolve_target) = Self::get_render_target(ctx);
        let depth_view = ctx.get_resource_view(GraphResource::SceneDepth);

        // Determine final store/resolve configuration
        let (store_op, final_resolve_target) = if resolve_target.is_some() {
            // MSAA: resolve at the end, do not store MSAA buffer
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
                    load: wgpu::LoadOp::Load, // Inherit results from OpaquePass
                    store: store_op,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load, // Inherit depth
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

        // Only draw when there are transparent objects
        if !render_lists.transparent.is_empty() {
            Self::draw_list(ctx, &mut tracked_pass, &render_lists.transparent);
        }
        // Even if there are no transparent objects, this pass is needed to complete MSAA resolve
    }
}
