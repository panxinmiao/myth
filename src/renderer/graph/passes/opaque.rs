//! Opaque Render Pass
//!
//! Only draws opaque objects, used in PBR/HDR rendering path.
//!
//! # Data Flow
//! ```text
//! RenderLists.opaque → OpaquePass → HDR Scene Color
//! ```
//!
//! # `RenderPass` Configuration
//! - `LoadOp`: Clear (clear color and depth)
//! - `StoreOp`: Store (store results for subsequent passes)

use crate::renderer::graph::frame::RenderCommand;
use crate::renderer::graph::{RenderNode, TrackedRenderPass};
use crate::renderer::graph::context::ExecuteContext;

/// Opaque Render Pass
///
/// Only draws objects in `render_lists.opaque`.
/// Clears color and depth buffers, results are stored for subsequent passes.
///
/// # Performance Considerations
/// - Command lists are sorted by Pipeline > Material > Depth to minimize state changes
/// - Front-to-Back sorting leverages Early-Z culling
pub struct OpaquePass {
    /// Clear color
    pub clear_color: wgpu::Color,
}

impl OpaquePass {
    #[must_use]
    pub fn new(clear_color: wgpu::Color) -> Self {
        Self { clear_color }
    }

    /// Determine render target views based on MSAA settings.
    fn get_render_target<'a>(
        ctx: &'a ExecuteContext,
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

impl Default for OpaquePass {
    fn default() -> Self {
        Self::new(wgpu::Color::BLACK)
    }
}

impl RenderNode for OpaquePass {
    fn name(&self) -> &'static str {
        "Opaque Pass"
    }

    fn run(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let render_lists = &ctx.render_lists;

        // Get global BindGroup
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
                resolve_target: None, // Opaque Pass does not resolve, wait for Transparent Pass to complete
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
