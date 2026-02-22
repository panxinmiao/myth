//! Simple Forward Render Pass
//!
//! Simplified Forward Pass for LDR/non-HDR rendering paths.
//! Draws opaque and transparent objects in a single `RenderPass`.
//!
//! # Data Flow
//! ```text
//! RenderLists (from SceneCullPass) → SimpleForwardPass → Surface/LDR Target
//! ```
//!
//! # Use Cases
//! - Low-end mode
//! - UI scenes
//! - Non-PBR scenes
//! - Scenes without Transmission effects

use crate::render::PrepareContext;
use crate::renderer::core::resources::Tracked;
use crate::renderer::graph::context::{ExecuteContext, GraphResource};
use crate::renderer::graph::frame::RenderCommand;
use crate::renderer::graph::{RenderNode, TrackedRenderPass};

/// Simple Forward Render Pass
///
/// In a single `RenderPass`, it completes all drawing:
/// 1. Clear color and depth buffers
/// 2. Draw opaque objects (Front-to-Back)
/// 3. Draw transparent objects (Back-to-Front)
///
/// # Performance Considerations
/// - Use `TrackedRenderPass` to avoid redundant state changes
/// - Command lists are pre-sorted, no additional sorting overhead
pub struct SimpleForwardPass {
    /// Clear color
    pub clear_color: wgpu::Color,

    /// Cached render target views during prepare phase.
    msaa_view: Option<Tracked<wgpu::TextureView>>,
    depth_view: Option<Tracked<wgpu::TextureView>>,
}

impl SimpleForwardPass {
    #[must_use]
    pub fn new(clear_color: wgpu::Color) -> Self {
        Self {
            clear_color,
            msaa_view: None,
            depth_view: None,
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

impl Default for SimpleForwardPass {
    fn default() -> Self {
        Self::new(wgpu::Color::BLACK)
    }
}

impl RenderNode for SimpleForwardPass {
    fn name(&self) -> &'static str {
        "Simple Forward Pass"
    }

    fn prepare(&mut self, ctx: &mut PrepareContext) {
        self.clear_color = ctx.extracted_scene.background.clear_color();

        self.msaa_view = ctx.try_get_resource_view(GraphResource::SceneMsaa).cloned();
        self.depth_view = Some(ctx.get_resource_view(GraphResource::SceneDepth).clone());
    }

    fn run(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let render_lists = &ctx.render_lists;

        // get global BindGroup
        let Some(gpu_global_bind_group) = &render_lists.gpu_global_bind_group else {
            log::warn!("SimpleForwardPass: gpu_global_bind_group missing, skipping");
            return;
        };

        let target_view = ctx.surface_view;

        let (color_view, resolve_target) = if let Some(msaa) = self.msaa_view.as_deref() {
            (msaa, Some(target_view))
        } else {
            (target_view, None)
        };

        let depth_view = self.depth_view.as_ref().unwrap();

        let store_op = if resolve_target.is_some() {
            wgpu::StoreOp::Discard
        } else {
            wgpu::StoreOp::Store
        };

        // Single RenderPass: Clear → Opaque → Transparent → Resolve
        let pass_desc = wgpu::RenderPassDescriptor {
            label: Some("Simple Forward Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
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

        // setup global bind group (camera, lights, etc.)
        tracked_pass.set_bind_group(
            0,
            render_lists.gpu_global_bind_group_id,
            gpu_global_bind_group,
            &[],
        );

        // draw opaque objects first (Front-to-Back)
        Self::draw_list(ctx, &mut tracked_pass, &render_lists.opaque);

        // ── Skybox: draw between opaque and transparent ──────────────
        //
        // In the LDR path, SkyboxPass::prepare() stores the prepared pipeline
        // and bind group in render_lists.prepared_skybox. We draw the skybox
        // here using the raw pass to bypass TrackedRenderPass state tracking,
        // then invalidate cached state so that transparent draws re-set
        // everything correctly.
        if let Some(skybox) = &ctx.render_lists.prepared_skybox {
            let raw = tracked_pass.raw_pass();

            skybox.draw(raw, gpu_global_bind_group);

            // Invalidate all tracked state — the skybox used a completely
            // different pipeline and bind group layout at slot 0.
            tracked_pass.invalidate_state();

            // Re-set global bind group for subsequent transparent draws.
            tracked_pass.set_bind_group(
                0,
                render_lists.gpu_global_bind_group_id,
                gpu_global_bind_group,
                &[],
            );
        }

        // Then draw transparent objects (Back-to-Front)
        Self::draw_list(ctx, &mut tracked_pass, &render_lists.transparent);

        // RenderPass will automatically resolve MSAA if resolve_target is Some, and end when dropped
    }
}
