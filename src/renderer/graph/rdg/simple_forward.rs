//! RDG Simple Forward Render Pass
//!
//! Single-pass LDR rendering for the [`BasicForward`] path. Combines opaque,
//! skybox, and transparent drawing into one `wgpu::RenderPass` with optional
//! hardware MSAA.
//!
//! # RDG Slots
//!
//! - `surface_out`: LDR colour output (write, Clear)
//! - `scene_depth`: Depth buffer (write, Clear)
//!
//! # Push Parameters
//!
//! - `clear_color`: Background clear colour (from scene settings)
//!
//! # Rendering Order
//!
//! 1. **Clear** colour and depth
//! 2. **Opaque** objects (front-to-back)
//! 3. **Skybox** (drawn behind opaque geometry via Reverse-Z)
//! 4. **Transparent** objects (back-to-front)
//!
//! [`BasicForward`]: crate::renderer::settings::RenderPath::BasicForward

use crate::renderer::core::resources::Tracked;
use crate::renderer::graph::frame::RenderCommand;
use crate::renderer::graph::rdg::builder::PassBuilder;
use crate::renderer::graph::rdg::context::{RdgExecuteContext, RdgPrepareContext};
use crate::renderer::graph::rdg::node::PassNode;
use crate::renderer::graph::rdg::types::TextureNodeId;
use crate::renderer::graph::TrackedRenderPass;

/// RDG Simple Forward Render Pass.
///
/// Draws the entire scene in a single LDR render pass with optional MSAA,
/// intended for the [`BasicForward`] rendering path. The pass writes
/// directly to the swap-chain surface via `surface_out`.
///
/// [`BasicForward`]: crate::renderer::settings::RenderPath::BasicForward
pub struct RdgSimpleForwardPass {
    // ─── RDG Resource Slots (set by Composer) ────────────────────────
    pub surface_out: TextureNodeId,
    pub scene_depth: TextureNodeId,

    // ─── Push Parameters ─────────────────────────────────────────────
    pub clear_color: wgpu::Color,

    // ─── Internal Cache ──────────────────────────────────────────────
    msaa_view: Option<Tracked<wgpu::TextureView>>,
    screen_bind_group: Option<wgpu::BindGroup>,
    screen_bind_group_id: u64,
}

impl RdgSimpleForwardPass {
    #[must_use]
    pub fn new() -> Self {
        Self {
            surface_out: TextureNodeId(0),
            scene_depth: TextureNodeId(0),
            clear_color: wgpu::Color::BLACK,
            msaa_view: None,
            screen_bind_group: None,
            screen_bind_group_id: 0,
        }
    }

    /// Draws a batch of render commands using [`TrackedRenderPass`] state tracking.
    fn draw_list<'pass>(
        ctx: &'pass RdgExecuteContext,
        pass: &mut TrackedRenderPass<'pass>,
        cmds: &'pass [RenderCommand],
        screen_bind_group: &'pass wgpu::BindGroup,
        screen_bind_group_id: u64,
    ) {
        if cmds.is_empty() {
            return;
        }

        for cmd in cmds {
            let pipeline = ctx.pipeline_cache.get_render_pipeline(cmd.pipeline_id);
            pass.set_pipeline(cmd.pipeline_id.0, pipeline);

            if let Some(gpu_material) = ctx.resource_manager.get_material(cmd.material_handle) {
                pass.set_bind_group(1, gpu_material.bind_group_id, &gpu_material.bind_group, &[]);
            }

            pass.set_bind_group(
                2,
                cmd.object_bind_group.bind_group_id,
                &cmd.object_bind_group.bind_group,
                &[cmd.dynamic_offset],
            );

            pass.set_bind_group(3, screen_bind_group_id, screen_bind_group, &[]);

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

impl PassNode for RdgSimpleForwardPass {
    fn name(&self) -> &'static str {
        "RDG_SimpleForward_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        builder.write_texture(self.surface_out);
        builder.write_texture(self.scene_depth);
    }

    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {
        self.clear_color = ctx.extracted_scene.background.clear_color();

        // Cache MSAA intermediate view (if MSAA is active)
        self.msaa_view = ctx.frame_resources.scene_msaa_view.clone();

        // Build screen bind group (group 3) with dummy textures (LDR path
        // has no SSAO or transmission).
        let (bg, bg_id) = ctx.frame_resources.build_screen_bind_group(
            ctx.device,
            ctx.global_bind_group_cache,
            &ctx.frame_resources.dummy_transmission_view,
            &ctx.frame_resources.ssao_dummy_view,
        );
        self.screen_bind_group = Some(bg);
        self.screen_bind_group_id = bg_id;
    }

    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let render_lists = &ctx.render_lists;

        let Some(gpu_global_bind_group) = &render_lists.gpu_global_bind_group else {
            log::warn!("RDG SimpleForwardPass: gpu_global_bind_group missing, skipping");
            return;
        };

        let target_view = ctx.get_texture_view(self.surface_out);
        let depth_view = ctx.get_texture_view(self.scene_depth);

        let (color_view, resolve_target) = if let Some(msaa) = self.msaa_view.as_deref() {
            (msaa as &wgpu::TextureView, Some(target_view))
        } else {
            (target_view, None)
        };

        let store_op = if resolve_target.is_some() {
            wgpu::StoreOp::Discard
        } else {
            wgpu::StoreOp::Store
        };

        // Single RenderPass: Clear → Opaque → Skybox → Transparent → Resolve
        let pass_desc = wgpu::RenderPassDescriptor {
            label: Some("RDG Simple Forward Pass"),
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
                    // Reverse-Z: far plane is 0.0
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

        // Set global bind group (group 0: camera, lights, etc.)
        tracked_pass.set_bind_group(
            0,
            render_lists.gpu_global_bind_group_id,
            gpu_global_bind_group,
            &[],
        );

        // 1. Opaque (front-to-back)
        let screen_bg = self.screen_bind_group.as_ref().unwrap();
        Self::draw_list(
            ctx,
            &mut tracked_pass,
            &render_lists.opaque,
            screen_bg,
            self.screen_bind_group_id,
        );

        // 2. Skybox (between opaque and transparent)
        if let Some(skybox) = &render_lists.prepared_skybox {
            let raw = tracked_pass.raw_pass();
            skybox.draw(raw, gpu_global_bind_group, ctx.pipeline_cache);

            // Invalidate tracked state — skybox uses a different pipeline and
            // bind group layout at slot 0.
            tracked_pass.invalidate_state();

            // Re-set global bind group for subsequent transparent draws.
            tracked_pass.set_bind_group(
                0,
                render_lists.gpu_global_bind_group_id,
                gpu_global_bind_group,
                &[],
            );
        }

        // 3. Transparent (back-to-front)
        Self::draw_list(
            ctx,
            &mut tracked_pass,
            &render_lists.transparent,
            screen_bg,
            self.screen_bind_group_id,
        );

        // RenderPass automatically resolves MSAA on drop if resolve_target is set.
    }
}
