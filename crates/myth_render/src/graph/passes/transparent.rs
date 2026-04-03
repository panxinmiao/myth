//! RDG Transparent Render Pass
//!
//! Draws transparent objects to the scene color buffer.  The colour target
//! is received as an SSA (Static Single Assignment) alias: the pass reads
//! the previous colour version and writes a *new* logical version that
//! shares the same physical GPU memory, producing a clean DAG edge
//! (e.g. Skybox → Transparent) without reliance on `add_pass` order.
//!
//! # MSAA Support
//!
//! When hardware MSAA is active in the HighFidelity path, this pass
//! continues rendering into the MSAA surface and resolves to a dedicated
//! single-sample HDR buffer (`Scene_Color_HDR_Final`).  If this pass is
//! the last user of the MSAA surface, the RDG lifetime system
//! automatically deduces `StoreOp::Discard`, releasing the large
//! multi-sampled allocation with zero VRAM bandwidth waste.
//!
//! # RDG Slots (explicit wiring, SSA model)
//!
//! | Slot              | Direction | Notes |
//! |--------------------|-----------|-------|
//! | `in_color`         | read      | Previous colour version (from Skybox / Opaque) |
//! | `out_color`        | write     | New colour version (SSA alias of `in_color`) |
//! | `depth_target`     | read      | Depth buffer for depth testing |
//! | `resolve_target`   | write     | Optional single-sample HDR for MSAA resolve |
//! | `transmission_input` | read    | Optional transmission texture |
//! | `ssao_input`       | read      | Optional SSAO texture |
//!
//! # Draw Order
//!
//! Transparent commands are sorted back-to-front for correct alpha blending.

use crate::core::gpu::Tracked;
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    ExecuteContext, PassNode, PrepareContext, RenderTargetOps, TextureNodeId,
    build_screen_bind_group,
};
use crate::graph::passes::draw::submit_draw_commands;

// ─── Feature ───────────────────────────────────────────────────────────

pub struct TransparentFeature;

impl Default for TransparentFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl TransparentFeature {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Builds the transparent pass node and inserts it into the graph.
    ///
    /// Creates an SSA alias of `color_target` so that the dependency on
    /// the previous colour writer (Skybox / Opaque) is locked by graph
    /// edges.  In MSAA mode a dedicated single-sample resolve target is
    /// also registered.
    ///
    /// Returns the [`TextureNodeId`] that downstream consumers (Bloom,
    /// ToneMap, hooks) should read:
    /// - **MSAA**: the resolve target (`Scene_Color_HDR_Final`).
    /// - **Non-MSAA**: the mutated colour alias.
    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        color_target: TextureNodeId,
        depth_target: TextureNodeId,
        transmission_tex: Option<TextureNodeId>,
        ssao_tex: Option<TextureNodeId>,
        shadow_tex: Option<TextureNodeId>,
        shadow_cube_tex: Option<TextureNodeId>,
    ) -> TextureNodeId {
        let fc = ctx.frame_config;

        ctx.graph.add_pass("Transparent_Pass", |builder| {
            let color_output = builder.mutate_texture(color_target, "Scene_Color_Transparent");

            let resolve_target = if fc.msaa_samples > 1 {
                let desc = fc.create_render_target_desc(
                    wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                );
                Some(builder.create_texture("Scene_Color_HDR_Final", desc))
            } else {
                None
            };

            builder.read_texture(depth_target);

            if let Some(tx) = transmission_tex {
                builder.read_texture(tx);
            }
            if let Some(ssao) = ssao_tex {
                builder.read_texture(ssao);
            }
            if let Some(shadow) = shadow_tex {
                builder.read_texture(shadow);
            }
            if let Some(shadow_cube) = shadow_cube_tex {
                builder.read_texture(shadow_cube);
            }

            let result = resolve_target.unwrap_or(color_output);

            let node = TransparentPassNode::new(
                color_target,
                color_output,
                depth_target,
                resolve_target,
                transmission_tex,
                ssao_tex,
                shadow_tex,
                shadow_cube_tex,
            );

            (node, result)
        })
    }
}

// ─── Pass Node ─────────────────────────────────────────────────────────

pub struct TransparentPassNode {
    out_color: TextureNodeId,
    depth_target: TextureNodeId,
    resolve_target: Option<TextureNodeId>,
    transmission_input: Option<TextureNodeId>,
    ssao_input: Option<TextureNodeId>,
    shadow_input: Option<TextureNodeId>,
    shadow_cube_input: Option<TextureNodeId>,
    screen_bind_group: Option<&'static wgpu::BindGroup>,
}

impl TransparentPassNode {
    #[must_use]
    fn new(
        _in_color: TextureNodeId,
        out_color: TextureNodeId,
        depth_target: TextureNodeId,
        resolve_target: Option<TextureNodeId>,
        transmission_input: Option<TextureNodeId>,
        ssao_input: Option<TextureNodeId>,
        shadow_input: Option<TextureNodeId>,
        shadow_cube_input: Option<TextureNodeId>,
    ) -> Self {
        Self {
            out_color,
            depth_target,
            resolve_target,
            transmission_input,
            ssao_input,
            shadow_input,
            shadow_cube_input,
            screen_bind_group: None,
        }
    }
}

impl<'a> PassNode<'a> for TransparentPassNode {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let PrepareContext {
            views,
            global_bind_group_cache: cache,
            device,
            system_textures: sys,
            ..
        } = ctx;
        let device = *device;

        let d2array_key = crate::graph::core::allocator::SubViewKey {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        };
        if let Some(id) = self.shadow_input {
            views.get_or_create_sub_view(id, &d2array_key);
        }

        // Pre-create CubeArray sub-view (mutable borrow, result dropped).
        let cube_key = crate::graph::core::allocator::SubViewKey {
            dimension: Some(wgpu::TextureViewDimension::CubeArray),
            ..Default::default()
        };
        if let Some(id) = self.shadow_cube_input {
            views.get_or_create_sub_view(id, &cube_key);
        }

        // All remaining borrows are immutable.
        let ssao_view: &Tracked<wgpu::TextureView> = match self.ssao_input {
            Some(id) => views.get_texture_view(id),
            None => &sys.white_r8,
        };

        let transmission_view: &Tracked<wgpu::TextureView> = match self.transmission_input {
            Some(id) => views.get_texture_view(id),
            None => &sys.black_hdr,
        };

        let shadow_view: &Tracked<wgpu::TextureView> = match self.shadow_input {
            Some(id) => views.get_sub_view(id, &d2array_key).unwrap(),
            None => &sys.depth_d2array,
        };

        let shadow_cube_view: &Tracked<wgpu::TextureView> = match self.shadow_cube_input {
            Some(id) => views.get_sub_view(id, &cube_key).unwrap(),
            None => &sys.depth_cube_array,
        };

        let bg = build_screen_bind_group(
            cache,
            device,
            sys,
            transmission_view,
            ssao_view,
            shadow_view,
            shadow_cube_view,
        );
        self.screen_bind_group = Some(bg);
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let gpu_global_bind_group = ctx.baked_lists.global_bind_group;

        let color_att =
            ctx.get_color_attachment(self.out_color, RenderTargetOps::Load, self.resolve_target);
        let depth_att = ctx.get_depth_stencil_attachment(self.depth_target, 0.0);

        let pass_desc = wgpu::RenderPassDescriptor {
            label: Some("Transparent Pass"),
            color_attachments: &[color_att],
            depth_stencil_attachment: depth_att,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };

        let raw_pass = encoder.begin_render_pass(&pass_desc);
        let mut pass = raw_pass;

        pass.set_bind_group(0, gpu_global_bind_group, &[]);

        if !ctx.baked_lists.transparent.is_empty() {
            let screen_bg = self.screen_bind_group.unwrap();
            pass.set_bind_group(3, screen_bg, &[]);

            submit_draw_commands(&mut pass, &ctx.baked_lists.transparent);
        }
    }
}
