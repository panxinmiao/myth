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

use crate::renderer::core::gpu::{ScreenBindGroupInfo, Tracked};
use crate::renderer::graph::core::{
    ExecuteContext, PassNode, PrepareContext, RenderGraph, RenderTargetOps, TextureNodeId,
};
use crate::renderer::graph::passes::draw::submit_draw_commands;

// ─── Feature ───────────────────────────────────────────────────────────

pub struct TransparentFeature {
    screen_info: Option<ScreenBindGroupInfo>,
}

impl Default for TransparentFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl TransparentFeature {
    #[must_use]
    pub fn new() -> Self {
        Self { screen_info: None }
    }

    /// Cache screen bind group info from ResourceManager.
    pub fn set_screen_info(&mut self, info: ScreenBindGroupInfo) {
        self.screen_info = Some(info);
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
    pub fn add_to_graph(
        &self,
        graph: &mut RenderGraph,
        color_target: TextureNodeId,
        depth_target: TextureNodeId,
        transmission_tex: Option<TextureNodeId>,
        ssao_tex: Option<TextureNodeId>,
        shadow_tex: Option<TextureNodeId>,
    ) -> TextureNodeId {
        let fc = *graph.frame_config();
        let screen_info = self
            .screen_info
            .clone()
            .expect("TransparentFeature: screen_info not set");

        graph.add_pass("Transparent_Pass", |builder| {
            // SSA colour relay: read the incoming version, write a new alias.
            let color_output =
                builder.mutate_and_export(color_target, "Scene_Color_Transparent");

            let resolve_target = if fc.msaa_samples > 1 {
                let desc = fc.create_render_target_desc(
                    wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                );
                Some(builder.create_and_export("Scene_Color_HDR_Final", desc))
            } else {
                None
            };

            // Depth — read-only for depth testing.
            builder.read_texture(depth_target);

            // Optional texture inputs.
            if let Some(tx) = transmission_tex {
                builder.read_texture(tx);
            }
            if let Some(ssao) = ssao_tex {
                builder.read_texture(ssao);
            }
            if let Some(shadow) = shadow_tex {
                builder.read_texture(shadow);
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
                screen_info,
            );

            (node, result)
        })
    }
}

// ─── Pass Node ─────────────────────────────────────────────────────────

/// RDG Transparent Render Pass.
///
/// Draws `render_lists.transparent` with back-to-front sorting.  Builds a
/// screen bind group (group 3) containing the real transmission texture
/// (if available) and SSAO view.  When MSAA is active, this pass is
/// typically the last user of the multi-sampled colour surface, triggering
/// automatic `StoreOp::Discard` via the RDG lifetime system.
pub struct TransparentPassNode {
    // ─── RDG Resource Slots (SSA model) ────────────────────────────
    /// New colour version — SSA alias of the input (write dependency).
    out_color: TextureNodeId,
    /// Depth buffer (read-only for depth testing).
    depth_target: TextureNodeId,
    /// Optional single-sample HDR texture for MSAA resolve.
    resolve_target: Option<TextureNodeId>,
    /// Optional transmission texture input.
    transmission_input: Option<TextureNodeId>,
    /// Optional SSAO texture input.
    ssao_input: Option<TextureNodeId>,
    /// Optional shadow map input (DAG dependency on ShadowPass).
    shadow_input: Option<TextureNodeId>,
    // ─── Screen Bind Group Infrastructure ──────────────────────────
    screen_info: ScreenBindGroupInfo,
    // ─── Internal Cache ────────────────────────────────────────────
    screen_bind_group: Option<wgpu::BindGroup>,
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
        screen_info: ScreenBindGroupInfo,
    ) -> Self {
        Self {
            out_color,
            depth_target,
            resolve_target,
            transmission_input,
            ssao_input,
            shadow_input,
            screen_info,
            screen_bind_group: None,
        }
    }
}

impl PassNode for TransparentPassNode {
    fn prepare(&mut self, ctx: &mut PrepareContext) {
        let ssao_view: &Tracked<wgpu::TextureView> = match self.ssao_input {
            Some(id) => ctx.views.get_texture_view(id),
            None => &self.screen_info.ssao_dummy_view,
        };

        let transmission_view: &Tracked<wgpu::TextureView> = match self.transmission_input {
            Some(id) => ctx.views.get_texture_view(id),
            None => &self.screen_info.dummy_transmission_view,
        };

        let shadow_view: &Tracked<wgpu::TextureView> = match self.shadow_input {
            Some(id) => ctx.views.get_texture_view(id),
            None => &self.screen_info.dummy_shadow_view,
        };

        let (bg, _) = self.screen_info.build_screen_bind_group(
            ctx.device,
            ctx.global_bind_group_cache,
            transmission_view,
            ssao_view,
            shadow_view,
        );
        self.screen_bind_group = Some(bg);
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let gpu_global_bind_group = ctx.baked_lists.global_bind_group;

        // `out_color` is an alias — LoadOp is auto-deduced to `Load`,
        // inheriting the content rendered by prior passes.
        let color_att = ctx.get_color_attachment(self.out_color, RenderTargetOps::Load, self.resolve_target);
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
            let screen_bg = self.screen_bind_group.as_ref().unwrap();
            pass.set_bind_group(3, screen_bg, &[]);

            submit_draw_commands(&mut pass, &ctx.baked_lists.transparent);
        }
    }
}
