//! RDG Transparent Render Pass
//!
//! Draws transparent objects to the scene color buffer. Inherits both
//! color and depth from prior passes (Opaque, Skybox) via `LoadOp::Load`.
//!
//! # MSAA Support
//!
//! When hardware MSAA is active in the HighFidelity path, the Composer
//! passes multi-sampled `color_target` / `depth_target` IDs and an optional
//! `resolve_target`.  This pass continues rendering into the same MSAA
//! surface used by Opaque and Skybox, then resolves to the single-sample
//! HDR buffer.  If this pass is the last user of the MSAA surface, the
//! RDG lifetime system automatically deduces `StoreOp::Discard`, releasing
//! the large multi-sampled allocation with zero VRAM bandwidth waste.
//!
//! # RDG Slots
//!
//! - `color_target`: Scene color buffer — `Scene_Color_HDR` or
//!   `Scene_Color_MSAA` (read + write, LoadOp::Load)
//! - `depth_target`: Scene depth buffer — `Scene_Depth` or
//!   `Scene_Depth_MSAA` (read, LoadOp::Load)
//! - `resolve_target`: Optional single-sample HDR to receive MSAA resolve
//! - `transmission_tex`: Optional transmission texture (read, for screen bind group)
//!
//! # Draw Order
//!
//! Transparent commands are sorted back-to-front for correct alpha blending.

use crate::renderer::core::resources::{ScreenBindGroupInfo, Tracked};
use crate::renderer::graph::rdg::builder::PassBuilder;
use crate::renderer::graph::rdg::context::{RdgExecuteContext, RdgPrepareContext};
use crate::renderer::graph::rdg::draw::submit_draw_commands;
use crate::renderer::graph::rdg::node::PassNode;
use crate::renderer::graph::rdg::types::TextureNodeId;

use super::graph::RenderGraph;

// ─── Feature ───────────────────────────────────────────────────────────

pub struct TransparentFeature {
    screen_info: Option<ScreenBindGroupInfo>,
}

impl TransparentFeature {
    pub fn new() -> Self {
        Self { screen_info: None }
    }

    /// Cache screen bind group info from ResourceManager.
    pub fn set_screen_info(&mut self, info: ScreenBindGroupInfo) {
        self.screen_info = Some(info);
    }

    pub fn add_to_graph(
        &self,
        rdg: &mut RenderGraph,
        color_target: TextureNodeId,
        depth_target: TextureNodeId,
        resolve_target: Option<TextureNodeId>,
    ) {
        let node = TransparentPassNode::new(
            color_target,
            depth_target,
            resolve_target,
            self.screen_info.clone().expect("TransparentFeature: screen_info not set"),
        );
        rdg.add_pass(Box::new(node));
    }
}

// ─── Pass Node ─────────────────────────────────────────────────────────

/// RDG Transparent Render Pass.
///
/// Draws `render_lists.transparent` with back-to-front sorting. Builds a
/// screen bind group (group 3) containing the real transmission texture
/// (if available) and SSAO view.  When MSAA is active, this pass is
/// typically the last user of `Scene_Color_MSAA` / `Scene_Depth_MSAA`,
/// triggering automatic `StoreOp::Discard` via the RDG lifetime system.
pub struct TransparentPassNode {
    // ─── RDG Resource Slots ────────────────────────────────────────
    /// Primary color target (`Scene_Color_HDR` or `Scene_Color_MSAA`).
    pub color_target: TextureNodeId,
    /// Primary depth target (`Scene_Depth` or `Scene_Depth_MSAA`).
    pub depth_target: TextureNodeId,
    /// Optional single-sample HDR texture for MSAA resolve.
    pub resolve_target: Option<TextureNodeId>,
    pub transmission_tex: TextureNodeId,
    pub ssao_tex: TextureNodeId,

    // ─── Push Parameters ───────────────────────────────────────────
    pub has_transmission: bool,
    pub ssao_enabled: bool,
    // ─── Screen Bind Group Infrastructure ──────────────────────────
    screen_info: ScreenBindGroupInfo,
    // ─── Internal Cache ────────────────────────────────────────────
    screen_bind_group: Option<wgpu::BindGroup>,
}

impl TransparentPassNode {
    #[must_use]
    pub fn new(
        color_target: TextureNodeId,
        depth_target: TextureNodeId,
        resolve_target: Option<TextureNodeId>,
        screen_info: ScreenBindGroupInfo,
    ) -> Self {
        Self {
            color_target,
            depth_target,
            resolve_target,
            transmission_tex: TextureNodeId(0),
            ssao_tex: TextureNodeId(0),
            has_transmission: false,
            ssao_enabled: false,
            screen_info,
            screen_bind_group: None,
        }
    }
}

impl PassNode for TransparentPassNode {
    fn name(&self) -> &'static str {
        "RDG_Transparent_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        // Primary color target — read + write (Load, then blend transparent).
        builder.read_texture(self.color_target);
        builder.write_texture(self.color_target);

        // Depth target — read-only for depth testing.
        builder.read_texture(self.depth_target);

        // Resolve target — declare write so the graph compiler allocates
        // physical memory and tracks dependencies for downstream consumers.
        if let Some(rt) = self.resolve_target {
            builder.write_texture(rt);
        }

        // Detect optional transmission resource.
        if let Some(tx) = builder.try_read_blackboard("Transmission_Tex") {
            self.transmission_tex = tx;
            self.has_transmission = true;
        } else {
            self.has_transmission = false;
        }

        // Detect optional SSAO resource.
        if let Some(ssao) = builder.try_read_blackboard("SSAO_Output") {
            self.ssao_tex = ssao;
            self.ssao_enabled = true;
        } else {
            self.ssao_enabled = false;
        }
    }

    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {
        // Build screen bind group (group 3)
        // TransparentPass uses the real transmission texture if available.
        let ssao_view: &Tracked<wgpu::TextureView> = if self.ssao_enabled {
            ctx.views.get_texture_view(self.ssao_tex)
        } else {
            &self.screen_info.ssao_dummy_view
        };

        let transmission_view: &Tracked<wgpu::TextureView> = if self.has_transmission {
            ctx.views.get_texture_view(self.transmission_tex)
        } else {
            &self.screen_info.dummy_transmission_view
        };

        let (bg, _) = self.screen_info.build_screen_bind_group(
            ctx.device,
            ctx.global_bind_group_cache,
            transmission_view,
            ssao_view,
        );
        self.screen_bind_group = Some(bg);
    }

    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let gpu_global_bind_group = ctx.baked_lists.global_bind_group;

        // Auto-deduced ops: color_target inherits from prior passes (Load),
        // and is Discarded on last use or Stored for downstream; depth is
        // read-only here.  The optional resolve_target receives the MSAA
        // resolve at the end of the render pass.
        let color_att = ctx.get_color_attachment(
            self.color_target,
            None,
            self.resolve_target,
        );
        let depth_att = ctx.get_depth_stencil_attachment(self.depth_target, 0.0);

        let pass_desc = wgpu::RenderPassDescriptor {
            label: Some("RDG Transparent Pass"),
            color_attachments: &[color_att],
            depth_stencil_attachment: depth_att,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };

        let raw_pass = encoder.begin_render_pass(&pass_desc);
        let mut pass = raw_pass;

        pass.set_bind_group(
            0,
            gpu_global_bind_group,
            &[],
        );

        if !ctx.baked_lists.transparent.is_empty() {
            let screen_bg = self.screen_bind_group.as_ref().unwrap();
            pass.set_bind_group(3, screen_bg, &[]);

            submit_draw_commands(&mut pass, &ctx.baked_lists.transparent);
        }
    }
}
