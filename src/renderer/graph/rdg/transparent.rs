//! RDG Transparent Render Pass
//!
//! Draws transparent objects to the HDR scene color buffer. Inherits both
//! color and depth from prior passes (Opaque, Skybox) via `LoadOp::Load`.
//!
//! # RDG Slots
//!
//! - `scene_color`: HDR color buffer (read + write, LoadOp::Load)
//! - `scene_depth`: Depth buffer (read, LoadOp::Load)
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

    pub fn add_to_graph(&self, rdg: &mut RenderGraph) {
        let node = TransparentPassNode::new(
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
/// (if available) and SSAO view.
pub struct TransparentPassNode {
    // ─── RDG Resource Slots (set by Composer) ──────────────────────
    pub scene_color: TextureNodeId,
    pub scene_depth: TextureNodeId,
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
    pub fn new(screen_info: ScreenBindGroupInfo) -> Self {
        Self {
            scene_color: TextureNodeId(0),
            scene_depth: TextureNodeId(0),
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
        self.scene_color = builder.read_blackboard("Scene_Color_HDR");

        builder.write_texture(self.scene_color);
        self.scene_depth = builder.read_blackboard("Scene_Depth");

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

        let color_view = ctx.get_texture_view(self.scene_color);
        let depth_view = ctx.get_texture_view(self.scene_depth);

        let pass_desc = wgpu::RenderPassDescriptor {
            label: Some("RDG Transparent Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None, // No MSAA in HighFidelity path
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
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
            // Set screen bind group (Group 3) at the pass level.
            let screen_bg = self.screen_bind_group.as_ref().unwrap();
            pass.set_bind_group(3, screen_bg, &[]);

            submit_draw_commands(&mut pass, &ctx.baked_lists.transparent);
        }
    }
}
