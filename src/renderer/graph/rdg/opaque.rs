//! RDG Opaque Render Pass
//!
//! Draws opaque objects to the HDR scene color buffer. Clears color to the
//! scene background color and conditionally clears or loads depth depending
//! on whether a Z-prepass has already written depth data.
//!
//! # RDG Slots
//!
//! - `scene_color`: HDR color output (write, Clear)
//! - `scene_depth`: Depth buffer (read if prepass, write if no prepass)
//! - `ssao_tex`: Optional SSAO texture (read, for screen bind group)
//!
//! # Push Parameters
//!
//! - `has_prepass`: Whether depth was already written by RdgPrepass
//! - `clear_color`: Background clear color
//! - `needs_specular`: Whether to output a specular MRT attachment

use crate::renderer::core::resources::{ScreenBindGroupInfo, Tracked};
use crate::renderer::graph::rdg::builder::PassBuilder;
use crate::renderer::graph::rdg::context::{RdgExecuteContext, RdgPrepareContext};
use crate::renderer::graph::rdg::draw::submit_draw_commands;
use crate::renderer::graph::rdg::node::PassNode;
use crate::renderer::graph::rdg::types::{RdgTextureDesc, TextureNodeId};

use super::graph::RenderGraph;

// ─── Feature ───────────────────────────────────────────────────────────

pub struct OpaqueFeature {
    screen_info: Option<ScreenBindGroupInfo>,
}

impl OpaqueFeature {
    pub fn new() -> Self {
        Self { screen_info: None }
    }

    /// Cache screen bind group info from ResourceManager.
    /// Called once (or on resize) before add_to_graph.
    pub fn set_screen_info(&mut self, info: ScreenBindGroupInfo) {
        self.screen_info = Some(info);
    }

    pub fn add_to_graph(
        &self,
        rdg: &mut RenderGraph,
        has_prepass: bool,
        clear_color: wgpu::Color,
        needs_specular: bool,
    ) {
        let node = OpaquePassNode::new(
            has_prepass,
            clear_color,
            needs_specular,
            self.screen_info.clone().expect("OpaqueFeature: screen_info not set"),
        );
        rdg.add_pass(Box::new(node));
    }
}

// ─── Pass Node ─────────────────────────────────────────────────────────

/// RDG Opaque Render Pass.
///
/// Draws `render_lists.opaque` to the HDR scene color buffer. Builds a
/// dynamic screen bind group (group 3) containing SSAO + transmission dummy
/// textures.
pub struct OpaquePassNode {
    // ─── RDG Resource Slots (set during setup) ─────────────────────
    pub scene_color: TextureNodeId,
    pub scene_depth: TextureNodeId,
    pub ssao_tex: TextureNodeId,
    pub specular_tex: TextureNodeId,

    // ─── Push Parameters ───────────────────────────────────────────
    pub has_prepass: bool,
    pub clear_color: wgpu::Color,
    pub needs_specular: bool,
    pub ssao_enabled: bool,
    // ─── Screen Bind Group Infrastructure ──────────────────────────
    screen_info: ScreenBindGroupInfo,
    // ─── Internal Cache ────────────────────────────────────────────
    screen_bind_group: Option<wgpu::BindGroup>,
}

impl OpaquePassNode {
    #[must_use]
    pub fn new(has_prepass: bool, clear_color: wgpu::Color, needs_specular: bool, screen_info: ScreenBindGroupInfo) -> Self {
        Self {
            scene_color: TextureNodeId(0),
            scene_depth: TextureNodeId(0),
            ssao_tex: TextureNodeId(0),
            specular_tex: TextureNodeId(0),
            has_prepass,
            clear_color,
            needs_specular,
            ssao_enabled: false,
            screen_info,
            screen_bind_group: None,
        }
    }
}

impl PassNode for OpaquePassNode {
    fn name(&self) -> &'static str {
        "RDG_Opaque_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        // Consumer: wire backbone resources.
        self.scene_color = builder.write_blackboard("Scene_Color_HDR");

        self.scene_depth = builder.write_blackboard("Scene_Depth");

        if self.has_prepass {
            builder.read_texture(self.scene_depth);
        }

        // self.ssao_tex = builder.try_read_blackboard("SSAO_Output");

        // Detect optional SSAO resource.
        if let Some(ssao) = builder.try_read_blackboard("SSAO_Output") {
            self.ssao_tex = ssao;
            self.ssao_enabled = true;
        } else {
            self.ssao_enabled = false;
        }

        // Producer: conditionally create specular MRT.
        // The Composer sets `needs_specular` before `add_pass`.
        if self.needs_specular {
            let (w, h) = builder.global_resolution();
            let hdr_format = builder.frame_config().hdr_format;
            let desc = RdgTextureDesc::new_2d(
                w,
                h,
                hdr_format,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            );
            self.specular_tex = builder.create_texture("Specular_MRT", desc);
        }
    }

    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {
        // Build screen bind group (group 3): SSAO + transmission dummy + sampler
        let ssao_view: &Tracked<wgpu::TextureView> = if self.ssao_enabled {
            ctx.views.get_texture_view(self.ssao_tex)
        } else {
            &self.screen_info.ssao_dummy_view
        };

        let transmission_view = &self.screen_info.dummy_transmission_view;

        let (bg, _) = self.screen_info.build_screen_bind_group(
            ctx.device,
            ctx.global_bind_group_cache,
            transmission_view,
            ssao_view,
        );
        self.screen_bind_group = Some(bg);
    }

    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let render_lists = &ctx.render_lists;

        let Some(gpu_global_bind_group) = &render_lists.gpu_global_bind_group else {
            log::warn!("RDG OpaquePass: gpu_global_bind_group missing, skipping");
            return;
        };

        let color_view = ctx.get_texture_view(self.scene_color);
        let depth_view = ctx.get_texture_view(self.scene_depth);

        let mut color_attachments: smallvec::SmallVec<
            [Option<wgpu::RenderPassColorAttachment>; 2],
        > = smallvec::smallvec![Some(wgpu::RenderPassColorAttachment {
            view: color_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(self.clear_color),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })];

        // Optional specular MRT
        if self.needs_specular {
            let specular_view = ctx.get_texture_view(self.specular_tex);
            color_attachments.push(Some(wgpu::RenderPassColorAttachment {
                view: specular_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            }));
        }

        let pass_desc = wgpu::RenderPassDescriptor {
            label: Some("RDG Opaque Pass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: if self.has_prepass {
                        wgpu::LoadOp::Load
                    } else {
                        wgpu::LoadOp::Clear(0.0) // Reverse-Z: 0.0 = far
                    },
                    store: wgpu::StoreOp::Store,
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

        // Set screen bind group (Group 3) at the pass level.
        let screen_bg = self.screen_bind_group.as_ref().unwrap();
        pass.set_bind_group(3, screen_bg, &[]);

        submit_draw_commands(&mut pass, &ctx.baked_lists.opaque);
    }
}
