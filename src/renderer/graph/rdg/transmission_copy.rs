//! RDG Transmission Copy Pass
//!
//! Copies the current scene color to a transmission texture for use by
//! transparent objects with refraction/transmission effects.
//!
//! # RDG Slots
//!
//! - `scene_color`: HDR scene color (read via blackboard)
//! - `transmission_tex`: Transmission output (created via `create_and_export`)
//!
//! # Notes
//!
//! - Uses `encoder.copy_texture_to_texture` (not a render pass)
//! - Only active when materials with Transmission exist in the scene
//! - Generates mipmaps after the copy for LOD-based blur

use crate::renderer::graph::rdg::builder::PassBuilder;
use crate::renderer::graph::rdg::context::{RdgExecuteContext, RdgPrepareContext};
use crate::renderer::graph::rdg::node::PassNode;
use crate::renderer::graph::rdg::types::{RdgTextureDesc, TextureNodeId};

use super::graph::RenderGraph;

// ─── Feature ───────────────────────────────────────────────────────────

pub struct TransmissionCopyFeature;

impl TransmissionCopyFeature {
    pub fn new() -> Self {
        Self
    }

    pub fn add_to_graph(&self, rdg: &mut RenderGraph, active: bool) {
        let node = TransmissionCopyPassNode {
            scene_color: TextureNodeId(0),
            transmission_tex: TextureNodeId(0),
            active,
        };
        rdg.add_pass(Box::new(node));
    }
}

// ─── Pass Node ─────────────────────────────────────────────────────────

/// RDG Transmission Copy Pass.
///
/// Copies scene color to the transmission texture and generates mipmaps.
/// Conditionally skipped if no materials use Transmission this frame.
pub struct TransmissionCopyPassNode {
    // ─── RDG Resource Slots ────────────────────────────────────────
    pub scene_color: TextureNodeId,
    pub transmission_tex: TextureNodeId,

    // ─── Push Parameters ───────────────────────────────────────────
    /// Whether any materials in the scene use transmission.
    pub active: bool,
}

impl TransmissionCopyPassNode {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scene_color: TextureNodeId(0),
            transmission_tex: TextureNodeId(0),
            active: false,
        }
    }
}

impl PassNode for TransmissionCopyPassNode {
    fn name(&self) -> &'static str {
        "RDG_TransmissionCopy_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        // Producer: create the transmission copy destination texture.
        let (w, h) = builder.global_resolution();
        let hdr_format = builder.frame_config().hdr_format;
        let desc = RdgTextureDesc::new_2d(
            w,
            h,
            hdr_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
        );
        self.transmission_tex = builder.create_texture("Transmission_Tex", desc);

        // Consumer: read the scene colour as copy source.
        self.scene_color = builder.read_blackboard("Scene_Color_HDR");
    }

    fn prepare(&mut self, _ctx: &mut RdgPrepareContext) {
        // No GPU resources to prepare — the copy is recorded directly in execute.
    }

    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        if !self.active {
            return;
        }

        let src_view = ctx.get_texture_view(self.scene_color);
        let src_texture = src_view.texture();
        let src_size = src_texture.size();

        let dst_view = ctx.get_texture_view(self.transmission_tex);
        let dst_texture = dst_view.texture();

        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: src_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: dst_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: src_size.width,
                height: src_size.height,
                depth_or_array_layers: 1,
            },
        );

        // Generate mipmaps for LOD-based transmission blur
        ctx.mipmap_generator
            .generate(ctx.device, encoder, dst_texture);
    }
}
