//! RDG Transmission Copy Pass
//!
//! Copies the current scene color to a transmission texture for use by
//! transparent objects with refraction/transmission effects.
//!
//! # RDG Slots (explicit wiring)
//!
//! - `scene_color`: HDR scene color (input)
//! - `transmission_tex`: Transmission output (created & returned by `add_to_graph`)
//!
//! # Notes
//!
//! - Uses `encoder.copy_texture_to_texture` (not a render pass)
//! - Only active when materials with Transmission exist in the scene
//! - Generates mipmaps after the copy for LOD-based blur

use crate::renderer::graph::{
    composer::GraphBuilderContext,
    core::{ExecuteContext, PassNode, TextureDesc, TextureNodeId},
};

// ─── Feature ───────────────────────────────────────────────────────────

pub struct TransmissionCopyFeature;

impl Default for TransmissionCopyFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl TransmissionCopyFeature {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    pub fn add_to_graph(
        &self,
        ctx: &mut GraphBuilderContext<'_, '_>,
        scene_color: TextureNodeId,
        active: bool,
    ) -> TextureNodeId {
        let fc = ctx.frame_config;
        let desc = TextureDesc::new_2d(
            fc.width,
            fc.height,
            fc.hdr_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
        );
        let transmission_tex = ctx.graph.register_resource("Transmission_Tex", desc, false);

        let node = TransmissionCopyPassNode {
            scene_color,
            transmission_tex,
            active,
        };
        ctx.graph.add_pass("TransmissionCopy_Pass", |builder| {
            builder.write_texture(transmission_tex);
            builder.read_texture(scene_color);
            (node, ())
        });
        transmission_tex
    }
}

// ─── Pass Node ─────────────────────────────────────────────────────────

/// RDG Transmission Copy Pass.
///
/// Copies scene color to the transmission texture and generates mipmaps.
/// Conditionally skipped if no materials use Transmission this frame.
pub struct TransmissionCopyPassNode {
    // ─── RDG Resource Slots (explicit wiring from add_to_graph) ────
    pub scene_color: TextureNodeId,
    pub transmission_tex: TextureNodeId,

    // ─── Push Parameters ───────────────────────────────────────────
    /// Whether any materials in the scene use transmission.
    pub active: bool,
}

impl PassNode<'_> for TransmissionCopyPassNode {
    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
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
