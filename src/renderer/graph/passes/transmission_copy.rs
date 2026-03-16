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
    core::{TextureDesc, TextureNodeId},
    passes::utils::{
        add_copy_texture_pass_into, add_generate_mipmap_pass, add_msaa_resolve_pass_into,
    },
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
    ) -> TextureNodeId {
        let fc = ctx.frame_config;
        let mip_count = (fc.width.max(fc.height) as f32).log2().floor() as u32 + 1;
        let desc = TextureDesc::new(
            fc.width,
            fc.height,
            1,
            mip_count,
            1,
            wgpu::TextureDimension::D2,
            fc.hdr_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
        );
        let transmission_tex = ctx.graph.register_resource("Transmission_Tex", desc, false);

        let is_msaa = fc.msaa_samples > 1;

        ctx.with_group("Transmission_Map", |ctx| {
            let out = if is_msaa {
                add_msaa_resolve_pass_into(ctx, scene_color, transmission_tex)
            } else {
                add_copy_texture_pass_into(ctx, scene_color, transmission_tex)
            };
            add_generate_mipmap_pass(ctx, out, "Transmission_Tex_Mipmapped")
        })
    }
}
