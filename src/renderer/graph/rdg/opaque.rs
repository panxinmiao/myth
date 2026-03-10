//! RDG Opaque Render Pass
//!
//! Draws opaque objects to the scene color buffer. Clears color to the
//! scene background color and conditionally clears or loads depth depending
//! on whether a Z-prepass has already written depth data.
//!
//! # MSAA Support
//!
//! When hardware MSAA is active in the HighFidelity path, the pass creates
//! multi-sampled color/depth targets and an HDR resolve target internally.
//! The GPU hardware resolves the result into the single-sample HDR buffer
//! at the end of the render pass.  The RDG lifetime system keeps the MSAA
//! surface alive (`StoreOp::Store`) as long as downstream passes (Skybox,
//! Transparent) still reference it.
//!
//! # RDG Slots (explicit wiring)
//!
//! - `color_target`: Scene color output — created internally
//! - `depth_target`: Scene depth — created or reused
//! - `resolve_target`: Optional single-sample HDR to receive MSAA resolve
//! - `ssao_tex`: Optional SSAO texture (explicit input)
//!
//! # Push Parameters
//!
//! - `has_prepass`: Whether depth was already written by RdgPrepass
//! - `clear_color`: Background clear color
//! - `needs_specular`: Whether to output a specular MRT attachment

use crate::renderer::HDR_TEXTURE_FORMAT;
use crate::renderer::core::resources::{ScreenBindGroupInfo, Tracked};
use crate::renderer::graph::rdg::builder::PassBuilder;
use crate::renderer::graph::rdg::context::{RdgExecuteContext, RdgPrepareContext};
use crate::renderer::graph::rdg::draw::submit_draw_commands;
use crate::renderer::graph::rdg::node::PassNode;
use crate::renderer::graph::rdg::types::{RdgTextureDesc, TextureNodeId};

use super::graph::RenderGraph;

/// Outputs produced by the Opaque pass, returned to the Composer for
/// explicit downstream wiring.
#[must_use = "SSA Graph: You must use the outputs of opaque pass to wire downstream passes!"]
pub struct OpaqueOutputs {
    /// Drawing surface for subsequent MSAA passes (Skybox, Transparent).
    /// In non-MSAA mode this IS `scene_color_hdr`.
    pub active_color: TextureNodeId,
    /// Depth surface for subsequent draws.  In MSAA mode this is a
    /// multi-sampled depth; in non-MSAA mode this is the Prepass's depth.
    pub active_depth: TextureNodeId,
    /// Single-sample HDR scene color (resolve target in MSAA, direct
    /// color target in non-MSAA).  Used by screen-space effects and
    /// the post-processing chain.
    pub scene_color_hdr: TextureNodeId,
    /// Resolved specular texture for SSSSS (`None` when specular is
    /// not enabled).
    pub specular_mrt: Option<TextureNodeId>,
}

// ─── Feature ───────────────────────────────────────────────────────────

pub struct OpaqueFeature {
    screen_info: Option<ScreenBindGroupInfo>,
}

impl Default for OpaqueFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl OpaqueFeature {
    #[must_use]
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
        scene_depth_ss: TextureNodeId,
        has_prepass: bool,
        clear_color: wgpu::Color,
        needs_specular: bool,
        ssao_tex: Option<TextureNodeId>,
    ) -> OpaqueOutputs {
        let fc = *rdg.frame_config();
        let is_msaa = fc.msaa_samples > 1;

        // ── Create color / depth / resolve targets ─────────────────
        let hdr_desc = RdgTextureDesc::new_2d(
            fc.width,
            fc.height,
            HDR_TEXTURE_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );

        let (color_target, depth_target, resolve_target, scene_color_hdr, in_depth) = if is_msaa {
            let msaa_color_desc = RdgTextureDesc::new(
                fc.width,
                fc.height,
                1,
                1,
                fc.msaa_samples,
                wgpu::TextureDimension::D2,
                HDR_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            );
            let msaa_depth_desc = RdgTextureDesc::new(
                fc.width,
                fc.height,
                1,
                1,
                fc.msaa_samples,
                wgpu::TextureDimension::D2,
                fc.depth_format,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            );

            let msaa_color = rdg.register_resource("Scene_Color_MSAA", msaa_color_desc, false);
            let msaa_depth = rdg.register_resource("Scene_Depth_MSAA", msaa_depth_desc, false);
            let scene_hdr = rdg.register_resource("Scene_Color_HDR", hdr_desc, false);

            (msaa_color, msaa_depth, Some(scene_hdr), scene_hdr, None)
        } else {
            let scene_hdr = rdg.register_resource("Scene_Color_HDR", hdr_desc, false);
            let depth_alias = rdg.create_alias(scene_depth_ss, "Scene_Depth_Opaque");
            (
                scene_hdr,
                depth_alias,
                None,
                scene_hdr,
                Some(scene_depth_ss),
            )
        };

        // ── Specular MRT (conditionally created) ───────────────────
        let (specular_tex, specular_resolve, specular_resolved) = if needs_specular {
            let spec_desc = RdgTextureDesc::new_2d(
                fc.width,
                fc.height,
                HDR_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            );
            let specular_single = rdg.register_resource("Specular_MRT", spec_desc, false);

            if is_msaa {
                let msaa_spec_desc = RdgTextureDesc::new(
                    fc.width,
                    fc.height,
                    1,
                    1,
                    fc.msaa_samples,
                    wgpu::TextureDimension::D2,
                    HDR_TEXTURE_FORMAT,
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                );
                let specular_msaa =
                    rdg.register_resource("Specular_MRT_MSAA", msaa_spec_desc, false);
                (specular_msaa, Some(specular_single), Some(specular_single))
            } else {
                (specular_single, None, Some(specular_single))
            }
        } else {
            (TextureNodeId(0), None, None)
        };

        let node = OpaquePassNode::new(
            color_target,
            depth_target,
            in_depth,
            has_prepass,
            clear_color,
            needs_specular,
            resolve_target,
            ssao_tex,
            specular_tex,
            specular_resolve,
            self.screen_info
                .clone()
                .expect("OpaqueFeature: screen_info not set"),
        );
        rdg.add_pass(Box::new(node));

        OpaqueOutputs {
            active_color: color_target,
            active_depth: depth_target,
            scene_color_hdr,
            specular_mrt: specular_resolved,
        }
    }
}

// ─── Pass Node ─────────────────────────────────────────────────────────

/// RDG Opaque Render Pass.
///
/// Draws `render_lists.opaque` to the scene color buffer. Builds a
/// dynamic screen bind group (group 3) containing SSAO + transmission dummy
/// textures.  When MSAA is active, the pass writes to a multi-sampled
/// color target and optionally resolves to a single-sample HDR texture.
pub struct OpaquePassNode {
    // ─── RDG Resource Slots ────────────────────────────────────────
    /// Primary color target (`Scene_Color_HDR` or `Scene_Color_MSAA`).
    pub color_target: TextureNodeId,
    /// Primary depth target (`Scene_Depth` or `Scene_Depth_MSAA`).
    pub depth_target: TextureNodeId,
    pub in_depth: Option<TextureNodeId>,
    /// Optional single-sample HDR texture for MSAA resolve.
    pub resolve_target: Option<TextureNodeId>,
    pub specular_tex: TextureNodeId,
    pub specular_resolve_target: Option<TextureNodeId>,

    // ─── Push Parameters ───────────────────────────────────────────
    pub has_prepass: bool,
    pub clear_color: wgpu::Color,
    pub needs_specular: bool,
    /// Explicit SSAO input (`None` when SSAO is disabled).
    pub ssao_input: Option<TextureNodeId>, // ─── Screen Bind Group Infrastructure ──────────────────────────
    screen_info: ScreenBindGroupInfo,
    // ─── Internal Cache ────────────────────────────────────────────
    screen_bind_group: Option<wgpu::BindGroup>,
}

impl OpaquePassNode {
    #[must_use]
    pub fn new(
        color_target: TextureNodeId,
        depth_target: TextureNodeId,
        in_depth: Option<TextureNodeId>,
        has_prepass: bool,
        clear_color: wgpu::Color,
        needs_specular: bool,
        resolve_target: Option<TextureNodeId>,
        ssao_input: Option<TextureNodeId>,
        specular_tex: TextureNodeId,
        specular_resolve_target: Option<TextureNodeId>,
        screen_info: ScreenBindGroupInfo,
    ) -> Self {
        Self {
            color_target,
            depth_target,
            in_depth,
            resolve_target,
            specular_tex,
            specular_resolve_target,
            has_prepass,
            clear_color,
            needs_specular,
            ssao_input,
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
        // Primary color target — may be single-sample HDR or MSAA.
        builder.declare_output(self.color_target);

        if let Some(in_depth) = self.in_depth {
            builder.read_texture(in_depth);
        }

        // Depth target.
        builder.declare_output(self.depth_target);
        // builder.read_texture(self.depth_target);

        // Resolve target — declare write so the graph compiler allocates
        // physical memory and tracks dependencies for downstream consumers.
        if let Some(rt) = self.resolve_target {
            builder.declare_output(rt);
        }

        // SSAO — explicit input wiring.
        if let Some(ssao) = self.ssao_input {
            builder.read_texture(ssao);
        }

        // Specular MRT (pre-registered in add_to_graph).
        if self.needs_specular {
            builder.declare_output(self.specular_tex);
            if self.specular_resolve_target.is_some() {
                // Self-read keeps the MSAA specular alive for the resolve.
                builder.read_texture(self.specular_tex);
            }
            if let Some(resolve) = self.specular_resolve_target {
                builder.declare_output(resolve);
            }
        }
    }

    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {
        // Build screen bind group (group 3): SSAO + transmission dummy + sampler
        let ssao_view: &Tracked<wgpu::TextureView> = match self.ssao_input {
            Some(id) => ctx.views.get_texture_view(id),
            None => &self.screen_info.ssao_dummy_view,
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
        let gpu_global_bind_group = ctx.baked_lists.global_bind_group;

        // ── Color attachments (auto-deduced LoadOp / StoreOp) ───────────
        let mut color_attachments: smallvec::SmallVec<
            [Option<wgpu::RenderPassColorAttachment>; 2],
        > = smallvec::smallvec![ctx.get_color_attachment(
            self.color_target,
            Some(self.clear_color),
            self.resolve_target
        )];

        // Specular MRT — may have been culled if no downstream consumer
        // (e.g. SSSSS disabled).  `get_color_attachment` returns `None`
        // for dead resources, naturally shrinking the MRT footprint.
        if self.needs_specular
            && let Some(att) = ctx.get_color_attachment(
                self.specular_tex,
                Some(wgpu::Color::TRANSPARENT),
                self.specular_resolve_target,
            )
        {
            color_attachments.push(Some(att));
        }

        // ── Depth/stencil attachment (auto-deduced ops) ─────────────────
        // Reverse-Z: clear to 0.0 (far plane) when this is the first use,
        // otherwise load the depth written by the prepass.
        let depth_stencil = ctx.get_depth_stencil_attachment(self.depth_target, 0.0);

        let pass_desc = wgpu::RenderPassDescriptor {
            label: Some("RDG Opaque Pass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: depth_stencil,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };

        let raw_pass = encoder.begin_render_pass(&pass_desc);
        let mut pass = raw_pass;

        pass.set_bind_group(0, gpu_global_bind_group, &[]);

        let screen_bg = self.screen_bind_group.as_ref().unwrap();
        pass.set_bind_group(3, screen_bg, &[]);

        submit_draw_commands(&mut pass, &ctx.baked_lists.opaque);
    }
}
