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

use crate::HDR_TEXTURE_FORMAT;
use crate::core::gpu::Tracked;
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    ExecuteContext, PassNode, PrepareContext, RenderTargetOps, TextureDesc, TextureNodeId,
    build_screen_bind_group,
};
use crate::graph::passes::draw::submit_draw_commands;

/// Outputs produced by the Opaque pass, returned to the Composer for
/// explicit downstream wiring.
#[must_use = "SSA Graph: You must use the outputs of opaque pass to wire downstream passes!"]
pub struct OpaqueOutputs {
    /// Drawing surface for subsequent MSAA passes (Skybox, Transparent).
    pub active_color: TextureNodeId,
    /// Depth surface for subsequent draws.  In MSAA mode this is a
    /// multi-sampled depth; in non-MSAA mode this is the Prepass's depth.
    pub active_depth: TextureNodeId,
    /// Resolved specular texture for SSSS (`None` when specular is
    /// not enabled).
    pub specular_mrt: Option<TextureNodeId>,
}

// ─── Feature ───────────────────────────────────────────────────────────

pub struct OpaqueFeature;

impl Default for OpaqueFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl OpaqueFeature {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        scene_depth_ss: TextureNodeId,
        clear_color: wgpu::Color,
        needs_specular: bool,
        ssao_tex: Option<TextureNodeId>,
        shadow_tex: Option<TextureNodeId>,
        shadow_cube_tex: Option<TextureNodeId>,
        env_map_tex: Option<TextureNodeId>,
        pmrem_tex: Option<TextureNodeId>,
    ) -> OpaqueOutputs {
        let fc = ctx.frame_config;
        let is_msaa = fc.msaa_samples > 1;

        let hdr_desc = TextureDesc::new_2d(
            fc.width,
            fc.height,
            HDR_TEXTURE_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );

        ctx.graph.add_pass("Opaque_Pass", |builder| {
            // ── Create color / depth / resolve targets ─────────────────
            let (color_target, depth_target) = if is_msaa {
                let msaa_color_desc = TextureDesc::new(
                    fc.width,
                    fc.height,
                    1,
                    1,
                    fc.msaa_samples,
                    wgpu::TextureDimension::D2,
                    HDR_TEXTURE_FORMAT,
                    wgpu::TextureUsages::RENDER_ATTACHMENT,
                );
                let msaa_depth_desc = TextureDesc::new(
                    fc.width,
                    fc.height,
                    1,
                    1,
                    fc.msaa_samples,
                    wgpu::TextureDimension::D2,
                    fc.depth_format,
                    wgpu::TextureUsages::RENDER_ATTACHMENT,
                );

                let msaa_color = builder.create_texture("Scene_Color_MSAA", msaa_color_desc);
                let msaa_depth = builder.create_texture("Scene_Depth_MSAA", msaa_depth_desc);

                (msaa_color, msaa_depth)
            } else {
                let scene_hdr = builder.create_texture("Scene_Color_HDR", hdr_desc);
                let depth = builder.read_texture(scene_depth_ss);
                (scene_hdr, depth)
            };

            // ── Specular MRT (conditionally created) ───────────────────
            let (specular_tex, specular_resolved) = if needs_specular {
                let spec_desc = TextureDesc::new_2d(
                    fc.width,
                    fc.height,
                    HDR_TEXTURE_FORMAT,
                    wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                );
                let specular_single = builder.create_texture("Specular_MRT", spec_desc);

                if is_msaa {
                    let msaa_spec_desc = TextureDesc::new(
                        fc.width,
                        fc.height,
                        1,
                        1,
                        fc.msaa_samples,
                        wgpu::TextureDimension::D2,
                        HDR_TEXTURE_FORMAT,
                        wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                    );
                    let specular_msaa = builder.create_texture("Specular_MRT_MSAA", msaa_spec_desc);
                    (specular_msaa, Some(specular_single))
                } else {
                    (specular_single, None)
                }
            } else {
                (TextureNodeId(0), None)
            };

            // ── Read dependencies ──────────────────────────────────────
            if let Some(ssao) = ssao_tex {
                builder.read_texture(ssao);
            }
            if let Some(shadow) = shadow_tex {
                builder.read_texture(shadow);
            }
            if let Some(shadow_cube) = shadow_cube_tex {
                builder.read_texture(shadow_cube);
            }
            if let Some(env_map) = env_map_tex {
                builder.read_texture(env_map);
            }
            if let Some(pmrem) = pmrem_tex {
                builder.read_texture(pmrem);
            }

            let node = OpaquePassNode::new(
                color_target,
                depth_target,
                clear_color,
                needs_specular,
                ssao_tex,
                shadow_tex,
                shadow_cube_tex,
                specular_tex,
                specular_resolved,
            );

            let specular_mrt = if needs_specular {
                Some(specular_resolved.unwrap_or(specular_tex))
            } else {
                None
            };

            (
                node,
                OpaqueOutputs {
                    active_color: color_target,
                    active_depth: depth_target,
                    specular_mrt,
                },
            )
        })
    }
}

// ─── Pass Node ─────────────────────────────────────────────────────────

/// RDG Opaque Render Pass.
///
/// Draws `render_lists.opaque` to the scene color buffer.  Builds a
/// dynamic screen bind group (Group 3) with SSAO, transmission, and shadow
/// textures resolved from [`SystemTextures`] fallbacks when inactive.
/// When MSAA is active, the pass writes to a multi-sampled color target
/// and optionally resolves to a single-sample HDR texture.
pub struct OpaquePassNode {
    // ─── RDG Resource Slots ────────────────────────────────────────
    pub color_target: TextureNodeId,
    pub depth_target: TextureNodeId,
    pub specular_tex: TextureNodeId,
    pub specular_resolve_target: Option<TextureNodeId>,

    // ─── Push Parameters ───────────────────────────────────────────
    pub clear_color: wgpu::Color,
    pub needs_specular: bool,
    pub ssao_input: Option<TextureNodeId>,
    pub shadow_input: Option<TextureNodeId>,
    pub shadow_cube_input: Option<TextureNodeId>,

    // ─── Internal Cache ────────────────────────────────────────────
    screen_bind_group: Option<&'static wgpu::BindGroup>,
}

impl OpaquePassNode {
    #[must_use]
    pub fn new(
        color_target: TextureNodeId,
        depth_target: TextureNodeId,
        clear_color: wgpu::Color,
        needs_specular: bool,
        ssao_input: Option<TextureNodeId>,
        shadow_input: Option<TextureNodeId>,
        shadow_cube_input: Option<TextureNodeId>,
        specular_tex: TextureNodeId,
        specular_resolve_target: Option<TextureNodeId>,
    ) -> Self {
        Self {
            color_target,
            depth_target,
            specular_tex,
            specular_resolve_target,
            clear_color,
            needs_specular,
            ssao_input,
            shadow_input,
            shadow_cube_input,
            screen_bind_group: None,
        }
    }
}

impl<'a> PassNode<'a> for OpaquePassNode {
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

        let transmission_view = &sys.black_hdr;

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

        // ── Color attachments (auto-deduced LoadOp / StoreOp) ───────────
        let mut color_attachments: smallvec::SmallVec<
            [Option<wgpu::RenderPassColorAttachment>; 2],
        > = smallvec::smallvec![ctx.get_color_attachment(
            self.color_target,
            RenderTargetOps::Clear(self.clear_color),
            None
        )];

        // Specular MRT — may have been culled if no downstream consumer
        // (e.g. SSSS disabled).  `get_color_attachment` returns `None`
        // for dead resources, naturally shrinking the MRT footprint.
        if self.needs_specular
            && let Some(att) = ctx.get_color_attachment(
                self.specular_tex,
                RenderTargetOps::Clear(wgpu::Color::TRANSPARENT),
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
            label: Some("Opaque Pass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: depth_stencil,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };

        let raw_pass = encoder.begin_render_pass(&pass_desc);
        let mut pass = raw_pass;

        pass.set_bind_group(0, gpu_global_bind_group, &[]);

        let screen_bg = self.screen_bind_group.unwrap();
        pass.set_bind_group(3, screen_bg, &[]);

        submit_draw_commands(&mut pass, &ctx.baked_lists.opaque);
    }
}
