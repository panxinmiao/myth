//! RDG Simple Forward Render Pass
//!
//! Single-pass LDR rendering for the [`BasicForward`] path. Combines opaque,
//! skybox, and transparent drawing into one `wgpu::RenderPass` with optional
//! hardware MSAA.
//!
//! # RDG Slots (explicit wiring)
//!
//! - `surface_out`: LDR colour output (input, from Composer)
//! - `scene_depth`: Depth buffer (created internally)
//!
//! # Push Parameters
//!
//! - `clear_color`: Background clear colour (from scene settings)
//!
//! # Rendering Order
//!
//! 1. **Clear** colour and depth
//! 2. **Opaque** objects (front-to-back)
//! 3. **Skybox** (drawn behind opaque geometry via Reverse-Z)
//! 4. **Transparent** objects (back-to-front)
//!
//! [`BasicForward`]: crate::settings::RenderPath::BasicForward

use crate::core::gpu::Tracked;
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    ExecuteContext, PassNode, PrepareContext, RenderTargetOps, TextureDesc, TextureNodeId,
    build_screen_bind_group,
};
use crate::graph::frame::PreparedSkyboxDraw;
use crate::graph::passes::draw::submit_draw_commands;

// ─── Feature ───────────────────────────────────────────────────────────

pub struct SimpleForwardFeature;

impl Default for SimpleForwardFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleForwardFeature {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        surface_out: TextureNodeId,
        clear_color: wgpu::Color,
        prepared_skybox: Option<PreparedSkyboxDraw<'a>>,
        shadow_tex: Option<TextureNodeId>,
        shadow_cube_tex: Option<TextureNodeId>,
        env_map_tex: Option<TextureNodeId>,
        pmrem_tex: Option<TextureNodeId>,
    ) {
        let fc = ctx.frame_config;

        let depth_desc = TextureDesc::new(
            fc.width,
            fc.height,
            1,
            1,
            fc.msaa_samples,
            wgpu::TextureDimension::D2,
            fc.depth_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        ctx.graph.add_pass("SimpleForward_Pass", |builder| {
            builder.write_texture(surface_out);
            let scene_depth = builder.create_texture("Scene_Depth", depth_desc);

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
            if let Some(skybox) = prepared_skybox {
                for dependency in skybox.sampled_textures.into_iter().flatten() {
                    builder.read_texture(dependency);
                }
            }

            let msaa_view = if fc.msaa_samples > 1 {
                let desc = TextureDesc::new(
                    fc.width,
                    fc.height,
                    1,
                    1,
                    fc.msaa_samples,
                    wgpu::TextureDimension::D2,
                    fc.surface_format,
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                );
                Some(builder.create_texture("Scene_Msaa", desc))
            } else {
                None
            };

            let node = SimpleForwardPassNode {
                surface_out,
                scene_depth,
                msaa_view,
                clear_color,
                prepared_skybox,
                shadow_input: shadow_tex,
                shadow_cube_input: shadow_cube_tex,
                screen_bind_group: None,
            };
            (node, ())
        });
    }
}

// ─── Pass Node ─────────────────────────────────────────────────────────

/// RDG Simple Forward Render Pass.
///
/// Draws the entire scene in a single LDR render pass with optional MSAA,
/// intended for the [`BasicForward`] rendering path. The pass writes
/// directly to the swap-chain surface via `surface_out`.
///
/// [`BasicForward`]: crate::settings::RenderPath::BasicForward
pub struct SimpleForwardPassNode<'a> {
    pub surface_out: TextureNodeId,
    pub scene_depth: TextureNodeId,
    pub msaa_view: Option<TextureNodeId>,
    pub clear_color: wgpu::Color,
    pub prepared_skybox: Option<PreparedSkyboxDraw<'a>>,
    pub shadow_input: Option<TextureNodeId>,
    pub shadow_cube_input: Option<TextureNodeId>,
    screen_bind_group: Option<&'static wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for SimpleForwardPassNode<'a> {
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
            &sys.black_hdr,
            &sys.white_r8,
            shadow_view,
            shadow_cube_view,
        );
        self.screen_bind_group = Some(bg);
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let gpu_global_bind_group = ctx.baked_lists.global_bind_group;

        let (color_view, resolve_target) = if let Some(msaa_view) = self.msaa_view {
            (msaa_view, Some(self.surface_out))
        } else {
            (self.surface_out, None)
        };

        let depth_att = ctx.get_depth_stencil_attachment(self.scene_depth, 0.0);
        let color_att = ctx.get_color_attachment(
            color_view,
            RenderTargetOps::Clear(self.clear_color),
            resolve_target,
        );

        let pass_desc = wgpu::RenderPassDescriptor {
            label: Some("RDG Simple Forward Pass"),
            color_attachments: &[color_att],
            depth_stencil_attachment: depth_att,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };

        let raw_pass = encoder.begin_render_pass(&pass_desc);
        let mut pass = raw_pass;

        pass.set_bind_group(0, gpu_global_bind_group, &[]);

        let screen_bg = self.screen_bind_group.unwrap();
        pass.set_bind_group(3, screen_bg, &[]);

        // 1. Opaque (front-to-back)
        submit_draw_commands(&mut pass, &ctx.baked_lists.opaque);

        // 2. Skybox (between opaque and transparent)
        if let Some(skybox) = &self.prepared_skybox {
            skybox.draw(&mut pass, gpu_global_bind_group);

            pass.set_bind_group(0, gpu_global_bind_group, &[]);
            pass.set_bind_group(3, screen_bg, &[]);
        }

        // 3. Transparent (back-to-front)
        submit_draw_commands(&mut pass, &ctx.baked_lists.transparent);
    }
}
