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
//! [`BasicForward`]: crate::renderer::settings::RenderPath::BasicForward

use crate::renderer::core::resources::ScreenBindGroupInfo;
use crate::renderer::graph::frame::PreparedSkyboxDraw;
use crate::renderer::graph::rdg::builder::PassBuilder;
use crate::renderer::graph::rdg::context::{RdgExecuteContext, RdgPrepareContext};
use crate::renderer::graph::rdg::draw::submit_draw_commands;
use crate::renderer::graph::rdg::node::PassNode;
use crate::renderer::graph::rdg::types::{RdgTextureDesc, TextureNodeId};

use super::graph::RenderGraph;

// ─── Feature ───────────────────────────────────────────────────────────

pub struct SimpleForwardFeature {
    screen_info: Option<ScreenBindGroupInfo>,
}

impl Default for SimpleForwardFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleForwardFeature {
    #[must_use]
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
        surface_out: TextureNodeId,
        clear_color: wgpu::Color,
        prepared_skybox: Option<PreparedSkyboxDraw>,
    ) {
        let fc = *rdg.frame_config();
        let depth_desc = RdgTextureDesc::new(
            fc.width,
            fc.height,
            1,
            1,
            fc.msaa_samples,
            wgpu::TextureDimension::D2,
            fc.depth_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let scene_depth = rdg.register_resource("Scene_Depth", depth_desc, false);

        let node = SimpleForwardPassNode::new(
            surface_out,
            scene_depth,
            clear_color,
            self.screen_info
                .clone()
                .expect("SimpleForwardFeature: screen_info not set"),
            prepared_skybox,
        );
        rdg.add_pass(Box::new(node));
    }
}

// ─── Pass Node ─────────────────────────────────────────────────────────

/// RDG Simple Forward Render Pass.
///
/// Draws the entire scene in a single LDR render pass with optional MSAA,
/// intended for the [`BasicForward`] rendering path. The pass writes
/// directly to the swap-chain surface via `surface_out`.
///
/// [`BasicForward`]: crate::renderer::settings::RenderPath::BasicForward
pub struct SimpleForwardPassNode {
    // ─── RDG Resource Slots (explicit wiring from add_to_graph) ───
    pub surface_out: TextureNodeId,
    pub scene_depth: TextureNodeId,
    pub msaa_view: Option<TextureNodeId>,

    // ─── Push Parameters ─────────────────────────────────────────────
    pub clear_color: wgpu::Color,
    pub prepared_skybox: Option<PreparedSkyboxDraw>,

    // ─── Screen Bind Group Infrastructure ──────────────────────────
    screen_info: ScreenBindGroupInfo,

    // ─── Internal Cache ──────────────────────────────────────────
    screen_bind_group: Option<wgpu::BindGroup>,
}

impl SimpleForwardPassNode {
    #[must_use]
    pub fn new(
        surface_out: TextureNodeId,
        scene_depth: TextureNodeId,
        clear_color: wgpu::Color,
        screen_info: ScreenBindGroupInfo,
        prepared_skybox: Option<PreparedSkyboxDraw>,
    ) -> Self {
        Self {
            surface_out,
            scene_depth,
            clear_color,
            prepared_skybox,
            msaa_view: None,
            screen_info,
            screen_bind_group: None,
        }
    }
}

impl PassNode for SimpleForwardPassNode {
    fn name(&self) -> &'static str {
        "RDG_SimpleForward_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        // Outputs (pre-registered in add_to_graph).
        builder.declare_output(self.surface_out);
        builder.declare_output(self.scene_depth);

        // MSAA intermediate (internal, conditionally created).
        let msaa_samples = builder.frame_config().msaa_samples;
        if msaa_samples > 1 {
            let (w, h) = builder.global_resolution();
            let surface_format = builder.frame_config().surface_format;
            let desc = RdgTextureDesc::new(
                w,
                h,
                1,
                1,
                msaa_samples,
                wgpu::TextureDimension::D2,
                surface_format,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );
            let msaa_id = builder.create_texture("Scene_Msaa", desc);
            self.msaa_view = Some(msaa_id);
        } else {
            self.msaa_view = None;
        }
    }

    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {
        // Build screen bind group (group 3) with dummy textures (LDR path
        // has no SSAO or transmission).
        let (bg, _) = self.screen_info.build_screen_bind_group(
            ctx.device,
            ctx.global_bind_group_cache,
            &self.screen_info.dummy_transmission_view.clone(),
            &self.screen_info.ssao_dummy_view.clone(),
        );
        self.screen_bind_group = Some(bg);
    }

    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let gpu_global_bind_group = ctx.baked_lists.global_bind_group;

        let (color_view, resolve_target) = if let Some(msaa_view) = self.msaa_view {
            (msaa_view, Some(self.surface_out))
        } else {
            (self.surface_out, None)
        };

        let depth_att = ctx.get_depth_stencil_attachment(self.scene_depth, 0.0);
        let color_att =
            ctx.get_color_attachment(color_view, Some(self.clear_color), resolve_target);

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

        // Set global bind group (group 0: camera, lights, etc.)
        pass.set_bind_group(0, gpu_global_bind_group, &[]);

        // Set screen bind group (Group 3) at the pass level.
        let screen_bg = self.screen_bind_group.as_ref().unwrap();
        pass.set_bind_group(3, screen_bg, &[]);

        // 1. Opaque (front-to-back)
        submit_draw_commands(&mut pass, &ctx.baked_lists.opaque);

        // 2. Skybox (between opaque and transparent)
        if let Some(skybox) = &self.prepared_skybox {
            skybox.draw(&mut pass, gpu_global_bind_group, ctx.pipeline_cache);

            // Re-set global and screen bind groups for subsequent transparent
            // draws — skybox uses a different pipeline and bind group layout.
            pass.set_bind_group(0, gpu_global_bind_group, &[]);
            pass.set_bind_group(3, screen_bg, &[]);
        }

        // 3. Transparent (back-to-front)
        submit_draw_commands(&mut pass, &ctx.baked_lists.transparent);

        // RenderPass automatically resolves MSAA on drop if resolve_target is set.
    }
}
