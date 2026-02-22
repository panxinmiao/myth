//! Transmission Copy Pass
//!

//! Copy the current scene color buffer to the Transmission texture.
//! will be used as the input for Transmission effects in the TransparentPass.
//!
//! # Data Flow
//! ```text
//! HDR Scene Color → TransmissionCopyPass → Transmission Texture
//! ```
//!
//! # Execution Timing
//! - After `OpaquePass`
//! - Before `TransparentPass`
//!
//! # Note
//! - This pass does not use a RenderPass, it directly uses `encoder.copy_texture_to_texture`
//! - Only executed if there are materials using Transmission in the scene

use crate::renderer::graph::RenderNode;
use crate::renderer::graph::context::{ExecuteContext, GraphResource};

/// Transmission Copy Pass
///
/// Copy the current scene color buffer to the Transmission texture.
/// This pass does not perform any drawing, only executes a texture copy.
///
/// # Conditional Execution
/// - Only executed if `render_lists.use_transmission` is true
/// - Only executed in `HighFidelity` path (Transmission requires HDR buffer)
/// - Only executed if `transmission_view` exists
pub struct TransmissionCopyPass;

impl TransmissionCopyPass {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for TransmissionCopyPass {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderNode for TransmissionCopyPass {
    fn name(&self) -> &'static str {
        "Transmission Copy Pass"
    }

    fn run(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let render_lists = &ctx.render_lists;

        // Check if execution is needed
        if !render_lists.use_transmission {
            return;
        }

        // Transmission requires HighFidelity path
        if !ctx.wgpu_ctx.render_path.supports_post_processing() {
            log::warn!("TransmissionCopyPass: Transmission requires HighFidelity path, skipping");
            return;
        }

        // Check if transmission texture exists
        let Some(transmission_view) = ctx.try_get_resource_view(GraphResource::Transmission) else {
            log::warn!("TransmissionCopyPass: transmission_view missing, skipping");
            return;
        };

        // Get source texture (scene color buffer)
        let color_view = ctx.get_scene_render_target_view();

        let is_msaa = ctx.wgpu_ctx.msaa_samples > 1;

        if is_msaa {
            // If MSAA is enabled, OpaquePass has not yet resolved, and the data is still in the MSAA View.
            // We need to manually perform a resolve to save the opaque results to the color_view.
            if let Some(msaa_view) = ctx.try_get_resource_view(GraphResource::SceneMsaa) {
                let pass_desc = wgpu::RenderPassDescriptor {
                    label: Some("Transmission MSAA Resolve"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: msaa_view,                  // Source: MSAA buffer
                        resolve_target: Some(color_view), // Target: regular texture
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,    // Load existing content
                            store: wgpu::StoreOp::Store, // Must store! Because TransparentPass will continue drawing on it
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None, // No depth needed, just resolving color
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                };

                // Execute an empty RenderPass to trigger the resolve
                let _ = encoder.begin_render_pass(&pass_desc);
            }
        }

        // Execute texture copy
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: color_view.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: transmission_view.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: color_view.texture().size().width,
                height: color_view.texture().size().height,
                depth_or_array_layers: 1,
            },
        );

        ctx.resource_manager.mipmap_generator.generate(
            &ctx.wgpu_ctx.device,
            encoder,
            transmission_view.texture(),
        );
    }
}
