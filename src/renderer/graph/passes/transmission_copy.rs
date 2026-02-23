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
use crate::renderer::graph::transient_pool::{TransientTextureDesc, TransientTextureId};

/// Transmission Copy Pass
///
/// Copy the current scene color buffer to the Transmission texture.
/// This pass does not perform any drawing, only executes a texture copy.
///
/// # Conditional Execution
/// - Only executed if `render_lists.use_transmission` is true
/// - Only executed in `HighFidelity` path (Transmission requires HDR buffer)
///
/// # Transient Resource
/// The transmission texture is allocated from the `TransientTexturePool`
/// during [`prepare`](RenderNode::prepare). The pool recycles textures
/// across frames, ensuring stable `TextureView` IDs for
/// `GlobalBindGroupCache` hits.
pub struct TransmissionCopyPass {
    /// Transient texture ID for the current frame's transmission buffer.
    pub(crate) transmission_texture_id: Option<TransientTextureId>,
}

impl TransmissionCopyPass {
    #[must_use]
    pub fn new() -> Self {
        Self {
            transmission_texture_id: None,
        }
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

    fn prepare(&mut self, ctx: &mut crate::renderer::graph::context::PrepareContext) {
        // Always allocate a transient transmission texture in HighFidelity mode.
        // If no material uses transmission this frame, the texture goes unused
        // and is cheaply recycled by the pool.
        if !ctx.wgpu_ctx.render_path.supports_post_processing() {
            self.transmission_texture_id = None;
            return;
        }

        let size = ctx.wgpu_ctx.size();
        let mip_level_count = ((size.0.max(size.1) as f32).log2().floor() as u32) + 1;

        let tex_id = ctx.transient_pool.allocate(
            &ctx.wgpu_ctx.device,
            &TransientTextureDesc {
                width: size.0,
                height: size.1,
                format: crate::renderer::HDR_TEXTURE_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                mip_level_count,
                label: "Transmission",
            },
        );

        self.transmission_texture_id = Some(tex_id);
        // Publish the ID so TransparentPass can build group 3 with the real transmission view.
        ctx.render_lists.transmission_texture_id = Some(tex_id);
    }

    fn run(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let render_lists = &ctx.render_lists;

        // Check if execution is needed
        if !render_lists.use_transmission {
            return;
        }

        // Transmission requires HighFidelity path
        if !ctx.wgpu_ctx.render_path.supports_post_processing() {
            return;
        }

        // Get transient transmission texture
        let Some(tex_id) = self.transmission_texture_id else {
            log::warn!("TransmissionCopyPass: transmission_texture_id missing, skipping");
            return;
        };
        let transmission_texture = ctx.transient_pool.get_texture(tex_id);

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
        let src_texture = color_view.texture();
        let src_size = src_texture.size();

        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: src_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: transmission_texture,
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

        ctx.resource_manager.mipmap_generator.generate(
            &ctx.wgpu_ctx.device,
            encoder,
            transmission_texture,
        );
    }
}
