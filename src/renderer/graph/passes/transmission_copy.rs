//! Transmission Copy Pass
//!
//! 将当前场景颜色缓冲复制到 Transmission 纹理，
//! 供透明物体的折射效果使用。
//!
//! # 数据流
//! ```text
//! HDR Scene Color → TransmissionCopyPass → Transmission Texture
//! ```
//!
//! # 执行时机
//! - 在 OpaquePass 之后
//! - 在 TransparentPass 之前
//!
//! # 注意
//! - 此 Pass 不使用 RenderPass，直接使用 `encoder.copy_texture_to_texture`
//! - 仅在场景中存在使用 Transmission 的材质时执行

use crate::renderer::graph::{RenderContext, RenderNode};

/// Transmission Copy Pass
///
/// 将场景颜色缓冲复制到 Transmission 纹理。
/// 此 Pass 不进行绘制，仅执行纹理拷贝。
///
/// # 条件执行
/// - 仅当 `render_lists.use_transmission` 为 true 时执行
/// - 仅当 HDR 模式启用时执行（Transmission 需要 HDR 缓冲）
/// - 仅当 `transmission_view` 存在时执行
pub struct TransmissionCopyPass;

impl TransmissionCopyPass {
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
    fn name(&self) -> &str {
        "Transmission Copy Pass"
    }

    fn run(&self, ctx: &mut RenderContext, encoder: &mut wgpu::CommandEncoder) {
        let render_lists = &ctx.render_frame.render_lists;

        // 检查是否需要执行
        if !render_lists.use_transmission {
            return;
        }


        // Transmission 需要 HDR 模式
        if !ctx.wgpu_ctx.enable_hdr {
            log::warn!(
                "TransmissionCopyPass: Transmission requires HDR mode, skipping"
            );
            return;
        }

        // 检查 Transmission 纹理是否存在
        let Some(transmission_view) = &ctx.frame_resources.transmission_view else {
            log::warn!(
                "TransmissionCopyPass: transmission_view missing, skipping"
            );
            return;
        };

        // 获取源纹理（场景颜色缓冲）
        let color_view = ctx.get_scene_render_target_view();

        // 执行纹理拷贝
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
            &transmission_view.texture(), 
        );
    }
}
