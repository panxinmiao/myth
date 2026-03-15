use crate::renderer::graph::core::{ExecuteContext, PassNode, TextureNodeId};

pub struct CopyTextureNode {
    pub src: TextureNodeId,
    pub dst: TextureNodeId,
}

impl<'a> PassNode<'a> for CopyTextureNode {

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let src_texture = ctx.get_texture(self.src); 
        let dst_texture = ctx.get_texture(self.dst);

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
                width: src_texture.size().width,
                height: src_texture.size().height,
                depth_or_array_layers: src_texture.size().depth_or_array_layers,
            },
        );
    }

}