use crate::renderer::graph::{
    composer::GraphBuilderContext,
    core::{ExecuteContext, PassNode, RenderTargetOps, TextureDesc, TextureNodeId},
};

pub struct CopyTextureNode {
    pub src: TextureNodeId,
    pub dst: TextureNodeId,
}

impl PassNode<'_> for CopyTextureNode {
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

pub fn add_copy_texture_pass(
    ctx: &mut GraphBuilderContext<'_, '_>,
    src_texture: TextureNodeId,
    dst_name: &'static str,
    dst_desc: TextureDesc,
) -> TextureNodeId {
    ctx.graph.add_pass("Copy_Texture_Pass", |builder| {
        builder.read_texture(src_texture);
        let dst_texture = builder.create_texture(dst_name, dst_desc);
        (
            CopyTextureNode {
                src: src_texture,
                dst: dst_texture,
            },
            dst_texture,
        )
    })
}

pub fn add_copy_texture_pass_into(
    ctx: &mut GraphBuilderContext<'_, '_>,
    src_texture: TextureNodeId,
    dst_texture: TextureNodeId,
) -> TextureNodeId {
    ctx.graph.add_pass("Copy_Texture_Pass", |builder| {
        builder.read_texture(src_texture);
        builder.write_texture(dst_texture);
        (
            CopyTextureNode {
                src: src_texture,
                dst: dst_texture,
            },
            dst_texture,
        )
    })
}

pub struct MsaaResolveNode {
    pub src_msaa: TextureNodeId,
    pub dst_resolved: TextureNodeId,
}

impl PassNode<'_> for MsaaResolveNode {
    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let msaa_att = ctx.get_color_attachment(
            self.src_msaa,
            RenderTargetOps::Load,
            Some(self.dst_resolved),
        );

        let pass_desc = wgpu::RenderPassDescriptor {
            label: Some("MSAA_Resolve_Pass"),
            color_attachments: &[msaa_att],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };

        let _ = encoder.begin_render_pass(&pass_desc);
    }
}

pub fn add_msaa_resolve_pass(
    ctx: &mut GraphBuilderContext<'_, '_>,
    src_msaa: TextureNodeId,
    desc: TextureDesc,
) -> TextureNodeId {
    ctx.graph.add_pass("Msaa_Resolve_Pass", |builder| {
        let dst_resolved = builder.create_texture("Resolved_Color", desc);
        builder.read_texture(src_msaa);
        (
            MsaaResolveNode {
                src_msaa,
                dst_resolved,
            },
            dst_resolved,
        )
    })
}

pub fn add_msaa_resolve_pass_into(
    ctx: &mut GraphBuilderContext<'_, '_>,
    src_msaa: TextureNodeId,
    dst_resolved: TextureNodeId,
) -> TextureNodeId {
    ctx.graph.add_pass("Msaa_Resolve_Pass", |builder| {
        builder.read_texture(src_msaa);
        builder.write_texture(dst_resolved);
        (
            MsaaResolveNode {
                src_msaa,
                dst_resolved,
            },
            dst_resolved,
        )
    })
}

pub struct GenerateMipmapNode {
    pub texture: TextureNodeId,
}

impl PassNode<'_> for GenerateMipmapNode {
    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let view = ctx.get_texture_view(self.texture);
        ctx.mipmap_generator
            .generate(ctx.device, encoder, view.texture());
    }
}

pub fn add_generate_mipmap_pass(
    ctx: &mut GraphBuilderContext<'_, '_>,
    texture: TextureNodeId,
    new_name: &'static str,
) -> TextureNodeId {
    ctx.graph.add_pass("Generate_Mipmap_Pass", |builder| {
        let out = builder.mutate_texture(texture, new_name);
        (GenerateMipmapNode { texture }, out)
    })
}
