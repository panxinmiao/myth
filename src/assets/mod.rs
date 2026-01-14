pub mod server;

// 重新导出 AssetServer 及相关类型
pub use server::{
    AssetServer,
    GeometryHandle, MaterialHandle, TextureHandle,
};


use image::{GenericImageView};
use std::path::Path;
use anyhow::Context;

pub fn load_image_from_file(path: impl AsRef<Path>) -> anyhow::Result<(Vec<u8>, u32, u32)> {
    let img = image::open(path).context("Failed to open image file")?;
    
    let (width, height) = img.dimensions();
    
    let rgba_image = img.into_rgba8();
    
    let data = rgba_image.into_raw();
    
    Ok((data, width, height))
}

pub enum ColorSpace {
    Srgb,
    Linear,
}

pub fn load_texture_from_file(path: impl AsRef<Path>, color_space: ColorSpace) -> anyhow::Result<crate::resources::texture::Texture> {
    let (data, width, height) = load_image_from_file(&path)?;
    
    let texture = crate::resources::texture::Texture::new_2d(
        path.as_ref().to_str(),
        width,
        height,
        Some(data),
        match color_space {
            ColorSpace::Srgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            ColorSpace::Linear => wgpu::TextureFormat::Rgba8Unorm,
        }
    );
    
    Ok(texture)
}