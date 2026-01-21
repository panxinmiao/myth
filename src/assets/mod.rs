pub mod server;
pub mod loaders;
pub mod skeleton_asset;
pub mod handle;

// 重新导出 AssetServer 及相关类型
pub use server::{
    AssetServer,
    GeometryHandle, MaterialHandle, TextureHandle, SamplerHandle,
};
pub use loaders::GltfLoader;
pub use handle::{AssetTracker, StrongHandle, WeakHandle, TrackedAsset};


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

pub fn load_cube_texture_from_files(paths: [impl AsRef<Path>; 6], color_space: ColorSpace) -> anyhow::Result<crate::resources::texture::Texture> {
    let mut face_data = Vec::with_capacity(6);
    let mut width = 0;
    let mut height = 0;
    
    for path in paths.iter() {
        let (data, w, h) = load_image_from_file(path)?;
        if width == 0 && height == 0 {
            width = w;
            height = h;
        } else {
            if width != w || height != h {
                return Err(anyhow::anyhow!("Cube texture faces must have the same dimensions"));
            }
        }
        face_data.push(data);
    }
    
    // 合并六个面的数据
    let mut combined_data = Vec::with_capacity((width * height * 4 * 6) as usize);
    for face in face_data.iter() {
        combined_data.extend_from_slice(face);
    }
    
    let texture = crate::resources::texture::Texture::new_cube(
        None,
        width,
        Some(combined_data),
        match color_space {
            ColorSpace::Srgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            ColorSpace::Linear => wgpu::TextureFormat::Rgba8Unorm,
        }
    );
    
    Ok(texture)
}   