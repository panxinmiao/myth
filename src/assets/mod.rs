pub mod server;
pub mod loaders;
pub mod skeleton_asset;
pub mod handle;
pub mod io;
pub mod storage;

pub use server::{
    AssetServer,
    GeometryHandle, MaterialHandle, TextureHandle, SamplerHandle,
};

#[cfg(feature = "gltf")]
pub use loaders::GltfLoader;
pub use handle::{AssetTracker, StrongHandle, WeakHandle, TrackedAsset};
pub use io::{AssetReader, AssetReaderVariant, FileAssetReader};
#[cfg(feature = "http")]
pub use io::HttpAssetReader;


use image::GenericImageView;
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

/// 加载 HDR 格式的环境贴图 (Equirectangular format)
/// 返回一个 2D 纹理，格式为 Rgba16Float，可用于 IBL
pub fn load_hdr_texture(path: impl AsRef<Path>) -> anyhow::Result<crate::resources::texture::Texture> {
    let img = image::open(&path).context("Failed to open HDR file")?;
    
    let width = img.width();
    let height = img.height();
    
    let rgb32f = img.into_rgb32f();
    
    let mut rgba_f16_data = Vec::with_capacity((width * height * 4) as usize * 2);
    
    for pixel in rgb32f.pixels() {
        let r = half::f16::from_f32(pixel[0]);
        let g = half::f16::from_f32(pixel[1]);
        let b = half::f16::from_f32(pixel[2]);
        let a = half::f16::from_f32(1.0);
        
        rgba_f16_data.extend_from_slice(&r.to_le_bytes());
        rgba_f16_data.extend_from_slice(&g.to_le_bytes());
        rgba_f16_data.extend_from_slice(&b.to_le_bytes());
        rgba_f16_data.extend_from_slice(&a.to_le_bytes());
    }

    let image = crate::resources::image::Image::new(
        path.as_ref().to_str(),
        width,
        height,
        1,
        wgpu::TextureDimension::D2,
        wgpu::TextureFormat::Rgba16Float,
        Some(rgba_f16_data),
    );

    let mut texture = crate::resources::texture::Texture::new(
        path.as_ref().to_str(),
        image,
        wgpu::TextureViewDimension::D2,
    );
    
    texture.sampler.address_mode_u = wgpu::AddressMode::ClampToEdge;
    texture.sampler.address_mode_v = wgpu::AddressMode::ClampToEdge;
    texture.sampler.mag_filter = wgpu::FilterMode::Linear;
    texture.sampler.min_filter = wgpu::FilterMode::Linear;
    
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