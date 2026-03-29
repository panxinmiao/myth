//! # Myth Assets
//!
//! Asset loading and management for the Myth engine.
//!
//! Provides [`AssetServer`] for centralised resource storage, loaders for
//! various formats (glTF, textures, HDR), and scene prefab/instantiation
//! helpers.

pub mod handle;
pub mod io;
pub mod loaders;
pub mod manager;
pub mod prefab;
pub mod resolve;
pub mod scene_ext;
pub mod server;
pub mod skeleton_asset;
pub mod storage;

pub use myth_resources::{
    GeometryHandle, ImageHandle, MaterialHandle, PrefabHandle, TextureHandle,
};
pub use server::AssetServer;

pub use handle::{AssetTracker, StrongHandle, TrackedAsset, WeakHandle};
pub use io::{AssetReader, AssetReaderVariant};
#[cfg(feature = "gltf")]
pub use loaders::GltfLoader;
pub use manager::{SceneHandle, SceneManager};
pub use myth_scene::GeometryQuery;
pub use prefab::{Prefab, PrefabNode, PrefabSkeleton, SharedPrefab};
pub use resolve::{ResolveGeometry, ResolveMaterial};
pub use scene_ext::SceneExt;
pub use storage::{AssetSlot, AssetStorage};

#[cfg(not(target_arch = "wasm32"))]
pub use io::FileAssetReader;

#[cfg(feature = "http")]
pub use io::HttpAssetReader;

use myth_core::{AssetError, Error, Result};
use myth_resources::image::{Image, ImageDimension, PixelFormat};
use myth_resources::texture::TextureSampler;
use std::path::Path;

pub use myth_resources::ColorSpace;

pub fn load_image_from_file(path: impl AsRef<Path>) -> Result<(Vec<u8>, u32, u32)> {
    use image::GenericImageView;

    let img = image::open(&path)
        .map_err(|e| Error::Asset(AssetError::Format(format!("Image error: {e}"))))?;

    let (width, height) = img.dimensions();

    let rgba_image = img.into_rgba8();

    let data = rgba_image.into_raw();

    Ok((data, width, height))
}

/// Returns `(Image, TextureSampler, generate_mipmaps)`.
pub fn load_texture_from_file(
    path: impl AsRef<Path>,
) -> Result<(Image, TextureSampler, bool)> {
    let (data, width, height) = load_image_from_file(&path)?;

    let image = Image::new(width, height, 1, ImageDimension::D2, PixelFormat::Rgba8Unorm, Some(data));

    Ok((image, TextureSampler::default(), false))
}

/// Loads an HDR environment map in Equirectangular format.
/// Returns `(Image, TextureSampler, generate_mipmaps)`.
pub fn load_hdr_texture_from_file(path: impl AsRef<Path>) -> Result<(Image, TextureSampler, bool)> {
    let img = image::open(&path)
        .map_err(|e| Error::Asset(AssetError::Format(format!("Image error: {e}"))))?;

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

    let image = Image::new(
        width,
        height,
        1,
        ImageDimension::D2,
        PixelFormat::Rgba16Float,
        Some(rgba_f16_data),
    );

    let sampler = TextureSampler {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..TextureSampler::default()
    };

    Ok((image, sampler, false))
}

/// Returns `(Image, TextureSampler, generate_mipmaps)`.
pub fn load_cube_texture_from_files(
    paths: &[impl AsRef<Path>; 6],
    _color_space: ColorSpace,
) -> Result<(Image, TextureSampler, bool)> {
    let mut face_data = Vec::with_capacity(6);
    let mut width = 0;
    let mut height = 0;

    for path in paths {
        let (data, w, h) = load_image_from_file(path)?;
        if width == 0 && height == 0 {
            width = w;
            height = h;
        } else if width != w || height != h {
            return Err(Error::Asset(AssetError::InvalidData(
                "Cube texture faces must have the same dimensions".to_string(),
            )));
        }
        face_data.push(data);
    }

    let mut combined_data = Vec::with_capacity((width * height * 4 * 6) as usize);
    for face in &face_data {
        combined_data.extend_from_slice(face);
    }

    let image = Image::new(
        width,
        height,
        6,
        ImageDimension::D2,
        PixelFormat::Rgba8Unorm,
        Some(combined_data),
    );

    Ok((image, TextureSampler::default(), false))
}

/// Loads a 3D LUT texture from a .cube file.
/// Returns `(Image, TextureSampler, generate_mipmaps)`.
pub fn load_lut_texture_from_file(path: impl AsRef<Path>) -> Result<(Image, TextureSampler, bool)> {
    let bytes = std::fs::read(&path)?;

    let image = server::AssetServer::decode_cube_cpu(&bytes)?;

    let sampler = TextureSampler {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..TextureSampler::default()
    };

    Ok((image, sampler, false))
}
