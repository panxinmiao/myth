//! Asset Loading and Management System
//!
//! This module provides centralized asset storage and loading capabilities:
//!
//! - [`AssetServer`]: Central registry for all asset types
//! - [`AssetStorage`]: Generic storage for asset collections
//! - [`Prefab`]: Scene prefab system for instancing
//! - Loaders for various formats (glTF, textures, HDR)
//!
//! # Supported Formats
//!
//! - **glTF/GLB**: Full glTF 2.0 support including PBR materials, animations, and skinning
//! - **Images**: PNG, JPEG, and other formats via the `image` crate
//! - **HDR**: High Dynamic Range environment maps (Equirectangular format)
//!
//! # Handle System
//!
//! All assets are referenced via strongly-typed handles:
//! - [`GeometryHandle`]: References to geometry data
//! - [`MaterialHandle`]: References to material instances
//! - [`TextureHandle`]: References to texture assets
//! - [`SamplerHandle`]: References to sampler configurations
//!
//! These handles are lightweight (8 bytes) and can be freely copied.
//!
//! # Example
//!
//! ```rust,ignore
//! // Load a texture
//! let handle = assets.load_texture_from_file("diffuse.png", ColorSpace::Srgb, true)?;
//!
//! // Load HDR environment
//! let env_handle = assets.load_hdr_texture("environment.hdr")?;
//!
//! // Load glTF model (returns a Prefab)
//! let prefab = GltfLoader::load(&assets, "model.gltf").await?;
//! prefab.instantiate(&mut scene);
//! ```

pub mod handle;
pub mod io;
pub mod loaders;
pub mod prefab;
pub mod server;
pub mod skeleton_asset;
pub mod storage;

pub use server::{AssetServer, GeometryHandle, MaterialHandle, SamplerHandle, TextureHandle};

pub use handle::{AssetTracker, StrongHandle, TrackedAsset, WeakHandle};
pub use io::{AssetReader, AssetReaderVariant};
#[cfg(feature = "gltf")]
pub use loaders::GltfLoader;
pub use prefab::{Prefab, PrefabNode, PrefabSkeleton, SharedPrefab};

#[cfg(not(target_arch = "wasm32"))]
pub use io::FileAssetReader;

#[cfg(feature = "http")]
pub use io::HttpAssetReader;

use crate::errors::{MythError, Result};
use image::GenericImageView;
use std::path::Path;

pub fn load_image_from_file(path: impl AsRef<Path>) -> Result<(Vec<u8>, u32, u32)> {
    let img = image::open(&path)?;

    let (width, height) = img.dimensions();

    let rgba_image = img.into_rgba8();

    let data = rgba_image.into_raw();

    Ok((data, width, height))
}

#[derive(Debug, Clone, Copy)]
pub enum ColorSpace {
    Srgb,
    Linear,
}

pub fn load_texture_from_file(
    path: impl AsRef<Path>,
    color_space: ColorSpace,
) -> Result<crate::resources::texture::Texture> {
    let (data, width, height) = load_image_from_file(&path)?;

    let texture = crate::resources::texture::Texture::new_2d(
        path.as_ref().to_str(),
        width,
        height,
        Some(data),
        match color_space {
            ColorSpace::Srgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            ColorSpace::Linear => wgpu::TextureFormat::Rgba8Unorm,
        },
    );

    Ok(texture)
}

/// Loads an HDR environment map in Equirectangular format.
///
/// Returns a 2D texture with `Rgba16Float` format suitable for IBL.
///
/// # Arguments
///
/// * `path` - Path to the HDR file
///
/// # Returns
///
/// A texture that can be used for image-based lighting.
pub fn load_hdr_texture_from_file(
    path: impl AsRef<Path>,
) -> Result<crate::resources::texture::Texture> {
    let img = image::open(&path)?;

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

pub fn load_cube_texture_from_files(
    paths: &[impl AsRef<Path>; 6],
    color_space: ColorSpace,
) -> Result<crate::resources::texture::Texture> {
    let mut face_data = Vec::with_capacity(6);
    let mut width = 0;
    let mut height = 0;

    for path in paths {
        let (data, w, h) = load_image_from_file(path)?;
        if width == 0 && height == 0 {
            width = w;
            height = h;
        } else if width != w || height != h {
            return Err(MythError::CubeMapError(
                "Cube texture faces must have the same dimensions".to_string(),
            ));
        }
        face_data.push(data);
    }

    // Merge all six faces' data
    let mut combined_data = Vec::with_capacity((width * height * 4 * 6) as usize);
    for face in &face_data {
        combined_data.extend_from_slice(face);
    }

    let texture = crate::resources::texture::Texture::new_cube(
        None,
        width,
        Some(combined_data),
        match color_space {
            ColorSpace::Srgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            ColorSpace::Linear => wgpu::TextureFormat::Rgba8Unorm,
        },
    );

    Ok(texture)
}
