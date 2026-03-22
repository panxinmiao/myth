//! Raw image data storage.
//!
//! [`Image`] is a pure CPU-side data container holding pixel bytes and
//! format metadata. It owns no GPU resources, locks, or atomic state —
//! all concurrency is managed externally by [`AssetStorage`].

use uuid::Uuid;

/// Compact format/dimension metadata for an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageDescriptor {
    pub dimension: wgpu::TextureDimension,
    pub format: wgpu::TextureFormat,
}

/// CPU-side image data.
///
/// A lightweight, purely owned container of pixel bytes plus the minimal
/// metadata the GPU uploader needs (size, format, dimension).
///
/// # Design
///
/// `Image` deliberately carries **no** `Arc`, `RwLock`, or atomic fields.
/// Thread-safe shared access is provided by the [`AssetStorage`] layer,
/// which wraps every stored asset in `Arc` and guards mutations with a
/// version counter. This keeps `Image` trivially movable and avoids
/// hidden synchronisation costs on the hot read path.
#[derive(Debug)]
pub struct Image {
    pub uuid: Uuid,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_level_count: u32,
    pub description: ImageDescriptor,
    /// Raw pixel bytes. `None` indicates a placeholder awaiting async load.
    pub data: Option<Vec<u8>>,
}

impl Image {
    /// Creates a new image with the given dimensions and optional pixel data.
    #[must_use]
    pub fn new(
        width: u32,
        height: u32,
        depth_or_array_layers: u32,
        dimension: wgpu::TextureDimension,
        format: wgpu::TextureFormat,
        data: Option<Vec<u8>>,
    ) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            width,
            height,
            depth: depth_or_array_layers,
            mip_level_count: 1,
            description: ImageDescriptor { dimension, format },
            data,
        }
    }

    /// Shorthand for the texel format.
    #[inline]
    #[must_use]
    pub fn format(&self) -> wgpu::TextureFormat {
        self.description.format
    }

    /// Shorthand for the texture dimension.
    #[inline]
    #[must_use]
    pub fn dimension(&self) -> wgpu::TextureDimension {
        self.description.dimension
    }

    /// Creates a 1×1 RGBA8 sRGB image with the specified color.
    #[must_use]
    pub fn solid_color(rgba: [u8; 4]) -> Self {
        Self::new(
            1,
            1,
            1,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            Some(rgba.to_vec()),
        )
    }

    /// Creates a checkerboard pattern image.
    #[must_use]
    pub fn checkerboard(width: u32, height: u32, cell_size: u32) -> Self {
        let mut data = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                let is_white = ((x / cell_size) + (y / cell_size)).is_multiple_of(2);
                let c = if is_white { 255u8 } else { 80u8 };
                data.extend_from_slice(&[c, c, c, 255]);
            }
        }
        Self::new(
            width,
            height,
            1,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            Some(data),
        )
    }
}
