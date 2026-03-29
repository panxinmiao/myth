//! Raw image data storage.
//!
//! [`Image`] is a pure CPU-side data container holding pixel bytes and
//! format metadata. It owns no GPU resources and carries no graphics API
//! dependencies — all format/dimension types are engine-native enums that
//! map to `wgpu` equivalents only at GPU upload time.
//!
//! # Key types
//!
//! * [`PixelFormat`] — describes the in-memory pixel layout without any
//!   colour-space semantics (sRGB vs Linear is decided by [`Texture`]).
//! * [`ImageDimension`] — spatial dimensionality (1D / 2D / 3D).
//! * [`ColorSpace`] — rendering intent; lives on [`Texture`] and is combined
//!   with `PixelFormat` at GPU upload time to produce the final
//!   `wgpu::TextureFormat`.

use uuid::Uuid;

// ────────────────────────────────────────────────────────────────────────────
// Engine-native pixel format
// ────────────────────────────────────────────────────────────────────────────

/// GPU-independent pixel format describing the in-memory byte layout.
///
/// This enum intentionally does **not** encode colour-space information
/// (sRGB vs Linear). Colour-space is a rendering decision owned by
/// [`Texture::color_space`](crate::texture::Texture::color_space) and
/// resolved at GPU upload time via [`PixelFormat::to_wgpu`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// 4 × 8-bit unsigned normalised RGBA.
    Rgba8Unorm,
    /// 4 × 16-bit IEEE 754 half-precision float RGBA.
    Rgba16Float,
    /// Single-channel 8-bit unsigned normalised.
    R8Unorm,
}

impl PixelFormat {
    /// Resolves the final `wgpu::TextureFormat` by combining the physical
    /// pixel layout with the requested colour space.
    ///
    /// Formats that have no sRGB variant (e.g. `Rgba16Float`) ignore the
    /// colour-space argument and always return the linear variant.
    #[must_use]
    pub fn to_wgpu(self, color_space: ColorSpace) -> wgpu::TextureFormat {
        match (self, color_space) {
            (Self::Rgba8Unorm, ColorSpace::Srgb) => wgpu::TextureFormat::Rgba8UnormSrgb,
            (Self::Rgba8Unorm, ColorSpace::Linear) => wgpu::TextureFormat::Rgba8Unorm,
            (Self::Rgba16Float, _) => wgpu::TextureFormat::Rgba16Float,
            (Self::R8Unorm, _) => wgpu::TextureFormat::R8Unorm,
        }
    }

    /// Bytes occupied by a single texel (or compressed block) in this format.
    #[inline]
    #[must_use]
    pub const fn block_copy_size(self) -> u32 {
        match self {
            Self::Rgba8Unorm => 4,
            Self::Rgba16Float => 8,
            Self::R8Unorm => 1,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Engine-native image dimension
// ────────────────────────────────────────────────────────────────────────────

/// Spatial dimensionality of an image's texel grid.
///
/// Maps 1:1 to `wgpu::TextureDimension` but lives in engine-native code
/// so that [`Image`] carries no `wgpu` dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageDimension {
    D1,
    D2,
    D3,
}

impl ImageDimension {
    /// Converts to the corresponding `wgpu::TextureDimension`.
    #[inline]
    #[must_use]
    pub fn to_wgpu(self) -> wgpu::TextureDimension {
        match self {
            Self::D1 => wgpu::TextureDimension::D1,
            Self::D2 => wgpu::TextureDimension::D2,
            Self::D3 => wgpu::TextureDimension::D3,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Colour space
// ────────────────────────────────────────────────────────────────────────────

/// Describes how pixel data should be interpreted in the GPU shader.
///
/// This is a **rendering intent** — the same physical pixel bytes can be
/// displayed as sRGB (e.g. base colour map) or treated as linear data
/// (e.g. roughness / normal map) depending on the material's needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    /// Non-linear sRGB encoding. Hardware performs automatic sRGB → linear
    /// conversion on texture fetch.
    Srgb,
    /// Linear encoding. No conversion is applied.
    Linear,
}

// ────────────────────────────────────────────────────────────────────────────
// Image
// ────────────────────────────────────────────────────────────────────────────

/// CPU-side image data.
///
/// A lightweight, purely owned container of pixel bytes plus the minimal
/// metadata the GPU uploader needs (size, format, dimension).
///
/// `Image` deliberately carries **no** `Arc`, `RwLock`, atomic fields, or
/// `wgpu` types. Thread-safe shared access is provided by the
/// [`AssetStorage`] layer, which wraps every stored asset in `Arc` and
/// guards mutations with a version counter.
///
/// Colour-space semantics are intentionally absent — the same image data
/// can be upload as sRGB or Linear depending on the [`Texture`] that
/// references it.
#[derive(Debug)]
pub struct Image {
    pub uuid: Uuid,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_level_count: u32,
    pub dimension: ImageDimension,
    pub format: PixelFormat,
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
        dimension: ImageDimension,
        format: PixelFormat,
        data: Option<Vec<u8>>,
    ) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            width,
            height,
            depth: depth_or_array_layers,
            mip_level_count: 1,
            dimension,
            format,
            data,
        }
    }

    /// Returns the pixel format.
    #[inline]
    #[must_use]
    pub fn format(&self) -> PixelFormat {
        self.format
    }

    /// Returns the spatial dimensionality.
    #[inline]
    #[must_use]
    pub fn dimension(&self) -> ImageDimension {
        self.dimension
    }

    /// Creates a 1×1 RGBA8 image with the specified colour.
    #[must_use]
    pub fn solid_color(rgba: [u8; 4]) -> Self {
        Self::new(1, 1, 1, ImageDimension::D2, PixelFormat::Rgba8Unorm, Some(rgba.to_vec()))
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
        Self::new(width, height, 1, ImageDimension::D2, PixelFormat::Rgba8Unorm, Some(data))
    }
}
