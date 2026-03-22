use crate::{ImageHandle, TextureHandle};
use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use uuid::Uuid;
use wgpu::{AddressMode, TextureViewDimension};

/// Texture source specifier.
///
/// Allows materials to reference textures from the `AssetServer` or
/// internal render target attachments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureSource {
    /// Asset from `AssetServer` (with version tracking and automatic upload)
    Asset(TextureHandle),
    /// Pure GPU resource (e.g., Render Target), directly using its Resource ID.
    /// This ID is typically assigned by `RenderGraph` or `TexturePool`.
    Attachment(u64, TextureViewDimension),
}

impl From<TextureHandle> for TextureSource {
    fn from(handle: TextureHandle) -> Self {
        Self::Asset(handle)
    }
}

impl From<TextureHandle> for Option<TextureSource> {
    fn from(handle: TextureHandle) -> Self {
        Some(TextureSource::Asset(handle))
    }
}

// ============================================================================
// Sampler configuration
// ============================================================================

/// Pure-data sampler descriptor used as both a configuration value on
/// [`Texture`] and a hash-map key in the render backend's sampler cache.
///
/// `PartialEq`, `Eq` and `Hash` are implemented manually so that the
/// floating-point LOD clamp fields are compared via their bit patterns,
/// making the type safe in hash-based collections.
#[derive(Debug, Clone, Copy)]
pub struct TextureSampler {
    pub address_mode_u: wgpu::AddressMode,
    pub address_mode_v: wgpu::AddressMode,
    pub address_mode_w: wgpu::AddressMode,
    pub mag_filter: wgpu::FilterMode,
    pub min_filter: wgpu::FilterMode,
    pub mipmap_filter: wgpu::MipmapFilterMode,
    /// Comparison function (for Shadow Map PCF).
    pub compare: Option<wgpu::CompareFunction>,
    /// Anisotropic filtering level (1 = disabled).
    pub anisotropy_clamp: Option<u16>,
    /// Minimum LOD clamp.
    pub lod_min_clamp: f32,
    /// Maximum LOD clamp.
    pub lod_max_clamp: f32,
    /// Border colour (only relevant with `ClampToBorder` address mode).
    pub border_color: Option<wgpu::SamplerBorderColor>,
}

impl Default for TextureSampler {
    fn default() -> Self {
        Self {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            compare: None,
            anisotropy_clamp: None,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            border_color: None,
        }
    }
}

impl TextureSampler {
    /// Trilinear filtering, all-axes clamped.
    pub const LINEAR_CLAMP: Self = Self {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 32.0,
        compare: None,
        anisotropy_clamp: Some(1),
        border_color: None,
    };
}

impl PartialEq for TextureSampler {
    fn eq(&self, other: &Self) -> bool {
        self.address_mode_u == other.address_mode_u
            && self.address_mode_v == other.address_mode_v
            && self.address_mode_w == other.address_mode_w
            && self.mag_filter == other.mag_filter
            && self.min_filter == other.min_filter
            && self.mipmap_filter == other.mipmap_filter
            && self.lod_min_clamp.to_bits() == other.lod_min_clamp.to_bits()
            && self.lod_max_clamp.to_bits() == other.lod_max_clamp.to_bits()
            && self.compare == other.compare
            && self.anisotropy_clamp == other.anisotropy_clamp
            && self.border_color == other.border_color
    }
}

impl Eq for TextureSampler {}

impl Hash for TextureSampler {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.address_mode_u.hash(state);
        self.address_mode_v.hash(state);
        self.address_mode_w.hash(state);
        self.mag_filter.hash(state);
        self.min_filter.hash(state);
        self.mipmap_filter.hash(state);
        self.lod_min_clamp.to_bits().hash(state);
        self.lod_max_clamp.to_bits().hash(state);
        self.compare.hash(state);
        self.anisotropy_clamp.hash(state);
        self.border_color.hash(state);
    }
}

// ============================================================================
// Texture Asset
// ============================================================================

/// Lightweight "glue" that pairs an [`Image`](crate::image::Image) (via
/// handle) with sampling and view configuration.
///
/// `Texture` is intentionally thin — the heavy pixel data lives in the
/// [`Image`] stored separately in `AssetServer.images`. This decoupling
/// enables multiple `Texture` assets to reference the same `Image` with
/// different sampler or view settings, and makes the version-driven GPU
/// upload path straightforward.
#[derive(Debug)]
pub struct Texture {
    pub uuid: Uuid,
    pub name: Option<Cow<'static, str>>,
    /// Handle into `AssetServer.images`.
    pub image: ImageHandle,
    pub view_dimension: TextureViewDimension,
    pub sampler: TextureSampler,
    pub generate_mipmaps: bool,
}

impl Texture {
    /// Returns the unique identifier for this texture.
    #[inline]
    #[must_use]
    pub fn uuid(&self) -> Uuid {
        self.uuid
    }

    /// Creates a `Texture` referencing the given image handle.
    #[must_use]
    pub fn new(
        name: Option<&str>,
        image: ImageHandle,
        view_dimension: TextureViewDimension,
    ) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            name: name.map(|s| Cow::Owned(s.to_string())),
            image,
            view_dimension,
            sampler: TextureSampler::default(),
            generate_mipmaps: false,
        }
    }

    /// Convenience: creates a 2D texture referencing the given image handle.
    #[must_use]
    pub fn new_2d(name: Option<&str>, image: ImageHandle) -> Self {
        Self::new(name, image, TextureViewDimension::D2)
    }

    /// Convenience: creates a 3D texture (e.g. LUT) referencing the given handle.
    #[must_use]
    pub fn new_3d(name: Option<&str>, image: ImageHandle) -> Self {
        let mut tex = Self::new(name, image, TextureViewDimension::D3);
        tex.sampler.address_mode_u = AddressMode::ClampToEdge;
        tex.sampler.address_mode_v = AddressMode::ClampToEdge;
        tex.sampler.address_mode_w = AddressMode::ClampToEdge;
        tex
    }

    /// Convenience: creates a Cube Map texture referencing the given handle.
    #[must_use]
    pub fn new_cube(name: Option<&str>, image: ImageHandle) -> Self {
        let mut tex = Self::new(name, image, TextureViewDimension::Cube);
        tex.sampler.address_mode_u = AddressMode::ClampToEdge;
        tex.sampler.address_mode_v = AddressMode::ClampToEdge;
        tex.sampler.address_mode_w = AddressMode::ClampToEdge;
        tex
    }

    /// Returns the name as a string slice, if present.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}
