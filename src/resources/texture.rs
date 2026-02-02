use uuid::Uuid;
use std::sync::atomic::{AtomicU64, Ordering};
use std::borrow::Cow;
use wgpu::{TextureFormat, TextureDimension, TextureViewDimension, AddressMode};
use crate::{assets::{TextureHandle, server::SamplerHandle}, resources::image::Image};


/// Texture source specifier.
///
/// Allows materials to reference textures from the AssetServer or
/// internal render target attachments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TextureSource {
    /// Asset from AssetServer (with version tracking and automatic upload)
    Asset(TextureHandle),
    /// Pure GPU resource (e.g., Render Target), directly using its Resource ID
    /// This ID is typically assigned by RenderGraph or TexturePool
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


/// Sampler source strategy for texture binding.
///
/// Specifies which sampler to use when binding a texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SamplerSource {
    /// Automatic matching: Uses the sampler settings associated with the texture asset.
    /// ResourceManager looks up the sampler from the TextureHandle's metadata.
    FromTexture(TextureHandle),

    /// Explicit specification: Uses a specific Sampler Asset.
    Asset(SamplerHandle),


    Default,

}

// Syntactic sugar: allows deriving SamplerSource directly from TextureHandle
impl From<TextureHandle> for SamplerSource {
    fn from(handle: TextureHandle) -> Self {
        Self::FromTexture(handle)
    }
}

impl From<SamplerHandle> for SamplerSource {
    fn from(handle: SamplerHandle) -> Self {
        Self::Asset(handle)
    }
}

impl From<TextureSource> for SamplerSource {
    fn from(texture_source: TextureSource) -> Self {
        match texture_source {
            TextureSource::Asset(handle) => Self::FromTexture(handle),
            TextureSource::Attachment(_, _) => Self::Default,
        }
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureSampler {
    pub address_mode_u: wgpu::AddressMode,
    pub address_mode_v: wgpu::AddressMode,
    pub address_mode_w: wgpu::AddressMode,
    pub mag_filter: wgpu::FilterMode,
    pub min_filter: wgpu::FilterMode,
    pub mipmap_filter: wgpu::MipmapFilterMode,

    /// Advanced: comparison function (for Shadow Map PCF)
    pub compare: Option<wgpu::CompareFunction>,
    /// Advanced: anisotropic filtering level (1 = disabled)
    pub anisotropy_clamp: u16,

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
            anisotropy_clamp: 1,
        }
    }
}


impl From<&Sampler> for TextureSampler {
    fn from(asset: &Sampler) -> Self {
        asset.descriptor
    }
}


// ============================================================================
// Sampler Asset (standalone sampler resource)
// ============================================================================

#[derive(Debug, Clone)]
pub struct Sampler {
    pub uuid: Uuid,

    pub name: Option<Cow<'static, str>>,
    
    /// Core sampling parameters
    pub descriptor: TextureSampler,
}

impl Sampler {
    pub fn new(descriptor: TextureSampler) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            // #[cfg(debug_assertions)]
            name: None,
            descriptor,
        }
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

}

// Since TextureSampler implements Default, Sampler can implement it too
impl Default for Sampler {
    fn default() -> Self {
        Self::new(TextureSampler::default())
    }
}

// ============================================================================
// 2. Texture Asset
// ============================================================================

#[derive(Debug)]
pub struct Texture {
    pub uuid: Uuid,

    // #[cfg(debug_assertions)]
    pub name: Option<Cow<'static, str>>,
    
    pub image: Image,

    pub view_dimension: TextureViewDimension,

    pub sampler: TextureSampler,
    //pub transform: TextureTransform,

    pub generate_mipmaps: bool,
    
    version: AtomicU64,
}

impl Texture {
    /// Creates a Texture from an existing Image.
    pub fn new(name: Option<&str>, image: Image, view_dimension: TextureViewDimension) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            // #[cfg(debug_assertions)]
            name: name.map(|s| Cow::Owned(s.to_string())),
            image,
            view_dimension,
            sampler: TextureSampler::default(),
            // transform: TextureTransform::default(),
            generate_mipmaps: false,
            version: AtomicU64::new(0),
        }
    }

    /// Creates a 2D texture (automatically creates the Image).
    pub fn new_2d(name: Option<&str>, width: u32, height: u32, data: Option<Vec<u8>>, format: TextureFormat) -> Self {
        let image = Image::new(
            name, width, height, 1, 
            TextureDimension::D2, 
            format, data
        );
        Self::new(name, image, TextureViewDimension::D2)
    }

    /// Creates a Cube Map texture.
    pub fn new_cube(name: Option<&str>, size: u32, data: Option<Vec<u8>>, format: TextureFormat) -> Self {
        let image = Image::new(
            name, size, size, 6, // 6 layers
            TextureDimension::D2, // Physical dimension is 2D
            format, data
        );
        let mut tex = Self::new(name, image, TextureViewDimension::Cube);
        // Cube maps default to Clamp sampling
        tex.sampler.address_mode_u = AddressMode::ClampToEdge;
        tex.sampler.address_mode_v = AddressMode::ClampToEdge;
        tex.sampler.address_mode_w = AddressMode::ClampToEdge;
        tex
    }

    pub fn mip_level_count(&self) -> u32 {
        if !self.generate_mipmaps {
            return 1;
        }
        let w = self.image.width();
        let h = self.image.height();
        let max_dim = std::cmp::max(w, h);
        if max_dim == 0 { return 1; }
        (max_dim as f32).log2().floor() as u32 + 1
    }

    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Creates a solid color texture (1x1).
    pub fn create_solid_color(name: Option<&str>, color: [u8; 4]) -> Texture {
        Self::new_2d(name, 1, 1, Some(color.to_vec()), wgpu::TextureFormat::Rgba8UnormSrgb)
    }

    pub fn needs_update(&mut self) {
        self.version.fetch_add(1, Ordering::Relaxed);
    }


    /// Creates a checkerboard test texture.
    pub fn create_checkerboard(name: Option<&str>, width: u32, height: u32, check_size: u32) -> Self {
        let mut data = Vec::with_capacity((width * height * 4) as usize);
        
        let color_a = [255, 255, 255, 255]; // White
        let color_b = [0, 0, 0, 255];       // Black (or use pink [255, 0, 255, 255] for debugging)

        for y in 0..height {
            for x in 0..width {
                // Simple XOR logic to generate checkerboard pattern
                let cx = x / check_size;
                let cy = y / check_size;
                let is_a = (cx + cy).is_multiple_of(2);
                
                if is_a {
                    data.extend_from_slice(&color_a);
                } else {
                    data.extend_from_slice(&color_b);
                }
            }
        }

        Self::new_2d(name, width, height, Some(data), wgpu::TextureFormat::Rgba8Unorm)
    }
}