use parking_lot::RwLock;
use std::sync::Arc;
use wgpu::TextureFormat;

use crate::ColorSpace;
use crate::io::{AssetReaderVariant, AssetSource};
use crate::storage::AssetStorage;
use myth_core::{AssetError, Error, Result};
use myth_resources::geometry::Geometry;
use myth_resources::material::Material;
use myth_resources::screen_space::SssRegistry;
use myth_resources::texture::{Sampler, Texture};
use myth_resources::{GeometryHandle, MaterialHandle, SamplerHandle, TextureHandle};

#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;
#[cfg(not(target_arch = "wasm32"))]
use tokio::runtime::Runtime;

#[cfg(not(target_arch = "wasm32"))]
fn get_asset_runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| Runtime::new().expect("Failed to create asset loader runtime"))
}

// Strongly-typed handles are defined in myth_resources::handles

// 2. Asset Server

#[derive(Clone)] // AssetServer is now lightweight and can be cloned freely
pub struct AssetServer {
    pub geometries: Arc<AssetStorage<GeometryHandle, Geometry>>,
    pub materials: Arc<AssetStorage<MaterialHandle, Material>>,
    pub textures: Arc<AssetStorage<TextureHandle, Texture>>,
    pub samplers: Arc<AssetStorage<SamplerHandle, Sampler>>,

    pub sss_registry: Arc<RwLock<SssRegistry>>,
}

impl Default for AssetServer {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetServer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            geometries: Arc::new(AssetStorage::new()),
            materials: Arc::new(AssetStorage::new()),
            textures: Arc::new(AssetStorage::new()),
            samplers: Arc::new(AssetStorage::new()),

            sss_registry: Arc::new(RwLock::new(SssRegistry::new())),
        }
    }

    // ========================================================================
    // Legacy / Synchronous Methods (Native Only)
    // ========================================================================
    // Todo: Keep these methods for backward compatibility, but on Native we could also consider refactoring them to call block_on(async_load)

    pub fn load_texture(
        &mut self,
        source: impl AssetSource,
        color_space: ColorSpace,
        generate_mipmaps: bool,
    ) -> Result<TextureHandle> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            get_asset_runtime().block_on(self.load_texture_async(
                source,
                color_space,
                generate_mipmaps,
            ))
        }
        #[cfg(target_arch = "wasm32")]
        {
            let mut texture = crate::load_texture_from_file(source.uri().to_string(), color_space)?;
            texture.generate_mipmaps = generate_mipmaps;
            let handle = self.textures.add(texture);
            Ok(handle)
        }
    }

    pub fn load_cube_texture(
        &mut self,
        sources: [impl AssetSource; 6],
        color_space: ColorSpace,
        generate_mipmaps: bool,
    ) -> Result<TextureHandle> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            get_asset_runtime().block_on(self.load_cube_texture_async(
                sources,
                color_space,
                generate_mipmaps,
            ))
        }
        #[cfg(target_arch = "wasm32")]
        {
            let mut texture = crate::load_cube_texture_from_files(
                &sources.map(|s| s.uri().to_string()),
                color_space,
            )?;
            texture.generate_mipmaps = generate_mipmaps;
            let handle = self.textures.add(texture);
            Ok(handle)
        }
    }

    /// Loads an HDR format environment map (Equirectangular format)
    pub fn load_hdr_texture(&mut self, source: impl AssetSource) -> Result<TextureHandle> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            get_asset_runtime().block_on(self.load_hdr_texture_async(source))
        }
        #[cfg(target_arch = "wasm32")]
        {
            let texture = crate::load_hdr_texture_from_file(source.uri().to_string())?;
            let handle = self.textures.add(texture);
            Ok(handle)
        }
    }

    // ========================================================================
    // Async Methods (Cross-Platform)
    // ========================================================================

    /// Asynchronously loads a 2D texture (supports local paths or HTTP URLs).
    pub async fn load_texture_async(
        &self,
        source: impl AssetSource,
        color_space: crate::ColorSpace,
        generate_mipmaps: bool,
    ) -> Result<TextureHandle> {
        let reader = AssetReaderVariant::new(&source)?;

        let uri = source.uri();
        let filename = source
            .filename()
            .unwrap_or(std::borrow::Cow::Borrowed("unknown"));

        // 1. IO: Read bytes
        let bytes = reader.read_bytes(&filename).await?;

        let image = Self::decode_image_async(bytes, color_space, filename.to_string()).await?;

        // 3. Build Texture resource
        let mut texture = Texture::new(Some(&uri), image, wgpu::TextureViewDimension::D2);

        texture.generate_mipmaps = generate_mipmaps;

        // 4. Store in AssetStorage
        let handle = self.textures.add(texture);
        Ok(handle)
    }

    /// Asynchronously loads a Cube Map (requires 6 images).
    pub async fn load_cube_texture_async(
        &self,
        sources: [impl AssetSource; 6],
        color_space: ColorSpace,
        generate_mipmaps: bool,
    ) -> Result<TextureHandle> {
        // Concurrently load 6 images
        let mut futures = Vec::with_capacity(6);

        for source in sources {
            futures.push(async move {
                let reader = AssetReaderVariant::new(&source)?;
                let filename = source
                    .filename()
                    .unwrap_or(std::borrow::Cow::Borrowed("unknown"));

                let bytes = reader.read_bytes(&filename).await?;
                let image =
                    Self::decode_image_async(bytes, color_space, filename.to_string()).await?;
                Ok::<myth_resources::image::Image, Error>(image)
            });
        }

        let images = futures::future::try_join_all(futures).await?;

        // Check dimension consistency
        let width: u32 = images[0].width();
        let height = images[0].height();
        if images
            .iter()
            .any(|img| img.width() != width || img.height() != height)
        {
            return Err(Error::Asset(AssetError::InvalidData(
                "Cube map images must have same dimensions".to_string(),
            )));
        }

        let mut combined_data = Vec::with_capacity((width * height * 4 * 6) as usize);

        for img in images {
            if let Some(data) = img.data.read().unwrap().as_ref() {
                combined_data.extend(data);
            }
        }

        let combined_image = myth_resources::image::Image::new(
            Some("CubeMap"),
            width,
            height,
            6,
            wgpu::TextureDimension::D2,
            match color_space {
                ColorSpace::Srgb => TextureFormat::Rgba8UnormSrgb,
                ColorSpace::Linear => TextureFormat::Rgba8Unorm,
            },
            Some(combined_data),
        );

        let mut texture = Texture::new(
            Some("CubeMap"),
            combined_image,
            wgpu::TextureViewDimension::Cube,
        );
        texture.generate_mipmaps = generate_mipmaps;

        let handle = self.textures.add(texture);
        Ok(handle)
    }

    /// Asynchronously loads an HDR environment map.
    pub async fn load_hdr_texture_async(&self, source: impl AssetSource) -> Result<TextureHandle> {
        let reader = AssetReaderVariant::new(&source)?;
        let filename = source
            .filename()
            .unwrap_or(std::borrow::Cow::Borrowed("unknown"));

        let bytes = reader.read_bytes(&filename).await?;

        // HDR decoding logic
        let image = Self::decode_hdr_async(bytes).await?;

        let mut texture = Texture::new(Some(&filename), image, wgpu::TextureViewDimension::D2);
        // HDR typically does not need mipmaps, or requires special handling
        texture.sampler.address_mode_u = wgpu::AddressMode::ClampToEdge;
        texture.sampler.address_mode_v = wgpu::AddressMode::ClampToEdge;
        texture.sampler.mag_filter = wgpu::FilterMode::Linear;
        texture.sampler.min_filter = wgpu::FilterMode::Linear;

        let handle = self.textures.add(texture);
        Ok(handle)
    }

    /// Loads a 2D texture from raw bytes (e.g. from a file dialog on WASM).
    pub async fn load_texture_from_bytes_async(
        &self,
        name: &str,
        bytes: Vec<u8>,
        color_space: ColorSpace,
        generate_mipmaps: bool,
    ) -> Result<TextureHandle> {
        let image = Self::decode_image_async(bytes, color_space, name.to_string()).await?;
        let mut texture = Texture::new(Some(name), image, wgpu::TextureViewDimension::D2);
        texture.generate_mipmaps = generate_mipmaps;
        let handle = self.textures.add(texture);
        Ok(handle)
    }

    /// Loads an HDR environment map from raw bytes (e.g. from a file dialog on WASM).
    pub async fn load_hdr_texture_from_bytes_async(
        &self,
        name: &str,
        bytes: Vec<u8>,
    ) -> Result<TextureHandle> {
        let image = Self::decode_hdr_async(bytes).await?;
        let mut texture = Texture::new(Some(name), image, wgpu::TextureViewDimension::D2);
        texture.sampler.address_mode_u = wgpu::AddressMode::ClampToEdge;
        texture.sampler.address_mode_v = wgpu::AddressMode::ClampToEdge;
        texture.sampler.mag_filter = wgpu::FilterMode::Linear;
        texture.sampler.min_filter = wgpu::FilterMode::Linear;
        let handle = self.textures.add(texture);
        Ok(handle)
    }

    // ========================================================================
    // LUT (3D Lookup Table) Loading
    // ========================================================================

    /// Loads a 3D LUT texture from a .cube file.
    pub fn load_lut_texture(&mut self, source: impl AssetSource) -> Result<TextureHandle> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            get_asset_runtime().block_on(self.load_lut_texture_async(source))
        }
        #[cfg(target_arch = "wasm32")]
        {
            let texture = crate::load_lut_texture_from_file(source.uri().to_string())?;
            let handle = self.textures.add(texture);
            Ok(handle)
        }
    }

    /// Asynchronously loads a 3D LUT from a .cube file.
    pub async fn load_lut_texture_async(&self, source: impl AssetSource) -> Result<TextureHandle> {
        let reader = AssetReaderVariant::new(&source)?;
        let filename = source
            .filename()
            .unwrap_or(std::borrow::Cow::Borrowed("unknown"));

        let bytes = reader.read_bytes(&filename).await?;

        let image = Self::decode_cube_async(bytes).await?;

        let mut texture = Texture::new(Some(&filename), image, wgpu::TextureViewDimension::D3);
        texture.sampler.address_mode_u = wgpu::AddressMode::ClampToEdge;
        texture.sampler.address_mode_v = wgpu::AddressMode::ClampToEdge;
        texture.sampler.address_mode_w = wgpu::AddressMode::ClampToEdge;
        texture.sampler.mag_filter = wgpu::FilterMode::Linear;
        texture.sampler.min_filter = wgpu::FilterMode::Linear;

        let handle = self.textures.add(texture);
        Ok(handle)
    }

    /// Loads a 3D LUT texture from raw bytes (e.g. from a file dialog on WASM).
    pub async fn load_lut_texture_from_bytes_async(
        &self,
        name: &str,
        bytes: Vec<u8>,
    ) -> Result<TextureHandle> {
        let image = Self::decode_cube_async(bytes).await?;
        let mut texture = Texture::new(Some(name), image, wgpu::TextureViewDimension::D3);
        texture.sampler.address_mode_u = wgpu::AddressMode::ClampToEdge;
        texture.sampler.address_mode_v = wgpu::AddressMode::ClampToEdge;
        texture.sampler.address_mode_w = wgpu::AddressMode::ClampToEdge;
        texture.sampler.mag_filter = wgpu::FilterMode::Linear;
        texture.sampler.min_filter = wgpu::FilterMode::Linear;
        let handle = self.textures.add(texture);
        Ok(handle)
    }

    // ========================================================================
    // Internal Helpers
    // ========================================================================

    /// Unified image decoding helper (automatically offloads to native thread pool).
    async fn decode_image_async(
        bytes: Vec<u8>,
        color_space: ColorSpace,
        label: String,
    ) -> Result<myth_resources::image::Image> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Native: Offload to blocking thread
            tokio::task::spawn_blocking(move || Self::decode_image_cpu(&bytes, color_space, &label))
                .await
                .map_err(|e| {
                    myth_core::Error::Asset(myth_core::AssetError::TaskJoin(e.to_string()))
                })?
        }
        #[cfg(target_arch = "wasm32")]
        {
            // WASM: Currently can only run on the main thread (unless WebWorker architecture is introduced)
            Self::decode_image_cpu(&bytes, color_space, &label)
        }
    }

    /// CPU image decoding logic.
    fn decode_image_cpu(
        bytes: &[u8],
        color_space: ColorSpace,
        label: &str,
    ) -> Result<myth_resources::image::Image> {
        use image::GenericImageView;

        let img = image::load_from_memory(bytes).map_err(|e| {
            Error::Asset(AssetError::Format(format!(
                "Failed to decode image {label}: {e}"
            )))
        })?;

        let (width, height) = img.dimensions();
        let rgba = img.to_rgba8();

        Ok(myth_resources::image::Image::new(
            Some(label),
            width,
            height,
            1,
            wgpu::TextureDimension::D2,
            match color_space {
                ColorSpace::Srgb => TextureFormat::Rgba8UnormSrgb,
                ColorSpace::Linear => TextureFormat::Rgba8Unorm,
            },
            Some(rgba.into_vec()),
        ))
    }

    /// Unified HDR decoding helper.
    async fn decode_hdr_async(bytes: Vec<u8>) -> Result<myth_resources::image::Image> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            tokio::task::spawn_blocking(move || Self::decode_hdr_cpu(&bytes))
                .await
                .map_err(|e| {
                    myth_core::Error::Asset(myth_core::AssetError::TaskJoin(e.to_string()))
                })?
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self::decode_hdr_cpu(&bytes)
        }
    }

    /// CPU HDR decoding logic (converts to `RGBA16Float`).
    fn decode_hdr_cpu(bytes: &[u8]) -> Result<myth_resources::image::Image> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| Error::Asset(AssetError::Format(format!("Failed to decode HDR: {e}"))))?;

        let width = img.width();
        let height = img.height();
        let rgb32f = img.into_rgb32f();

        // Convert RGB32F to RGBA16F (half float) for GPU
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

        Ok(myth_resources::image::Image::new(
            Some("HDR_Texture"),
            width,
            height,
            1,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba16Float,
            Some(rgba_f16_data),
        ))
    }

    // ========================================================================
    // .cube LUT Decoding
    // ========================================================================

    /// Unified .cube decoding helper (automatically offloads to native thread pool).
    async fn decode_cube_async(bytes: Vec<u8>) -> Result<myth_resources::image::Image> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            tokio::task::spawn_blocking(move || Self::decode_cube_cpu(&bytes))
                .await
                .map_err(|e| {
                    myth_core::Error::Asset(myth_core::AssetError::TaskJoin(e.to_string()))
                })?
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self::decode_cube_cpu(&bytes)
        }
    }

    /// CPU .cube file decoding logic (parses text, converts to `Rgba16Float` 3D texture).
    pub(crate) fn decode_cube_cpu(bytes: &[u8]) -> Result<myth_resources::image::Image> {
        let raw_text = std::str::from_utf8(bytes).map_err(|e| {
            Error::Asset(AssetError::Format(format!(
                "Failed to parse .cube file as UTF-8: {e}"
            )))
        })?;

        // 绉婚櫎鍙兘瀛樺湪鐨?UTF-8 BOM 瀛楃 (Windows 涓嬪父瑙?
        let text = raw_text.strip_prefix('\u{FEFF}').unwrap_or(raw_text);

        let mut size = 0;
        let mut data = Vec::new();

        for line in text.lines() {
            let line = line.trim();
            // 璺宠繃绌鸿鍜屾敞閲?
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // 瑙ｆ瀽 LUT 灏哄
            if line.starts_with("LUT_3D_SIZE") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() == 2 {
                    size = parts[1].parse::<u32>().map_err(|_| {
                        Error::Asset(AssetError::Format("Invalid LUT_3D_SIZE".to_string()))
                    })?;
                }
                continue;
            }

            // 璺宠繃甯歌鐨?.cube 鍏冩暟鎹厤缃锛岄槻姝㈣鍒?
            if line.starts_with("TITLE")
                || line.starts_with("DOMAIN_")
                || line.starts_with("LUT_1D_")
                || line.starts_with("LUT_3D_INPUT_RANGE")
            {
                continue;
            }

            // 灏濊瘯灏嗚瑙ｆ瀽涓?3 涓诞鐐规暟 (RGB)
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 3 {
                // 瀹夊叏瑙ｆ瀽锛氬彧鏈夊綋杩?3 涓儴鍒嗗叏閮借兘鎴愬姛杞负 f32 鏃讹紝鎵嶅瓨鍏ユ暟鎹?
                if let (Ok(r), Ok(g), Ok(b)) = (
                    parts[0].parse::<f32>(),
                    parts[1].parse::<f32>(),
                    parts[2].parse::<f32>(),
                ) {
                    data.push(r);
                    data.push(g);
                    data.push(b);
                }
            }
        }

        if size == 0 {
            return Err(Error::Asset(AssetError::Format(
                "Missing LUT_3D_SIZE in .cube file. (Did you accidentally download an HTML file?)"
                    .to_string(),
            )));
        }

        let expected_len = (size * size * size * 3) as usize;
        if data.len() < expected_len {
            return Err(Error::Asset(AssetError::Format(format!(
                "LUT data too short! Expected {} float values, but found {}.",
                expected_len,
                data.len()
            ))));
        }

        let start_index = data.len() - expected_len;
        let lut_3d_data = &data[start_index..];

        // Convert RGB32F to RGBA16F (half float) for GPU usage
        let mut rgba_f16_data = Vec::with_capacity((size * size * size * 4) as usize * 2);
        for chunk in lut_3d_data.chunks_exact(3) {
            let r = half::f16::from_f32(chunk[0]);
            let g = half::f16::from_f32(chunk[1]);
            let b = half::f16::from_f32(chunk[2]);
            let a = half::f16::from_f32(1.0); // Alpha is fully opaque

            rgba_f16_data.extend_from_slice(&r.to_le_bytes());
            rgba_f16_data.extend_from_slice(&g.to_le_bytes());
            rgba_f16_data.extend_from_slice(&b.to_le_bytes());
            rgba_f16_data.extend_from_slice(&a.to_le_bytes());
        }

        Ok(myth_resources::image::Image::new(
            Some("LUT_3D"),
            size,
            size,
            size, // Depth 涓?size
            wgpu::TextureDimension::D3,
            wgpu::TextureFormat::Rgba16Float,
            Some(rgba_f16_data),
        ))
    }
}

impl myth_scene::GeometryQuery for AssetServer {
    fn get_geometry_bbox(&self, handle: GeometryHandle) -> Option<myth_resources::BoundingBox> {
        self.geometries.get(handle).map(|g| g.bounding_box)
    }
}
