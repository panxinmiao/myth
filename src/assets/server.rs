use slotmap::{KeyData, new_key_type};
use std::sync::Arc;
use wgpu::TextureFormat;

use crate::ColorSpace;
use crate::assets::AssetReaderVariant;
use crate::assets::io::AssetSource;
use crate::assets::storage::AssetStorage;
use crate::resources::geometry::Geometry;
use crate::resources::material::Material;
use crate::resources::texture::{Sampler, Texture};

#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;
#[cfg(not(target_arch = "wasm32"))]
use tokio::runtime::Runtime;

#[cfg(not(target_arch = "wasm32"))]
fn get_asset_runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| Runtime::new().expect("Failed to create asset loader runtime"))
}

// Strongly-typed handles
new_key_type! {
    pub struct GeometryHandle;
    pub struct MaterialHandle;
    pub struct TextureHandle;
    pub struct SamplerHandle;
}

const DUMMY_ENV_MAP_ID: u64 = 0xFFFF_FFFF_FFFF_FFFF;
// const DUMMY_SAMPLER_ID: u64 = 0xFFFF_FFFF_FFFF_FFFE;

impl TextureHandle {
    /// Creates a reserved handle for internal system use
    #[inline]
    #[must_use]
    pub fn system_reserved(index: u64) -> Self {
        let data = KeyData::from_ffi(index);
        Self::from(data)
    }

    #[inline]
    #[must_use]
    pub fn dummy_env_map() -> Self {
        // Construct a Handle pointing to a specific ID
        let data = KeyData::from_ffi(DUMMY_ENV_MAP_ID);
        Self::from(data)
    }
}

// 2. Asset Server

#[derive(Clone)] // AssetServer is now lightweight and can be cloned freely
pub struct AssetServer {
    pub geometries: Arc<AssetStorage<GeometryHandle, Geometry>>,
    pub materials: Arc<AssetStorage<MaterialHandle, Material>>,
    pub textures: Arc<AssetStorage<TextureHandle, Texture>>,
    pub samplers: Arc<AssetStorage<SamplerHandle, Sampler>>,
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
        }
    }

    // ========================================================================
    // Legacy / Synchronous Methods (Native Only)
    // ========================================================================
    // Todo: 保留这些方法以兼容旧代码，但在 Native 上也可以考虑重构为调用 block_on(async_load)

    pub fn load_texture(
        &mut self,
        source: impl AssetSource,
        color_space: ColorSpace,
        generate_mipmaps: bool,
    ) -> anyhow::Result<TextureHandle> {
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
            let mut texture =
                crate::assets::load_texture_from_file(source.uri().to_string(), color_space)?;
            texture.generate_mipmaps = generate_mipmaps;
            let handle = self.textures.add(texture);
            Ok(handle)

            // panic!("Synchronous loading not supported on WASM");
        }
    }

    pub fn load_cube_texture(
        &mut self,
        sources: [impl AssetSource; 6],
        color_space: ColorSpace,
        generate_mipmaps: bool,
    ) -> anyhow::Result<TextureHandle> {
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
            let mut texture = crate::assets::load_cube_texture_from_files(
                sources.map(|s| s.uri().to_string()),
                color_space,
            )?;
            texture.generate_mipmaps = generate_mipmaps;
            let handle = self.textures.add(texture);
            Ok(handle)
            // panic!("Synchronous loading not supported on WASM");
        }
    }

    /// Loads an HDR format environment map (Equirectangular format)
    pub fn load_hdr_texture(&mut self, source: impl AssetSource) -> anyhow::Result<TextureHandle> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            get_asset_runtime().block_on(self.load_hdr_texture_async(source))
        }
        #[cfg(target_arch = "wasm32")]
        {
            let texture = crate::assets::load_hdr_texture_from_file(source.uri().to_string())?;
            let handle = self.textures.add(texture);
            Ok(handle)
            // panic!("Synchronous loading not supported on WASM");
        }
    }

    // ========================================================================
    // Async Methods (Cross-Platform)
    // ========================================================================

    /// 异步加载 2D 纹理 (支持本地路径或 HTTP URL)
    pub async fn load_texture_async(
        &self,
        source: impl AssetSource,
        color_space: crate::assets::ColorSpace,
        generate_mipmaps: bool,
    ) -> anyhow::Result<TextureHandle> {
        let reader = AssetReaderVariant::new(&source)?;

        let uri = source.uri();
        let filename = source
            .filename()
            .unwrap_or(std::borrow::Cow::Borrowed("unknown"));

        // 1. IO: 读取字节
        let bytes = reader.read_bytes(&filename).await?;

        let image = Self::decode_image_async(bytes, color_space, filename.to_string()).await?;

        // 3. 构建 Texture 资源
        let mut texture = Texture::new(Some(&uri), image, wgpu::TextureViewDimension::D2);

        texture.generate_mipmaps = generate_mipmaps;

        // 4. 存入 AssetStorage
        let handle = self.textures.add(texture);
        Ok(handle)
    }

    /// 异步加载 Cube Map (需要 6 张图)
    pub async fn load_cube_texture_async(
        &self,
        sources: [impl AssetSource; 6],
        color_space: ColorSpace,
        generate_mipmaps: bool,
    ) -> anyhow::Result<TextureHandle> {
        // 并发加载 6 张图
        let mut futures = Vec::with_capacity(6);

        // let uris: Vec<String> = sources.into_iter().map(|s| s.to_uri()).collect();

        for source in sources {
            futures.push(async move {
                let reader = AssetReaderVariant::new(&source)?;
                // let uri = source.uri();
                let filename = source
                    .filename()
                    .unwrap_or(std::borrow::Cow::Borrowed("unknown"));

                let bytes = reader.read_bytes(&filename).await?;
                let image =
                    Self::decode_image_async(bytes, color_space, filename.to_string()).await?;
                Ok::<crate::resources::image::Image, anyhow::Error>(image)
            });
        }

        let images = futures::future::try_join_all(futures).await?;

        // 检查尺寸一致性
        let width: u32 = images[0].width();
        let height = images[0].height();
        if images
            .iter()
            .any(|img| img.width() != width || img.height() != height)
        {
            return Err(anyhow::anyhow!("Cube map images must have same dimensions"));
        }

        let mut combined_data = Vec::with_capacity((width * height * 4 * 6) as usize);

        for img in images {
            if let Some(data) = img.data.read().unwrap().as_ref() {
                combined_data.extend(data);
            }
        }

        let combined_image = crate::resources::image::Image::new(
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

    /// 异步加载 HDR 环境贴图
    pub async fn load_hdr_texture_async(
        &self,
        source: impl AssetSource,
    ) -> anyhow::Result<TextureHandle> {
        let reader = AssetReaderVariant::new(&source)?;
        let filename = source
            .filename()
            .unwrap_or(std::borrow::Cow::Borrowed("unknown"));

        let bytes = reader.read_bytes(&filename).await?;

        // HDR 解码逻辑 (参考你之前的示例)
        let image = Self::decode_hdr_async(bytes).await?;

        let mut texture = Texture::new(Some(&filename), image, wgpu::TextureViewDimension::D2);
        // HDR 通常不需要 mipmap，或者需要特殊处理
        texture.sampler.address_mode_u = wgpu::AddressMode::ClampToEdge;
        texture.sampler.address_mode_v = wgpu::AddressMode::ClampToEdge;
        texture.sampler.mag_filter = wgpu::FilterMode::Linear;
        texture.sampler.min_filter = wgpu::FilterMode::Linear;

        let handle = self.textures.add(texture);
        Ok(handle)
    }

    // ========================================================================
    // Internal Helpers
    // ========================================================================

    /// 统一的图片解码帮助函数 (自动处理 Native 线程池卸载)
    async fn decode_image_async(
        bytes: Vec<u8>,
        color_space: ColorSpace,
        label: String,
    ) -> anyhow::Result<crate::resources::image::Image> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Native: 放到 blocking thread 执行
            tokio::task::spawn_blocking(move || Self::decode_image_cpu(bytes, color_space, label))
                .await?
        }
        #[cfg(target_arch = "wasm32")]
        {
            // WASM: 目前只能在主线程执行 (除非引入 WebWorker 架构)
            Self::decode_image_cpu(bytes, color_space, label)
        }
    }

    /// CPU 图片解码逻辑
    fn decode_image_cpu(
        bytes: Vec<u8>,
        color_space: ColorSpace,
        label: String,
    ) -> anyhow::Result<crate::resources::image::Image> {
        use image::GenericImageView;

        let img = image::load_from_memory(&bytes)
            .map_err(|e| anyhow::anyhow!("Failed to decode image {label}: {e}"))?;

        let (width, height) = img.dimensions();
        let rgba = img.to_rgba8();

        Ok(crate::resources::image::Image::new(
            Some(&label),
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

    /// 统一的 HDR 解码帮助函数
    async fn decode_hdr_async(bytes: Vec<u8>) -> anyhow::Result<crate::resources::image::Image> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            tokio::task::spawn_blocking(move || Self::decode_hdr_cpu(bytes)).await?
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self::decode_hdr_cpu(bytes)
        }
    }

    /// CPU HDR 解码逻辑 (转换为 `RGBA16Float`)
    fn decode_hdr_cpu(bytes: Vec<u8>) -> anyhow::Result<crate::resources::image::Image> {
        // use image::GenericImageView;

        let img = image::load_from_memory(&bytes)
            .map_err(|e| anyhow::anyhow!("Failed to decode HDR: {e}"))?;

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

        Ok(crate::resources::image::Image::new(
            Some("HDR_Texture"),
            width,
            height,
            1,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba16Float,
            Some(rgba_f16_data),
        ))
    }
}
