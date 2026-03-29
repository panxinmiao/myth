use flume::{Receiver, Sender, unbounded};
use parking_lot::RwLock;
use std::sync::Arc;
use uuid::Uuid;
use wgpu::TextureFormat;

use crate::ColorSpace;
use crate::io::{AssetReaderVariant, AssetSource};
use crate::prefab::SharedPrefab;
use crate::storage::AssetStorage;
use myth_core::{AssetError, Error, Result};
use myth_resources::geometry::Geometry;
use myth_resources::image::Image;
use myth_resources::material::Material;
use myth_resources::screen_space::SssRegistry;
use myth_resources::texture::{Texture, TextureSampler};
use myth_resources::{GeometryHandle, ImageHandle, MaterialHandle, PrefabHandle, TextureHandle};

#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;
#[cfg(not(target_arch = "wasm32"))]
use tokio::runtime::Runtime;

#[cfg(not(target_arch = "wasm32"))]
fn get_asset_runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| Runtime::new().expect("Failed to create asset loader runtime"))
}

// ────────────────────────────────────────────────────────────────────────────
// Cross-platform async task spawning
// ────────────────────────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
fn spawn_asset_task<F>(f: F)
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    get_asset_runtime().spawn(f);
}

#[cfg(target_arch = "wasm32")]
fn spawn_asset_task<F>(f: F)
where
    F: std::future::Future<Output = ()> + 'static,
{
    wasm_bindgen_futures::spawn_local(f);
}

// ────────────────────────────────────────────────────────────────────────────
// Internal loading events
// ────────────────────────────────────────────────────────────────────────────

/// Completed texture data delivered by a background task.
struct TextureLoadResult {
    image: Image,
    sampler: TextureSampler,
    view_dimension: wgpu::TextureViewDimension,
    generate_mipmaps: bool,
    name: Option<String>,
}

/// Completed or failed texture load event.
struct TextureLoadEvent {
    handle: TextureHandle,
    result: std::result::Result<TextureLoadResult, String>,
}

/// Completed glTF prefab load event.
struct PrefabLoadEvent {
    handle: PrefabHandle,
    source: String,
    result: std::result::Result<SharedPrefab, String>,
}

/// Internal channel pair for background → main thread communication.
struct LoadingChannel<T> {
    tx: Sender<T>,
    rx: Receiver<T>,
}

impl<T> LoadingChannel<T> {
    fn new() -> Self {
        let (tx, rx) = unbounded();
        Self { tx, rx }
    }

    fn sender(&self) -> Sender<T> {
        self.tx.clone()
    }
}

/// Shared state for the internal background-loading pipeline.
struct LoadingPipeline {
    texture_channel: LoadingChannel<TextureLoadEvent>,
    prefab_channel: LoadingChannel<PrefabLoadEvent>,
}

// ────────────────────────────────────────────────────────────────────────────
// AssetServer
// ────────────────────────────────────────────────────────────────────────────

// ────────────────────────────────────────────────────────────────────────────
// UUID v5 namespace for deterministic asset deduplication
// ────────────────────────────────────────────────────────────────────────────

/// Fixed namespace UUID for asset signature hashing (DNS namespace from RFC 4122).
const MYTH_ASSET_NAMESPACE: Uuid = uuid::uuid!("6ba7b810-9dad-11d1-80b4-00c04fd430c8");

/// Central asset manager for the engine.
///
/// `AssetServer` is lightweight and `Clone`-friendly — all inner state lives
/// behind `Arc`. Cloning the server gives access to the same storage,
/// channels, and default resources.
///
/// # Fire-and-Forget Loading
///
/// The primary loading API ([`load_texture`](Self::load_texture),
/// [`load_hdr_texture`](Self::load_hdr_texture), etc.) returns a handle
/// **immediately** and schedules the actual I/O + decode on a background
/// task.  Call [`process_loading_events`](Self::process_loading_events)
/// once per frame (the engine does this automatically) to promote completed
/// loads into the storage.
///
/// Until the data arrives the handle's slot is `Loading`, so
/// [`AssetStorage::get`] returns `None`. The render pipeline is designed to
/// fall back to default placeholder textures in this case.
///
/// # UUID-Based Deduplication
///
/// Every fire-and-forget load generates a deterministic UUID v5 from the
/// resource URI and its loading parameters (color space, mipmap, etc.).
/// Requesting the same resource twice returns the same handle without
/// spawning a redundant background task.
#[derive(Clone)]
pub struct AssetServer {
    pub geometries: Arc<AssetStorage<GeometryHandle, Geometry>>,
    pub materials: Arc<AssetStorage<MaterialHandle, Material>>,
    pub images: Arc<AssetStorage<ImageHandle, Image>>,
    pub textures: Arc<AssetStorage<TextureHandle, Texture>>,
    pub prefabs: Arc<AssetStorage<PrefabHandle, SharedPrefab>>,

    pub sss_registry: Arc<RwLock<SssRegistry>>,

    /// Internal background-loading infrastructure (shared across clones).
    loading: Arc<LoadingPipeline>,

    /// 1×1 white RGBA texture, used as fallback for albedo maps.
    pub default_white_texture: TextureHandle,
    /// 1×1 black RGBA texture, used as fallback for emission / AO maps.
    pub default_black_texture: TextureHandle,
    /// 1×1 flat normal map (`[128, 128, 255, 255]`).
    pub default_normal_texture: TextureHandle,
}

impl Default for AssetServer {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetServer {
    #[must_use]
    pub fn new() -> Self {
        let images = Arc::new(AssetStorage::new());
        let textures = Arc::new(AssetStorage::new());

        let white_img = images.add(Image::solid_color([255, 255, 255, 255]));
        let default_white_texture = textures.add(Texture::new_2d(Some("default_white"), white_img));

        let black_img = images.add(Image::solid_color([0, 0, 0, 255]));
        let default_black_texture = textures.add(Texture::new_2d(Some("default_black"), black_img));

        let normal_img = images.add(Image::solid_color([128, 128, 255, 255]));
        let default_normal_texture =
            textures.add(Texture::new_2d(Some("default_normal"), normal_img));

        Self {
            geometries: Arc::new(AssetStorage::new()),
            materials: Arc::new(AssetStorage::new()),
            images,
            textures,
            prefabs: Arc::new(AssetStorage::new()),

            sss_registry: Arc::new(RwLock::new(SssRegistry::new())),

            loading: Arc::new(LoadingPipeline {
                texture_channel: LoadingChannel::new(),
                prefab_channel: LoadingChannel::new(),
            }),

            default_white_texture,
            default_black_texture,
            default_normal_texture,
        }
    }

    // ========================================================================
    // Fire-and-Forget Loading API (Primary)
    // ========================================================================

    /// Generates a deterministic UUID v5 from the asset type, URI, and
    /// loading parameters to serve as a deduplication key.
    fn generate_asset_uuid(type_tag: &str, uri: &str, params: &str) -> Uuid {
        let signature = format!("{type_tag}|{uri}|{params}");
        Uuid::new_v5(&MYTH_ASSET_NAMESPACE, signature.as_bytes())
    }

    /// Loads a 2D texture, returning a handle immediately.
    ///
    /// The handle is valid the instant it's returned and can be bound into
    /// materials or scene properties right away. The actual data will be
    /// filled in asynchronously; until then the render pipeline substitutes
    /// a default placeholder.
    ///
    /// Duplicate requests for the same URI + parameters are deduplicated
    /// via UUID — no redundant I/O or GPU memory is created.
    #[allow(clippy::needless_pass_by_value)]
    pub fn load_texture(
        &self,
        source: impl AssetSource,
        color_space: ColorSpace,
        generate_mipmaps: bool,
    ) -> TextureHandle {
        let uri = source.uri().to_string();
        let uuid = Self::generate_asset_uuid(
            "Tex2D",
            &uri,
            &format!("{color_space:?}|{generate_mipmaps}"),
        );
        let (handle, is_new) = self.textures.reserve_with_uuid(uuid);
        if !is_new {
            return handle;
        }

        let tx = self.loading.texture_channel.sender();
        let filename = source
            .filename()
            .map_or_else(|| "unknown".to_string(), |c| c.to_string());

        spawn_asset_task(async move {
            let result =
                Self::load_texture_task(&uri, &filename, color_space, generate_mipmaps).await;
            let event = match result {
                Ok(data) => TextureLoadEvent {
                    handle,
                    result: Ok(data),
                },
                Err(e) => TextureLoadEvent {
                    handle,
                    result: Err(e.to_string()),
                },
            };
            let _ = tx.send(event);
        });

        handle
    }

    /// Loads an HDR environment map, returning a handle immediately.
    ///
    /// Deduplicated by URI.
    #[allow(clippy::needless_pass_by_value)]
    pub fn load_hdr_texture(&self, source: impl AssetSource) -> TextureHandle {
        let uri = source.uri().to_string();
        let uuid = Self::generate_asset_uuid("HDR", &uri, "");
        let (handle, is_new) = self.textures.reserve_with_uuid(uuid);
        if !is_new {
            return handle;
        }

        let tx = self.loading.texture_channel.sender();
        let filename = source
            .filename()
            .map_or_else(|| "unknown".to_string(), |c| c.to_string());

        spawn_asset_task(async move {
            let result = Self::load_hdr_texture_task(&uri, &filename).await;
            let event = match result {
                Ok(data) => TextureLoadEvent {
                    handle,
                    result: Ok(data),
                },
                Err(e) => TextureLoadEvent {
                    handle,
                    result: Err(e.to_string()),
                },
            };
            let _ = tx.send(event);
        });

        handle
    }

    /// Loads a 3D LUT from a `.cube` file, returning a handle immediately.
    ///
    /// Deduplicated by URI.
    #[allow(clippy::needless_pass_by_value)]
    pub fn load_lut_texture(&self, source: impl AssetSource) -> TextureHandle {
        let uri = source.uri().to_string();
        let uuid = Self::generate_asset_uuid("LUT", &uri, "");
        let (handle, is_new) = self.textures.reserve_with_uuid(uuid);
        if !is_new {
            return handle;
        }

        let tx = self.loading.texture_channel.sender();
        let filename = source
            .filename()
            .map_or_else(|| "unknown".to_string(), |c| c.to_string());

        spawn_asset_task(async move {
            let result = Self::load_lut_texture_task(&uri, &filename).await;
            let event = match result {
                Ok(data) => TextureLoadEvent {
                    handle,
                    result: Ok(data),
                },
                Err(e) => TextureLoadEvent {
                    handle,
                    result: Err(e.to_string()),
                },
            };
            let _ = tx.send(event);
        });

        handle
    }

    /// Loads a glTF/GLB model, returning a [`PrefabHandle`] immediately.
    ///
    /// The handle can be polled via [`AssetStorage::get`] on
    /// [`prefabs`](Self::prefabs) to check when loading completes.
    /// Completed loads are promoted automatically by
    /// [`process_loading_events`](Self::process_loading_events).
    ///
    /// Deduplicated by URI — loading the same model twice returns the same
    /// handle without spawning a redundant parsing task.
    #[cfg(feature = "gltf")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn load_gltf(&self, source: impl AssetSource) -> PrefabHandle {
        let uri = source.uri().to_string();
        let uuid = Self::generate_asset_uuid("GLTF", &uri, "");
        let (handle, is_new) = self.prefabs.reserve_with_uuid(uuid);
        if !is_new {
            return handle;
        }

        let tx = self.loading.prefab_channel.sender();
        let assets = self.clone();

        spawn_asset_task(async move {
            let source_str = uri.clone();
            let result = crate::loaders::GltfLoader::load_async(uri, assets).await;
            let event = PrefabLoadEvent {
                handle,
                source: source_str,
                result: result.map_err(|e| e.to_string()),
            };
            let _ = tx.send(event);
        });

        handle
    }

    // ========================================================================
    // Event Processing (called once per frame by Engine)
    // ========================================================================

    /// Processes all completed background loads (textures and prefabs),
    /// promoting `Loading` slots to `Loaded` (or `Failed`).
    ///
    /// This is called automatically by [`Engine::update`] each frame.
    pub fn process_loading_events(&self) {
        // Drain texture completions.
        while let Ok(event) = self.loading.texture_channel.rx.try_recv() {
            match event.result {
                Ok(data) => {
                    let image_handle = self.images.add(data.image);
                    let mut texture =
                        Texture::new(data.name.as_deref(), image_handle, data.view_dimension);
                    texture.sampler = data.sampler;
                    texture.generate_mipmaps = data.generate_mipmaps;
                    self.textures.insert_ready(event.handle, texture);
                }
                Err(ref msg) => {
                    log::error!("Texture load failed: {msg}");
                    self.textures.mark_failed(event.handle, msg.clone());
                }
            }
        }

        // Drain prefab completions into unified AssetStorage.
        while let Ok(event) = self.loading.prefab_channel.rx.try_recv() {
            match event.result {
                Ok(prefab) => {
                    self.prefabs.insert_ready(event.handle, prefab);
                    log::info!("Prefab loaded: {}", event.source);
                }
                Err(ref msg) => {
                    log::error!("glTF load failed ({}): {msg}", event.source);
                    self.prefabs.mark_failed(event.handle, msg.clone());
                }
            }
        }
    }

    // ========================================================================
    // Blocking (Synchronous) Loading — Native Only
    // ========================================================================

    /// Loads a 2D texture synchronously, blocking the calling thread.
    ///
    /// Prefer [`load_texture`](Self::load_texture) for non-blocking loads.
    pub fn load_texture_blocking(
        &self,
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
            let (image, sampler_cfg, gen_mips) =
                crate::load_texture_from_file(source.uri().to_string(), color_space)?;
            let image_handle = self.images.add(image);
            let mut texture = Texture::new_2d(None, image_handle);
            texture.sampler = sampler_cfg;
            texture.generate_mipmaps = gen_mips || generate_mipmaps;
            let handle = self.textures.add(texture);
            Ok(handle)
        }
    }

    /// Loads a cube map synchronously, blocking the calling thread.
    pub fn load_cube_texture_blocking(
        &self,
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
            let (image, sampler_cfg, gen_mips) = crate::load_cube_texture_from_files(
                &sources.map(|s| s.uri().to_string()),
                color_space,
            )?;
            let image_handle = self.images.add(image);
            let mut texture = Texture::new_cube(None, image_handle);
            texture.sampler = sampler_cfg;
            texture.generate_mipmaps = gen_mips || generate_mipmaps;
            let handle = self.textures.add(texture);
            Ok(handle)
        }
    }

    /// Loads an HDR texture synchronously, blocking the calling thread.
    pub fn load_hdr_texture_blocking(&self, source: impl AssetSource) -> Result<TextureHandle> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            get_asset_runtime().block_on(self.load_hdr_texture_async(source))
        }
        #[cfg(target_arch = "wasm32")]
        {
            let (image, sampler_cfg, _) =
                crate::load_hdr_texture_from_file(source.uri().to_string())?;
            let image_handle = self.images.add(image);
            let mut texture = Texture::new_2d(None, image_handle);
            texture.sampler = sampler_cfg;
            let handle = self.textures.add(texture);
            Ok(handle)
        }
    }

    /// Loads a 3D LUT synchronously, blocking the calling thread.
    pub fn load_lut_texture_blocking(&self, source: impl AssetSource) -> Result<TextureHandle> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            get_asset_runtime().block_on(self.load_lut_texture_async(source))
        }
        #[cfg(target_arch = "wasm32")]
        {
            let (image, sampler_cfg, _) =
                crate::load_lut_texture_from_file(source.uri().to_string())?;
            let image_handle = self.images.add(image);
            let mut texture = Texture::new_3d(None, image_handle);
            texture.sampler = sampler_cfg;
            let handle = self.textures.add(texture);
            Ok(handle)
        }
    }

    // ========================================================================
    // Async Methods (Cross-Platform, full control)
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

        let bytes = reader.read_bytes(&filename).await?;
        let image = Self::decode_image_async(bytes, color_space, filename.to_string()).await?;
        let image_handle = self.images.add(image);

        let mut texture = Texture::new(Some(&uri), image_handle, wgpu::TextureViewDimension::D2);
        texture.generate_mipmaps = generate_mipmaps;

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
                Ok::<Image, Error>(image)
            });
        }

        let images = futures::future::try_join_all(futures).await?;

        let width: u32 = images[0].width;
        let height = images[0].height;
        if images
            .iter()
            .any(|img| img.width != width || img.height != height)
        {
            return Err(Error::Asset(AssetError::InvalidData(
                "Cube map images must have same dimensions".to_string(),
            )));
        }

        let mut combined_data = Vec::with_capacity((width * height * 4 * 6) as usize);
        for img in &images {
            if let Some(data) = &img.data {
                combined_data.extend(data);
            }
        }

        let combined_image = Image::new(
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

        let image_handle = self.images.add(combined_image);
        let mut texture = Texture::new(
            Some("CubeMap"),
            image_handle,
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
        let image = Self::decode_hdr_async(bytes).await?;
        let image_handle = self.images.add(image);

        let mut texture = Texture::new(
            Some(&filename),
            image_handle,
            wgpu::TextureViewDimension::D2,
        );
        texture.sampler.address_mode_u = wgpu::AddressMode::ClampToEdge;
        texture.sampler.address_mode_v = wgpu::AddressMode::ClampToEdge;
        texture.sampler.mag_filter = wgpu::FilterMode::Linear;
        texture.sampler.min_filter = wgpu::FilterMode::Linear;

        let handle = self.textures.add(texture);
        Ok(handle)
    }

    /// Loads a 2D texture from raw bytes.
    pub async fn load_texture_from_bytes_async(
        &self,
        name: &str,
        bytes: Vec<u8>,
        color_space: ColorSpace,
        generate_mipmaps: bool,
    ) -> Result<TextureHandle> {
        let image = Self::decode_image_async(bytes, color_space, name.to_string()).await?;
        let image_handle = self.images.add(image);
        let mut texture = Texture::new(Some(name), image_handle, wgpu::TextureViewDimension::D2);
        texture.generate_mipmaps = generate_mipmaps;
        let handle = self.textures.add(texture);
        Ok(handle)
    }

    /// Loads an HDR environment map from raw bytes.
    pub async fn load_hdr_texture_from_bytes_async(
        &self,
        name: &str,
        bytes: Vec<u8>,
    ) -> Result<TextureHandle> {
        let image = Self::decode_hdr_async(bytes).await?;
        let image_handle = self.images.add(image);
        let mut texture = Texture::new(Some(name), image_handle, wgpu::TextureViewDimension::D2);
        texture.sampler.address_mode_u = wgpu::AddressMode::ClampToEdge;
        texture.sampler.address_mode_v = wgpu::AddressMode::ClampToEdge;
        texture.sampler.mag_filter = wgpu::FilterMode::Linear;
        texture.sampler.min_filter = wgpu::FilterMode::Linear;
        let handle = self.textures.add(texture);
        Ok(handle)
    }

    /// Asynchronously loads a 3D LUT from a `.cube` file.
    pub async fn load_lut_texture_async(&self, source: impl AssetSource) -> Result<TextureHandle> {
        let reader = AssetReaderVariant::new(&source)?;
        let filename = source
            .filename()
            .unwrap_or(std::borrow::Cow::Borrowed("unknown"));

        let bytes = reader.read_bytes(&filename).await?;
        let image = Self::decode_cube_async(bytes).await?;
        let image_handle = self.images.add(image);

        let mut texture = Texture::new(
            Some(&filename),
            image_handle,
            wgpu::TextureViewDimension::D3,
        );
        texture.sampler.address_mode_u = wgpu::AddressMode::ClampToEdge;
        texture.sampler.address_mode_v = wgpu::AddressMode::ClampToEdge;
        texture.sampler.address_mode_w = wgpu::AddressMode::ClampToEdge;
        texture.sampler.mag_filter = wgpu::FilterMode::Linear;
        texture.sampler.min_filter = wgpu::FilterMode::Linear;

        let handle = self.textures.add(texture);
        Ok(handle)
    }

    /// Loads a 3D LUT from raw bytes.
    pub async fn load_lut_texture_from_bytes_async(
        &self,
        name: &str,
        bytes: Vec<u8>,
    ) -> Result<TextureHandle> {
        let image = Self::decode_cube_async(bytes).await?;
        let image_handle = self.images.add(image);
        let mut texture = Texture::new(Some(name), image_handle, wgpu::TextureViewDimension::D3);
        texture.sampler.address_mode_u = wgpu::AddressMode::ClampToEdge;
        texture.sampler.address_mode_v = wgpu::AddressMode::ClampToEdge;
        texture.sampler.address_mode_w = wgpu::AddressMode::ClampToEdge;
        texture.sampler.mag_filter = wgpu::FilterMode::Linear;
        texture.sampler.min_filter = wgpu::FilterMode::Linear;
        let handle = self.textures.add(texture);
        Ok(handle)
    }

    // ========================================================================
    // Utility
    // ========================================================================

    /// Creates a simple checkerboard texture (useful for testing).
    #[must_use]
    pub fn checkerboard(&self, size: u32, squares: u32) -> TextureHandle {
        let image = Image::checkerboard(size, size, squares);
        let image_handle = self.images.add(image);
        let texture = Texture::new_2d(Some("Checkerboard"), image_handle);
        self.textures.add(texture)
    }

    // ========================================================================
    // Cache Invalidation
    // ========================================================================

    /// Invalidates a cached texture so a fresh reload can be dispatched.
    ///
    /// Use this when the underlying file has been replaced on disk (same URI
    /// but different content). The next call to [`load_texture`] with the
    /// same parameters will trigger a new background I/O task.
    pub fn invalidate_texture(&self, type_tag: &str, uri: &str, params: &str) {
        let uuid = Self::generate_asset_uuid(type_tag, uri, params);
        self.textures.invalidate_uuid(&uuid);
    }

    /// Convenience wrapper: invalidate and immediately re-dispatch a 2D
    /// texture load. Returns the (same or new) handle.
    #[allow(clippy::needless_pass_by_value)]
    pub fn reload_texture(
        &self,
        source: impl AssetSource,
        color_space: ColorSpace,
        generate_mipmaps: bool,
    ) -> TextureHandle {
        let uri = source.uri().to_string();
        let params = format!("{color_space:?}|{generate_mipmaps}");
        self.invalidate_texture("Tex2D", &uri, &params);
        self.load_texture(uri, color_space, generate_mipmaps)
    }

    /// Invalidates **all** UUID-cached textures, forcing a full reload on
    /// subsequent load requests.
    pub fn invalidate_all_textures(&self) {
        self.textures.invalidate_all_uuids();
    }

    /// Invalidates a cached prefab so a fresh reload can be dispatched.
    pub fn invalidate_prefab(&self, uri: &str) {
        let uuid = Self::generate_asset_uuid("GLTF", uri, "");
        self.prefabs.invalidate_uuid(&uuid);
    }

    // ========================================================================
    // Internal Task Implementations
    // ========================================================================

    /// Background task: read + decode a 2D texture.
    async fn load_texture_task(
        uri: &str,
        filename: &str,
        color_space: ColorSpace,
        generate_mipmaps: bool,
    ) -> Result<TextureLoadResult> {
        let reader = AssetReaderVariant::new(&uri)?;
        let bytes = reader.read_bytes(filename).await?;
        let image = Self::decode_image_async(bytes, color_space, filename.to_string()).await?;
        Ok(TextureLoadResult {
            image,
            sampler: TextureSampler::default(),
            view_dimension: wgpu::TextureViewDimension::D2,
            generate_mipmaps,
            name: Some(uri.to_string()),
        })
    }

    /// Background task: read + decode an HDR texture.
    async fn load_hdr_texture_task(uri: &str, filename: &str) -> Result<TextureLoadResult> {
        let reader = AssetReaderVariant::new(&uri)?;
        let bytes = reader.read_bytes(filename).await?;
        let image = Self::decode_hdr_async(bytes).await?;
        Ok(TextureLoadResult {
            image,
            sampler: TextureSampler {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..TextureSampler::default()
            },
            view_dimension: wgpu::TextureViewDimension::D2,
            generate_mipmaps: false,
            name: Some(filename.to_string()),
        })
    }

    /// Background task: read + decode a .cube LUT.
    async fn load_lut_texture_task(uri: &str, filename: &str) -> Result<TextureLoadResult> {
        let reader = AssetReaderVariant::new(&uri)?;
        let bytes = reader.read_bytes(filename).await?;
        let image = Self::decode_cube_async(bytes).await?;
        Ok(TextureLoadResult {
            image,
            sampler: TextureSampler {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..TextureSampler::default()
            },
            view_dimension: wgpu::TextureViewDimension::D3,
            generate_mipmaps: false,
            name: Some(filename.to_string()),
        })
    }

    // ========================================================================
    // Internal Decode Helpers
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

        // Remove potential UTF-8 BOM (Windows specific)
        let text = raw_text.strip_prefix('\u{FEFF}').unwrap_or(raw_text);

        let mut size = 0;
        let mut data = Vec::new();

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if line.starts_with("LUT_3D_SIZE") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() == 2 {
                    size = parts[1].parse::<u32>().map_err(|_| {
                        Error::Asset(AssetError::Format("Invalid LUT_3D_SIZE".to_string()))
                    })?;
                }
                continue;
            }

            if line.starts_with("TITLE")
                || line.starts_with("DOMAIN_")
                || line.starts_with("LUT_1D_")
                || line.starts_with("LUT_3D_INPUT_RANGE")
            {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 3
                && let (Ok(r), Ok(g), Ok(b)) = (
                    parts[0].parse::<f32>(),
                    parts[1].parse::<f32>(),
                    parts[2].parse::<f32>(),
                )
            {
                data.push(r);
                data.push(g);
                data.push(b);
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
            size,
            size,
            size,
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
