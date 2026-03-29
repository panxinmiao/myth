use flume::{Receiver, Sender, unbounded};
use parking_lot::RwLock;
use std::sync::Arc;
use uuid::Uuid;

use crate::io::{AssetReaderVariant, AssetSource};
use crate::prefab::SharedPrefab;
use crate::storage::AssetStorage;
use myth_core::{AssetError, Error, Result};
use myth_resources::geometry::Geometry;
use myth_resources::image::{ColorSpace, Image, ImageDimension, PixelFormat};
use myth_resources::material::Material;
use myth_resources::screen_space::SssRegistry;
use myth_resources::texture::Texture;
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

/// Completed or failed image load event from a background I/O + decode task.
struct ImageLoadEvent {
    handle: ImageHandle,
    result: std::result::Result<Image, String>,
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
    image_channel: LoadingChannel<ImageLoadEvent>,
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
        let mut white_tex = Texture::new_2d(Some("default_white"), white_img);
        white_tex.color_space = ColorSpace::Srgb;
        let default_white_texture = textures.add(white_tex);

        let black_img = images.add(Image::solid_color([0, 0, 0, 255]));
        let mut black_tex = Texture::new_2d(Some("default_black"), black_img);
        black_tex.color_space = ColorSpace::Srgb;
        let default_black_texture = textures.add(black_tex);

        let normal_img = images.add(Image::solid_color([128, 128, 255, 255]));
        let mut normal_tex = Texture::new_2d(Some("default_normal"), normal_img);
        normal_tex.color_space = ColorSpace::Linear;
        let default_normal_texture = textures.add(normal_tex);

        Self {
            geometries: Arc::new(AssetStorage::new()),
            materials: Arc::new(AssetStorage::new()),
            images,
            textures,
            prefabs: Arc::new(AssetStorage::new()),

            sss_registry: Arc::new(RwLock::new(SssRegistry::new())),

            loading: Arc::new(LoadingPipeline {
                image_channel: LoadingChannel::new(),
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

    // ── Image loading (fire-and-forget, URI-only deduplication) ─────────

    /// Loads a raw [`Image`] asset from the given URI, returning a handle
    /// immediately.
    ///
    /// The underlying I/O and CPU decode run on a background task.
    /// Duplicate requests for the **same URI** (regardless of colour-space
    /// or mipmap parameters) are deduplicated — at most one decode task
    /// runs per file.
    fn load_image(&self, uri: &str, filename: &str, pixel_format: PixelFormat) -> ImageHandle {
        let uuid = Self::generate_asset_uuid("Image", uri, "");
        let (handle, is_new) = self.images.reserve_with_uuid(uuid);
        if !is_new {
            return handle;
        }

        let tx = self.loading.image_channel.sender();
        let uri_owned = uri.to_string();
        let filename_owned = filename.to_string();

        spawn_asset_task(async move {
            let result =
                Self::load_image_task(&uri_owned, &filename_owned, pixel_format).await;
            let event = match result {
                Ok(image) => ImageLoadEvent {
                    handle,
                    result: Ok(image),
                },
                Err(e) => ImageLoadEvent {
                    handle,
                    result: Err(e.to_string()),
                },
            };
            let _ = tx.send(event);
        });

        handle
    }

    /// Loads a raw [`Image`] from an HDR file, returning a handle
    /// immediately.  Deduplicated by URI.
    fn load_hdr_image(&self, uri: &str, filename: &str) -> ImageHandle {
        let uuid = Self::generate_asset_uuid("Image", uri, "HDR");
        let (handle, is_new) = self.images.reserve_with_uuid(uuid);
        if !is_new {
            return handle;
        }

        let tx = self.loading.image_channel.sender();
        let uri_owned = uri.to_string();
        let filename_owned = filename.to_string();

        spawn_asset_task(async move {
            let result = Self::load_hdr_image_task(&uri_owned, &filename_owned).await;
            let event = match result {
                Ok(image) => ImageLoadEvent {
                    handle,
                    result: Ok(image),
                },
                Err(e) => ImageLoadEvent {
                    handle,
                    result: Err(e.to_string()),
                },
            };
            let _ = tx.send(event);
        });

        handle
    }

    /// Loads a raw [`Image`] from a `.cube` LUT file, returning a handle
    /// immediately.  Deduplicated by URI.
    fn load_lut_image(&self, uri: &str, filename: &str) -> ImageHandle {
        let uuid = Self::generate_asset_uuid("Image", uri, "LUT");
        let (handle, is_new) = self.images.reserve_with_uuid(uuid);
        if !is_new {
            return handle;
        }

        let tx = self.loading.image_channel.sender();
        let uri_owned = uri.to_string();
        let filename_owned = filename.to_string();

        spawn_asset_task(async move {
            let result = Self::load_lut_image_task(&uri_owned, &filename_owned).await;
            let event = match result {
                Ok(image) => ImageLoadEvent {
                    handle,
                    result: Ok(image),
                },
                Err(e) => ImageLoadEvent {
                    handle,
                    result: Err(e.to_string()),
                },
            };
            let _ = tx.send(event);
        });

        handle
    }

    // ── Texture loading (synchronous, referencing async Image) ──────────

    /// Loads a 2D texture, returning a handle immediately.
    ///
    /// The underlying [`Image`] is loaded asynchronously and deduplicated
    /// by URI alone. The `Texture` descriptor is created **synchronously**
    /// with the requested colour-space and mipmap settings, and is
    /// immediately available for material binding.
    ///
    /// Until the `Image` data arrives the render pipeline substitutes a
    /// default placeholder texture.
    #[allow(clippy::needless_pass_by_value)]
    pub fn load_texture(
        &self,
        source: impl AssetSource,
        color_space: ColorSpace,
        generate_mipmaps: bool,
    ) -> TextureHandle {
        let uri = source.uri().to_string();
        let filename = source
            .filename()
            .map_or_else(|| "unknown".to_string(), |c| c.to_string());

        let tex_uuid = Self::generate_asset_uuid(
            "Tex2D",
            &uri,
            &format!("{color_space:?}|{generate_mipmaps}"),
        );
        let (tex_handle, is_new) = self.textures.reserve_with_uuid(tex_uuid);
        if !is_new {
            return tex_handle;
        }

        let image_handle = self.load_image(&uri, &filename, PixelFormat::Rgba8Unorm);

        let mut texture = Texture::new_2d(Some(&uri), image_handle);
        texture.color_space = color_space;
        texture.generate_mipmaps = generate_mipmaps;
        self.textures.insert_ready(tex_handle, texture);

        tex_handle
    }

    /// Loads an HDR environment map, returning a handle immediately.
    ///
    /// Deduplicated by URI.
    #[allow(clippy::needless_pass_by_value)]
    pub fn load_hdr_texture(&self, source: impl AssetSource) -> TextureHandle {
        let uri = source.uri().to_string();
        let filename = source
            .filename()
            .map_or_else(|| "unknown".to_string(), |c| c.to_string());

        let tex_uuid = Self::generate_asset_uuid("HDR", &uri, "");
        let (tex_handle, is_new) = self.textures.reserve_with_uuid(tex_uuid);
        if !is_new {
            return tex_handle;
        }

        let image_handle = self.load_hdr_image(&uri, &filename);

        let mut texture = Texture::new_2d(Some(&uri), image_handle);
        texture.color_space = ColorSpace::Linear;
        texture.sampler.address_mode_u = wgpu::AddressMode::ClampToEdge;
        texture.sampler.address_mode_v = wgpu::AddressMode::ClampToEdge;
        texture.sampler.mag_filter = wgpu::FilterMode::Linear;
        texture.sampler.min_filter = wgpu::FilterMode::Linear;
        self.textures.insert_ready(tex_handle, texture);

        tex_handle
    }

    /// Loads a 3D LUT from a `.cube` file, returning a handle immediately.
    ///
    /// Deduplicated by URI.
    #[allow(clippy::needless_pass_by_value)]
    pub fn load_lut_texture(&self, source: impl AssetSource) -> TextureHandle {
        let uri = source.uri().to_string();
        let filename = source
            .filename()
            .map_or_else(|| "unknown".to_string(), |c| c.to_string());

        let tex_uuid = Self::generate_asset_uuid("LUT", &uri, "");
        let (tex_handle, is_new) = self.textures.reserve_with_uuid(tex_uuid);
        if !is_new {
            return tex_handle;
        }

        let image_handle = self.load_lut_image(&uri, &filename);

        let mut texture = Texture::new_3d(Some(&uri), image_handle);
        texture.color_space = ColorSpace::Linear;
        self.textures.insert_ready(tex_handle, texture);

        tex_handle
    }

    /// Loads a cube map texture from 6 face images, returning a handle
    /// immediately.
    ///
    /// The six underlying face images are combined into a single [`Image`]
    /// asset, loaded asynchronously and deduplicated by the composite URI
    /// of all six faces.
    pub fn load_cube_texture(
        &self,
        sources: [impl AssetSource; 6],
        color_space: ColorSpace,
        generate_mipmaps: bool,
    ) -> TextureHandle {
        let uris: Vec<String> = sources.iter().map(|s| s.uri().to_string()).collect();
        let filenames: Vec<String> = sources
            .iter()
            .map(|s| {
                s.filename()
                    .map_or_else(|| "unknown".to_string(), |c| c.to_string())
            })
            .collect();
        let combined_uri = uris.join("|");

        let tex_uuid = Self::generate_asset_uuid(
            "CubeMap",
            &combined_uri,
            &format!("{color_space:?}|{generate_mipmaps}"),
        );
        let (tex_handle, is_new) = self.textures.reserve_with_uuid(tex_uuid);
        if !is_new {
            return tex_handle;
        }

        let img_uuid = Self::generate_asset_uuid("Image", &combined_uri, "Cube");
        let (image_handle, img_is_new) = self.images.reserve_with_uuid(img_uuid);

        if img_is_new {
            let tx = self.loading.image_channel.sender();

            spawn_asset_task(async move {
                let result = Self::load_cube_image_task(&uris, &filenames).await;
                let event = match result {
                    Ok(image) => ImageLoadEvent {
                        handle: image_handle,
                        result: Ok(image),
                    },
                    Err(e) => ImageLoadEvent {
                        handle: image_handle,
                        result: Err(e.to_string()),
                    },
                };
                let _ = tx.send(event);
            });
        }

        let mut texture = Texture::new_cube(Some(&combined_uri), image_handle);
        texture.color_space = color_space;
        texture.generate_mipmaps = generate_mipmaps;
        self.textures.insert_ready(tex_handle, texture);

        tex_handle
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

    /// Processes all completed background loads (images and prefabs),
    /// promoting `Loading` slots to `Loaded` (or `Failed`).
    ///
    /// This is called automatically by [`Engine::update`] each frame.
    pub fn process_loading_events(&self) {
        // Drain image completions.
        while let Ok(event) = self.loading.image_channel.rx.try_recv() {
            match event.result {
                Ok(image) => {
                    self.images.insert_ready(event.handle, image);
                }
                Err(ref msg) => {
                    log::error!("Image load failed: {msg}");
                    self.images.mark_failed(event.handle, msg.clone());
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
    /// Delegates to the fire-and-forget entry point for full UUID
    /// deduplication, then blocks until the underlying [`Image`] is ready.
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
            self.load_texture_blocking_wasm(source, color_space, generate_mipmaps)
        }
    }

    /// Loads a cube map synchronously, blocking the calling thread.
    ///
    /// Delegates to the fire-and-forget entry point for full UUID
    /// deduplication, then blocks until the underlying [`Image`] is ready.
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
            self.load_cube_texture_blocking_wasm(sources, color_space, generate_mipmaps)
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
            self.load_hdr_texture_blocking_wasm(source)
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
            self.load_lut_texture_blocking_wasm(source)
        }
    }

    // ========================================================================
    // Async Methods (Cross-Platform, full control)
    // ========================================================================

    /// Asynchronously loads a 2D texture and waits for the underlying
    /// [`Image`] to finish decoding.
    ///
    /// Internally delegates to the fire-and-forget [`load_texture`](Self::load_texture)
    /// and then polls until the image data is available, ensuring full
    /// UUID-based deduplication.
    pub async fn load_texture_async(
        &self,
        source: impl AssetSource,
        color_space: ColorSpace,
        generate_mipmaps: bool,
    ) -> Result<TextureHandle> {
        let handle = self.load_texture(source, color_space, generate_mipmaps);
        let texture = self
            .textures
            .get(handle)
            .expect("Texture is immediately available after load_texture");
        self.wait_for_image(texture.image).await?;
        Ok(handle)
    }

    /// Asynchronously loads a cube map and waits for the underlying
    /// combined [`Image`] to finish decoding.
    ///
    /// Delegates to the fire-and-forget [`load_cube_texture`](Self::load_cube_texture).
    pub async fn load_cube_texture_async(
        &self,
        sources: [impl AssetSource; 6],
        color_space: ColorSpace,
        generate_mipmaps: bool,
    ) -> Result<TextureHandle> {
        let handle = self.load_cube_texture(sources, color_space, generate_mipmaps);
        let texture = self
            .textures
            .get(handle)
            .expect("Texture is immediately available after load_cube_texture");
        self.wait_for_image(texture.image).await?;
        Ok(handle)
    }

    /// Asynchronously loads an HDR environment map and waits for the
    /// underlying [`Image`] to finish decoding.
    ///
    /// Delegates to the fire-and-forget [`load_hdr_texture`](Self::load_hdr_texture).
    pub async fn load_hdr_texture_async(&self, source: impl AssetSource) -> Result<TextureHandle> {
        let handle = self.load_hdr_texture(source);
        let texture = self
            .textures
            .get(handle)
            .expect("Texture is immediately available after load_hdr_texture");
        self.wait_for_image(texture.image).await?;
        Ok(handle)
    }

    /// Loads a 2D texture from raw bytes.
    ///
    /// Bytes-based loads cannot be URI-deduplicated, but still honour the
    /// decoupled Image / Texture architecture.
    pub async fn load_texture_from_bytes_async(
        &self,
        name: &str,
        bytes: Vec<u8>,
        color_space: ColorSpace,
        generate_mipmaps: bool,
    ) -> Result<TextureHandle> {
        let image = Self::decode_image_async(bytes, name.to_string()).await?;
        let image_handle = self.images.add(image);
        let mut texture = Texture::new_2d(Some(name), image_handle);
        texture.color_space = color_space;
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
        let mut texture = Texture::new_2d(Some(name), image_handle);
        texture.color_space = ColorSpace::Linear;
        texture.sampler.address_mode_u = wgpu::AddressMode::ClampToEdge;
        texture.sampler.address_mode_v = wgpu::AddressMode::ClampToEdge;
        texture.sampler.mag_filter = wgpu::FilterMode::Linear;
        texture.sampler.min_filter = wgpu::FilterMode::Linear;
        let handle = self.textures.add(texture);
        Ok(handle)
    }

    /// Asynchronously loads a 3D LUT from a `.cube` file and waits for
    /// the underlying [`Image`] to finish decoding.
    ///
    /// Delegates to the fire-and-forget [`load_lut_texture`](Self::load_lut_texture).
    pub async fn load_lut_texture_async(&self, source: impl AssetSource) -> Result<TextureHandle> {
        let handle = self.load_lut_texture(source);
        let texture = self
            .textures
            .get(handle)
            .expect("Texture is immediately available after load_lut_texture");
        self.wait_for_image(texture.image).await?;
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
        let mut texture = Texture::new_3d(Some(name), image_handle);
        texture.color_space = ColorSpace::Linear;
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
    // WASM Blocking Helpers (synchronous I/O with UUID deduplication)
    // ========================================================================

    #[cfg(target_arch = "wasm32")]
    fn load_texture_blocking_wasm(
        &self,
        source: impl AssetSource,
        color_space: ColorSpace,
        generate_mipmaps: bool,
    ) -> Result<TextureHandle> {
        let uri = source.uri().to_string();

        let tex_uuid = Self::generate_asset_uuid(
            "Tex2D",
            &uri,
            &format!("{color_space:?}|{generate_mipmaps}"),
        );
        let (tex_handle, tex_is_new) = self.textures.reserve_with_uuid(tex_uuid);
        if !tex_is_new {
            return Ok(tex_handle);
        }

        let img_uuid = Self::generate_asset_uuid("Image", &uri, "");
        let image_handle = self.get_or_load_image_sync(img_uuid, || {
            let (data, width, height) = crate::load_image_from_file(&uri)?;
            Ok(Image::new(
                width, height, 1, ImageDimension::D2, PixelFormat::Rgba8Unorm, Some(data),
            ))
        })?;

        let mut texture = Texture::new_2d(Some(&uri), image_handle);
        texture.color_space = color_space;
        texture.generate_mipmaps = generate_mipmaps;
        self.textures.insert_ready(tex_handle, texture);
        Ok(tex_handle)
    }

    #[cfg(target_arch = "wasm32")]
    fn load_cube_texture_blocking_wasm(
        &self,
        sources: [impl AssetSource; 6],
        color_space: ColorSpace,
        generate_mipmaps: bool,
    ) -> Result<TextureHandle> {
        let uris: Vec<String> = sources.iter().map(|s| s.uri().to_string()).collect();
        let combined_uri = uris.join("|");

        let tex_uuid = Self::generate_asset_uuid(
            "CubeMap",
            &combined_uri,
            &format!("{color_space:?}|{generate_mipmaps}"),
        );
        let (tex_handle, tex_is_new) = self.textures.reserve_with_uuid(tex_uuid);
        if !tex_is_new {
            return Ok(tex_handle);
        }

        let img_uuid = Self::generate_asset_uuid("Image", &combined_uri, "Cube");
        let image_handle = self.get_or_load_image_sync(img_uuid, || {
            let paths: Vec<String> = sources.iter().map(|s| s.uri().to_string()).collect();
            let paths_arr: [String; 6] = paths.try_into().unwrap();
            let (image, _, _) = crate::load_cube_texture_from_files(&paths_arr, ColorSpace::Linear)?;
            Ok(image)
        })?;

        let mut texture = Texture::new_cube(Some(&combined_uri), image_handle);
        texture.color_space = color_space;
        texture.generate_mipmaps = generate_mipmaps;
        self.textures.insert_ready(tex_handle, texture);
        Ok(tex_handle)
    }

    #[cfg(target_arch = "wasm32")]
    fn load_hdr_texture_blocking_wasm(&self, source: impl AssetSource) -> Result<TextureHandle> {
        let uri = source.uri().to_string();

        let tex_uuid = Self::generate_asset_uuid("HDR", &uri, "");
        let (tex_handle, tex_is_new) = self.textures.reserve_with_uuid(tex_uuid);
        if !tex_is_new {
            return Ok(tex_handle);
        }

        let img_uuid = Self::generate_asset_uuid("Image", &uri, "HDR");
        let image_handle = self.get_or_load_image_sync(img_uuid, || {
            let (image, _, _) = crate::load_hdr_texture_from_file(&uri)?;
            Ok(image)
        })?;

        let mut texture = Texture::new_2d(Some(&uri), image_handle);
        texture.color_space = ColorSpace::Linear;
        texture.sampler.address_mode_u = wgpu::AddressMode::ClampToEdge;
        texture.sampler.address_mode_v = wgpu::AddressMode::ClampToEdge;
        texture.sampler.mag_filter = wgpu::FilterMode::Linear;
        texture.sampler.min_filter = wgpu::FilterMode::Linear;
        self.textures.insert_ready(tex_handle, texture);
        Ok(tex_handle)
    }

    #[cfg(target_arch = "wasm32")]
    fn load_lut_texture_blocking_wasm(&self, source: impl AssetSource) -> Result<TextureHandle> {
        let uri = source.uri().to_string();

        let tex_uuid = Self::generate_asset_uuid("LUT", &uri, "");
        let (tex_handle, tex_is_new) = self.textures.reserve_with_uuid(tex_uuid);
        if !tex_is_new {
            return Ok(tex_handle);
        }

        let img_uuid = Self::generate_asset_uuid("Image", &uri, "LUT");
        let image_handle = self.get_or_load_image_sync(img_uuid, || {
            let (image, _, _) = crate::load_lut_texture_from_file(&uri)?;
            Ok(image)
        })?;

        let mut texture = Texture::new_3d(Some(&uri), image_handle);
        texture.color_space = ColorSpace::Linear;
        self.textures.insert_ready(tex_handle, texture);
        Ok(tex_handle)
    }

    /// Helper: retrieve cached image or load and insert synchronously.
    #[cfg(target_arch = "wasm32")]
    fn get_or_load_image_sync(
        &self,
        uuid: Uuid,
        load_fn: impl FnOnce() -> Result<Image>,
    ) -> Result<ImageHandle> {
        let (handle, is_new) = self.images.reserve_with_uuid(uuid);
        if is_new {
            match load_fn() {
                Ok(image) => self.images.insert_ready(handle, image),
                Err(e) => {
                    self.images.mark_failed(handle, e.to_string());
                    return Err(e);
                }
            }
        }
        Ok(handle)
    }

    // ========================================================================
    // Internal Task Implementations
    // ========================================================================

    /// Background task: read + decode a standard image.
    async fn load_image_task(
        uri: &str,
        filename: &str,
        _pixel_format: PixelFormat,
    ) -> Result<Image> {
        let reader = AssetReaderVariant::new(&uri)?;
        let bytes = reader.read_bytes(filename).await?;
        Self::decode_image_async(bytes, filename.to_string()).await
    }

    /// Background task: read + decode an HDR image.
    async fn load_hdr_image_task(uri: &str, filename: &str) -> Result<Image> {
        let reader = AssetReaderVariant::new(&uri)?;
        let bytes = reader.read_bytes(filename).await?;
        Self::decode_hdr_async(bytes).await
    }

    /// Background task: read + decode a .cube LUT image.
    async fn load_lut_image_task(uri: &str, filename: &str) -> Result<Image> {
        let reader = AssetReaderVariant::new(&uri)?;
        let bytes = reader.read_bytes(filename).await?;
        Self::decode_cube_async(bytes).await
    }

    /// Background task: read + decode 6 cube-map face images and combine
    /// them into a single [`Image`] with `depth = 6`.
    async fn load_cube_image_task(uris: &[String], filenames: &[String]) -> Result<Image> {
        let mut futures = Vec::with_capacity(6);

        for (uri, filename) in uris.iter().zip(filenames.iter()) {
            let uri = uri.clone();
            let filename = filename.clone();
            futures.push(async move {
                let reader = AssetReaderVariant::new(&uri)?;
                let bytes = reader.read_bytes(&filename).await?;
                Self::decode_image_async(bytes, filename).await
            });
        }

        let face_images = futures::future::try_join_all(futures).await?;

        let width = face_images[0].width;
        let height = face_images[0].height;
        if face_images
            .iter()
            .any(|img| img.width != width || img.height != height)
        {
            return Err(Error::Asset(AssetError::InvalidData(
                "Cube map faces must have the same dimensions".to_string(),
            )));
        }

        let mut combined_data = Vec::with_capacity((width * height * 4 * 6) as usize);
        for img in &face_images {
            if let Some(data) = &img.data {
                combined_data.extend_from_slice(data);
            }
        }

        Ok(Image::new(
            width,
            height,
            6,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(combined_data),
        ))
    }

    // ========================================================================
    // Async Wait Helpers
    // ========================================================================

    /// Polls the loading channel and waits until the [`Image`] behind
    /// `handle` transitions out of the `Loading` state.
    ///
    /// On success the image data is available via
    /// [`AssetStorage::get`](crate::storage::AssetStorage::get).
    /// Returns an error if the background decode task failed.
    async fn wait_for_image(&self, handle: ImageHandle) -> Result<()> {
        loop {
            self.process_loading_events();

            if self.images.is_loaded(handle) {
                return Ok(());
            }
            if let Some(msg) = self.images.get_error(handle) {
                return Err(Error::Asset(AssetError::Format(msg)));
            }

            #[cfg(not(target_arch = "wasm32"))]
            tokio::task::yield_now().await;

            #[cfg(target_arch = "wasm32")]
            {
                wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(
                    &wasm_bindgen::JsValue::NULL,
                ))
                .await
                .ok();
            }
        }
    }

    // ========================================================================
    // Internal Decode Helpers
    // ========================================================================

    /// Unified image decoding helper (automatically offloads to native thread pool).
    ///
    /// Decodes to `PixelFormat::Rgba8Unorm` — colour-space is not baked in.
    async fn decode_image_async(
        bytes: Vec<u8>,
        label: String,
    ) -> Result<Image> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            tokio::task::spawn_blocking(move || Self::decode_image_cpu(&bytes, &label))
                .await
                .map_err(|e| {
                    myth_core::Error::Asset(myth_core::AssetError::TaskJoin(e.to_string()))
                })?
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self::decode_image_cpu(&bytes, &label)
        }
    }

    /// CPU image decoding logic.
    ///
    /// Always produces `PixelFormat::Rgba8Unorm`; colour-space interpretation
    /// is deferred to the [`Texture`] that references this image.
    fn decode_image_cpu(
        bytes: &[u8],
        label: &str,
    ) -> Result<Image> {
        use image::GenericImageView;

        let img = image::load_from_memory(bytes).map_err(|e| {
            Error::Asset(AssetError::Format(format!(
                "Failed to decode image {label}: {e}"
            )))
        })?;

        let (width, height) = img.dimensions();
        let rgba = img.to_rgba8();

        Ok(Image::new(
            width,
            height,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba8Unorm,
            Some(rgba.into_vec()),
        ))
    }

    /// Unified HDR decoding helper.
    async fn decode_hdr_async(bytes: Vec<u8>) -> Result<Image> {
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

    /// CPU HDR decoding logic (converts to `PixelFormat::Rgba16Float`).
    fn decode_hdr_cpu(bytes: &[u8]) -> Result<Image> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| Error::Asset(AssetError::Format(format!("Failed to decode HDR: {e}"))))?;

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

        Ok(Image::new(
            width,
            height,
            1,
            ImageDimension::D2,
            PixelFormat::Rgba16Float,
            Some(rgba_f16_data),
        ))
    }

    // ========================================================================
    // .cube LUT Decoding
    // ========================================================================

    /// Unified .cube decoding helper (automatically offloads to native thread pool).
    async fn decode_cube_async(bytes: Vec<u8>) -> Result<Image> {
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

    /// CPU .cube file decoding logic (parses text, converts to `PixelFormat::Rgba16Float` 3D image).
    pub(crate) fn decode_cube_cpu(bytes: &[u8]) -> Result<Image> {
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

        Ok(Image::new(
            size,
            size,
            size,
            ImageDimension::D3,
            PixelFormat::Rgba16Float,
            Some(rgba_f16_data),
        ))
    }
}

impl myth_scene::GeometryQuery for AssetServer {
    fn get_geometry_bbox(&self, handle: GeometryHandle) -> Option<myth_resources::BoundingBox> {
        self.geometries.get(handle).map(|g| g.bounding_box)
    }
}
