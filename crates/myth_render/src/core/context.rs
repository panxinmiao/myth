//! wgpu Context
//!
//! The [`WgpuContext`] holds core GPU handles: device, queue, and render target state.
//! It supports two modes of operation:
//!
//! - **Windowed mode**: A `wgpu::Surface` is created from a window handle for
//!   on-screen presentation. The surface owns the swap-chain back buffers.
//! - **Headless mode**: No surface is created. The engine maintains an offscreen
//!   `wgpu::Texture` as the render target, suitable for server-side rendering,
//!   GPU readback, or CI/CD pipelines.

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::settings::{RenderPath, RendererInitConfig, RendererSettings};
use myth_core::{Error, PlatformError, Result};

/// Core wgpu context holding GPU handles and render target state.
///
/// This struct owns the fundamental wgpu resources needed for rendering:
/// - `device`: GPU device for resource creation
/// - `queue`: Command submission queue
/// - `surface` (optional): Window surface for on-screen presentation
/// - `headless_texture` (optional): Offscreen texture for headless rendering
///
/// The render target dimensions and format are always available via
/// [`target_width`], [`target_height`], and [`surface_view_format`],
/// regardless of the active mode.
pub struct WgpuContext {
    /// The wgpu device for GPU operations.
    pub device: wgpu::Device,
    /// The command queue for submitting work.
    pub queue: wgpu::Queue,

    /// Window surface for on-screen presentation (`None` in headless mode).
    pub surface: Option<wgpu::Surface<'static>>,
    /// Surface configuration (`None` in headless mode).
    pub config: Option<wgpu::SurfaceConfiguration>,

    /// Offscreen render target for headless mode (`None` in windowed mode).
    ///
    /// Created with `RENDER_ATTACHMENT | COPY_SRC` usage to support both
    /// rendering and GPU-to-CPU readback.
    pub headless_texture: Option<wgpu::Texture>,

    /// Render target width in pixels.
    pub target_width: u32,
    /// Render target height in pixels.
    pub target_height: u32,

    /// Depth buffer format.
    pub depth_format: wgpu::TextureFormat,

    /// The view format used for render target views (always sRGB-qualified).
    pub surface_view_format: wgpu::TextureFormat,

    pub msaa_samples: u32,

    pub anisotropy_clamp: u16,

    /// The active render path. Stored for runtime branching in the frame graph.
    pub render_path: RenderPath,

    /// Version counter for pipeline-affecting settings (HDR, MSAA, RenderPath).
    /// Incremented when these settings change, used to invalidate L1 pipeline cache.
    pub pipeline_settings_version: u64,
}

impl WgpuContext {
    pub async fn new<W>(
        window: W,
        init_config: &RendererInitConfig,
        settings: &RendererSettings,
        width: u32,
        height: u32,
    ) -> Result<Self>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        let instance_desc = match init_config.backends {
            Some(backends) => wgpu::InstanceDescriptor {
                backends,
                ..wgpu::InstanceDescriptor::new_without_display_handle_from_env()
            },
            None => wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
        };
        let instance = wgpu::Instance::new(instance_desc);

        let surface = instance
            .create_surface(window)
            .map_err(|e| Error::Platform(PlatformError::SurfaceConfigFailed(e.to_string())))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: init_config.power_preference,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| Error::Platform(PlatformError::AdapterNotFound(e.to_string())))?;

        let info = adapter.get_info();

        log::debug!("Backend: {:?}", info.backend);
        log::debug!("Device: {}", info.name);
        log::debug!("Vendor: {:x}", info.vendor);

        // === Query Surface-supported formats ===
        let caps = surface.get_capabilities(&adapter);

        // Print debug info showing formats supported on the current platform
        log::debug!("Surface Supported Formats: {:?}", caps.formats);

        // Prefer sRGB format (Native); if unavailable (Web), select the first available format (usually Linear)
        // Note: On the Web, an sRGB format will definitely not be found here, falling back to caps.formats[0]
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(caps.formats[0]);

        log::debug!("Selected Surface Format: {surface_format:?}");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: init_config.required_features,
                required_limits: init_config.required_limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .map_err(|e| {
                Error::Render(myth_core::RenderError::RequestDeviceFailed(e.to_string()))
            })?;

        let view_format = surface_format.add_srgb_suffix();

        let present_mode = if settings.vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            desired_maximum_frame_latency: 2,
            present_mode,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![view_format],
        };

        surface.configure(&device, &config);

        Ok(Self {
            device,
            queue,
            surface: Some(surface),
            config: Some(config),
            headless_texture: None,
            target_width: width,
            target_height: height,
            depth_format: init_config.depth_format,
            surface_view_format: view_format,
            msaa_samples: 1,
            anisotropy_clamp: settings.anisotropy_clamp,
            render_path: settings.path,
            pipeline_settings_version: 0,
        })
    }

    /// Creates a headless (offscreen) GPU context without a window surface.
    ///
    /// This constructor skips surface creation and instead allocates an
    /// offscreen `wgpu::Texture` as the render target. The texture is created
    /// with `RENDER_ATTACHMENT | COPY_SRC` usage, enabling both rendering and
    /// GPU-to-CPU readback.
    ///
    /// # Arguments
    ///
    /// * `target_format` — Desired render target format. Pass `None` to use
    ///   the default `Rgba8UnormSrgb`, which provides broad compatibility and
    ///   straightforward CPU readback. Use `Rgba16Float` for HDR workflows or
    ///   `Rgba32Float` for maximum precision.
    pub async fn new_headless(
        init_config: &RendererInitConfig,
        settings: &RendererSettings,
        width: u32,
        height: u32,
        target_format: Option<wgpu::TextureFormat>,
    ) -> Result<Self> {
        let instance_desc = match init_config.backends {
            Some(backends) => wgpu::InstanceDescriptor {
                backends,
                ..wgpu::InstanceDescriptor::new_without_display_handle_from_env()
            },
            None => wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
        };
        let instance = wgpu::Instance::new(instance_desc);

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: init_config.power_preference,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| Error::Platform(PlatformError::AdapterNotFound(e.to_string())))?;

        let info = adapter.get_info();
        log::debug!("Backend (headless): {:?}", info.backend);
        log::debug!("Device: {}", info.name);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Headless Device"),
                required_features: init_config.required_features,
                required_limits: init_config.required_limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .map_err(|e| {
                Error::Render(myth_core::RenderError::RequestDeviceFailed(e.to_string()))
            })?;

        let view_format = target_format.unwrap_or(wgpu::TextureFormat::Rgba8UnormSrgb);

        let headless_texture = Self::create_headless_texture(&device, width, height, view_format);

        Ok(Self {
            device,
            queue,
            surface: None,
            config: None,
            headless_texture: Some(headless_texture),
            target_width: width,
            target_height: height,
            depth_format: init_config.depth_format,
            surface_view_format: view_format,
            msaa_samples: 1,
            anisotropy_clamp: settings.anisotropy_clamp,
            render_path: settings.path,
            pipeline_settings_version: 0,
        })
    }

    /// Creates the offscreen render target texture for headless mode.
    fn create_headless_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Headless Render Target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.target_width = width;
            self.target_height = height;

            if let Some(config) = &mut self.config {
                config.width = width;
                config.height = height;
                if let Some(surface) = &self.surface {
                    surface.configure(&self.device, config);
                }
            }

            if self.headless_texture.is_some() {
                self.headless_texture = Some(Self::create_headless_texture(
                    &self.device,
                    width,
                    height,
                    self.surface_view_format,
                ));
            }
        }
    }

    /// Dynamically reconfigure the surface present mode for VSync toggling.
    ///
    /// This is a no-op in headless mode where no surface exists.
    pub fn set_vsync(&mut self, vsync: bool) {
        let Some(config) = &mut self.config else {
            return;
        };

        let present_mode = if vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };

        if config.present_mode != present_mode {
            config.present_mode = present_mode;
            if let Some(surface) = &self.surface {
                surface.configure(&self.device, config);
            }
            log::info!("Surface reconfigured — VSync: {vsync}");
        }
    }

    #[must_use]
    pub fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> wgpu::TextureView {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Returns the current render target dimensions.
    #[inline]
    pub fn size(&self) -> (u32, u32) {
        (self.target_width, self.target_height)
    }

    /// Returns `true` if the context is running in headless (offscreen) mode.
    #[inline]
    #[must_use]
    pub fn is_headless(&self) -> bool {
        self.surface.is_none()
    }
}
