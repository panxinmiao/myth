//! wgpu Context
//!
//! The [`WgpuContext`] holds core GPU handles: device, queue, surface, and config.
//! It is responsible for window surface management and resize handling.

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::errors::{Error, PlatformError, Result};
use crate::renderer::settings::{RenderPath, RendererSettings};

/// Core wgpu context holding GPU handles.
///
/// This struct owns the fundamental wgpu resources needed for rendering:
/// - `device`: GPU device for resource creation
/// - `queue`: Command submission queue
/// - `surface`: Window surface for presentation
/// - `config`: Surface configuration (format, present mode, etc.)
///
/// It also manages the depth buffer texture which is recreated on resize.
pub struct WgpuContext {
    /// The wgpu device for GPU operations
    pub device: wgpu::Device,
    /// The command queue for submitting work
    pub queue: wgpu::Queue,
    /// The window surface for presentation
    pub surface: wgpu::Surface<'static>,
    /// Surface configuration
    pub config: wgpu::SurfaceConfiguration,

    /// Depth buffer format
    pub depth_format: wgpu::TextureFormat,

    pub surface_view_format: wgpu::TextureFormat,

    pub msaa_samples: u32,

    pub enable_hdr: bool,

    /// The active render path. Stored for runtime branching in the frame graph.
    pub render_path: RenderPath,

    /// Version counter for pipeline-affecting settings (HDR, MSAA, RenderPath).
    /// Incremented when these settings change, used to invalidate L1 pipeline cache.
    pub pipeline_settings_version: u64,
}

impl WgpuContext {
    pub async fn new<W>(
        window: W,
        settings: &RendererSettings,
        width: u32,
        height: u32,
    ) -> Result<Self>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        let instance_desc = match settings.backends {
            Some(backends) => wgpu::InstanceDescriptor {
                backends,
                ..wgpu::InstanceDescriptor::from_env_or_default()
            },
            None => wgpu::InstanceDescriptor::from_env_or_default(),
        };
        let instance = wgpu::Instance::new(&instance_desc);
        let surface = instance
            .create_surface(window)
            .map_err(|e| Error::Platform(PlatformError::SurfaceConfigFailed(e.to_string())))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: settings.power_preference,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| Error::Platform(PlatformError::AdapterNotFound(e.to_string())))?;

        let info = adapter.get_info();

        log::debug!("Backend: {:?}", info.backend);
        log::debug!("Device: {}", info.name);
        log::debug!("Vendor: {:x}", info.vendor);

        // ===  查询 Surface 支持的格式 ===
        let caps = surface.get_capabilities(&adapter);

        // 打印调试信息，查看当前平台支持哪些格式
        log::debug!("Surface Supported Formats: {:?}", caps.formats);

        // 优先选择 sRGB 格式 (Native)，如果没有 (Web)，则选择第一个可用格式 (通常是 Linear)
        // 注意：在 Web 上，这里肯定找不到 Srgb 格式，会回退到 caps.formats[0]
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
                required_features: settings.required_features,
                required_limits: settings.required_limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await?;

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
            surface,
            config,
            depth_format: settings.depth_format,
            surface_view_format: view_format,
            msaa_samples: settings.msaa_samples(),
            enable_hdr: settings.is_hdr(),
            render_path: settings.path.clone(),
            pipeline_settings_version: 0,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    #[must_use]
    pub fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        format: wgpu::TextureFormat,
    ) -> wgpu::TextureView {
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
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

    /// Returns the current surface dimensions.
    #[inline]
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }
}
