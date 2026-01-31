//! wgpu Context
//!
//! The [`WgpuContext`] holds core GPU handles: device, queue, surface, and config.
//! It is responsible for window surface management and resize handling.

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::errors::{Result, ThreeError};
use crate::renderer::settings::RenderSettings;

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
    /// Depth buffer texture view (recreated on resize)
    pub depth_texture_view: wgpu::TextureView,
    /// Clear color for the frame
    pub clear_color: wgpu::Color,
}

impl WgpuContext {
    pub async fn new<W>(
        window: W,
        settings: &RenderSettings,
        width: u32,
        height: u32,
    ) -> Result<Self>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window)
            .map_err(|e| ThreeError::AdapterRequestFailed(e.to_string()))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: settings.power_preference,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| ThreeError::AdapterRequestFailed(e.to_string()))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: settings.required_features,
                required_limits: settings.required_limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await?;

        let mut config = surface
            .get_default_config(&adapter, width, height)
            .ok_or_else(|| {
                ThreeError::AdapterRequestFailed("Surface not supported by adapter".to_string())
            })?;

        config.present_mode = if settings.vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        surface.configure(&device, &config);

        let depth_texture_view =
            Self::create_depth_texture(&device, &config, settings.depth_format);

        Ok(Self {
            device,
            queue,
            surface,
            config,
            depth_format: settings.depth_format,
            depth_texture_view,
            clear_color: settings.clear_color,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture_view = Self::create_depth_texture(&self.device, &self.config, self.depth_format);
        }
    }

    pub fn create_depth_texture(
        device: &wgpu::Device, 
        config: &wgpu::SurfaceConfiguration, 
        format: wgpu::TextureFormat
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

    /// Returns the surface color format.
    pub fn color_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Returns the depth texture view.
    ///
    /// The depth texture is automatically recreated on resize.
    #[inline]
    pub fn get_depth_view(&self) -> &wgpu::TextureView {
        &self.depth_texture_view
    }

    /// Returns the current surface dimensions.
    #[inline]
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }
}
