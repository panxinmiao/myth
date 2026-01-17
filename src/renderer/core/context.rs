//! WGPU 上下文
//!
//! WgpuContext 只持有 device, queue, surface, config
//! 负责 Resize 和 Present

use std::sync::Arc;
use winit::window::Window;

use crate::errors::{ThreeError, Result};
use crate::renderer::settings::RenderSettings;

/// WGPU 核心上下文
pub struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    
    pub depth_format: wgpu::TextureFormat,
    pub depth_texture_view: wgpu::TextureView,
    pub clear_color: wgpu::Color,
}

impl WgpuContext {
    pub async fn new(window: Arc<Window>, settings: &RenderSettings) -> Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone())
            .map_err(|e| ThreeError::AdapterRequestFailed(e.to_string()))?;
        
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: settings.power_preference,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await
        .map_err(|e| ThreeError::AdapterRequestFailed(e.to_string()))?;

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: settings.required_features,
                required_limits: settings.required_limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            },
        ).await?;

        let mut config = surface.get_default_config(&adapter, size.width, size.height)
            .ok_or_else(|| ThreeError::AdapterRequestFailed("Surface not supported by adapter".to_string()))?;

        config.present_mode = if settings.vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        surface.configure(&device, &config);

        let depth_texture_view = Self::create_depth_texture(&device, &config, settings.depth_format);

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

    pub fn color_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }
}
