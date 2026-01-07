// src/renderer/gpu_texture.rs

use crate::core::texture::Texture;

/// GPU Texture 抽象
/// 管理 Texture, View, Sampler 的生命周期
pub struct GpuTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    
    pub last_used_frame: u64,

    pub version: u64,       // 数据内容版本
    pub generation_id: u64, // 结构版本 (Resize/Format change)
    
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
}

impl GpuTexture {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, texture: &Texture) -> Self {
        Self::create_internal(device, queue, texture)
    }

    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, texture: &Texture) {
        // 1. 结构变更 -> 重建
        if texture.generation_id != self.generation_id {
            *self = Self::create_internal(device, queue, texture);
            return;
        }

        // 2. 数据变更 -> 上传
        if texture.version > self.version {
            Self::upload_data(queue, &self.texture, texture);
            self.version = texture.version;
        }
    }

    fn create_internal(device: &wgpu::Device, queue: &wgpu::Queue, texture: &Texture) -> Self {
        let size = wgpu::Extent3d {
            width: texture.source.width,
            height: texture.source.height,
            depth_or_array_layers: 1,
        };

        let gpu_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&texture.name),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture.source.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        Self::upload_data(queue, &gpu_texture, texture);

        let view = gpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: texture.sampler.address_mode_u,
            address_mode_v: texture.sampler.address_mode_v,
            mag_filter: texture.sampler.mag_filter,
            min_filter: texture.sampler.min_filter,
            ..Default::default()
        });

        Self {
            texture: gpu_texture,
            view,
            sampler,
            last_used_frame: 0,
            version: texture.version,
            generation_id: texture.generation_id,
            width: texture.source.width,
            height: texture.source.height,
            format: texture.source.format,
        }
    }

    fn upload_data(queue: &wgpu::Queue, gpu_texture: &wgpu::Texture, texture: &Texture) {
        if let Some(data) = &texture.source.data {
            let size = wgpu::Extent3d {
                width: texture.source.width,
                height: texture.source.height,
                depth_or_array_layers: 1,
            };
            let bytes_per_pixel = texture.source.bytes_per_pixel();
            let unpadded_bytes_per_row = texture.source.width * bytes_per_pixel;
            
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: gpu_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(unpadded_bytes_per_row),
                    rows_per_image: Some(texture.source.height),
                },
                size,
            );
        }
    }
}