use wgpu;
use crate::resources::texture::Texture;
use crate::render::resources::image::GpuImage;

pub struct GpuTexture {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,

    // 依赖追踪
    pub image_id: u64,
    pub image_generation_id: u64,
    
    pub version: u64, // Texture 配置版本
    pub last_used_frame: u64,
}

impl GpuTexture {
    pub fn new(device: &wgpu::Device, texture: &Texture, gpu_image: &GpuImage) -> Self {
        let view = gpu_image.texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("{}_view", texture.name)),
            format: Some(gpu_image.texture.format()),
            dimension: Some(texture.view_dimension), 
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&format!("{}_sampler", texture.name)),
            address_mode_u: texture.sampler.address_mode_u,
            address_mode_v: texture.sampler.address_mode_v,
            address_mode_w: texture.sampler.address_mode_w,
            mag_filter: texture.sampler.mag_filter,
            min_filter: texture.sampler.min_filter,
            mipmap_filter: texture.sampler.mipmap_filter,
            compare: texture.sampler.compare,
            anisotropy_clamp: texture.sampler.anisotropy_clamp,
            ..Default::default()
        });

        Self {
            view,
            sampler,
            image_id: gpu_image.id,
            image_generation_id: gpu_image.generation_id,
            version: texture.version(),
            last_used_frame: 0,
        }
    }
}