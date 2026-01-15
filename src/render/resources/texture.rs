use wgpu;
use crate::resources::texture::Texture;
use crate::render::resources::image::GpuImage;

pub struct GpuTexture {
    pub view: wgpu::TextureView,

    pub image_id: u64,
    pub image_generation_id: u64,
    
    pub version: u64,

    pub image_data_version: u64,
    pub last_used_frame: u64,
}

impl GpuTexture {
    pub fn new(texture: &Texture, gpu_image: &GpuImage) -> Self {
        let view = gpu_image.texture.create_view(&wgpu::TextureViewDescriptor {
            label:  texture.name(),
            format: Some(gpu_image.texture.format()),
            dimension: Some(texture.view_dimension), 
            ..Default::default()
        });

        Self {
            view,
            image_id: gpu_image.id,
            image_generation_id: gpu_image.generation_id,
            version: texture.version(),
            image_data_version: gpu_image.version,
            last_used_frame: 0,
        }
    }
}