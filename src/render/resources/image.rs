use crate::resources::image::ImageInner;
use std::sync::atomic::Ordering;

pub struct GpuImage {
    pub texture: wgpu::Texture,
    
    pub id: u64, // 对应的 Image ID
    pub version: u64,
    pub generation_id: u64,

    pub last_used_frame: u64,
}

impl GpuImage {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, image: &ImageInner) -> Self {

        let desc = *image.descriptor.read().expect("Failed to read image descriptor");

        let size = wgpu::Extent3d {
            width: desc.width,
            height: desc.height,
            depth_or_array_layers: desc.depth_or_array_layers,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: image.label(),
            size,
            mip_level_count: desc.mip_level_count,
            sample_count: 1,
            dimension: desc.dimension,
            format: desc.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        Self::upload_data(queue, &texture, image);

        Self {
            texture,
            id: image.id,
            version: image.version.load(Ordering::Relaxed),
            generation_id: image.generation_id.load(Ordering::Relaxed),
            last_used_frame: 0,
        }
    }

    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, image: &ImageInner) {
        // 1. 检查是否需要重建 (Resize)
        let gen_id = image.generation_id.load(Ordering::Relaxed);
        if self.generation_id != gen_id {
             // 简单的处理：重建整个对象
             // 在实际工程中，ResourceManager 应该负责替换这个 struct
             *self = Self::new(device, queue, image);
             return;
        }

        // 2. 检查是否需要上传数据
        let ver = image.version.load(Ordering::Relaxed);
        if self.version < ver {
            Self::upload_data(queue, &self.texture, image);
            self.version = ver;
        }
    }

    fn upload_data(queue: &wgpu::Queue, texture: &wgpu::Texture, image: &ImageInner) {
        let data_guard = image.data.read().expect("Failed to read image data");
        if let Some(data) = &*data_guard {
            let width = texture.width();
            let height = texture.height();
            let depth = texture.depth_or_array_layers();
            let format = texture.format();

            // 计算布局
            let block_size = format.block_copy_size(None).unwrap_or(4);
            let bytes_per_row = width * block_size;
            
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: depth,
                }
            );
        }
    }
}