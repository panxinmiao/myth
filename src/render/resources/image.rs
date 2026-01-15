use crate::resources::image::ImageInner;
use std::sync::atomic::Ordering;

pub struct GpuImage {
    pub texture: wgpu::Texture,
    
    pub id: u64,
    pub version: u64,
    pub generation_id: u64,

    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: wgpu::TextureFormat,
    pub mip_level_count: u32,
    pub usage: wgpu::TextureUsages,

    pub mipmaps_generated: bool,
    pub last_used_frame: u64,
}

impl GpuImage {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, image: &ImageInner, mip_level_count: u32, usage: wgpu::TextureUsages) -> Self {

        // let desc = *image.descriptor.read().expect("Failed to read image descriptor");

        let width = image.width.load(Ordering::Relaxed);
        let height = image.height.load(Ordering::Relaxed);
        let depth = image.depth.load(Ordering::Relaxed);
        let desc = image.description.read().expect("Failed to read image descriptor");
        let size = wgpu::Extent3d {
            width: width,
            height: height,
            depth_or_array_layers: depth,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: image.label(),
            size,
            mip_level_count,
            sample_count: 1,
            dimension: desc.dimension,
            format: desc.format,
            usage,
            view_formats: &[],
        });

        Self::upload_data(queue, &texture, image, width, height, depth, desc.format);

        let mipmaps_generated = mip_level_count <= 1;
        Self {
            texture,
            id: image.id,
            version: image.version.load(Ordering::Relaxed),
            generation_id: image.generation_id.load(Ordering::Relaxed),

            width,
            height,
            depth,
            format: desc.format,
            mip_level_count,
            usage,

            mipmaps_generated,
            last_used_frame: 0,
        }
    }

    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, image: &ImageInner) {
        // 1. 检查是否需要重建 (Resize)
        let gen_id = image.generation_id.load(Ordering::Relaxed);
        if self.generation_id != gen_id {
             // 简单的处理：重建整个对象
             // 在实际工程中，ResourceManager 应该负责替换这个 struct
             *self = Self::new(device, queue, image, self.mip_level_count, self.usage);
             return;
        }

        // 2. 检查是否需要上传数据
        let ver = image.version.load(Ordering::Relaxed);
        if self.version < ver {
            Self::upload_data(queue, &self.texture, image, self.width, self.height, self.depth, self.format);
            self.version = ver;
            if self.mip_level_count > 1 {
                self.mipmaps_generated = false; 
            }
        }
    }

    fn upload_data(queue: &wgpu::Queue, texture: &wgpu::Texture, image: &ImageInner, src_width: u32, src_height: u32, src_depth: u32, src_format: wgpu::TextureFormat) {
        let data_guard = image.data.read().expect("Failed to read image data");
        if let Some(data) = &*data_guard {
            // 计算布局
            let block_size = src_format.block_copy_size(None).unwrap_or(4);
            let bytes_per_row = src_width * block_size;
            
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
                    rows_per_image: Some(src_height),
                },
                wgpu::Extent3d {
                    width: src_width,
                    height: src_height,
                    depth_or_array_layers: src_depth,
                }
            );
        }
    }
}