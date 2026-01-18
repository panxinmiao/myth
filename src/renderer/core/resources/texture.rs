//! Texture 和 Image 相关操作

use std::sync::atomic::Ordering;

use crate::assets::{AssetServer, TextureHandle};
use crate::resources::image::{Image, ImageInner};
use crate::resources::texture::{Texture, TextureSampler};

use super::{ResourceManager, GpuTexture, GpuImage};

impl GpuTexture {
    pub fn new(texture: &Texture, gpu_image: &GpuImage) -> Self {
        let view = gpu_image.texture.create_view(&wgpu::TextureViewDescriptor {
            label: texture.name(),
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

impl GpuImage {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, image: &ImageInner, mip_level_count: u32, usage: wgpu::TextureUsages) -> Self {
        let width = image.width.load(Ordering::Relaxed);
        let height = image.height.load(Ordering::Relaxed);
        let depth = image.depth.load(Ordering::Relaxed);
        let desc = image.description.read().expect("Failed to read image descriptor");
        
        let size = wgpu::Extent3d {
            width,
            height,
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
        let gen_id = image.generation_id.load(Ordering::Relaxed);
        if self.generation_id != gen_id {
            *self = Self::new(device, queue, image, self.mip_level_count, self.usage);
            return;
        }

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

impl ResourceManager {
    pub(crate) fn prepare_image(&mut self, image: &Image, required_mip_count: u32, required_usage: wgpu::TextureUsages) {
        let id = image.id();
        let mut needs_recreate = false;

        if let Some(gpu_img) = self.gpu_images.get(&id) {
            if gpu_img.mip_level_count < required_mip_count || !gpu_img.usage.contains(required_usage) {
                needs_recreate = true;
            }
        } else {
            needs_recreate = true;
        }

        if needs_recreate {
            self.gpu_images.remove(&id);
            let mut gpu_img = GpuImage::new(&self.device, &self.queue, image, required_mip_count, required_usage);
            gpu_img.last_used_frame = self.frame_index;
            self.gpu_images.insert(id, gpu_img);
        } else if let Some(gpu_img) = self.gpu_images.get_mut(&id) {
            gpu_img.update(&self.device, &self.queue, image);
            gpu_img.last_used_frame = self.frame_index;
        }
    }

    pub fn prepare_texture(&mut self, assets: &AssetServer, handle: TextureHandle) {

        if handle == TextureHandle::dummy_env_map(){
            return; 
        }


        let Some(texture_asset) = assets.get_texture(handle) else {
            log::warn!("Texture asset not found for handle: {:?}", handle);
            return;
        };

        if let Some(gpu_tex) = self.gpu_textures.get_mut(handle) {
            let tex_ver_match = gpu_tex.version == texture_asset.version();
            let img_id_match = gpu_tex.image_id == texture_asset.image.id();
            let img_gen_match = gpu_tex.image_generation_id == texture_asset.image.generation_id();
            let img_data_match = gpu_tex.image_data_version == texture_asset.image.version();

            if tex_ver_match && img_id_match && img_gen_match && img_data_match {
                gpu_tex.last_used_frame = self.frame_index;
                if let Some(gpu_img) = self.gpu_images.get_mut(&gpu_tex.image_id) {
                    gpu_img.last_used_frame = self.frame_index;
                }
                return;
            }
        }

        let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let image_mips = 1;
        let generated_mips = if texture_asset.generate_mipmaps { texture_asset.mip_level_count() } else { 1 };
        let final_mip_count = std::cmp::max(image_mips, generated_mips);

        if final_mip_count > 1 {
            usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
        }

        self.prepare_image(&texture_asset.image, final_mip_count, usage);

        let image_id = texture_asset.image.id();
        let gpu_image = self.gpu_images.get(&image_id).expect("GpuImage should be ready");

        if texture_asset.generate_mipmaps && !gpu_image.mipmaps_generated {
            let gpu_img_mut = self.gpu_images.get_mut(&image_id).unwrap();
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Mipmap Gen") });
            self.mipmap_generator.generate(&self.device, &mut encoder, &gpu_img_mut.texture, gpu_img_mut.mip_level_count);
            self.queue.submit(Some(encoder.finish()));
            gpu_img_mut.mipmaps_generated = true;
        }

        let gpu_image = self.gpu_images.get(&image_id).unwrap();

        let mut needs_update_texture = false;
        if let Some(gpu_tex) = self.gpu_textures.get_mut(handle) {
            let config_changed = gpu_tex.version != texture_asset.version();
            let image_recreated = gpu_tex.image_generation_id != gpu_image.generation_id;
            let image_swapped = gpu_tex.image_id != image_id;

            if config_changed || image_recreated || image_swapped {
                needs_update_texture = true;
            }
            gpu_tex.last_used_frame = self.frame_index;
        } else {
            needs_update_texture = true;
        }

        if needs_update_texture {
            let gpu_tex = GpuTexture::new(texture_asset, gpu_image);
            self.gpu_textures.insert(handle, gpu_tex);

            let sampler = self.get_or_create_sampler(&texture_asset.sampler, texture_asset.name());
            self.gpu_samplers.insert(handle, sampler);
        }

        if let Some(gpu_tex) = self.gpu_textures.get_mut(handle) {
            gpu_tex.last_used_frame = self.frame_index;
        }
    }

    pub(crate) fn get_or_create_sampler(&mut self, config: &TextureSampler, label: Option<&str>) -> wgpu::Sampler {
        if let Some(sampler) = self.sampler_cache.get(config) {
            return sampler.clone();
        }

        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label,
            address_mode_u: config.address_mode_u,
            address_mode_v: config.address_mode_v,
            address_mode_w: config.address_mode_w,
            mag_filter: config.mag_filter,
            min_filter: config.min_filter,
            mipmap_filter: config.mipmap_filter,
            compare: config.compare,
            anisotropy_clamp: config.anisotropy_clamp,
            ..Default::default()
        });

        self.sampler_cache.insert(*config, sampler.clone());
        sampler
    }
}
