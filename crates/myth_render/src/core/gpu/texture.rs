//! Texture and Image operations
//!
//! Core concepts:
//! - `GpuImage`: Physical texture resource, containing `wgpu::Texture` and default view
//! - `GpuSampler`: Sampler state, globally cached for reuse
//! - `TextureBinding`: Maps `TextureHandle` to (`ImageId`, `ViewId`, `SamplerId`)
use crate::core::gpu::generate_gpu_resource_id;
use myth_assets::{AssetServer, ImageHandle, TextureHandle};
use myth_resources::image::Image;
use myth_resources::texture::TextureSampler;

use super::ResourceManager;

/// Texture resource mapping
///
/// Maps `TextureHandle` to the corresponding `GpuImage` ID, View ID, and `GpuSampler` ID
#[derive(Debug, Clone, Copy)]
pub struct TextureBinding {
    /// GPU-side image view ID
    pub view_id: u64,
    /// CPU-side image ID
    pub image_handle: ImageHandle,
    pub sampler_id: usize,
    /// CPU-side Texture version (used to detect sampler parameter changes)
    pub texture_version: u64,
}

/// GPU-side image resource
///
/// Contains the physical texture and default view, excluding sampler
pub struct GpuImage {
    pub id: u64,
    pub texture: wgpu::Texture,
    pub default_view: wgpu::TextureView,
    pub default_view_dimension: wgpu::TextureViewDimension,
    pub size: wgpu::Extent3d,
    pub format: wgpu::TextureFormat,
    pub mip_level_count: u32,
    pub usage: wgpu::TextureUsages,
    pub version: u32,
    pub generation_id: u64,
    pub mipmaps_generated: bool,
    pub last_used_frame: u64,
}

impl GpuImage {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &Image,
        view_dimension: wgpu::TextureViewDimension,
        mip_level_count: u32,
        usage: wgpu::TextureUsages,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: image.width,
            height: image.height,
            depth_or_array_layers: image.depth,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count,
            sample_count: 1,
            dimension: image.description.dimension,
            format: image.description.format,
            usage,
            view_formats: &[],
        });

        Self::upload_data(
            queue,
            &texture,
            image,
            image.width,
            image.height,
            image.depth,
            image.description.format,
        );

        let default_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: Some(image.description.format),
            dimension: Some(view_dimension),
            ..Default::default()
        });

        let mipmaps_generated = mip_level_count <= 1;
        Self {
            id: generate_gpu_resource_id(),
            texture,
            default_view,
            default_view_dimension: view_dimension,
            size,
            format: image.description.format,
            mip_level_count,
            usage,
            version: 0,
            generation_id: 0,
            mipmaps_generated,
            last_used_frame: 0,
        }
    }

    /// Check if the image data has changed and re-upload if needed.
    ///
    /// If the image dimensions or format changed (generation change),
    /// the entire GPU texture is rebuilt.
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &Image,
        view_dimension: wgpu::TextureViewDimension,
        image_version: u32,
    ) {
        // Dimension/format changed → full rebuild
        if self.size.width != image.width
            || self.size.height != image.height
            || self.size.depth_or_array_layers != image.depth
            || self.format != image.description.format
        {
            *self = Self::new(
                device,
                queue,
                image,
                view_dimension,
                self.mip_level_count,
                self.usage,
            );
            self.version = image_version;
            return;
        }

        // Data-only update
        if self.version < image_version {
            Self::upload_data(
                queue,
                &self.texture,
                image,
                self.size.width,
                self.size.height,
                self.size.depth_or_array_layers,
                self.format,
            );
            self.version = image_version;
            if self.mip_level_count > 1 {
                self.mipmaps_generated = false;
            }
        }
    }

    fn upload_data(
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        image: &Image,
        src_width: u32,
        src_height: u32,
        src_depth: u32,
        src_format: wgpu::TextureFormat,
    ) {
        if let Some(data) = &image.data {
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
                },
            );
        }
    }
}

impl ResourceManager {
    /// Ensure a GPU image exists for the given CPU `Image`, creating or
    /// updating as needed.  Returns the physical GPU image ID.
    pub(crate) fn prepare_image(
        &mut self,
        image: &Image,
        image_handle: ImageHandle,
        image_version: u32,
        view_dimension: wgpu::TextureViewDimension,
        required_mip_count: u32,
        required_usage: wgpu::TextureUsages,
    ) -> u64 {
        // let uuid = image.uuid;
        // let id = uuid.as_u128() as u64;
        let mut needs_recreate = false;

        if let Some(gpu_img) = self.gpu_images.get(image_handle) {
            if gpu_img.mip_level_count < required_mip_count
                || !gpu_img.usage.contains(required_usage)
            {
                needs_recreate = true;
            }
        } else {
            needs_recreate = true;
        }

        if needs_recreate {
            self.gpu_images.remove(image_handle);
            let mut gpu_img = GpuImage::new(
                &self.device,
                &self.queue,
                image,
                view_dimension,
                required_mip_count,
                required_usage,
            );
            gpu_img.version = image_version;
            gpu_img.last_used_frame = self.frame_index;
            let new_id = gpu_img.id;
            self.gpu_images.insert(image_handle, gpu_img);
            new_id
        } else if let Some(gpu_img) = self.gpu_images.get_mut(image_handle) {
            gpu_img.update(
                &self.device,
                &self.queue,
                image,
                view_dimension,
                image_version,
            );
            gpu_img.last_used_frame = self.frame_index;
            gpu_img.id
        } else {
            0
        }
    }

    /// Prepare GPU resources for a `Texture` asset.
    ///
    /// Looks up the Image via `AssetServer::images`, builds or updates the
    /// `GpuImage`, resolves the sampler, and records the `TextureBinding`.
    /// Version tracking ensures GPU resources are only rebuilt when the
    /// underlying data actually changes.
    pub fn prepare_texture(&mut self, assets: &AssetServer, handle: TextureHandle) {
        if handle == TextureHandle::dummy_env_map() {
            return;
        }

        let Some(texture_asset) = assets.textures.get(handle) else {
            log::warn!("Texture asset not found for handle: {handle:?}");
            return;
        };

        // Look up Image from storage (with version)
        let Some((image_arc, image_version)) = assets.images.get_entry(texture_asset.image) else {
            log::warn!(
                "Image asset not found for handle: {:?}",
                texture_asset.image
            );
            return;
        };

        let sampler_id = self.get_or_create_sampler(texture_asset.sampler);

        let image_handle = texture_asset.image;

        // Fast path: skip if nothing changed
        if let Some(binding) = self.texture_bindings.get(handle)
            && let Some(gpu_img) = self.gpu_images.get(image_handle)
        {
            let version_match = (binding.texture_version as u32) >= image_version;
            let image_match = binding.image_handle == image_handle && binding.view_id == gpu_img.id;

            let sampler_match = binding.sampler_id == sampler_id;

            if version_match && image_match && sampler_match {
                if let Some(gpu_img) = self.gpu_images.get_mut(image_handle) {
                    gpu_img.last_used_frame = self.frame_index;
                }
                return;
            }
        }

        let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let generated_mips = if texture_asset.generate_mipmaps {
            let max_dim = std::cmp::max(image_arc.width, image_arc.height);
            max_dim.ilog2() + 1
        } else {
            1
        };
        let final_mip_count = std::cmp::max(1, generated_mips);

        if final_mip_count > 1 {
            usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
        }

        let gpu_image_id = self.prepare_image(
            &image_arc,
            image_handle,
            image_version,
            texture_asset.view_dimension,
            final_mip_count,
            usage,
        );

        if texture_asset.generate_mipmaps
            && let Some(gpu_img) = self.gpu_images.get_mut(image_handle)
            && !gpu_img.mipmaps_generated
        {
            // let gpu_img_mut = self.gpu_images.get_mut(image_handle).unwrap();
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Mipmap Gen"),
                });
            self.mipmap_generator
                .generate(&self.device, &mut encoder, &gpu_img.texture);
            self.queue.submit(Some(encoder.finish()));
            gpu_img.mipmaps_generated = true;
        }

        let binding = TextureBinding {
            view_id: gpu_image_id,
            image_handle,
            sampler_id,
            texture_version: u64::from(image_version),
        };
        self.texture_bindings.insert(handle, binding);
    }

    // /// Get a `TextureView` with the specified configuration
    // ///
    // /// Fast path: if desc is None, return the default view directly
    // /// Cache path: look up/create view based on `TextureViewKey`
    // #[inline]
    // pub fn get_texture_view_desc(
    //     &mut self,
    //     cpu_image_id: u64,
    //     desc: Option<&wgpu::TextureViewDescriptor>,
    // ) -> Option<(&wgpu::TextureView, u64)> {
    //     let gpu_image = self.gpu_images.get(&cpu_image_id)?;
    //     let image_id = gpu_image.id;

    //     if desc.is_none() {
    //         return Some((&gpu_image.default_view, image_id));
    //     }

    //     let desc = desc.unwrap();
    //     let key = TextureViewKey::new(image_id, desc);

    //     if !self.view_cache.contains_key(&key) {
    //         let gpu_image = self.gpu_images.get(&cpu_image_id)?;
    //         let view = gpu_image.texture.create_view(desc);
    //         let view_id = generate_gpu_resource_id();
    //         self.view_cache.insert(key, (view, view_id));
    //     }

    //     let (view, id) = self.view_cache.get(&key)?;
    //     Some((view, *id))
    // }

    // /// Get a `TextureView` with the specified configuration (immutable version, cache-only lookup)
    // #[inline]
    // pub fn get_texture_view_cached(
    //     &self,
    //     cpu_image_id: u64,
    //     desc: Option<&wgpu::TextureViewDescriptor>,
    // ) -> Option<(&wgpu::TextureView, u64)> {
    //     let gpu_image = self.gpu_images.get(&cpu_image_id)?;
    //     let image_id = gpu_image.id;

    //     if desc.is_none() {
    //         return Some((&gpu_image.default_view, image_id));
    //     }

    //     let desc = desc.unwrap();
    //     let key = TextureViewKey::new(image_id, desc);
    //     let (view, id) = self.view_cache.get(&key)?;
    //     Some((view, *id))
    // }

    pub(crate) fn get_or_create_sampler(&mut self, descriptor: TextureSampler) -> usize {
        self.sampler_registry
            .get_custom(&self.device, &descriptor)
            .0
    }

    #[inline]
    pub fn get_texture_binding(&self, handle: TextureHandle) -> Option<&TextureBinding> {
        self.texture_bindings.get(handle)
    }

    #[inline]
    pub fn get_image(&self, image_handle: ImageHandle) -> Option<&GpuImage> {
        self.gpu_images.get(image_handle)
    }
}
