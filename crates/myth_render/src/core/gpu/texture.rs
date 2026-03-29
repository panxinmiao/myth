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
        resolved_format: wgpu::TextureFormat,
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
            dimension: image.dimension.to_wgpu(),
            format: resolved_format,
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
            resolved_format,
        );

        let default_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: Some(resolved_format),
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
            format: resolved_format,
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
        resolved_format: wgpu::TextureFormat,
        view_dimension: wgpu::TextureViewDimension,
        image_version: u32,
    ) {
        // Dimension/format changed → full rebuild
        if self.size.width != image.width
            || self.size.height != image.height
            || self.size.depth_or_array_layers != image.depth
            || self.format != resolved_format
        {
            *self = Self::new(
                device,
                queue,
                image,
                resolved_format,
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
        resolved_format: wgpu::TextureFormat,
        view_dimension: wgpu::TextureViewDimension,
        required_mip_count: u32,
        required_usage: wgpu::TextureUsages,
    ) -> u64 {
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
                resolved_format,
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
                resolved_format,
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
    /// Performs a **two-level query**: first retrieves the `Texture` config
    /// (always immediately available after [`AssetServer::load_texture`]),
    /// then checks whether the underlying `Image` has finished decoding.
    /// If the image is not yet ready, no binding is created and the
    /// material system falls back to a placeholder texture.
    ///
    /// Version tracking ensures GPU resources are only rebuilt when the
    /// underlying data actually changes.
    pub fn prepare_texture(&mut self, assets: &AssetServer, handle: TextureHandle) {
        if handle == TextureHandle::dummy_env_map() {
            return;
        }

        let Some(texture_asset) = assets.textures.get(handle) else {
            return;
        };

        let image_handle: ImageHandle = texture_asset.image;

        // ── Fast path: skip if nothing changed (no RwLock / Arc / hash) ──
        if let Some(binding) = self.texture_bindings.get(handle)
            && let Some(gpu_img) = self.gpu_images.get_mut(image_handle)
        {
            // Lightweight version check — single RwLock read, no Arc::clone
            if let Some(img_ver) = assets.images.get_version(image_handle) {
                let version_match = (binding.texture_version as u32) >= img_ver;
                let image_match =
                    binding.image_handle == image_handle && binding.view_id == gpu_img.id;
                let sampler_match = self
                    .sampler_registry
                    .lookup_index(&texture_asset.sampler)
                    .is_some_and(|idx| idx == binding.sampler_id);

                if version_match && image_match && sampler_match {
                    gpu_img.last_used_frame = self.frame_index;
                    return;
                }
            }
        }

        // ── Slow path: something changed, do the full update ──
        let Some((image_arc, image_version)) = assets.images.get_entry(image_handle) else {
            // Image still decoding in the background — no binding is created,
            // the material system will use a fallback placeholder texture.
            return;
        };

        let resolved_format = texture_asset.resolve_wgpu_format(image_arc.format);
        let sampler_id = self.get_or_create_sampler(texture_asset.sampler);

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
            resolved_format,
            texture_asset.view_dimension,
            final_mip_count,
            usage,
        );

        if texture_asset.generate_mipmaps
            && let Some(gpu_img) = self.gpu_images.get_mut(image_handle)
            && !gpu_img.mipmaps_generated
        {
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
