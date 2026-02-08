//! Texture 和 Image 相关操作
//!
//! 核心概念：
//! - `GpuImage`: 物理纹理资源，包含 `wgpu::Texture` 和默认视图
//! - `GpuSampler`: 采样器状态，全局缓存复用
//! - `TextureBinding`: 将 `TextureHandle` 映射到 (`ImageId`, `ViewId`, `SamplerId`)
//! - `TextureViewKey`: 视图缓存键，支持"一份数据，多种视图"

use std::sync::atomic::Ordering;

use crate::assets::server::SamplerHandle;
use crate::assets::{AssetServer, TextureHandle};
use crate::renderer::core::resources::generate_gpu_resource_id;
use crate::resources::image::{Image, ImageInner};
use crate::resources::texture::{SamplerSource, TextureSampler};

use super::{ResourceManager};

/// 纹理资源映射
///
/// 将 `TextureHandle` 映射到对应的 `GpuImage` ID、View ID 和 `GpuSampler` ID
#[derive(Debug, Clone, Copy)]
pub struct TextureBinding {
    /// GPU 端图像视图 ID
    pub view_id: u64,
    /// CPU 端图像 ID
    pub cpu_image_id: u64,
    pub sampler_id: u64,
    /// CPU 端 Texture 版本（用于检测采样参数变化）
    pub texture_version: u64,
}

/// 纹理视图缓存键
///
/// 用于按需创建和缓存不同配置的 `TextureView`。
/// Key 包含 `view_id，确保底层` Image 重建时自动失效。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureViewKey {
    pub view_id: u64,
    pub format: Option<wgpu::TextureFormat>,
    pub dimension: Option<wgpu::TextureViewDimension>,
    pub base_mip_level: u32,
    pub mip_level_count: Option<u32>,
    pub base_array_layer: u32,
    pub array_layer_count: Option<u32>,
    pub aspect: wgpu::TextureAspect,
}

impl TextureViewKey {
    #[inline]
    #[must_use]
    pub fn new(view_id: u64, desc: &wgpu::TextureViewDescriptor) -> Self {
        Self {
            view_id,
            format: desc.format,
            dimension: desc.dimension,
            base_mip_level: desc.base_mip_level,
            mip_level_count: desc.mip_level_count,
            base_array_layer: desc.base_array_layer,
            array_layer_count: desc.array_layer_count,
            aspect: desc.aspect,
        }
    }

    // #[inline]
    // #[must_use]
    // pub fn default_for_view(view_id: u64) -> Self {
    //     Self {
    //         view_id,
    //         format: None,
    //         dimension: None,
    //         base_mip_level: 0,
    //         mip_level_count: None,
    //         base_array_layer: 0,
    //         array_layer_count: None,
    //         aspect: wgpu::TextureAspect::All,
    //     }
    // }
}

/// GPU 端图像资源
///
/// 包含物理纹理和默认视图，不包含采样器
pub struct GpuImage {
    pub id: u64,
    pub texture: wgpu::Texture,
    pub default_view: wgpu::TextureView,
    pub default_view_dimension: wgpu::TextureViewDimension,
    pub size: wgpu::Extent3d,
    pub format: wgpu::TextureFormat,
    pub mip_level_count: u32,
    pub usage: wgpu::TextureUsages,
    pub version: u64,
    pub generation_id: u64,
    pub mipmaps_generated: bool,
    pub last_used_frame: u64,
}

/// GPU 端采样器资源
///
/// 与 `GpuImage` 分离，实现全局缓存和复用
pub struct GpuSampler {
    pub id: u64,
    pub sampler: wgpu::Sampler,
}

impl GpuImage {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &ImageInner,
        view_dimension: wgpu::TextureViewDimension,
        mip_level_count: u32,
        usage: wgpu::TextureUsages,
    ) -> Self {
        let width = image.width.load(Ordering::Relaxed);
        let height = image.height.load(Ordering::Relaxed);
        let depth = image.depth.load(Ordering::Relaxed);
        let desc = image
            .description
            .read()
            .expect("Failed to read image descriptor");

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

        let default_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: image.label(),
            format: Some(desc.format),
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
            format: desc.format,
            mip_level_count,
            usage,
            version: image.version.load(Ordering::Relaxed),
            generation_id: image.generation_id.load(Ordering::Relaxed),
            mipmaps_generated,
            last_used_frame: 0,
        }
    }

    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &ImageInner,
        view_dimension: wgpu::TextureViewDimension,
    ) {
        let gen_id = image.generation_id.load(Ordering::Relaxed);
        if self.generation_id != gen_id {
            *self = Self::new(
                device,
                queue,
                image,
                view_dimension,
                self.mip_level_count,
                self.usage,
            );
            return;
        }

        let ver = image.version.load(Ordering::Relaxed);
        if self.version < ver {
            Self::upload_data(
                queue,
                &self.texture,
                image,
                self.size.width,
                self.size.height,
                self.size.depth_or_array_layers,
                self.format,
            );
            self.version = ver;
            if self.mip_level_count > 1 {
                self.mipmaps_generated = false;
            }
        }
    }

    fn upload_data(
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        image: &ImageInner,
        src_width: u32,
        src_height: u32,
        src_depth: u32,
        src_format: wgpu::TextureFormat,
    ) {
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
                },
            );
        }
    }
}

impl ResourceManager {
    pub(crate) fn prepare_image(
        &mut self,
        image: &Image,
        view_dimension: wgpu::TextureViewDimension,
        required_mip_count: u32,
        required_usage: wgpu::TextureUsages,
    ) -> u64 {
        let id = image.id();
        let mut needs_recreate = false;

        if let Some(gpu_img) = self.gpu_images.get(&id) {
            if gpu_img.mip_level_count < required_mip_count
                || !gpu_img.usage.contains(required_usage)
            {
                needs_recreate = true;
            }
        } else {
            needs_recreate = true;
        }

        if needs_recreate {
            self.gpu_images.remove(&id);
            let mut gpu_img = GpuImage::new(
                &self.device,
                &self.queue,
                image,
                view_dimension,
                required_mip_count,
                required_usage,
            );
            gpu_img.last_used_frame = self.frame_index;
            let new_id = gpu_img.id;
            self.gpu_images.insert(id, gpu_img);
            new_id
        } else if let Some(gpu_img) = self.gpu_images.get_mut(&id) {
            gpu_img.update(&self.device, &self.queue, image, view_dimension);
            gpu_img.last_used_frame = self.frame_index;
            gpu_img.id
        } else {
            0
        }
    }

    pub fn prepare_texture(&mut self, assets: &AssetServer, handle: TextureHandle) {
        if handle == TextureHandle::dummy_env_map() {
            return;
        }

        let Some(texture_asset) = assets.textures.get(handle) else {
            log::warn!("Texture asset not found for handle: {handle:?}");
            return;
        };

        if let Some(binding) = self.texture_bindings.get(handle) {
            let image_id = texture_asset.image.id();
            if let Some(gpu_img) = self.gpu_images.get(&image_id) {
                let version_match = binding.texture_version == texture_asset.version();
                let image_match = binding.view_id == gpu_img.id
                    && gpu_img.generation_id == texture_asset.image.generation_id();

                if version_match && image_match {
                    if let Some(gpu_img) = self.gpu_images.get_mut(&image_id) {
                        gpu_img.last_used_frame = self.frame_index;
                    }
                    return;
                }
            }
        }

        let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let generated_mips = if texture_asset.generate_mipmaps {
            texture_asset.mip_level_count()
        } else {
            1
        };
        let final_mip_count = std::cmp::max(1, generated_mips);

        if final_mip_count > 1 {
            usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
        }

        let gpu_image_id = self.prepare_image(
            &texture_asset.image,
            texture_asset.view_dimension,
            final_mip_count,
            usage,
        );

        let image_id = texture_asset.image.id();

        if texture_asset.generate_mipmaps
            && let Some(gpu_img) = self.gpu_images.get(&image_id)
            && !gpu_img.mipmaps_generated
        {
            let gpu_img_mut = self.gpu_images.get_mut(&image_id).unwrap();
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Mipmap Gen"),
                });
            self.mipmap_generator
                .generate(&self.device, &mut encoder, &gpu_img_mut.texture);
            self.queue.submit(Some(encoder.finish()));
            gpu_img_mut.mipmaps_generated = true;
        }

        let sampler_id = self.get_or_create_sampler(texture_asset.sampler, texture_asset.name());

        let binding = TextureBinding {
            view_id: gpu_image_id,
            cpu_image_id: image_id,
            sampler_id,
            texture_version: texture_asset.version(),
        };
        self.texture_bindings.insert(handle, binding);
    }

    /// 准备 Sampler 资源
    ///
    /// 逻辑：
    /// 1. 从 `AssetServer` 获取 Sampler 数据
    /// 2. 构建 Descriptor Key
    /// 3. 查 `sampler_cache` (去重)
    ///    - 命中：直接复用 ID
    ///    - 未命中：创建新 `wgpu::Sampler，存入` cache 和 lookup
    /// 4. 更新 `sampler_bindings` 映射
    pub fn prepare_sampler(&mut self, assets: &AssetServer, handle: SamplerHandle) -> u64 {
        // 1. 如果已经绑定且 Asset 版本没变，直接返回 (优化)
        if let Some(&id) = self.sampler_bindings.get(handle) {
            // 这里可以加版本检查逻辑，类似 prepare_texture
            return id;
        }

        // 2. 获取 Asset 数据
        let sampler_asset = assets
            .samplers
            .get(handle)
            .expect("Sampler asset not found");

        // 3. 构建 Key
        let key = sampler_asset.descriptor;

        // 4. 查找或创建 GPU 资源 (Flyweight 模式)
        let id = if let Some(gpu_sampler) = self.sampler_cache.get(&key) {
            gpu_sampler.id
        } else {
            // 创建新的 wgpu::Sampler
            let desc = wgpu::SamplerDescriptor {
                label: Some("Cached Sampler"),
                address_mode_u: key.address_mode_u,
                address_mode_v: key.address_mode_v,
                address_mode_w: key.address_mode_w,
                mag_filter: key.mag_filter,
                min_filter: key.min_filter,
                mipmap_filter: key.mipmap_filter,
                lod_min_clamp: 0.0, // 这些参数如果在 Key 里没有，就用默认值
                lod_max_clamp: 32.0,
                compare: key.compare,
                anisotropy_clamp: key.anisotropy_clamp,
                border_color: None,
            };
            let sampler = self.device.create_sampler(&desc);
            let new_id = generate_gpu_resource_id();

            let gpu_sampler = GpuSampler {
                id: new_id,
                sampler: sampler.clone(),
            };

            self.sampler_cache.insert(key, gpu_sampler);
            self.sampler_id_lookup.insert(new_id, sampler);
            new_id
        };

        // 5. 记录绑定关系
        self.sampler_bindings.insert(handle, id);

        id
    }

    pub fn resolve_sampler_id(&mut self, assets: &AssetServer, source: SamplerSource) -> u64 {
        match source {
            SamplerSource::FromTexture(tex_handle) => {
                if let Some(binding) = self.texture_bindings.get(tex_handle) {
                    return binding.sampler_id;
                }

                if let Some(texture) = assets.textures.get(tex_handle) {
                    // [修改] 不需要转换 Key，直接传 sampler
                    self.get_or_create_sampler(texture.sampler, texture.name())
                } else {
                    self.dummy_sampler.id
                }
            }

            SamplerSource::Asset(sampler_handle) => self.prepare_sampler(assets, sampler_handle),

            SamplerSource::Default => self.dummy_sampler.id,
        }
    }

    /// 获取指定配置的 `TextureView`
    ///
    /// 极速路径：如果 desc 为 None，直接返回默认视图
    /// 缓存路径：根据 `TextureViewKey` 查找/创建视图
    #[inline]
    pub fn get_texture_view_desc(
        &mut self,
        cpu_image_id: u64,
        desc: Option<&wgpu::TextureViewDescriptor>,
    ) -> Option<(&wgpu::TextureView, u64)> {
        let gpu_image = self.gpu_images.get(&cpu_image_id)?;
        let image_id = gpu_image.id;

        if desc.is_none() {
            return Some((&gpu_image.default_view, image_id));
        }

        let desc = desc.unwrap();
        let key = TextureViewKey::new(image_id, desc);

        if !self.view_cache.contains_key(&key) {
            let gpu_image = self.gpu_images.get(&cpu_image_id)?;
            let view = gpu_image.texture.create_view(desc);
            let view_id = generate_gpu_resource_id();
            self.view_cache.insert(key, (view, view_id));
        }

        let (view, id) = self.view_cache.get(&key)?;
        Some((view, *id))
    }

    /// 获取指定配置的 TextureView（不可变版本，仅查缓存）
    #[inline]
    pub fn get_texture_view_cached(
        &self,
        cpu_image_id: u64,
        desc: Option<&wgpu::TextureViewDescriptor>,
    ) -> Option<(&wgpu::TextureView, u64)> {
        let gpu_image = self.gpu_images.get(&cpu_image_id)?;
        let image_id = gpu_image.id;

        if desc.is_none() {
            return Some((&gpu_image.default_view, image_id));
        }

        let desc = desc.unwrap();
        let key = TextureViewKey::new(image_id, desc);
        let (view, id) = self.view_cache.get(&key)?;
        Some((view, *id))
    }

    pub(crate) fn get_or_create_sampler(
        &mut self,
        descriptor: TextureSampler,
        label: Option<&str>,
    ) -> u64 {
        // 1. 直接用 descriptor 查表
        if let Some(gpu_sampler) = self.sampler_cache.get(&descriptor) {
            return gpu_sampler.id;
        }

        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label,
            address_mode_u: descriptor.address_mode_u,
            address_mode_v: descriptor.address_mode_v,
            address_mode_w: descriptor.address_mode_w,
            mag_filter: descriptor.mag_filter,
            min_filter: descriptor.min_filter,
            mipmap_filter: descriptor.mipmap_filter,
            compare: descriptor.compare,
            anisotropy_clamp: descriptor.anisotropy_clamp,
            ..Default::default()
        });

        let id = generate_gpu_resource_id();
        let gpu_sampler = GpuSampler {
            id,
            sampler: sampler.clone(),
        };

        self.sampler_cache.insert(descriptor, gpu_sampler);
        self.sampler_id_lookup.insert(id, sampler);

        id
    }

    #[inline]
    pub fn get_sampler_by_id(&self, sampler_id: u64) -> Option<&wgpu::Sampler> {
        self.sampler_id_lookup.get(&sampler_id)
    }

    #[inline]
    pub fn get_texture_binding(&self, handle: TextureHandle) -> Option<&TextureBinding> {
        self.texture_bindings.get(handle)
    }

    #[inline]
    pub fn get_image(&self, id: u64) -> Option<&GpuImage> {
        self.gpu_images.get(&id)
    }

    #[inline]
    pub fn get_image_by_cpu_id(&self, cpu_image_id: u64) -> Option<&GpuImage> {
        self.gpu_images.get(&cpu_image_id)
    }
}
