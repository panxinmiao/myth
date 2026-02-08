use wgpu::TextureViewDimension;

use crate::assets::AssetServer;
use crate::resources::texture::TextureSource;

use super::{ResourceManager, generate_gpu_resource_id};

const EQUIRECT_CUBE_SIZE: u32 = 1024;
pub(crate) const PMREM_SIZE: u32 = 512;

/// How the environment source needs to be processed by `IBLComputePass`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeSourceType {
    /// 2D equirectangular HDR → equirect_to_cube + mipmap_gen + PMREM
    Equirectangular,
    /// Cube map without mipmaps → blit to owned cube + mipmap_gen + PMREM
    CubeNoMipmaps,
    /// Cube map with mipmaps → PMREM only (uses source cube directly)
    CubeWithMipmaps,
}

/// GPU-side environment map resources.
///
/// Created during `resolve_gpu_environment` (prepare phase) and written by
/// `IBLComputePass` (compute phase). The pass only writes into pre-created
/// textures; it never creates or removes cache entries.
#[derive(Debug)]
pub struct GpuEnvironment {
    /// Version of the source texture when this entry was last (re)created
    pub source_version: u64,
    /// Whether compute pass needs to (re)generate the textures
    pub needs_compute: bool,
    /// How the source needs to be processed
    pub source_type: CubeSourceType,
    /// Cube texture (owned; created when source is 2D or cube without mipmaps)
    pub cube_texture: Option<wgpu::Texture>,
    /// PMREM texture (always owned)
    pub pmrem_texture: wgpu::Texture,
    /// Resource ID for the cube view registered in `internal_resources`
    pub cube_view_id: u64,
    /// Resource ID for the PMREM view registered in `internal_resources`
    pub pmrem_view_id: u64,
    /// Maximum mip level (`mip_levels - 1`) for roughness LOD
    pub env_map_max_mip_level: f32,
}

impl ResourceManager {
    /// Resolve (or create) the `GpuEnvironment` for the current scene environment.
    ///
    /// This must be called before `prepare_global` so that the uniform buffer
    /// can be populated with the correct `env_map_max_mip_level`, and so that
    /// real resource IDs are available for BindGroup creation.
    ///
    /// All GPU textures (cube, PMREM) are created here; `IBLComputePass` only
    /// writes into them — it never creates or removes cache entries.
    ///
    /// Returns the resolved `env_map_max_mip_level` (0.0 if no env map).
    pub fn resolve_gpu_environment(
        &mut self,
        assets: &AssetServer,
        environment: &crate::scene::environment::Environment,
    ) -> f32 {
        let source = match environment.source_env_map {
            Some(s) => s,
            None => return 0.0,
        };

        let mut current_version: u64 = 0;
        if let TextureSource::Asset(handle) = &source {
            self.prepare_texture(assets, *handle);
            if let Some(tex) = assets.textures.get(*handle) {
                current_version = tex.version();
            }
        }

        // --- Check existing cache entry ---
        if let Some(gpu_env) = self.environment_map_cache.get_mut(&source) {
            if gpu_env.source_version == current_version && !gpu_env.needs_compute {
                return gpu_env.env_map_max_mip_level;
            }
            if gpu_env.source_version != current_version {
                // Source texture content changed — need to regenerate
                gpu_env.source_version = current_version;
                gpu_env.needs_compute = true;
                self.pending_ibl_source = Some(source);
            }
            return gpu_env.env_map_max_mip_level;
        }

        // --- No cached entry — determine source type ---
        let (is_2d_source, source_cube_size, source_mip_count) = match &source {
            TextureSource::Asset(handle) => {
                if let Some(binding) = self.texture_bindings.get(*handle) {
                    if let Some(img) = self.gpu_images.get(&binding.cpu_image_id) {
                        let is_2d = img.default_view_dimension == TextureViewDimension::D2;
                        (is_2d, img.size.width, img.mip_level_count)
                    } else {
                        (true, EQUIRECT_CUBE_SIZE, 1)
                    }
                } else {
                    (true, EQUIRECT_CUBE_SIZE, 1)
                }
            }
            TextureSource::Attachment(_, dim) => {
                let is_2d = *dim == TextureViewDimension::D2;
                // Cannot determine mip count for attachments; assume cube has mipmaps
                (is_2d, EQUIRECT_CUBE_SIZE, if is_2d { 1 } else { 2 })
            }
        };

        let source_type = if is_2d_source {
            CubeSourceType::Equirectangular
        } else if source_mip_count <= 1 {
            CubeSourceType::CubeNoMipmaps
        } else {
            CubeSourceType::CubeWithMipmaps
        };

        // --- Create owned cube texture (Equirectangular & CubeNoMipmaps) ---
        let needs_owned_cube = matches!(
            source_type,
            CubeSourceType::Equirectangular | CubeSourceType::CubeNoMipmaps
        );

        let owned_cube_size = match source_type {
            CubeSourceType::Equirectangular => EQUIRECT_CUBE_SIZE,
            CubeSourceType::CubeNoMipmaps => source_cube_size,
            CubeSourceType::CubeWithMipmaps => 0, // unused
        };

        let cube_texture = if needs_owned_cube {
            let mip_levels = (owned_cube_size as f32).log2().floor() as u32 + 1;
            Some(self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Env Cube (Owned)"),
                size: wgpu::Extent3d {
                    width: owned_cube_size,
                    height: owned_cube_size,
                    depth_or_array_layers: 6,
                },
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                mip_level_count: mip_levels,
                sample_count: 1,
                view_formats: &[],
            }))
        } else {
            None
        };

        // --- Register cube view ---
        let cube_view_id = if let Some(ref cube_tex) = cube_texture {
            let view = cube_tex.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(TextureViewDimension::Cube),
                ..Default::default()
            });
            let id = generate_gpu_resource_id();
            self.internal_resources.insert(id, view);
            id
        } else {
            // CubeWithMipmaps — resolve from the asset
            match &source {
                TextureSource::Asset(handle) => {
                    if let Some(binding) = self.texture_bindings.get(*handle) {
                        if let Some(img) = self.gpu_images.get(&binding.cpu_image_id) {
                            let view = img.texture.create_view(&wgpu::TextureViewDescriptor {
                                dimension: Some(TextureViewDimension::Cube),
                                ..Default::default()
                            });
                            let id = generate_gpu_resource_id();
                            self.internal_resources.insert(id, view);
                            id
                        } else {
                            self.dummy_env_image.id
                        }
                    } else {
                        self.dummy_env_image.id
                    }
                }
                TextureSource::Attachment(id, _) => *id,
            }
        };

        // --- Create PMREM texture ---
        let pmrem_mip_levels = (PMREM_SIZE as f32).log2().floor() as u32 + 1;
        let pmrem_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PMREM Cubemap"),
            size: wgpu::Extent3d {
                width: PMREM_SIZE,
                height: PMREM_SIZE,
                depth_or_array_layers: 6,
            },
            mip_level_count: pmrem_mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Register PMREM view
        let pmrem_view = pmrem_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("PMREM Cube View"),
            dimension: Some(TextureViewDimension::Cube),
            ..Default::default()
        });
        let pmrem_view_id = generate_gpu_resource_id();
        self.internal_resources.insert(pmrem_view_id, pmrem_view);

        let env_map_max_mip_level = (pmrem_mip_levels - 1) as f32;

        let gpu_env = GpuEnvironment {
            source_version: current_version,
            needs_compute: true,
            source_type,
            cube_texture,
            pmrem_texture,
            cube_view_id,
            pmrem_view_id,
            env_map_max_mip_level,
        };

        self.pending_ibl_source = Some(source);
        self.environment_map_cache.insert(source, gpu_env);

        env_map_max_mip_level
    }

    /// Ensure the global BRDF LUT texture exists.
    ///
    /// Creates the texture on first call and sets `needs_brdf_compute`.
    /// Returns the resource ID of the BRDF LUT view.
    pub fn ensure_brdf_lut(&mut self) -> u64 {
        if let Some(id) = self.brdf_lut_view_id {
            return id;
        }

        let size = 512u32;
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("BRDF LUT"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let id = self.register_internal_texture_by_name("BRDF_LUT", view);

        self.brdf_lut_texture = Some(texture);
        self.brdf_lut_view_id = Some(id);
        self.needs_brdf_compute = true;

        id
    }

    /// Get the `env_map_max_mip_level` for a given environment source.
    pub fn get_env_map_max_mip_level(
        &self,
        source: Option<TextureSource>,
    ) -> f32 {
        if let Some(src) = source {
            if let Some(gpu_env) = self.environment_map_cache.get(&src) {
                return gpu_env.env_map_max_mip_level;
            }
        }
        0.0
    }
}