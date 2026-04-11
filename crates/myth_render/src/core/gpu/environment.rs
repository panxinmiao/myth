use std::hash::{Hash, Hasher};

use glam::Vec3;
use wgpu::TextureViewDimension;

use myth_assets::AssetServer;
use myth_resources::texture::TextureSource;
use myth_scene::background::{BackgroundMode, ProceduralSkyParams};
use myth_scene::environment::Environment;

use crate::core::gpu::{ResourceState, Tracked};

use super::ResourceManager;

pub const BRDF_LUT_SIZE: u32 = 128;
const PROCEDURAL_IBL_SUN_UPDATE_THRESHOLD_DEGREES: f32 = 0.5;

/// How the scene environment source must be converted before PMREM filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeSourceType {
    /// Procedural atmosphere bake populates the base cube directly.
    Procedural,
    /// 2D equirectangular HDR panorama.
    Equirectangular,
    /// Cubemap source texture.
    Cubemap,
}

/// Persistent GPU environment resources owned by a scene.
///
/// The texture objects and tracked views stay stable across source changes so
/// downstream bind-group caches can keep hitting without rebuilds.
#[derive(Debug)]
pub struct GpuEnvironment {
    pub base_cube_texture: wgpu::Texture,
    pub pmrem_texture: wgpu::Texture,
    pub base_cube_view: Tracked<wgpu::TextureView>,
    pub base_cube_storage_view: Tracked<wgpu::TextureView>,
    pub pmrem_view: Tracked<wgpu::TextureView>,
    pub pmrem_storage_views: Vec<Tracked<wgpu::TextureView>>,
    pub source_version: u64,
    pub needs_compute: bool,
    pub source_type: CubeSourceType,
    pub source_key: Option<TextureSource>,
    pub source_ready: bool,
    pub last_baked_sun_direction: Option<Vec3>,
    pending_bake_sun_direction: Option<Vec3>,
    procedural_physics_hash: u64,
    procedural_bake_hash: u64,
    pub last_used_frame: u64,
}

impl GpuEnvironment {
    #[inline]
    #[must_use]
    pub fn env_map_max_mip_level(&self) -> f32 {
        (self.pmrem_texture.mip_level_count() - 1) as f32
    }

    #[inline]
    #[must_use]
    pub fn pmrem_mip_view(&self, mip: u32) -> &Tracked<wgpu::TextureView> {
        &self.pmrem_storage_views[mip as usize]
    }
}

struct ResolvedEnvironmentSource {
    source_type: CubeSourceType,
    source_key: Option<TextureSource>,
    source_version: u64,
    source_ready: bool,
}

fn hash_f32<H: Hasher>(hasher: &mut H, value: f32) {
    value.to_bits().hash(hasher);
}

fn hash_vec3<H: Hasher>(hasher: &mut H, value: Vec3) {
    hash_f32(hasher, value.x);
    hash_f32(hasher, value.y);
    hash_f32(hasher, value.z);
}

fn resolved_procedural_starbox_hash(
    resource_manager: &mut ResourceManager,
    assets: &AssetServer,
    params: &ProceduralSkyParams,
) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();

    let Some(source) = params.starbox_texture else {
        0u8.hash(&mut hasher);
        return hasher.finish();
    };

    source.hash(&mut hasher);

    match source {
        TextureSource::Asset(handle) => {
            let state = resource_manager.prepare_texture(assets, handle);
            match state {
                ResourceState::Ready => {
                    if let Some(texture) = assets.textures.get(handle) {
                        u64::from(assets.images.get_version(texture.image).unwrap_or(0))
                            .hash(&mut hasher);
                    }
                    if let Some(binding) = resource_manager.texture_bindings.get(handle) {
                        binding.view_id.hash(&mut hasher);
                    }
                }
                ResourceState::Pending | ResourceState::Unknown => {
                    0u8.hash(&mut hasher);
                }
            }
        }
        TextureSource::Attachment(id, dim) => {
            id.hash(&mut hasher);
            dim.hash(&mut hasher);
        }
    }

    hasher.finish()
}

fn procedural_physics_hash(params: &ProceduralSkyParams) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    hash_vec3(&mut hasher, params.rayleigh_scattering);
    hash_f32(&mut hasher, params.rayleigh_scale_height);
    hash_f32(&mut hasher, params.mie_scattering);
    hash_f32(&mut hasher, params.mie_absorption);
    hash_f32(&mut hasher, params.mie_scale_height);
    hash_f32(&mut hasher, params.mie_anisotropy);
    hash_vec3(&mut hasher, params.ozone_absorption);
    hash_f32(&mut hasher, params.planet_radius);
    hash_f32(&mut hasher, params.atmosphere_radius);
    hasher.finish()
}

fn procedural_bake_hash(params: &ProceduralSkyParams, starbox_hash: u64) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    hash_f32(&mut hasher, params.sun_intensity);
    hash_f32(&mut hasher, params.sun_disk_size);
    hash_f32(&mut hasher, params.moon_intensity);
    hash_f32(&mut hasher, params.moon_disk_size);
    hash_f32(&mut hasher, params.star_intensity);
    hash_vec3(&mut hasher, params.star_axis);
    starbox_hash.hash(&mut hasher);
    hasher.finish()
}

fn procedural_source_version(physics_hash: u64, bake_hash: u64, sun_direction: Vec3) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    physics_hash.hash(&mut hasher);
    bake_hash.hash(&mut hasher);
    hash_vec3(&mut hasher, sun_direction);
    hasher.finish()
}

fn sun_direction_requires_rebake(last_baked: Option<Vec3>, current: Vec3) -> bool {
    let Some(last_baked) = last_baked else {
        return true;
    };

    let threshold_cos = PROCEDURAL_IBL_SUN_UPDATE_THRESHOLD_DEGREES
        .to_radians()
        .cos();
    last_baked.normalize().dot(current.normalize()) <= threshold_cos
}

impl ResourceManager {
    /// Resolve or create the persistent GPU environment state for a scene.
    ///
    /// This stage only updates CPU-side dirty flags and guarantees stable
    /// texture ownership. Actual cube conversion / atmosphere bake / PMREM
    /// generation is deferred to RenderGraph pass nodes.
    pub fn resolve_gpu_environment(
        &mut self,
        scene_id: u32,
        assets: &AssetServer,
        environment: &Environment,
        background: &BackgroundMode,
    ) -> f32 {
        let config = environment.map_config();

        let needs_recreate = self
            .scene_gpu_environments
            .get(&scene_id)
            .is_none_or(|gpu_env| {
                gpu_env.base_cube_texture.width() != config.base_cube_size
                    || gpu_env.pmrem_texture.width() != config.pmrem_size
            });

        if needs_recreate {
            self.recreate_scene_gpu_environment(scene_id, config.base_cube_size, config.pmrem_size);
        }

        if let BackgroundMode::Procedural(params) = background {
            // Intentionally exclude `star_rotation` from PMREM rebake hashing:
            // the day/night system updates it every frame, and rebaking IBL at
            // that cadence would be too expensive. Static night-sky changes and
            // starbox asset changes still trigger rebakes through `bake_hash`.
            let starbox_hash = resolved_procedural_starbox_hash(self, assets, params);
            let gpu_env = self
                .scene_gpu_environments
                .get_mut(&scene_id)
                .expect("scene gpu environment must exist after recreation");

            gpu_env.last_used_frame = self.frame_index;

            let physics_hash = procedural_physics_hash(params);
            let bake_hash = procedural_bake_hash(params, starbox_hash);
            let source_kind_changed = gpu_env.source_type != CubeSourceType::Procedural
                || gpu_env.source_key.is_some()
                || !gpu_env.source_ready;
            let physics_changed = gpu_env.procedural_physics_hash != physics_hash;
            let bake_changed = gpu_env.procedural_bake_hash != bake_hash;
            let sun_changed = sun_direction_requires_rebake(
                gpu_env.last_baked_sun_direction,
                params.sun_direction,
            );

            let needs_rebake = gpu_env.needs_compute
                || source_kind_changed
                || physics_changed
                || bake_changed
                || sun_changed;

            gpu_env.source_type = CubeSourceType::Procedural;
            gpu_env.source_key = None;
            gpu_env.source_ready = true;
            gpu_env.procedural_physics_hash = physics_hash;
            gpu_env.procedural_bake_hash = bake_hash;

            if needs_rebake {
                gpu_env.source_version =
                    procedural_source_version(physics_hash, bake_hash, params.sun_direction);
                gpu_env.pending_bake_sun_direction = Some(params.sun_direction);
                gpu_env.needs_compute = true;
            } else {
                gpu_env.pending_bake_sun_direction = None;
            }

            return gpu_env.env_map_max_mip_level();
        }

        let resolved = match background {
            BackgroundMode::Procedural(_) => {
                unreachable!("procedural backgrounds are handled above")
            }
            _ => {
                let Some(source) = environment.source_env_map().cloned() else {
                    if let Some(gpu_env) = self.scene_gpu_environments.get_mut(&scene_id) {
                        gpu_env.needs_compute = false;
                        gpu_env.source_key = None;
                        gpu_env.source_ready = false;
                        gpu_env.pending_bake_sun_direction = None;
                        gpu_env.last_used_frame = self.frame_index;
                    }
                    return 0.0;
                };

                let mut source_version = environment.source_version();
                let mut source_ready = true;

                if let TextureSource::Asset(handle) = &source {
                    let state = self.prepare_texture(assets, *handle);
                    source_ready =
                        !matches!(state, ResourceState::Pending | ResourceState::Unknown);

                    if source_ready && let Some(tex) = assets.textures.get(*handle) {
                        source_version = source_version.wrapping_shl(32)
                            ^ u64::from(assets.images.get_version(tex.image).unwrap_or(0));
                    }
                }

                let source_type = match &source {
                    TextureSource::Asset(handle) => self
                        .texture_bindings
                        .get(*handle)
                        .and_then(|binding| self.gpu_images.get(binding.image_handle))
                        .map_or(CubeSourceType::Equirectangular, |img| {
                            if img.default_view_dimension == TextureViewDimension::D2 {
                                CubeSourceType::Equirectangular
                            } else {
                                CubeSourceType::Cubemap
                            }
                        }),
                    TextureSource::Attachment(_, dim) => {
                        if *dim == TextureViewDimension::D2 {
                            CubeSourceType::Equirectangular
                        } else {
                            CubeSourceType::Cubemap
                        }
                    }
                };

                ResolvedEnvironmentSource {
                    source_type,
                    source_key: Some(source),
                    source_version,
                    source_ready,
                }
            }
        };

        let gpu_env = self
            .scene_gpu_environments
            .get_mut(&scene_id)
            .expect("scene gpu environment must exist after recreation");

        gpu_env.last_used_frame = self.frame_index;

        let source_changed = gpu_env.source_version != resolved.source_version
            || gpu_env.source_type != resolved.source_type
            || gpu_env.source_key != resolved.source_key
            || gpu_env.source_ready != resolved.source_ready;

        if source_changed {
            gpu_env.source_version = resolved.source_version;
            gpu_env.source_type = resolved.source_type;
            gpu_env.source_key = resolved.source_key;
            gpu_env.source_ready = resolved.source_ready;
            gpu_env.pending_bake_sun_direction = None;
            gpu_env.needs_compute = resolved.source_ready;
        } else if !resolved.source_ready {
            gpu_env.needs_compute = false;
            gpu_env.pending_bake_sun_direction = None;
        }

        gpu_env.env_map_max_mip_level()
    }

    fn recreate_scene_gpu_environment(
        &mut self,
        scene_id: u32,
        base_cube_size: u32,
        pmrem_size: u32,
    ) {
        if let Some(old) = self.scene_gpu_environments.remove(&scene_id) {
            self.internal_resources.remove(&old.base_cube_view.id());
            self.internal_resources.remove(&old.pmrem_view.id());
        }

        let base_cube_mips = (base_cube_size as f32).log2().floor() as u32 + 1;
        let base_cube_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Scene Environment Base Cube"),
            size: wgpu::Extent3d {
                width: base_cube_size,
                height: base_cube_size,
                depth_or_array_layers: 6,
            },
            mip_level_count: base_cube_mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let base_cube_view =
            Tracked::new(base_cube_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Scene Environment Base Cube View"),
                dimension: Some(TextureViewDimension::Cube),
                ..Default::default()
            }));
        let base_cube_storage_view =
            Tracked::new(base_cube_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Scene Environment Base Cube Mip0"),
                dimension: Some(TextureViewDimension::D2Array),
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(6),
                usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                ..Default::default()
            }));
        self.internal_resources
            .insert(base_cube_view.id(), (*base_cube_view).clone());

        let pmrem_mips = (pmrem_size as f32).log2().floor() as u32 + 1;
        let pmrem_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Scene Environment PMREM"),
            size: wgpu::Extent3d {
                width: pmrem_size,
                height: pmrem_size,
                depth_or_array_layers: 6,
            },
            mip_level_count: pmrem_mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let pmrem_view = Tracked::new(pmrem_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Scene Environment PMREM View"),
            dimension: Some(TextureViewDimension::Cube),
            ..Default::default()
        }));
        let pmrem_storage_views = (0..pmrem_mips)
            .map(|mip| {
                Tracked::new(pmrem_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("Scene Environment PMREM Mip"),
                    format: Some(wgpu::TextureFormat::Rgba16Float),
                    dimension: Some(TextureViewDimension::D2Array),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(6),
                    usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                }))
            })
            .collect();
        self.internal_resources
            .insert(pmrem_view.id(), (*pmrem_view).clone());

        self.scene_gpu_environments.insert(
            scene_id,
            GpuEnvironment {
                base_cube_texture,
                pmrem_texture,
                base_cube_view,
                base_cube_storage_view,
                pmrem_view,
                pmrem_storage_views,
                source_version: u64::MAX,
                needs_compute: true,
                source_type: CubeSourceType::Equirectangular,
                source_key: None,
                source_ready: false,
                last_baked_sun_direction: None,
                pending_bake_sun_direction: None,
                procedural_physics_hash: u64::MAX,
                procedural_bake_hash: u64::MAX,
                last_used_frame: self.frame_index,
            },
        );
    }

    #[inline]
    #[must_use]
    pub fn gpu_environment(&self, scene_id: u32) -> Option<&GpuEnvironment> {
        self.scene_gpu_environments.get(&scene_id)
    }

    #[inline]
    #[must_use]
    pub fn gpu_environment_mut(&mut self, scene_id: u32) -> Option<&mut GpuEnvironment> {
        self.scene_gpu_environments.get_mut(&scene_id)
    }

    #[inline]
    #[must_use]
    pub fn scene_environment_ready(&self, scene_id: u32) -> bool {
        self.scene_gpu_environments
            .get(&scene_id)
            .is_some_and(|gpu_env| !gpu_env.needs_compute)
    }

    #[inline]
    #[must_use]
    pub fn scene_environment_source_ready(&self, scene_id: u32) -> bool {
        self.scene_gpu_environments
            .get(&scene_id)
            .is_some_and(|gpu_env| match gpu_env.source_type {
                CubeSourceType::Procedural => true,
                _ => gpu_env.source_ready,
            })
    }

    #[inline]
    pub fn mark_scene_environment_clean(&mut self, scene_id: u32) {
        if let Some(gpu_env) = self.scene_gpu_environments.get_mut(&scene_id) {
            if gpu_env.source_type == CubeSourceType::Procedural {
                gpu_env.last_baked_sun_direction = gpu_env.pending_bake_sun_direction.take();
            }
            gpu_env.needs_compute = false;
        }
    }

    #[inline]
    #[must_use]
    pub fn get_env_map_max_mip_level(&self, scene_id: u32) -> f32 {
        self.scene_gpu_environments
            .get(&scene_id)
            .map_or(0.0, GpuEnvironment::env_map_max_mip_level)
    }

    /// Ensure the global BRDF LUT texture exists.
    ///
    /// Creates the texture on first call and sets `needs_brdf_compute`.
    /// Returns the resource ID of the BRDF LUT view.
    pub fn ensure_brdf_lut(&mut self) -> u64 {
        if let Some(id) = self.brdf_lut_view_id {
            return id;
        }

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("BRDF LUT"),
            size: wgpu::Extent3d {
                width: BRDF_LUT_SIZE,
                height: BRDF_LUT_SIZE,
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
}
