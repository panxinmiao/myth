//! Procedural sky atmosphere bake passes.
//!
//! The atmosphere pipeline is decomposed into four atomic RenderGraph nodes:
//!
//! - Transmittance LUT
//! - Multi-scatter LUT
//! - Sky-view LUT
//! - Sky-view -> persistent scene base cubemap bake
//!
//! The LUTs are transient RDG resources while all persistent buffers,
//! layouts, pipelines, and samplers live on the long-lived feature.

use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

use rustc_hash::FxHashMap;

use crate::core::binding::BindGroupKey;
use crate::core::gpu::{ResourceState, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    ExecuteContext, ExtractContext, PassNode, PrepareContext, TextureDesc, TextureNodeId,
};
use crate::pipeline::{
    ComputePipelineId, ComputePipelineKey, ShaderCompilationOptions, ShaderSource,
};
use myth_resources::shader_defines::ShaderDefines;
use myth_resources::texture::TextureSource;
use myth_scene::background::ProceduralSkyParams;

const TRANSMITTANCE_WIDTH: u32 = 256;
const TRANSMITTANCE_HEIGHT: u32 = 64;
const MULTI_SCATTER_SIZE: u32 = 32;
const SKY_VIEW_WIDTH: u32 = 192;
const SKY_VIEW_HEIGHT: u32 = 108;
const SCENE_CACHE_TTL: u64 = 120;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum AtmosphereStarboxKind {
    None,
    Equirectangular,
    Cube,
}

impl AtmosphereStarboxKind {
    fn from_view_dimension(view_dimension: wgpu::TextureViewDimension) -> Option<Self> {
        match view_dimension {
            wgpu::TextureViewDimension::D2 => Some(Self::Equirectangular),
            wgpu::TextureViewDimension::Cube => Some(Self::Cube),
            _ => None,
        }
    }

    fn apply_shader_defines(self, defines: &mut ShaderDefines) {
        match self {
            Self::None => {}
            Self::Equirectangular => {
                defines.set("CELESTIAL_STARBOX_EQUIRECT", "1");
            }
            Self::Cube => {
                defines.set("CELESTIAL_STARBOX_CUBE", "1");
            }
        }
    }
}

struct ResolvedAtmosphereStarbox {
    view: wgpu::TextureView,
    resource_id: u64,
    kind: AtmosphereStarboxKind,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuAtmosphereParams {
    rayleigh_scattering: [f32; 3],
    rayleigh_scale_height: f32,
    mie_scattering: f32,
    mie_absorption: f32,
    mie_scale_height: f32,
    mie_anisotropy: f32,
    ozone_absorption: [f32; 3],
    _pad0: f32,
    planet_radius: f32,
    atmosphere_radius: f32,
    sun_intensity: f32,
    sun_cos_angle: f32,
    sun_direction: [f32; 3],
    _pad1: f32,
}

impl GpuAtmosphereParams {
    fn from_scene(params: &ProceduralSkyParams) -> Self {
        Self {
            rayleigh_scattering: params.rayleigh_scattering.into(),
            rayleigh_scale_height: params.rayleigh_scale_height,
            mie_scattering: params.mie_scattering,
            mie_absorption: params.mie_absorption,
            mie_scale_height: params.mie_scale_height,
            mie_anisotropy: params.mie_anisotropy,
            ozone_absorption: params.ozone_absorption.into(),
            _pad0: 0.0,
            planet_radius: params.planet_radius,
            atmosphere_radius: params.atmosphere_radius,
            sun_intensity: params.sun_intensity,
            sun_cos_angle: params.sun_direction.y,
            sun_direction: params.sun_direction.into(),
            _pad1: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuBakeParams {
    sun_direction: [f32; 3],
    sun_intensity: f32,
    moon_direction: [f32; 3],
    moon_intensity: f32,
    star_axis: [f32; 3],
    sun_disk_size: f32,
    moon_disk_size: f32,
    planet_radius: f32,
    atmosphere_radius: f32,
    star_intensity: f32,
    star_rotation: f32,
    _pad3: [f32; 3],
}

impl GpuBakeParams {
    fn from_scene(params: &ProceduralSkyParams) -> Self {
        Self {
            sun_direction: params.sun_direction.into(),
            sun_intensity: params.sun_intensity,
            moon_direction: params.moon_direction.into(),
            moon_intensity: params.moon_intensity,
            star_axis: params.star_axis.into(),
            sun_disk_size: params.sun_disk_size,
            moon_disk_size: params.moon_disk_size,
            planet_radius: params.planet_radius,
            atmosphere_radius: params.atmosphere_radius,
            star_intensity: params.star_intensity,
            star_rotation: params.star_rotation,
            _pad3: [0.0; 3],
        }
    }
}

struct AtmosphereSceneState {
    atmosphere_params_buffer: Tracked<wgpu::Buffer>,
    bake_params_buffer: Tracked<wgpu::Buffer>,
    _transmittance_texture: wgpu::Texture,
    transmittance_view: Tracked<wgpu::TextureView>,
    _multi_scatter_texture: wgpu::Texture,
    multi_scatter_view: Tracked<wgpu::TextureView>,
    _sky_view_texture: wgpu::Texture,
    sky_view_view: Tracked<wgpu::TextureView>,
    physics_hash: u64,
    physics_dirty: bool,
    uploaded_version: AtomicU64,
    last_used_frame: u64,
    starbox: Option<ResolvedAtmosphereStarbox>,
}

#[derive(Clone, Copy)]
pub(crate) struct ProceduralSkyboxResources<'a> {
    pub sky_view_view: &'a Tracked<wgpu::TextureView>,
    pub transmittance_view: &'a Tracked<wgpu::TextureView>,
    pub bake_params_buffer: &'a Tracked<wgpu::Buffer>,
}

pub(crate) struct AtmosphereGraphOutput {
    pub sky_view: TextureNodeId,
    pub transmittance: TextureNodeId,
}

impl AtmosphereGraphOutput {
    #[must_use]
    pub fn skybox_dependencies(&self) -> [Option<TextureNodeId>; 2] {
        [Some(self.sky_view), Some(self.transmittance)]
    }
}

fn hash_f32<H: Hasher>(hasher: &mut H, value: f32) {
    value.to_bits().hash(hasher);
}

fn hash_vec3<H: Hasher>(hasher: &mut H, value: glam::Vec3) {
    hash_f32(hasher, value.x);
    hash_f32(hasher, value.y);
    hash_f32(hasher, value.z);
}

fn physics_hash(params: &ProceduralSkyParams) -> u64 {
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

impl AtmosphereSceneState {
    fn update_if_needed(
        &self,
        queue: &wgpu::Queue,
        version: u64,
        atmosphere_params: &GpuAtmosphereParams,
        bake_params: &GpuBakeParams,
    ) {
        if self.uploaded_version.load(Ordering::Relaxed) == version {
            return;
        }

        queue.write_buffer(
            &self.atmosphere_params_buffer,
            0,
            bytemuck::bytes_of(atmosphere_params),
        );
        queue.write_buffer(&self.bake_params_buffer, 0, bytemuck::bytes_of(bake_params));
        self.uploaded_version.store(version, Ordering::Relaxed);
    }
}

pub struct AtmosphereFeature {
    transmittance_pipeline: Option<ComputePipelineId>,
    multi_scatter_pipeline: Option<ComputePipelineId>,
    sky_view_pipeline: Option<ComputePipelineId>,
    sky_to_cube_pipelines: FxHashMap<AtmosphereStarboxKind, ComputePipelineId>,
    transmittance_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    multi_scatter_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    sky_view_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    sky_to_cube_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    sky_to_cube_eq_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    sky_to_cube_cube_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    sampler: Option<wgpu::Sampler>,
    scene_states: FxHashMap<u32, AtmosphereSceneState>,
}

impl Default for AtmosphereFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl AtmosphereFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            transmittance_pipeline: None,
            multi_scatter_pipeline: None,
            sky_view_pipeline: None,
            sky_to_cube_pipelines: FxHashMap::default(),
            transmittance_layout: None,
            multi_scatter_layout: None,
            sky_view_layout: None,
            sky_to_cube_layout: None,
            sky_to_cube_eq_layout: None,
            sky_to_cube_cube_layout: None,
            sampler: None,
            scene_states: FxHashMap::default(),
        }
    }

    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        scene_id: u32,
        params: &ProceduralSkyParams,
    ) {
        self.ensure_layouts(ctx.device);
        self.ensure_sampler(ctx.device);
        self.ensure_pipelines(ctx);
        self.ensure_scene_state(ctx, scene_id, params);
        self.update_starbox_state(ctx, scene_id, params);
        self.prune_scene_states(ctx.resource_manager.frame_index());
    }

    pub(crate) fn procedural_skybox_resources(
        &self,
        scene_id: u32,
    ) -> Option<ProceduralSkyboxResources<'_>> {
        self.scene_states
            .get(&scene_id)
            .map(|state| ProceduralSkyboxResources {
                sky_view_view: &state.sky_view_view,
                transmittance_view: &state.transmittance_view,
                bake_params_buffer: &state.bake_params_buffer,
            })
    }

    pub(crate) fn add_to_graph<'a>(
        &'a mut self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        scene_id: u32,
        params: &ProceduralSkyParams,
        base_cube: TextureNodeId,
        base_cube_storage_view: &'a Tracked<wgpu::TextureView>,
        bake_environment: bool,
    ) -> AtmosphereGraphOutput {
        let rebuild_physics = self
            .scene_states
            .get(&scene_id)
            .expect("scene atmosphere state must be prepared before graph build")
            .physics_dirty;

        if rebuild_physics {
            self.scene_states
                .get_mut(&scene_id)
                .expect("scene atmosphere state must exist")
                .physics_dirty = false;
        }

        let state = self
            .scene_states
            .get(&scene_id)
            .expect("scene atmosphere state must be prepared before graph build");

        let transmittance_pipeline = self
            .transmittance_pipeline
            .map(|id| ctx.pipeline_cache.get_compute_pipeline(id))
            .expect("atmosphere transmittance pipeline must exist");
        let multi_scatter_pipeline = self
            .multi_scatter_pipeline
            .map(|id| ctx.pipeline_cache.get_compute_pipeline(id))
            .expect("atmosphere multi scatter pipeline must exist");
        let sky_view_pipeline = self
            .sky_view_pipeline
            .map(|id| ctx.pipeline_cache.get_compute_pipeline(id))
            .expect("atmosphere sky view pipeline must exist");
        let sky_to_cube_kind = state
            .starbox
            .as_ref()
            .map_or(AtmosphereStarboxKind::None, |starbox| starbox.kind);
        let sky_to_cube_pipeline = self
            .sky_to_cube_pipelines
            .get(&sky_to_cube_kind)
            .copied()
            .map(|id| ctx.pipeline_cache.get_compute_pipeline(id))
            .expect("atmosphere sky-to-cube pipeline must exist");

        let gpu_params = GpuAtmosphereParams::from_scene(params);
        let bake_params = GpuBakeParams::from_scene(params);
        let params_version = params.version();
        let transmittance_layout = self
            .transmittance_layout
            .as_ref()
            .expect("atmosphere transmittance layout must exist");
        let multi_scatter_layout = self
            .multi_scatter_layout
            .as_ref()
            .expect("atmosphere multi-scatter layout must exist");
        let sky_view_layout = self
            .sky_view_layout
            .as_ref()
            .expect("atmosphere sky-view layout must exist");
        let sky_to_cube_layout = self.sky_to_cube_layout_for_kind(sky_to_cube_kind);
        let sampler = self
            .sampler
            .as_ref()
            .expect("atmosphere sampler must exist");

        let transmittance_desc = TextureDesc::new_2d(
            TRANSMITTANCE_WIDTH,
            TRANSMITTANCE_HEIGHT,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let multi_scatter_desc = TextureDesc::new_2d(
            MULTI_SCATTER_SIZE,
            MULTI_SCATTER_SIZE,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let sky_view_desc = TextureDesc::new_2d(
            SKY_VIEW_WIDTH,
            SKY_VIEW_HEIGHT,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        ctx.with_group("Atmosphere_System", |ctx| {
            let transmittance = if rebuild_physics {
                ctx.graph.add_pass("Atmosphere_Transmittance", |builder| {
                    let output = builder.write_external_texture(
                        "Atmosphere_Transmittance",
                        transmittance_desc,
                        &state.transmittance_view,
                    );
                    let node = AtmosphereTransmittanceNode {
                        output_tex: output,
                        params_version,
                        gpu_params,
                        bake_params,
                        state,
                        pipeline: transmittance_pipeline,
                        layout: transmittance_layout,
                        bind_group: None,
                    };
                    (node, output)
                })
            } else {
                ctx.graph.import_external_resource(
                    "Atmosphere_Transmittance",
                    transmittance_desc,
                    &state.transmittance_view,
                )
            };

            let multi_scatter = if rebuild_physics {
                ctx.graph.add_pass("Atmosphere_MultiScatter", |builder| {
                    builder.read_texture(transmittance);
                    let output = builder.write_external_texture(
                        "Atmosphere_MultiScatter",
                        multi_scatter_desc,
                        &state.multi_scatter_view,
                    );
                    let node = AtmosphereMultiScatterNode {
                        transmittance_tex: transmittance,
                        output_tex: output,
                        params_version,
                        gpu_params,
                        bake_params,
                        state,
                        pipeline: multi_scatter_pipeline,
                        layout: multi_scatter_layout,
                        sampler,
                        bind_group: None,
                    };
                    (node, output)
                })
            } else {
                ctx.graph.import_external_resource(
                    "Atmosphere_MultiScatter",
                    multi_scatter_desc,
                    &state.multi_scatter_view,
                )
            };

            let sky_view = ctx.graph.add_pass("Atmosphere_SkyView", |builder| {
                builder.read_texture(transmittance);
                builder.read_texture(multi_scatter);
                let output = builder.write_external_texture(
                    "Atmosphere_SkyView",
                    sky_view_desc,
                    &state.sky_view_view,
                );
                let node = AtmosphereSkyViewNode {
                    transmittance_tex: transmittance,
                    multi_scatter_tex: multi_scatter,
                    output_tex: output,
                    params_version,
                    gpu_params,
                    bake_params,
                    state,
                    pipeline: sky_view_pipeline,
                    layout: sky_view_layout,
                    sampler,
                    bind_group: None,
                };
                (node, output)
            });

            if bake_environment {
                ctx.graph.add_pass("Atmosphere_SkyToCube", |builder| {
                    builder.read_texture(transmittance);
                    builder.read_texture(sky_view);
                    builder.write_texture(base_cube);
                    let node = AtmosphereSkyToCubeNode {
                        base_cube,
                        transmittance_tex: transmittance,
                        sky_view_tex: sky_view,
                        params_version,
                        gpu_params,
                        bake_params,
                        state,
                        pipeline: sky_to_cube_pipeline,
                        layout: sky_to_cube_layout,
                        sampler,
                        base_cube_storage_view,
                        starbox: state.starbox.as_ref(),
                        bind_group: None,
                    };
                    (node, ())
                });
            }

            AtmosphereGraphOutput {
                sky_view,
                transmittance,
            }
        })
    }

    fn ensure_scene_state(
        &mut self,
        ctx: &mut ExtractContext,
        scene_id: u32,
        params: &ProceduralSkyParams,
    ) {
        let frame_index = ctx.resource_manager.frame_index();
        let state = self.scene_states.entry(scene_id).or_insert_with(|| {
            let atmosphere_params_buffer =
                Tracked::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Atmosphere Params"),
                    size: std::mem::size_of::<GpuAtmosphereParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            let bake_params_buffer =
                Tracked::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Atmosphere Bake Params"),
                    size: std::mem::size_of::<GpuBakeParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));

            let transmittance_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Atmosphere Transmittance LUT"),
                size: wgpu::Extent3d {
                    width: TRANSMITTANCE_WIDTH,
                    height: TRANSMITTANCE_HEIGHT,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let transmittance_view = Tracked::new(
                transmittance_texture.create_view(&wgpu::TextureViewDescriptor::default()),
            );

            let multi_scatter_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Atmosphere Multi-Scatter LUT"),
                size: wgpu::Extent3d {
                    width: MULTI_SCATTER_SIZE,
                    height: MULTI_SCATTER_SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let multi_scatter_view = Tracked::new(
                multi_scatter_texture.create_view(&wgpu::TextureViewDescriptor::default()),
            );

            let sky_view_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Atmosphere Sky-View LUT"),
                size: wgpu::Extent3d {
                    width: SKY_VIEW_WIDTH,
                    height: SKY_VIEW_HEIGHT,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let sky_view_view =
                Tracked::new(sky_view_texture.create_view(&wgpu::TextureViewDescriptor::default()));

            AtmosphereSceneState {
                atmosphere_params_buffer,
                bake_params_buffer,
                _transmittance_texture: transmittance_texture,
                transmittance_view,
                _multi_scatter_texture: multi_scatter_texture,
                multi_scatter_view,
                _sky_view_texture: sky_view_texture,
                sky_view_view,
                physics_hash: u64::MAX,
                physics_dirty: true,
                uploaded_version: AtomicU64::new(u64::MAX),
                last_used_frame: frame_index,
                starbox: None,
            }
        });

        state.last_used_frame = frame_index;

        let next_physics_hash = physics_hash(params);
        if state.physics_hash != next_physics_hash {
            state.physics_hash = next_physics_hash;
            state.physics_dirty = true;
        }
    }

    fn prune_scene_states(&mut self, current_frame: u64) {
        self.scene_states.retain(|_, state| {
            current_frame.saturating_sub(state.last_used_frame) <= SCENE_CACHE_TTL
        });
    }

    fn update_starbox_state(
        &mut self,
        ctx: &mut ExtractContext,
        scene_id: u32,
        params: &ProceduralSkyParams,
    ) {
        let starbox = params
            .starbox_texture
            .and_then(|source| Self::resolve_starbox(ctx, &source));

        if let Some(state) = self.scene_states.get_mut(&scene_id) {
            state.starbox = starbox;
        }
    }

    fn resolve_starbox(
        ctx: &mut ExtractContext,
        source: &TextureSource,
    ) -> Option<ResolvedAtmosphereStarbox> {
        match source {
            TextureSource::Asset(handle) => {
                let state = ctx.resource_manager.prepare_texture(ctx.assets, *handle);
                if !matches!(state, ResourceState::Ready) {
                    return None;
                }

                let binding = ctx.resource_manager.texture_bindings.get(*handle)?;
                let image = ctx.resource_manager.gpu_images.get(binding.image_handle)?;
                let kind =
                    AtmosphereStarboxKind::from_view_dimension(image.default_view_dimension)?;
                Some(ResolvedAtmosphereStarbox {
                    view: image.default_view.clone(),
                    resource_id: binding.view_id,
                    kind,
                })
            }
            TextureSource::Attachment(id, dimension) => {
                let kind = AtmosphereStarboxKind::from_view_dimension(*dimension)?;
                let view = ctx.resource_manager.internal_resources.get(id)?.clone();
                Some(ResolvedAtmosphereStarbox {
                    view,
                    resource_id: *id,
                    kind,
                })
            }
        }
    }

    fn ensure_sampler(&mut self, device: &wgpu::Device) {
        if self.sampler.is_some() {
            return;
        }

        self.sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Atmosphere Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        }));
    }

    fn ensure_layouts(&mut self, device: &wgpu::Device) {
        if self.transmittance_layout.is_some() {
            return;
        }

        self.transmittance_layout = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Atmo Transmittance BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            },
        )));

        self.multi_scatter_layout = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Atmo Multi-Scatter BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            },
        )));

        self.sky_view_layout = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Atmo Sky-View BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            },
        )));

        self.sky_to_cube_layout = Some(Tracked::new(create_sky_to_cube_layout(
            device,
            None,
            "Atmo Sky-to-Cube BGL",
        )));
        self.sky_to_cube_eq_layout = Some(Tracked::new(create_sky_to_cube_layout(
            device,
            Some(wgpu::TextureViewDimension::D2),
            "Atmo Sky-to-Cube Eq BGL",
        )));
        self.sky_to_cube_cube_layout = Some(Tracked::new(create_sky_to_cube_layout(
            device,
            Some(wgpu::TextureViewDimension::Cube),
            "Atmo Sky-to-Cube Cube BGL",
        )));
    }

    fn sky_to_cube_layout_for_kind(
        &self,
        kind: AtmosphereStarboxKind,
    ) -> &Tracked<wgpu::BindGroupLayout> {
        match kind {
            AtmosphereStarboxKind::None => self.sky_to_cube_layout.as_ref().unwrap(),
            AtmosphereStarboxKind::Equirectangular => self.sky_to_cube_eq_layout.as_ref().unwrap(),
            AtmosphereStarboxKind::Cube => self.sky_to_cube_cube_layout.as_ref().unwrap(),
        }
    }

    fn ensure_pipelines(&mut self, ctx: &mut ExtractContext) {
        if self.transmittance_pipeline.is_some() {
            return;
        }

        let opts = ShaderCompilationOptions::default();
        let compilation_options = wgpu::PipelineCompilationOptions::default();

        let (trans_module, trans_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/utility/atmosphere/transmittance_lut"),
            &opts,
        );
        let trans_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Atmo Transmittance PL"),
                bind_group_layouts: &[Some(self.transmittance_layout.as_deref().unwrap())],
                immediate_size: 0,
            });
        self.transmittance_pipeline = Some(ctx.pipeline_cache.get_or_create_compute(
            ctx.device,
            trans_module,
            &trans_layout,
            &ComputePipelineKey::new(trans_hash).with_compilation_options(&compilation_options),
            &compilation_options,
            "Atmo Transmittance Pipeline",
        ));

        let (multi_module, multi_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/utility/atmosphere/multi_scatter_lut"),
            &opts,
        );
        let multi_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Atmo Multi-Scatter PL"),
                bind_group_layouts: &[Some(self.multi_scatter_layout.as_deref().unwrap())],
                immediate_size: 0,
            });
        self.multi_scatter_pipeline = Some(ctx.pipeline_cache.get_or_create_compute(
            ctx.device,
            multi_module,
            &multi_layout,
            &ComputePipelineKey::new(multi_hash).with_compilation_options(&compilation_options),
            &compilation_options,
            "Atmo Multi-Scatter Pipeline",
        ));

        let (sky_view_module, sky_view_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/utility/atmosphere/sky_view_lut"),
            &opts,
        );
        let sky_view_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Atmo Sky-View PL"),
                bind_group_layouts: &[Some(self.sky_view_layout.as_deref().unwrap())],
                immediate_size: 0,
            });
        self.sky_view_pipeline = Some(ctx.pipeline_cache.get_or_create_compute(
            ctx.device,
            sky_view_module,
            &sky_view_layout,
            &ComputePipelineKey::new(sky_view_hash).with_compilation_options(&compilation_options),
            &compilation_options,
            "Atmo Sky-View Pipeline",
        ));

        for kind in [
            AtmosphereStarboxKind::None,
            AtmosphereStarboxKind::Equirectangular,
            AtmosphereStarboxKind::Cube,
        ] {
            let mut sky_to_cube_opts = ShaderCompilationOptions::default();
            kind.apply_shader_defines(&mut sky_to_cube_opts.defines);
            let (sky_to_cube_module, sky_to_cube_hash) = ctx.shader_manager.get_or_compile(
                ctx.device,
                ShaderSource::File("entry/utility/atmosphere/sky_to_cube"),
                &sky_to_cube_opts,
            );
            let sky_to_cube_bind_group_layout: &wgpu::BindGroupLayout =
                self.sky_to_cube_layout_for_kind(kind);
            let sky_to_cube_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Atmo Sky-to-Cube PL"),
                        bind_group_layouts: &[Some(sky_to_cube_bind_group_layout)],
                        immediate_size: 0,
                    });
            let pipeline = ctx.pipeline_cache.get_or_create_compute(
                ctx.device,
                sky_to_cube_module,
                &sky_to_cube_layout,
                &ComputePipelineKey::new(sky_to_cube_hash)
                    .with_compilation_options(&compilation_options),
                &compilation_options,
                "Atmo Sky-to-Cube Pipeline",
            );
            self.sky_to_cube_pipelines.insert(kind, pipeline);
        }
    }
}

fn create_sky_to_cube_layout(
    device: &wgpu::Device,
    starbox_view_dimension: Option<wgpu::TextureViewDimension>,
    label: &str,
) -> wgpu::BindGroupLayout {
    let mut entries = vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: wgpu::TextureViewDimension::D2Array,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
    ];

    if let Some(view_dimension) = starbox_view_dimension {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 5,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension,
                multisampled: false,
            },
            count: None,
        });
    }

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

trait AtmosphereNodeState {
    fn update_params(&self, ctx: &PrepareContext<'_>);
}

struct AtmosphereTransmittanceNode<'a> {
    output_tex: TextureNodeId,
    params_version: u64,
    gpu_params: GpuAtmosphereParams,
    bake_params: GpuBakeParams,
    state: &'a AtmosphereSceneState,
    pipeline: &'a wgpu::ComputePipeline,
    layout: &'a Tracked<wgpu::BindGroupLayout>,
    bind_group: Option<&'a wgpu::BindGroup>,
}

impl AtmosphereNodeState for AtmosphereTransmittanceNode<'_> {
    fn update_params(&self, ctx: &PrepareContext<'_>) {
        self.state.update_if_needed(
            ctx.queue,
            self.params_version,
            &self.gpu_params,
            &self.bake_params,
        );
    }
}

impl<'a> PassNode<'a> for AtmosphereTransmittanceNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.update_params(ctx);

        let PrepareContext {
            views,
            global_bind_group_cache: cache,
            device,
            ..
        } = ctx;
        let device = *device;

        let output_view = views.get_texture_view(self.output_tex);
        let key = BindGroupKey::new(self.layout.id())
            .with_resource(self.state.atmosphere_params_buffer.id())
            .with_resource(output_view.id());

        let bg = cache.get_or_create_bg(key, || {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atmo Transmittance BG"),
                layout: self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.state.atmosphere_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(output_view),
                    },
                ],
            })
        });
        self.bind_group = Some(bg);
    }

    fn execute(&self, _ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Atmosphere Transmittance LUT"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(self.pipeline);
        cpass.set_bind_group(
            0,
            self.bind_group.expect("atmo transmittance bg missing"),
            &[],
        );
        cpass.dispatch_workgroups(
            TRANSMITTANCE_WIDTH.div_ceil(8),
            TRANSMITTANCE_HEIGHT.div_ceil(8),
            1,
        );
    }
}

struct AtmosphereMultiScatterNode<'a> {
    transmittance_tex: TextureNodeId,
    output_tex: TextureNodeId,
    params_version: u64,
    gpu_params: GpuAtmosphereParams,
    bake_params: GpuBakeParams,
    state: &'a AtmosphereSceneState,
    pipeline: &'a wgpu::ComputePipeline,
    layout: &'a Tracked<wgpu::BindGroupLayout>,
    sampler: &'a wgpu::Sampler,
    bind_group: Option<&'a wgpu::BindGroup>,
}

impl AtmosphereNodeState for AtmosphereMultiScatterNode<'_> {
    fn update_params(&self, ctx: &PrepareContext<'_>) {
        self.state.update_if_needed(
            ctx.queue,
            self.params_version,
            &self.gpu_params,
            &self.bake_params,
        );
    }
}

impl<'a> PassNode<'a> for AtmosphereMultiScatterNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.update_params(ctx);

        let PrepareContext {
            views,
            global_bind_group_cache: cache,
            device,
            ..
        } = ctx;
        let device = *device;

        let transmittance_view = views.get_texture_view(self.transmittance_tex);
        let output_view = views.get_texture_view(self.output_tex);
        let key = BindGroupKey::new(self.layout.id())
            .with_resource(self.state.atmosphere_params_buffer.id())
            .with_resource(transmittance_view.id())
            .with_resource(output_view.id());

        let bg = cache.get_or_create_bg(key, || {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atmo Multi-Scatter BG"),
                layout: self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.state.atmosphere_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(transmittance_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(output_view),
                    },
                ],
            })
        });
        self.bind_group = Some(bg);
    }

    fn execute(&self, _ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Atmosphere Multi-Scatter LUT"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(self.pipeline);
        cpass.set_bind_group(
            0,
            self.bind_group.expect("atmo multi-scatter bg missing"),
            &[],
        );
        cpass.dispatch_workgroups(
            MULTI_SCATTER_SIZE.div_ceil(8),
            MULTI_SCATTER_SIZE.div_ceil(8),
            1,
        );
    }
}

struct AtmosphereSkyViewNode<'a> {
    transmittance_tex: TextureNodeId,
    multi_scatter_tex: TextureNodeId,
    output_tex: TextureNodeId,
    params_version: u64,
    gpu_params: GpuAtmosphereParams,
    bake_params: GpuBakeParams,
    state: &'a AtmosphereSceneState,
    pipeline: &'a wgpu::ComputePipeline,
    layout: &'a Tracked<wgpu::BindGroupLayout>,
    sampler: &'a wgpu::Sampler,
    bind_group: Option<&'a wgpu::BindGroup>,
}

impl AtmosphereNodeState for AtmosphereSkyViewNode<'_> {
    fn update_params(&self, ctx: &PrepareContext<'_>) {
        self.state.update_if_needed(
            ctx.queue,
            self.params_version,
            &self.gpu_params,
            &self.bake_params,
        );
    }
}

impl<'a> PassNode<'a> for AtmosphereSkyViewNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.update_params(ctx);

        let PrepareContext {
            views,
            global_bind_group_cache: cache,
            device,
            ..
        } = ctx;
        let device = *device;

        let transmittance_view = views.get_texture_view(self.transmittance_tex);
        let multi_scatter_view = views.get_texture_view(self.multi_scatter_tex);
        let output_view = views.get_texture_view(self.output_tex);
        let key = BindGroupKey::new(self.layout.id())
            .with_resource(self.state.atmosphere_params_buffer.id())
            .with_resource(transmittance_view.id())
            .with_resource(multi_scatter_view.id())
            .with_resource(output_view.id());

        let bg = cache.get_or_create_bg(key, || {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atmo Sky-View BG"),
                layout: self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.state.atmosphere_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(transmittance_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(multi_scatter_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(output_view),
                    },
                ],
            })
        });
        self.bind_group = Some(bg);
    }

    fn execute(&self, _ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Atmosphere Sky-View LUT"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(self.pipeline);
        cpass.set_bind_group(0, self.bind_group.expect("atmo sky-view bg missing"), &[]);
        cpass.dispatch_workgroups(SKY_VIEW_WIDTH.div_ceil(8), SKY_VIEW_HEIGHT.div_ceil(8), 1);
    }
}

struct AtmosphereSkyToCubeNode<'a> {
    base_cube: TextureNodeId,
    transmittance_tex: TextureNodeId,
    sky_view_tex: TextureNodeId,
    params_version: u64,
    gpu_params: GpuAtmosphereParams,
    bake_params: GpuBakeParams,
    state: &'a AtmosphereSceneState,
    pipeline: &'a wgpu::ComputePipeline,
    layout: &'a Tracked<wgpu::BindGroupLayout>,
    sampler: &'a wgpu::Sampler,
    base_cube_storage_view: &'a Tracked<wgpu::TextureView>,
    starbox: Option<&'a ResolvedAtmosphereStarbox>,
    bind_group: Option<&'a wgpu::BindGroup>,
}

impl AtmosphereNodeState for AtmosphereSkyToCubeNode<'_> {
    fn update_params(&self, ctx: &PrepareContext<'_>) {
        self.state.update_if_needed(
            ctx.queue,
            self.params_version,
            &self.gpu_params,
            &self.bake_params,
        );
    }
}

impl<'a> PassNode<'a> for AtmosphereSkyToCubeNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.update_params(ctx);

        let PrepareContext {
            views,
            global_bind_group_cache: cache,
            device,
            ..
        } = ctx;
        let device = *device;

        let transmittance_view = views.get_texture_view(self.transmittance_tex);
        let sky_view = views.get_texture_view(self.sky_view_tex);
        let key = BindGroupKey::new(self.layout.id())
            .with_resource(sky_view.id())
            .with_resource(self.state.bake_params_buffer.id())
            .with_resource(self.base_cube_storage_view.id())
            .with_resource(transmittance_view.id());

        let key = if let Some(starbox) = self.starbox {
            key.with_resource(starbox.resource_id)
        } else {
            key
        };

        let bg = cache.get_or_create_bg(key, || match self.starbox {
            Some(starbox) => device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atmo Sky-to-Cube BG + Starbox"),
                layout: self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(sky_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.state.bake_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(self.base_cube_storage_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(transmittance_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&starbox.view),
                    },
                ],
            }),
            None => device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atmo Sky-to-Cube BG"),
                layout: self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(sky_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.state.bake_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(self.base_cube_storage_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(transmittance_view),
                    },
                ],
            }),
        });
        self.bind_group = Some(bg);
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let dispatch = ctx.get_texture(self.base_cube).width().div_ceil(8);

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Atmosphere Sky-to-Cube Bake"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(self.pipeline);
            cpass.set_bind_group(
                0,
                self.bind_group.expect("atmo sky-to-cube bg missing"),
                &[],
            );
            cpass.dispatch_workgroups(dispatch, dispatch, 6);
        }

        ctx.mipmap_generator
            .generate(ctx.device, encoder, ctx.get_texture(self.base_cube));
    }
}
