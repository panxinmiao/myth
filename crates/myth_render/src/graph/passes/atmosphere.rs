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

use std::sync::atomic::{AtomicU64, Ordering};

use rustc_hash::FxHashMap;

use crate::core::binding::BindGroupKey;
use crate::core::gpu::Tracked;
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    ExecuteContext, ExtractContext, PassNode, PrepareContext, TextureDesc, TextureNodeId,
};
use crate::pipeline::{
    ComputePipelineId, ComputePipelineKey, ShaderCompilationOptions, ShaderSource,
};
use myth_scene::background::ProceduralSkyParams;

const TRANSMITTANCE_WIDTH: u32 = 256;
const TRANSMITTANCE_HEIGHT: u32 = 64;
const MULTI_SCATTER_SIZE: u32 = 32;
const SKY_VIEW_WIDTH: u32 = 192;
const SKY_VIEW_HEIGHT: u32 = 108;
const SCENE_CACHE_TTL: u64 = 120;

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
    sun_disk_size: f32,
    exposure: f32,
    _pad0: f32,
    _pad1: f32,
}

impl GpuBakeParams {
    fn from_scene(params: &ProceduralSkyParams) -> Self {
        Self {
            sun_direction: params.sun_direction.into(),
            sun_intensity: params.sun_intensity,
            sun_disk_size: params.sun_disk_size,
            exposure: params.exposure,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

struct AtmosphereSceneState {
    atmosphere_params_buffer: Tracked<wgpu::Buffer>,
    bake_params_buffer: Tracked<wgpu::Buffer>,
    uploaded_version: AtomicU64,
    last_used_frame: u64,
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
        queue.write_buffer(
            &self.bake_params_buffer,
            0,
            bytemuck::bytes_of(bake_params),
        );
        self.uploaded_version.store(version, Ordering::Relaxed);
    }
}

pub struct AtmosphereFeature {
    transmittance_pipeline: Option<ComputePipelineId>,
    multi_scatter_pipeline: Option<ComputePipelineId>,
    sky_view_pipeline: Option<ComputePipelineId>,
    sky_to_cube_pipeline: Option<ComputePipelineId>,
    transmittance_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    multi_scatter_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    sky_view_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    sky_to_cube_layout: Option<Tracked<wgpu::BindGroupLayout>>,
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
            sky_to_cube_pipeline: None,
            transmittance_layout: None,
            multi_scatter_layout: None,
            sky_view_layout: None,
            sky_to_cube_layout: None,
            sampler: None,
            scene_states: FxHashMap::default(),
        }
    }

    pub fn extract_and_prepare(&mut self, ctx: &mut ExtractContext, scene_id: u32) {
        self.ensure_layouts(ctx.device);
        self.ensure_sampler(ctx.device);
        self.ensure_pipelines(ctx);
        self.ensure_scene_state(ctx, scene_id);
        self.prune_scene_states(ctx.resource_manager.frame_index());
    }

    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        scene_id: u32,
        params: &ProceduralSkyParams,
        base_cube: TextureNodeId,
        base_cube_storage_view: &'a Tracked<wgpu::TextureView>,
    ) {
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
        let sky_to_cube_pipeline = self
            .sky_to_cube_pipeline
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
        let sky_to_cube_layout = self
            .sky_to_cube_layout
            .as_ref()
            .expect("atmosphere sky-to-cube layout must exist");
        let sampler = self
            .sampler
            .as_ref()
            .expect("atmosphere sampler must exist");

        ctx.with_group("Atmosphere_System", |ctx| {
            let transmittance_desc = TextureDesc::new_2d(
                TRANSMITTANCE_WIDTH,
                TRANSMITTANCE_HEIGHT,
                wgpu::TextureFormat::Rgba16Float,
                wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            );

            let transmittance = ctx.graph.add_pass("Atmosphere_Transmittance", |builder| {
                let output = builder.create_texture("Atmosphere_Transmittance", transmittance_desc);
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
            });

            let multi_scatter_desc = TextureDesc::new_2d(
                MULTI_SCATTER_SIZE,
                MULTI_SCATTER_SIZE,
                wgpu::TextureFormat::Rgba16Float,
                wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            );

            let multi_scatter = ctx.graph.add_pass("Atmosphere_MultiScatter", |builder| {
                builder.read_texture(transmittance);
                let output = builder.create_texture("Atmosphere_MultiScatter", multi_scatter_desc);
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
            });

            let sky_view_desc = TextureDesc::new_2d(
                SKY_VIEW_WIDTH,
                SKY_VIEW_HEIGHT,
                wgpu::TextureFormat::Rgba16Float,
                wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            );

            let sky_view = ctx.graph.add_pass("Atmosphere_SkyView", |builder| {
                builder.read_texture(transmittance);
                builder.read_texture(multi_scatter);
                let output = builder.create_texture("Atmosphere_SkyView", sky_view_desc);
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
                    bind_group: None,
                };
                (node, ())
            });
        });
    }

    fn ensure_scene_state(&mut self, ctx: &mut ExtractContext, scene_id: u32) {
        let frame_index = ctx.resource_manager.frame_index();
        let state = self.scene_states.entry(scene_id).or_insert_with(|| AtmosphereSceneState {
            atmosphere_params_buffer: Tracked::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Atmosphere Params"),
                size: std::mem::size_of::<GpuAtmosphereParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            bake_params_buffer: Tracked::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Atmosphere Bake Params"),
                size: std::mem::size_of::<GpuBakeParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            uploaded_version: AtomicU64::new(u64::MAX),
            last_used_frame: frame_index,
        });

        state.last_used_frame = frame_index;
    }

    fn prune_scene_states(&mut self, current_frame: u64) {
        self.scene_states.retain(|_, state| {
            current_frame.saturating_sub(state.last_used_frame) <= SCENE_CACHE_TTL
        });
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

        self.sky_to_cube_layout = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Atmo Sky-to-Cube BGL"),
                entries: &[
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
                ],
            },
        )));
    }

    fn ensure_pipelines(&mut self, ctx: &mut ExtractContext) {
        if self.transmittance_pipeline.is_some() {
            return;
        }

        let opts = ShaderCompilationOptions::default();

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
            &ComputePipelineKey {
                shader_hash: trans_hash,
            },
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
            &ComputePipelineKey {
                shader_hash: multi_hash,
            },
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
            &ComputePipelineKey {
                shader_hash: sky_view_hash,
            },
            "Atmo Sky-View Pipeline",
        ));

        let (sky_to_cube_module, sky_to_cube_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/utility/atmosphere/sky_to_cube"),
            &opts,
        );
        let sky_to_cube_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Atmo Sky-to-Cube PL"),
                bind_group_layouts: &[Some(self.sky_to_cube_layout.as_deref().unwrap())],
                immediate_size: 0,
            });
        self.sky_to_cube_pipeline = Some(ctx.pipeline_cache.get_or_create_compute(
            ctx.device,
            sky_to_cube_module,
            &sky_to_cube_layout,
            &ComputePipelineKey {
                shader_hash: sky_to_cube_hash,
            },
            "Atmo Sky-to-Cube Pipeline",
        ));
    }
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
        cpass.set_bind_group(0, self.bind_group.expect("atmo transmittance bg missing"), &[]);
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
        cpass.set_bind_group(0, self.bind_group.expect("atmo multi-scatter bg missing"), &[]);
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

        let bg = cache.get_or_create_bg(key, || {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            })
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
            cpass.set_bind_group(0, self.bind_group.expect("atmo sky-to-cube bg missing"), &[]);
            cpass.dispatch_workgroups(dispatch, dispatch, 6);
        }

        ctx.mipmap_generator
            .generate(ctx.device, encoder, ctx.get_texture(self.base_cube));
    }
}