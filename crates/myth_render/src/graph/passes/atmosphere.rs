//! Procedural sky atmosphere bake pass.
//!
//! Implements the Hillaire 2020 atmosphere model as a pure RenderGraph node.
//! The node owns only the GPU command recording for:
//!
//! - Transmittance LUT
//! - Multi-scatter LUT
//! - Sky-view LUT
//! - Sky-view -> persistent scene base cubemap bake
//!
//! PMREM generation is intentionally handled by the shared IBL node so the
//! graph topology stays explicit and reusable.

use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::TextureNodeId;
use crate::graph::core::context::{ExecuteContext, ExtractContext};
use crate::graph::core::node::PassNode;
use crate::pipeline::{
    ComputePipelineId, ComputePipelineKey, ShaderCompilationOptions, ShaderSource,
};
use myth_scene::background::ProceduralSkyParams;
use wgpu::TextureViewDimension;

const TRANSMITTANCE_WIDTH: u32 = 256;
const TRANSMITTANCE_HEIGHT: u32 = 64;
const MULTI_SCATTER_SIZE: u32 = 32;
const SKY_VIEW_WIDTH: u32 = 192;
const SKY_VIEW_HEIGHT: u32 = 108;

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

pub struct AtmosphereFeature {
    transmittance_pipeline: Option<ComputePipelineId>,
    multi_scatter_pipeline: Option<ComputePipelineId>,
    sky_view_pipeline: Option<ComputePipelineId>,
    sky_to_cube_pipeline: Option<ComputePipelineId>,
    transmittance_layout: Option<wgpu::BindGroupLayout>,
    multi_scatter_layout: Option<wgpu::BindGroupLayout>,
    sky_view_layout: Option<wgpu::BindGroupLayout>,
    sky_to_cube_layout: Option<wgpu::BindGroupLayout>,
    transmittance_texture: Option<wgpu::Texture>,
    multi_scatter_texture: Option<wgpu::Texture>,
    sky_view_texture: Option<wgpu::Texture>,
    sampler: Option<wgpu::Sampler>,
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
            transmittance_texture: None,
            multi_scatter_texture: None,
            sky_view_texture: None,
            sampler: None,
        }
    }

    pub fn extract_and_prepare(&mut self, ctx: &mut ExtractContext) {
        self.ensure_textures(ctx.device);
        self.ensure_layouts(ctx.device);
        self.ensure_sampler(ctx.device);
        self.ensure_pipelines(ctx);
    }

    pub fn add_bake_node<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        params: &ProceduralSkyParams,
        base_cube: TextureNodeId,
        base_cube_texture: &'a wgpu::Texture,
    ) {
        let node = AtmosphereBakeNode {
            gpu_params: GpuAtmosphereParams::from_scene(params),
            bake_params: GpuBakeParams::from_scene(params),
            transmittance_pipeline: self
                .transmittance_pipeline
                .map(|id| ctx.pipeline_cache.get_compute_pipeline(id)),
            multi_scatter_pipeline: self
                .multi_scatter_pipeline
                .map(|id| ctx.pipeline_cache.get_compute_pipeline(id)),
            sky_view_pipeline: self
                .sky_view_pipeline
                .map(|id| ctx.pipeline_cache.get_compute_pipeline(id)),
            sky_to_cube_pipeline: self
                .sky_to_cube_pipeline
                .map(|id| ctx.pipeline_cache.get_compute_pipeline(id)),
            transmittance_layout: self
                .transmittance_layout
                .as_ref()
                .expect("transmittance layout must exist"),
            multi_scatter_layout: self
                .multi_scatter_layout
                .as_ref()
                .expect("multi scatter layout must exist"),
            sky_view_layout: self.sky_view_layout.as_ref().expect("sky view layout must exist"),
            sky_to_cube_layout: self
                .sky_to_cube_layout
                .as_ref()
                .expect("sky to cube layout must exist"),
            transmittance_texture: self
                .transmittance_texture
                .as_ref()
                .expect("transmittance texture must exist"),
            multi_scatter_texture: self
                .multi_scatter_texture
                .as_ref()
                .expect("multi scatter texture must exist"),
            sky_view_texture: self.sky_view_texture.as_ref().expect("sky view texture must exist"),
            sampler: self.sampler.as_ref().expect("atmosphere sampler must exist"),
            base_cube_texture,
        };

        ctx.graph.add_pass("Atmosphere_Bake", |builder| {
            builder.write_texture(base_cube);
            (node, ())
        });
    }

    fn ensure_textures(&mut self, device: &wgpu::Device) {
        if self.transmittance_texture.is_some() {
            return;
        }

        let storage_usage =
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;

        self.transmittance_texture = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Atmo Transmittance LUT"),
            size: wgpu::Extent3d {
                width: TRANSMITTANCE_WIDTH,
                height: TRANSMITTANCE_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: storage_usage,
            view_formats: &[],
        }));

        self.multi_scatter_texture = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Atmo Multi-Scatter LUT"),
            size: wgpu::Extent3d {
                width: MULTI_SCATTER_SIZE,
                height: MULTI_SCATTER_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: storage_usage,
            view_formats: &[],
        }));

        self.sky_view_texture = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Atmo Sky-View LUT"),
            size: wgpu::Extent3d {
                width: SKY_VIEW_WIDTH,
                height: SKY_VIEW_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: storage_usage,
            view_formats: &[],
        }));
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

        self.transmittance_layout = Some(device.create_bind_group_layout(
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
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            },
        ));

        self.multi_scatter_layout = Some(device.create_bind_group_layout(
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
                            view_dimension: TextureViewDimension::D2,
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
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            },
        ));

        self.sky_view_layout = Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
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
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        }));

        self.sky_to_cube_layout = Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Atmo Sky-to-Cube BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
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
                        view_dimension: TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        }));
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
                bind_group_layouts: &[Some(self.transmittance_layout.as_ref().unwrap())],
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
                bind_group_layouts: &[Some(self.multi_scatter_layout.as_ref().unwrap())],
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
                bind_group_layouts: &[Some(self.sky_view_layout.as_ref().unwrap())],
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
                bind_group_layouts: &[Some(self.sky_to_cube_layout.as_ref().unwrap())],
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

struct AtmosphereBakeNode<'a> {
    gpu_params: GpuAtmosphereParams,
    bake_params: GpuBakeParams,
    transmittance_pipeline: Option<&'a wgpu::ComputePipeline>,
    multi_scatter_pipeline: Option<&'a wgpu::ComputePipeline>,
    sky_view_pipeline: Option<&'a wgpu::ComputePipeline>,
    sky_to_cube_pipeline: Option<&'a wgpu::ComputePipeline>,
    transmittance_layout: &'a wgpu::BindGroupLayout,
    multi_scatter_layout: &'a wgpu::BindGroupLayout,
    sky_view_layout: &'a wgpu::BindGroupLayout,
    sky_to_cube_layout: &'a wgpu::BindGroupLayout,
    transmittance_texture: &'a wgpu::Texture,
    multi_scatter_texture: &'a wgpu::Texture,
    sky_view_texture: &'a wgpu::Texture,
    sampler: &'a wgpu::Sampler,
    base_cube_texture: &'a wgpu::Texture,
}

impl AtmosphereBakeNode<'_> {
    fn create_mapped_buffer(device: &wgpu::Device, label: &str, data: &[u8]) -> wgpu::Buffer {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: data.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        buffer.slice(..).get_mapped_range_mut().copy_from_slice(data);
        buffer.unmap();
        buffer
    }
}

impl PassNode<'_> for AtmosphereBakeNode<'_> {
    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let transmittance_pipeline = self
            .transmittance_pipeline
            .expect("atmosphere transmittance pipeline must exist");
        let multi_scatter_pipeline = self
            .multi_scatter_pipeline
            .expect("atmosphere multi scatter pipeline must exist");
        let sky_view_pipeline = self
            .sky_view_pipeline
            .expect("atmosphere sky view pipeline must exist");
        let sky_to_cube_pipeline = self
            .sky_to_cube_pipeline
            .expect("atmosphere sky-to-cube pipeline must exist");

        let params_buffer = Self::create_mapped_buffer(
            ctx.device,
            "Atmo Params",
            bytemuck::bytes_of(&self.gpu_params),
        );

        let transmittance_view = self
            .transmittance_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let multi_scatter_view = self
            .multi_scatter_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let sky_view = self
            .sky_view_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        {
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atmo Transmittance BG"),
                layout: self.transmittance_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&transmittance_view),
                    },
                ],
            });
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Atmosphere Transmittance LUT"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(transmittance_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(
                TRANSMITTANCE_WIDTH.div_ceil(8),
                TRANSMITTANCE_HEIGHT.div_ceil(8),
                1,
            );
        }

        {
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atmo Multi-Scatter BG"),
                layout: self.multi_scatter_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&transmittance_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&multi_scatter_view),
                    },
                ],
            });
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Atmosphere Multi-Scatter LUT"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(multi_scatter_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(
                MULTI_SCATTER_SIZE.div_ceil(8),
                MULTI_SCATTER_SIZE.div_ceil(8),
                1,
            );
        }

        {
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atmo Sky-View BG"),
                layout: self.sky_view_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&transmittance_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&multi_scatter_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&sky_view),
                    },
                ],
            });
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Atmosphere Sky-View LUT"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(sky_view_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(SKY_VIEW_WIDTH.div_ceil(8), SKY_VIEW_HEIGHT.div_ceil(8), 1);
        }

        {
            let bake_buffer = Self::create_mapped_buffer(
                ctx.device,
                "Atmo Bake Params",
                bytemuck::bytes_of(&self.bake_params),
            );
            let cube_dest = self
                .base_cube_texture
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some("Atmo Base Cube Mip0"),
                    dimension: Some(TextureViewDimension::D2Array),
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(6),
                    usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                    ..Default::default()
                });
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atmo Sky-to-Cube BG"),
                layout: self.sky_to_cube_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&sky_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bake_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&cube_dest),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&transmittance_view),
                    },
                ],
            });
            let dispatch = self.base_cube_texture.width().div_ceil(8);
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Atmosphere Sky-to-Cube Bake"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(sky_to_cube_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(dispatch, dispatch, 6);
        }

        ctx.mipmap_generator
            .generate(ctx.device, encoder, self.base_cube_texture);
    }
}