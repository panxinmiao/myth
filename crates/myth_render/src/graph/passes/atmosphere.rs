//! Procedural Sky Atmosphere Feature
//!
//! Implements the Hillaire 2020 atmosphere model via three precomputed LUTs:
//!
//! 1. **Transmittance LUT** (256×64) — optical depth along rays
//! 2. **Multi-Scattering LUT** (32×32) — isotropic bounce lighting
//! 3. **Sky-View LUT** (192×108) — full sky hemisphere radiance
//!
//! After LUT generation, the sky-view is baked into a cubemap with sun disk,
//! and that cubemap is PMREM-prefiltered for IBL. The entire pipeline is
//! self-contained: no dependency on [`IblComputeFeature`].
//!
//! All GPU work is performed with a dedicated command encoder during the
//! extract-and-prepare phase, matching the existing IBL compute pattern.

use crate::core::gpu::{CubeSourceType, GpuEnvironment, Tracked, generate_gpu_resource_id};
use crate::graph::core::context::ExtractContext;
use crate::pipeline::{
    ComputePipelineId, ComputePipelineKey, ShaderCompilationOptions, ShaderSource,
};
use myth_resources::texture::TextureSource;
use myth_scene::background::ProceduralSkyParams;
use wgpu::TextureViewDimension;

// ─── LUT Dimensions ────────────────────────────────────────────────────────

const TRANSMITTANCE_WIDTH: u32 = 256;
const TRANSMITTANCE_HEIGHT: u32 = 64;
const MULTI_SCATTER_SIZE: u32 = 32;
const SKY_VIEW_WIDTH: u32 = 192;
const SKY_VIEW_HEIGHT: u32 = 108;
const IBL_CUBE_SIZE: u32 = 256;
const PMREM_SIZE: u32 = 256;

// ─── GPU Uniforms (must match WGSL struct layouts) ─────────────────────────

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

// ─── Atmosphere Feature ────────────────────────────────────────────────────

/// Procedural atmosphere rendering feature.
///
/// Manages the complete atmosphere pipeline from LUT generation through PMREM
/// prefiltering. All persistent GPU state (textures, layouts, pipelines) lives
/// here; recomputation is triggered only when `ProceduralSkyParams::version()`
/// changes.
pub struct AtmosphereFeature {
    // ─── Pipeline IDs ──────────────────────────────────────────────
    transmittance_pipeline: Option<ComputePipelineId>,
    multi_scatter_pipeline: Option<ComputePipelineId>,
    sky_view_pipeline: Option<ComputePipelineId>,
    sky_to_cube_pipeline: Option<ComputePipelineId>,
    pmrem_pipeline: Option<ComputePipelineId>,

    // ─── Bind Group Layouts ────────────────────────────────────────
    transmittance_layout: Option<wgpu::BindGroupLayout>,
    multi_scatter_layout: Option<wgpu::BindGroupLayout>,
    sky_view_layout: Option<wgpu::BindGroupLayout>,
    sky_to_cube_layout: Option<wgpu::BindGroupLayout>,
    pmrem_src_layout: Option<wgpu::BindGroupLayout>,
    pmrem_dst_layout: Option<wgpu::BindGroupLayout>,

    // ─── GPU Textures ──────────────────────────────────────────────
    transmittance_texture: Option<wgpu::Texture>,
    multi_scatter_texture: Option<wgpu::Texture>,
    sky_view_texture: Option<wgpu::Texture>,
    cube_texture: Option<wgpu::Texture>,

    // ─── Tracked Views (for downstream consumption) ────────────────
    sky_view_view: Option<Tracked<wgpu::TextureView>>,

    // ─── Environment Cache Integration ─────────────────────────────
    /// Stable resource ID for the cube view in `internal_resources`.
    cube_view_id: u64,
    /// Stable resource ID for the PMREM view in `internal_resources`.
    pmrem_view_id: u64,
    /// The `TextureSource` key used in `environment_map_cache`.
    cache_key: TextureSource,
    /// Maximum PMREM mip level for roughness mapping.
    env_map_max_mip: f32,

    // ─── Version Tracking ──────────────────────────────────────────
    last_params_version: u64,
    /// Whether the pipeline has produced valid results at least once.
    pub(crate) ready: bool,
}

impl Default for AtmosphereFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl AtmosphereFeature {
    #[must_use]
    pub fn new() -> Self {
        let cube_view_id = generate_gpu_resource_id();
        let pmrem_view_id = generate_gpu_resource_id();
        let cache_key = TextureSource::Attachment(cube_view_id, TextureViewDimension::Cube);

        let pmrem_mips = (PMREM_SIZE as f32).log2().floor() as u32 + 1;

        Self {
            transmittance_pipeline: None,
            multi_scatter_pipeline: None,
            sky_view_pipeline: None,
            sky_to_cube_pipeline: None,
            pmrem_pipeline: None,
            transmittance_layout: None,
            multi_scatter_layout: None,
            sky_view_layout: None,
            sky_to_cube_layout: None,
            pmrem_src_layout: None,
            pmrem_dst_layout: None,
            transmittance_texture: None,
            multi_scatter_texture: None,
            sky_view_texture: None,
            cube_texture: None,
            sky_view_view: None,
            cube_view_id,
            pmrem_view_id,
            cache_key,
            env_map_max_mip: (pmrem_mips - 1) as f32,
            last_params_version: u64::MAX,
            ready: false,
        }
    }

    /// Returns the sky-view LUT texture view for procedural skybox rendering.
    #[inline]
    pub fn sky_view_view(&self) -> Option<&Tracked<wgpu::TextureView>> {
        self.sky_view_view.as_ref()
    }

    /// Returns the `TextureSource` key used for environment cache lookups.
    ///
    /// The renderer sets `scene.environment.source_env_map` to this value
    /// before `resolve_gpu_environment` runs, enabling the standard bind-group
    /// path to pick up our cube + PMREM views.
    #[inline]
    pub fn env_source(&self) -> TextureSource {
        self.cache_key
    }

    /// Returns the maximum PMREM mip level for uniform buffer writes.
    #[inline]
    pub fn env_map_max_mip(&self) -> f32 {
        self.env_map_max_mip
    }

    // ─── Lazy Initialization ───────────────────────────────────────────────

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

        self.sky_view_view = Some(Tracked::new(
            self.sky_view_texture
                .as_ref()
                .unwrap()
                .create_view(&Default::default()),
        ));

        let cube_mips = (IBL_CUBE_SIZE as f32).log2().floor() as u32 + 1;
        self.cube_texture = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Atmo Sky Cube"),
            size: wgpu::Extent3d {
                width: IBL_CUBE_SIZE,
                height: IBL_CUBE_SIZE,
                depth_or_array_layers: 6,
            },
            mip_level_count: cube_mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        }));
    }

    fn ensure_layouts(&mut self, device: &wgpu::Device) {
        if self.transmittance_layout.is_some() {
            return;
        }

        // Transmittance: uniform + storage output
        self.transmittance_layout =
            Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            }));

        // Multi-scatter: uniform + transmittance + sampler + storage output
        self.multi_scatter_layout =
            Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            }));

        // Sky-view: uniform + transmittance + multi-scatter + sampler + storage
        self.sky_view_layout =
            Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        // Sky-to-cube: sky-view + sampler + bake params + cube storage
        self.sky_to_cube_layout =
            Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        // PMREM source: cube texture + sampler + params
        self.pmrem_src_layout =
            Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Atmo PMREM Src BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::Cube,
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
                ],
            }));

        // PMREM dest: storage texture (2D array)
        self.pmrem_dst_layout =
            Some(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Atmo PMREM Dst BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2Array,
                    },
                    count: None,
                }],
            }));
    }

    fn ensure_pipelines(&mut self, ctx: &mut ExtractContext) {
        if self.transmittance_pipeline.is_some() {
            return;
        }

        let device = ctx.device;
        let opts = ShaderCompilationOptions::default();

        // Transmittance pipeline
        {
            let (module, hash) = ctx.shader_manager.get_or_compile(
                device,
                ShaderSource::File("entry/utility/atmosphere/transmittance_lut"),
                &opts,
            );
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Atmo Transmittance PL"),
                bind_group_layouts: &[Some(self.transmittance_layout.as_ref().unwrap())],
                immediate_size: 0,
            });
            self.transmittance_pipeline = Some(ctx.pipeline_cache.get_or_create_compute(
                device,
                module,
                &layout,
                &ComputePipelineKey { shader_hash: hash },
                "Atmo Transmittance Pipeline",
            ));
        }

        // Multi-scatter pipeline
        {
            let (module, hash) = ctx.shader_manager.get_or_compile(
                device,
                ShaderSource::File("entry/utility/atmosphere/multi_scatter_lut"),
                &opts,
            );
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Atmo Multi-Scatter PL"),
                bind_group_layouts: &[Some(self.multi_scatter_layout.as_ref().unwrap())],
                immediate_size: 0,
            });
            self.multi_scatter_pipeline = Some(ctx.pipeline_cache.get_or_create_compute(
                device,
                module,
                &layout,
                &ComputePipelineKey { shader_hash: hash },
                "Atmo Multi-Scatter Pipeline",
            ));
        }

        // Sky-view pipeline
        {
            let (module, hash) = ctx.shader_manager.get_or_compile(
                device,
                ShaderSource::File("entry/utility/atmosphere/sky_view_lut"),
                &opts,
            );
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Atmo Sky-View PL"),
                bind_group_layouts: &[Some(self.sky_view_layout.as_ref().unwrap())],
                immediate_size: 0,
            });
            self.sky_view_pipeline = Some(ctx.pipeline_cache.get_or_create_compute(
                device,
                module,
                &layout,
                &ComputePipelineKey { shader_hash: hash },
                "Atmo Sky-View Pipeline",
            ));
        }

        // Sky-to-cube pipeline
        {
            let (module, hash) = ctx.shader_manager.get_or_compile(
                device,
                ShaderSource::File("entry/utility/atmosphere/sky_to_cube"),
                &opts,
            );
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Atmo Sky-to-Cube PL"),
                bind_group_layouts: &[Some(self.sky_to_cube_layout.as_ref().unwrap())],
                immediate_size: 0,
            });
            self.sky_to_cube_pipeline = Some(ctx.pipeline_cache.get_or_create_compute(
                device,
                module,
                &layout,
                &ComputePipelineKey { shader_hash: hash },
                "Atmo Sky-to-Cube Pipeline",
            ));
        }

        // PMREM prefilter pipeline (reuses ibl.wgsl)
        {
            let (module, hash) = ctx.shader_manager.get_or_compile(
                device,
                ShaderSource::File("entry/utility/ibl.wgsl"),
                &opts,
            );
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Atmo PMREM PL"),
                bind_group_layouts: &[
                    Some(self.pmrem_src_layout.as_ref().unwrap()),
                    Some(self.pmrem_dst_layout.as_ref().unwrap()),
                ],
                immediate_size: 0,
            });
            self.pmrem_pipeline = Some(ctx.pipeline_cache.get_or_create_compute(
                device,
                module,
                &layout,
                &ComputePipelineKey { shader_hash: hash },
                "Atmo PMREM Pipeline",
            ));
        }
    }

    // ─── Utility ───────────────────────────────────────────────────────────

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

    // ─── Public API ────────────────────────────────────────────────────────

    /// Regenerate atmosphere LUTs and PMREM if parameters have changed.
    ///
    /// All compute work is submitted via a dedicated command encoder during the
    /// prepare phase. After completion, a `GpuEnvironment` entry is inserted
    /// into the `environment_map_cache` with `needs_compute: false`, so the
    /// standard `prepare_global` bind-group path picks up our textures.
    #[allow(clippy::too_many_lines)]
    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        params: &ProceduralSkyParams,
    ) {
        if self.last_params_version == params.version() && self.ready {
            return;
        }

        let device = ctx.device;
        self.ensure_textures(device);
        self.ensure_layouts(device);
        self.ensure_pipelines(ctx);

        let gpu_params = GpuAtmosphereParams::from_scene(params);
        let param_buffer =
            Self::create_mapped_buffer(device, "Atmo Params", bytemuck::bytes_of(&gpu_params));

        let sampler = ctx.resource_manager.sampler_registry.default_sampler().1;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Atmosphere Encoder"),
        });

        // ── Phase 1: Transmittance LUT ─────────────────────────────────
        {
            let dest_view = self
                .transmittance_texture
                .as_ref()
                .unwrap()
                .create_view(&Default::default());
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atmo Transmittance BG"),
                layout: self.transmittance_layout.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: param_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&dest_view),
                    },
                ],
            });
            let pipeline = ctx
                .pipeline_cache
                .get_compute_pipeline(self.transmittance_pipeline.unwrap());
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Transmittance LUT"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(
                TRANSMITTANCE_WIDTH.div_ceil(8),
                TRANSMITTANCE_HEIGHT.div_ceil(8),
                1,
            );
        }

        // ── Phase 2: Multi-Scattering LUT ──────────────────────────────
        {
            let trans_view = self
                .transmittance_texture
                .as_ref()
                .unwrap()
                .create_view(&Default::default());
            let dest_view = self
                .multi_scatter_texture
                .as_ref()
                .unwrap()
                .create_view(&Default::default());
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atmo Multi-Scatter BG"),
                layout: self.multi_scatter_layout.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: param_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&trans_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&dest_view),
                    },
                ],
            });
            let pipeline = ctx
                .pipeline_cache
                .get_compute_pipeline(self.multi_scatter_pipeline.unwrap());
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Multi-Scatter LUT"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(
                MULTI_SCATTER_SIZE.div_ceil(8),
                MULTI_SCATTER_SIZE.div_ceil(8),
                1,
            );
        }

        // ── Phase 3: Sky-View LUT ──────────────────────────────────────
        {
            let trans_view = self
                .transmittance_texture
                .as_ref()
                .unwrap()
                .create_view(&Default::default());
            let ms_view = self
                .multi_scatter_texture
                .as_ref()
                .unwrap()
                .create_view(&Default::default());
            let dest_view = self
                .sky_view_texture
                .as_ref()
                .unwrap()
                .create_view(&Default::default());
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atmo Sky-View BG"),
                layout: self.sky_view_layout.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: param_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&trans_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&ms_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&dest_view),
                    },
                ],
            });
            let pipeline = ctx
                .pipeline_cache
                .get_compute_pipeline(self.sky_view_pipeline.unwrap());
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sky-View LUT"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(
                SKY_VIEW_WIDTH.div_ceil(8),
                SKY_VIEW_HEIGHT.div_ceil(8),
                1,
            );
        }

        // ── Phase 4: Sky-to-Cubemap Bake ───────────────────────────────
        {
            let sky_view = self
                .sky_view_texture
                .as_ref()
                .unwrap()
                .create_view(&Default::default());
            let cube_dest = self
                .cube_texture
                .as_ref()
                .unwrap()
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some("Atmo Cube Mip0 Dest"),
                    dimension: Some(TextureViewDimension::D2Array),
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(6),
                    usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                    ..Default::default()
                });
            let bake_params = GpuBakeParams {
                sun_direction: params.sun_direction.into(),
                sun_intensity: params.sun_intensity,
                sun_disk_size: params.sun_disk_size,
                exposure: params.exposure,
                _pad0: 0.0,
                _pad1: 0.0,
            };
            let bake_buffer = Self::create_mapped_buffer(
                device,
                "Atmo Bake Params",
                bytemuck::bytes_of(&bake_params),
            );

            let trans_view = self.transmittance_texture.as_ref().unwrap().create_view(&Default::default());

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Atmo Sky-to-Cube BG"),
                layout: self.sky_to_cube_layout.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&sky_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
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
                        resource: wgpu::BindingResource::TextureView(&trans_view),
                    },
                ],
            });
            let pipeline = ctx
                .pipeline_cache
                .get_compute_pipeline(self.sky_to_cube_pipeline.unwrap());
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sky-to-Cube Bake"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(
                IBL_CUBE_SIZE.div_ceil(8),
                IBL_CUBE_SIZE.div_ceil(8),
                6,
            );
        }

        // ── Phase 5: Generate mipmaps for the baked cubemap ────────────
        let cube_tex = self.cube_texture.as_ref().unwrap();
        ctx.resource_manager
            .mipmap_generator
            .generate(device, &mut encoder, cube_tex);

        // ── Phase 6: PMREM prefiltering ────────────────────────────────
        let pmrem_mips = (PMREM_SIZE as f32).log2().floor() as u32 + 1;
        let pmrem_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Atmo PMREM"),
            size: wgpu::Extent3d {
                width: PMREM_SIZE,
                height: PMREM_SIZE,
                depth_or_array_layers: 6,
            },
            mip_level_count: pmrem_mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let cube_src_view = cube_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Atmo PMREM Src Cube"),
            dimension: Some(TextureViewDimension::Cube),
            ..Default::default()
        });

        let pmrem_pipeline = ctx
            .pipeline_cache
            .get_compute_pipeline(self.pmrem_pipeline.unwrap());

        for mip in 0..pmrem_mips {
            let mip_size = (PMREM_SIZE >> mip).max(1);
            let roughness = mip as f32 / (pmrem_mips - 1) as f32;
            let params = [roughness, mip_size as f32, 0.0, 0.0];
            let param_buf =
                Self::create_mapped_buffer(device, "PMREM Params", bytemuck::cast_slice(&params));

            let bg_src = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: self.pmrem_src_layout.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&cube_src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: param_buf.as_entire_binding(),
                    },
                ],
            });

            let pmrem_dest = pmrem_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("PMREM Mip Dest"),
                format: Some(wgpu::TextureFormat::Rgba16Float),
                dimension: Some(TextureViewDimension::D2Array),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(6),
                usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
            });

            let bg_dst = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: self.pmrem_dst_layout.as_ref().unwrap(),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&pmrem_dest),
                }],
            });

            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Atmo PMREM"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(pmrem_pipeline);
            cpass.set_bind_group(0, &bg_src, &[]);
            cpass.set_bind_group(1, &bg_dst, &[]);
            let groups = mip_size.div_ceil(8);
            cpass.dispatch_workgroups(groups, groups, 6);
        }

        // Submit all atmosphere + PMREM work
        ctx.queue.submit(Some(encoder.finish()));

        // ── Register views in ResourceManager ──────────────────────────
        let cube_view = cube_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Atmo Cube View"),
            dimension: Some(TextureViewDimension::Cube),
            ..Default::default()
        });
        ctx.resource_manager
            .register_internal_texture_direct(self.cube_view_id, cube_view);

        let pmrem_view = pmrem_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Atmo PMREM View"),
            dimension: Some(TextureViewDimension::Cube),
            ..Default::default()
        });
        ctx.resource_manager
            .register_internal_texture_direct(self.pmrem_view_id, pmrem_view);

        // ── Insert GpuEnvironment into the cache ───────────────────────
        // Remove any stale entry first (the old PMREM texture is dropped here).
        ctx.resource_manager
            .environment_map_cache
            .remove(&self.cache_key);

        let gpu_env = GpuEnvironment {
            source_version: 0,
            needs_compute: false,
            source_type: CubeSourceType::CubeWithMipmaps,
            cube_texture: None,
            pmrem_texture,
            cube_view_id: self.cube_view_id,
            pmrem_view_id: self.pmrem_view_id,
            env_map_max_mip_level: self.env_map_max_mip,
        };
        ctx.resource_manager
            .environment_map_cache
            .insert(self.cache_key, gpu_env);

        self.last_params_version = params.version();
        self.ready = true;

        log::info!(
            "Atmosphere LUTs + PMREM regenerated (version {})",
            params.version(),
        );
    }
}
