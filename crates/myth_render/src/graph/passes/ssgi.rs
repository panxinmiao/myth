//! Screen Space Global Illumination feature.
//!
//! SSGI is implemented as a scene-level screen-space system with four stages:
//! raw Hi-Z ray march, temporal accumulation, multi-stage A-Trous cleanup,
//! and material-modulated merge back into the HDR scene color.

use rustc_hash::FxHashMap;

use crate::HDR_TEXTURE_FORMAT;
use crate::core::gpu::{CommonSampler, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    BufferDesc, BufferNodeId, ExecuteContext, ExtractContext, PassNode, PrepareContext,
    RenderTargetOps, TextureDesc, TextureNodeId,
};
use crate::graph::passes::utils::CopyTextureNode;
use crate::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions, ShaderSource,
};
use myth_resources::buffer::CpuBuffer;
use myth_resources::ssgi::SsgiUniforms;
use myth_resources::uniforms::WgslStruct;

const SSGI_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
#[allow(dead_code)]
const SSGI_WORKGROUP_SIZE: u32 = 8;
const MAX_ATROUS_PASSES: usize = 4;
const HISTORY_FLAG_INDIRECT_VALID: u32 = 1 << 0;
const HISTORY_FLAG_SOURCE_VALID: u32 = 1 << 1;
const SSGI_ATROUS_PASS_NAMES: [&str; MAX_ATROUS_PASSES] = [
    "SSGI_ATrous_Step_1",
    "SSGI_ATrous_Step_2",
    "SSGI_ATrous_Step_4",
    "SSGI_ATrous_Step_8",
];
const SSGI_ATROUS_OUTPUT_NAMES: [&str; MAX_ATROUS_PASSES] = [
    "SSGI_ATrous_Step_1_Output",
    "SSGI_ATrous_Step_2_Output",
    "SSGI_ATrous_Step_4_Output",
    "SSGI_ATrous_Step_8_Output",
];
const SSGI_ATROUS_UNIFORM_NAMES: [&str; MAX_ATROUS_PASSES] = [
    "SSGI_ATrous_Uniforms_1",
    "SSGI_ATrous_Uniforms_2",
    "SSGI_ATrous_Uniforms_4",
    "SSGI_ATrous_Uniforms_8",
];
const SSGI_ATROUS_BUFFER_LABELS: [&str; MAX_ATROUS_PASSES] = [
    "SSGI A-Trous Uniform Step 1",
    "SSGI A-Trous Uniform Step 2",
    "SSGI A-Trous Uniform Step 4",
    "SSGI A-Trous Uniform Step 8",
];

fn blue_noise_view_dimension() -> wgpu::TextureViewDimension {
    if cfg!(feature = "advanced_noise") {
        wgpu::TextureViewDimension::D2Array
    } else {
        wgpu::TextureViewDimension::D2
    }
}

#[must_use = "SSA Graph: consume the merged color output from SSGI"]
pub struct SsgiOutputs {
    pub merged_color: TextureNodeId,
    pub raw_indirect: TextureNodeId,
    pub clean_indirect: TextureNodeId,
}

pub struct SsgiFeature {
    raw_pipelines: FxHashMap<u32, RenderPipelineId>,
    temporal_pipeline: Option<RenderPipelineId>,
    atrous_pipeline: Option<RenderPipelineId>,
    merge_pipeline: Option<RenderPipelineId>,

    raw_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    temporal_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    atrous_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    merge_layout: Option<Tracked<wgpu::BindGroupLayout>>,

    uniforms_buffer: Option<Tracked<wgpu::Buffer>>,
    black_cube_view: Option<Tracked<wgpu::TextureView>>,
    blue_noise_view: Option<Tracked<wgpu::TextureView>>,
    blue_noise_sampler: Option<Tracked<wgpu::Sampler>>,

    indirect_history_view: Option<Tracked<wgpu::TextureView>>,
    history_meta_view: Option<Tracked<wgpu::TextureView>>,
    source_history_view: Option<Tracked<wgpu::TextureView>>,
    source_history_input_view: Option<Tracked<wgpu::TextureView>>,
    atrous_uniform_buffers: Vec<Tracked<wgpu::Buffer>>,

    prepared_max_steps: u32,
    prepared_atrous_passes: u32,
    full_resolution: (u32, u32),
    half_resolution: (u32, u32),
    history_valid: bool,
    source_history_valid: bool,
    prev_frame_taa_enabled: bool,
}

impl Default for SsgiFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl SsgiFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            raw_pipelines: FxHashMap::default(),
            temporal_pipeline: None,
            atrous_pipeline: None,
            merge_pipeline: None,

            raw_layout: None,
            temporal_layout: None,
            atrous_layout: None,
            merge_layout: None,

            uniforms_buffer: None,
            black_cube_view: None,
            blue_noise_view: None,
            blue_noise_sampler: None,

            indirect_history_view: None,
            history_meta_view: None,
            source_history_view: None,
            source_history_input_view: None,
            atrous_uniform_buffers: Vec::with_capacity(MAX_ATROUS_PASSES),

            prepared_max_steps: 16,
            prepared_atrous_passes: 3,
            full_resolution: (0, 0),
            half_resolution: (0, 0),
            history_valid: false,
            source_history_valid: false,
            prev_frame_taa_enabled: false,
        }
    }

    #[must_use]
    pub fn history_flags(&self) -> u32 {
        let mut flags = 0;
        if self.history_valid {
            flags |= HISTORY_FLAG_INDIRECT_VALID;
        }
        if self.source_history_valid {
            flags |= HISTORY_FLAG_SOURCE_VALID;
        }
        flags
    }

    pub fn invalidate_history(&mut self) {
        self.history_valid = false;
        self.source_history_valid = false;
        self.prev_frame_taa_enabled = false;
    }

    fn ensure_layouts(&mut self, device: &wgpu::Device) {
        if self.raw_layout.is_some() {
            return;
        }

        let raw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSGI Raw Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: blue_noise_view_dimension(),
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let temporal_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSGI Temporal Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let atrous_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSGI A-Trous Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let merge_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSGI Merge Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        self.raw_layout = Some(Tracked::new(raw_layout));
        self.temporal_layout = Some(Tracked::new(temporal_layout));
        self.atrous_layout = Some(Tracked::new(atrous_layout));
        self.merge_layout = Some(Tracked::new(merge_layout));
    }

    fn ensure_atrous_uniform_buffers(
        &mut self,
        ctx: &mut ExtractContext,
        base_uniforms: SsgiUniforms,
    ) {
        while self.atrous_uniform_buffers.len() < MAX_ATROUS_PASSES {
            let label = SSGI_ATROUS_BUFFER_LABELS[self.atrous_uniform_buffers.len()];
            let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<SsgiUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.atrous_uniform_buffers.push(Tracked::new(buffer));
        }

        self.prepared_atrous_passes = base_uniforms
            .denoise_params
            .x
            .clamp(1, MAX_ATROUS_PASSES as u32);

        for (pass_index, buffer) in self
            .atrous_uniform_buffers
            .iter()
            .enumerate()
            .take(self.prepared_atrous_passes as usize)
        {
            let mut pass_uniforms = base_uniforms;
            pass_uniforms.denoise_params.y = 1u32 << pass_index;
            ctx.queue
                .write_buffer(buffer, 0, bytemuck::bytes_of(&pass_uniforms));
        }
    }

    fn ensure_history_buffers(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let half_w = (width / 2).max(1);
        let half_h = (height / 2).max(1);

        if self.full_resolution == (width, height)
            && self.half_resolution == (half_w, half_h)
            && self.indirect_history_view.is_some()
            && self.history_meta_view.is_some()
            && self.source_history_view.is_some()
        {
            return;
        }

        let half_extent = wgpu::Extent3d {
            width: half_w,
            height: half_h,
            depth_or_array_layers: 1,
        };
        let full_extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let indirect_history = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSGI History Indirect"),
            size: half_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SSGI_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let history_meta = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSGI History Meta"),
            size: half_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SSGI_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let source_history = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSGI Source History"),
            size: full_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.indirect_history_view = Some(Tracked::new(
            indirect_history.create_view(&wgpu::TextureViewDescriptor::default()),
        ));
        self.history_meta_view = Some(Tracked::new(
            history_meta.create_view(&wgpu::TextureViewDescriptor::default()),
        ));
        self.source_history_view = Some(Tracked::new(
            source_history.create_view(&wgpu::TextureViewDescriptor::default()),
        ));

        self.full_resolution = (width, height);
        self.half_resolution = (half_w, half_h);
        self.invalidate_history();
    }

    fn ensure_pipelines(&mut self, ctx: &mut ExtractContext, max_steps: u32) {
        if !self.raw_pipelines.contains_key(&max_steps) {
            let global_state_key = (ctx.render_state.id, ctx.extracted_scene.scene_id);
            let gpu_world = ctx
                .resource_manager
                .get_global_state(global_state_key.0, global_state_key.1)
                .expect("SSGI: GpuGlobalState must exist");

            let mut options = ShaderCompilationOptions::default();
            let max_steps_define = max_steps.to_string();
            options.add_define(
                "struct_definitions",
                SsgiUniforms::wgsl_struct_def("SsgiUniforms").as_str(),
            );
            options.inject_code("binding_code", &gpu_world.binding_wgsl);
            options.inject_code(
                "scene_lighting_structs",
                myth_resources::uniforms::scene_lighting_structs_wgsl(),
            );
            options.inject_code("ssgi_max_steps", &max_steps_define);

            let (module, hash) = ctx.shader_manager.get_or_compile(
                ctx.device,
                ShaderSource::File("entry/features/ssgi/raw"),
                &options,
            );

            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("SSGI Raw Pipeline Layout"),
                        bind_group_layouts: &[Some(&gpu_world.layout), self.raw_layout.as_deref()],
                        immediate_size: 0,
                    });

            let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
                format: SSGI_TEXTURE_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            });

            let key =
                FullscreenPipelineKey::fullscreen(hash, smallvec::smallvec![color_target], None);

            let pipeline = ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                module,
                &pipeline_layout,
                &key,
                "SSGI Raw Pipeline",
            );

            self.raw_pipelines.insert(max_steps, pipeline);
        }

        if self.temporal_pipeline.is_none() {
            let mut options = ShaderCompilationOptions::default();
            options.add_define(
                "struct_definitions",
                SsgiUniforms::wgsl_struct_def("SsgiUniforms").as_str(),
            );

            let (module, hash) = ctx.shader_manager.get_or_compile(
                ctx.device,
                ShaderSource::File("entry/features/ssgi/temporal"),
                &options,
            );

            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("SSGI Temporal Pipeline Layout"),
                        bind_group_layouts: &[self.temporal_layout.as_deref()],
                        immediate_size: 0,
                    });

            let indirect_target = ColorTargetKey::from(wgpu::ColorTargetState {
                format: SSGI_TEXTURE_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            });
            let meta_target = ColorTargetKey::from(wgpu::ColorTargetState {
                format: SSGI_TEXTURE_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            });

            let key = FullscreenPipelineKey::fullscreen(
                hash,
                smallvec::smallvec![indirect_target, meta_target],
                None,
            );

            self.temporal_pipeline = Some(ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                module,
                &pipeline_layout,
                &key,
                "SSGI Temporal Pipeline",
            ));
        }

        if self.atrous_pipeline.is_none() {
            let mut options = ShaderCompilationOptions::default();
            options.add_define(
                "struct_definitions",
                SsgiUniforms::wgsl_struct_def("SsgiUniforms").as_str(),
            );

            let (module, hash) = ctx.shader_manager.get_or_compile(
                ctx.device,
                ShaderSource::File("entry/features/ssgi/atrous_blur"),
                &options,
            );

            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("SSGI A-Trous Pipeline Layout"),
                        bind_group_layouts: &[self.atrous_layout.as_deref()],
                        immediate_size: 0,
                    });

            let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
                format: SSGI_TEXTURE_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            });

            let key =
                FullscreenPipelineKey::fullscreen(hash, smallvec::smallvec![color_target], None);

            self.atrous_pipeline = Some(ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                module,
                &pipeline_layout,
                &key,
                "SSGI A-Trous Pipeline",
            ));
        }

        if self.merge_pipeline.is_none() {
            let mut options = ShaderCompilationOptions::default();
            options.add_define(
                "struct_definitions",
                SsgiUniforms::wgsl_struct_def("SsgiUniforms").as_str(),
            );

            let (module, hash) = ctx.shader_manager.get_or_compile(
                ctx.device,
                ShaderSource::File("entry/features/ssgi/merge"),
                &options,
            );

            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("SSGI Merge Pipeline Layout"),
                        bind_group_layouts: &[self.merge_layout.as_deref()],
                        immediate_size: 0,
                    });

            let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
                format: HDR_TEXTURE_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            });

            let key =
                FullscreenPipelineKey::fullscreen(hash, smallvec::smallvec![color_target], None);

            self.merge_pipeline = Some(ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                module,
                &pipeline_layout,
                &key,
                "SSGI Merge Pipeline",
            ));
        }
    }

    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        ssgi_uniforms: &CpuBuffer<SsgiUniforms>,
        size: (u32, u32),
    ) {
        self.ensure_layouts(ctx.device);
        self.ensure_history_buffers(ctx.device, size.0, size.1);

        let uniforms = *ssgi_uniforms.read();
        let max_steps = uniforms.frame_params.y;
        self.ensure_pipelines(ctx, max_steps);
        self.prepared_max_steps = max_steps;
        self.ensure_atrous_uniform_buffers(ctx, uniforms);

        ctx.resource_manager.ensure_buffer(ssgi_uniforms);

        self.black_cube_view = Some(ctx.resource_manager.system_textures().black_cube.clone());
        self.blue_noise_view = Some(ctx.resource_manager.system_textures().blue_noise.clone());
        self.blue_noise_sampler = Some(
            ctx.resource_manager
                .system_textures()
                .blue_noise_sampler
                .clone(),
        );
        self.uniforms_buffer = ssgi_uniforms.gpu_handle().and_then(|handle| {
            ctx.resource_manager
                .get_gpu_buffer(handle)
                .map(|gpu| Tracked::with_id(gpu.buffer.clone(), gpu.id))
        });
    }

    pub fn add_to_graph<'a>(
        &'a mut self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        current_color: TextureNodeId,
        scene_depth: TextureNodeId,
        scene_hiz: TextureNodeId,
        scene_normals: TextureNodeId,
        velocity: TextureNodeId,
        material_mrt: TextureNodeId,
        pmrem_tex: Option<TextureNodeId>,
        taa_history_view: Option<Tracked<wgpu::TextureView>>,
    ) -> SsgiOutputs {
        let raw_pipeline = ctx.pipeline_cache.get_render_pipeline(
            self.raw_pipelines
                .get(&self.prepared_max_steps)
                .copied()
                .expect("SSGI raw pipeline missing"),
        );
        let temporal_pipeline = ctx.pipeline_cache.get_render_pipeline(
            self.temporal_pipeline
                .expect("SSGI temporal pipeline missing"),
        );
        let atrous_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.atrous_pipeline.expect("SSGI A-Trous pipeline missing"));
        let merge_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.merge_pipeline.expect("SSGI merge pipeline missing"));

        let raw_layout = self.raw_layout.as_ref().unwrap();
        let temporal_layout = self.temporal_layout.as_ref().unwrap();
        let atrous_layout = self.atrous_layout.as_ref().unwrap();
        let merge_layout = self.merge_layout.as_ref().unwrap();

        let uniforms_buffer = self
            .uniforms_buffer
            .as_ref()
            .expect("SSGI uniforms buffer missing");
        let indirect_history_view = self.indirect_history_view.as_ref().unwrap();
        let history_meta_view = self.history_meta_view.as_ref().unwrap();
        let source_history_view = self.source_history_view.as_ref().unwrap();
        let black_cube_view = self.black_cube_view.as_ref().unwrap();
        let blue_noise_view = self.blue_noise_view.as_ref().unwrap();
        let blue_noise_sampler = self.blue_noise_sampler.as_ref().unwrap();
        let atrous_uniform_buffers = &self.atrous_uniform_buffers;

        let half_w = (ctx.frame_config.width / 2).max(1);
        let half_h = (ctx.frame_config.height / 2).max(1);
        let atrous_pass_count = self
            .prepared_atrous_passes
            .clamp(1, MAX_ATROUS_PASSES as u32) as usize;

        let taa_enabled = taa_history_view.is_some();
        let use_taa_source_history =
            self.source_history_valid && self.prev_frame_taa_enabled && taa_enabled;

        self.source_history_input_view = Some(if use_taa_source_history {
            taa_history_view.expect("SSGI TAA history view missing")
        } else {
            source_history_view.clone()
        });
        let source_history_input_view = self
            .source_history_input_view
            .as_ref()
            .expect("SSGI source history input view missing");

        let indirect_history_desc = persistent_texture_desc(
            indirect_history_view.texture().width(),
            indirect_history_view.texture().height(),
            SSGI_TEXTURE_FORMAT,
        );
        let history_meta_desc = persistent_texture_desc(
            history_meta_view.texture().width(),
            history_meta_view.texture().height(),
            SSGI_TEXTURE_FORMAT,
        );
        let source_history_desc = persistent_texture_desc(
            source_history_input_view.texture().width(),
            source_history_input_view.texture().height(),
            HDR_TEXTURE_FORMAT,
        );
        let uniforms_desc = BufferDesc::new(
            std::mem::size_of::<SsgiUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let outputs = ctx.with_group("SSGI_System", |ctx| {
            let raw_desc = TextureDesc::new_2d(
                half_w,
                half_h,
                SSGI_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );

            let raw_indirect: TextureNodeId = ctx.graph.add_pass("SSGI_Raw", |builder| {
                builder.read_texture(scene_depth);
                builder.read_texture(scene_hiz);
                builder.read_texture(scene_normals);

                let source_history = builder.read_external_texture(
                    "SSGI_Source_History_Read",
                    source_history_desc,
                    source_history_input_view,
                );
                let uniforms =
                    builder.read_external_buffer("SSGI_Uniforms", uniforms_desc, uniforms_buffer);
                let pmrem_input = if let Some(pmrem) = pmrem_tex {
                    builder.read_texture(pmrem)
                } else {
                    builder.read_external_texture(
                        "SSGI_PMREM_Fallback",
                        TextureDesc::new(
                            1,
                            1,
                            6,
                            1,
                            1,
                            wgpu::TextureDimension::D2,
                            wgpu::TextureFormat::Rgba8Unorm,
                            wgpu::TextureUsages::TEXTURE_BINDING,
                        ),
                        black_cube_view,
                    )
                };

                let out = builder.create_texture("SSGI_Raw_Indirect", raw_desc);
                let node = SsgiRawNode {
                    scene_depth,
                    scene_hiz,
                    scene_normals,
                    source_history,
                    pmrem_texture: pmrem_input,
                    uniforms,
                    blue_noise_view,
                    blue_noise_sampler,
                    output_tex: out,
                    pipeline: raw_pipeline,
                    layout: raw_layout,
                    transient_bg: None,
                };

                (node, out)
            });

            let temporal_desc = TextureDesc::new_2d(
                half_w,
                half_h,
                SSGI_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            );

            let (temporal_indirect, temporal_meta) =
                ctx.graph.add_pass("SSGI_Temporal", |builder| {
                    let indirect_history = builder.read_external_texture(
                        "SSGI_History_Indirect_Read",
                        indirect_history_desc,
                        indirect_history_view,
                    );
                    let history_meta = builder.read_external_texture(
                        "SSGI_History_Meta_Read",
                        history_meta_desc,
                        history_meta_view,
                    );

                    builder.read_texture(raw_indirect);
                    builder.read_texture(scene_depth);
                    builder.read_texture(scene_normals);
                    builder.read_texture(velocity);

                    let uniforms = builder.read_external_buffer(
                        "SSGI_Uniforms",
                        uniforms_desc,
                        uniforms_buffer,
                    );

                    let temporal_indirect =
                        builder.create_texture("SSGI_Temporal_Indirect", temporal_desc);
                    let temporal_meta = builder.create_texture("SSGI_Temporal_Meta", temporal_desc);

                    let node = SsgiTemporalNode {
                        raw_indirect,
                        indirect_history,
                        scene_depth,
                        scene_normals,
                        history_meta,
                        velocity,
                        uniforms,
                        output_indirect: temporal_indirect,
                        output_meta: temporal_meta,
                        pipeline: temporal_pipeline,
                        layout: temporal_layout,
                        transient_bg: None,
                    };

                    (node, (temporal_indirect, temporal_meta))
                });

            let atrous_desc = TextureDesc::new_2d(
                half_w,
                half_h,
                SSGI_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );

            let mut clean_indirect = temporal_indirect;
            for pass_index in 0..atrous_pass_count {
                let pass_name = SSGI_ATROUS_PASS_NAMES[pass_index];
                let output_name = SSGI_ATROUS_OUTPUT_NAMES[pass_index];
                let uniform_name = SSGI_ATROUS_UNIFORM_NAMES[pass_index];
                let pass_uniform_buffer = atrous_uniform_buffers
                    .get(pass_index)
                    .expect("SSGI A-Trous uniform buffer missing");

                clean_indirect = ctx.graph.add_pass(pass_name, |builder| {
                    builder.read_texture(clean_indirect);
                    builder.read_texture(scene_depth);
                    builder.read_texture(scene_normals);
                    builder.read_texture(temporal_meta);
                    let uniforms = builder.read_external_buffer(
                        uniform_name,
                        uniforms_desc,
                        pass_uniform_buffer,
                    );

                    let out = builder.create_texture(output_name, atrous_desc);
                    let node = SsgiAtrousNode {
                        input_indirect: clean_indirect,
                        scene_depth,
                        scene_normals,
                        variance_meta: temporal_meta,
                        uniforms,
                        output_tex: out,
                        pipeline: atrous_pipeline,
                        layout: atrous_layout,
                        transient_bg: None,
                    };

                    (node, out)
                });
            }

            let merged_desc = TextureDesc::new_2d(
                ctx.frame_config.width,
                ctx.frame_config.height,
                HDR_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            );

            let merged_color: TextureNodeId = ctx.graph.add_pass("SSGI_Merge", |builder| {
                builder.read_texture(current_color);
                builder.read_texture(clean_indirect);
                builder.read_texture(material_mrt);
                builder.read_texture(scene_depth);
                builder.read_texture(scene_normals);
                let uniforms =
                    builder.read_external_buffer("SSGI_Uniforms", uniforms_desc, uniforms_buffer);

                let out = builder.create_texture("SSGI_Merged_Color", merged_desc);
                let node = SsgiMergeNode {
                    current_color,
                    clean_indirect,
                    material_mrt,
                    scene_depth,
                    scene_normals,
                    uniforms,
                    output_tex: out,
                    pipeline: merge_pipeline,
                    layout: merge_layout,
                    transient_bg: None,
                };

                (node, out)
            });

            ctx.graph.add_pass("SSGI_Save_History_Indirect", |builder| {
                builder.read_texture(temporal_indirect);
                let history_out = builder.write_external_texture(
                    "SSGI_History_Indirect_Write",
                    indirect_history_desc,
                    indirect_history_view,
                );
                (
                    CopyTextureNode {
                        src: temporal_indirect,
                        dst: history_out,
                    },
                    (),
                )
            });

            ctx.graph.add_pass("SSGI_Save_History_Meta", |builder| {
                builder.read_texture(temporal_meta);
                let history_out = builder.write_external_texture(
                    "SSGI_History_Meta_Write",
                    history_meta_desc,
                    history_meta_view,
                );
                (
                    CopyTextureNode {
                        src: temporal_meta,
                        dst: history_out,
                    },
                    (),
                )
            });

            ctx.graph.add_pass("SSGI_Save_Source_History", |builder| {
                builder.read_texture(merged_color);
                let history_out = builder.write_external_texture(
                    "SSGI_Source_History_Write",
                    persistent_texture_desc(
                        source_history_view.texture().width(),
                        source_history_view.texture().height(),
                        HDR_TEXTURE_FORMAT,
                    ),
                    source_history_view,
                );
                (
                    CopyTextureNode {
                        src: merged_color,
                        dst: history_out,
                    },
                    (),
                )
            });

            SsgiOutputs {
                merged_color,
                raw_indirect,
                clean_indirect,
            }
        });

        self.history_valid = true;
        self.source_history_valid = true;
        self.prev_frame_taa_enabled = taa_enabled;
        outputs
    }
}

struct SsgiRawNode<'a> {
    scene_depth: TextureNodeId,
    scene_hiz: TextureNodeId,
    scene_normals: TextureNodeId,
    source_history: TextureNodeId,
    pmrem_texture: TextureNodeId,
    uniforms: BufferNodeId,
    blue_noise_view: &'a Tracked<wgpu::TextureView>,
    blue_noise_sampler: &'a Tracked<wgpu::Sampler>,
    output_tex: TextureNodeId,
    pipeline: &'a wgpu::RenderPipeline,
    layout: &'a Tracked<wgpu::BindGroupLayout>,
    transient_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for SsgiRawNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let builder = ctx
            .build_bind_group(self.layout, Some("SSGI Raw BG"))
            .bind_texture(0, self.scene_depth)
            .bind_texture(1, self.scene_normals)
            .bind_texture(2, self.source_history)
            .bind_texture(3, self.scene_hiz)
            .bind_texture(4, self.pmrem_texture)
            .bind_common_sampler(5, CommonSampler::LinearClamp)
            .bind_common_sampler(6, CommonSampler::NearestClamp)
            .bind_buffer(7, self.uniforms)
            .bind_tracked_texture_view(8, self.blue_noise_view)
            .bind_tracked_sampler(9, self.blue_noise_sampler);

        self.transient_bg = Some(builder.build());
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let output = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);
        let global_bg = ctx.baked_lists.global_bind_group;

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSGI Raw Pass"),
            color_attachments: &[output],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(self.pipeline);
        rpass.set_bind_group(0, global_bg, &[]);
        rpass.set_bind_group(1, self.transient_bg.expect("SSGI raw BG missing"), &[]);
        rpass.draw(0..3, 0..1);
    }
}

struct SsgiTemporalNode<'a> {
    raw_indirect: TextureNodeId,
    indirect_history: TextureNodeId,
    scene_depth: TextureNodeId,
    scene_normals: TextureNodeId,
    history_meta: TextureNodeId,
    velocity: TextureNodeId,
    uniforms: BufferNodeId,
    output_indirect: TextureNodeId,
    output_meta: TextureNodeId,
    pipeline: &'a wgpu::RenderPipeline,
    layout: &'a Tracked<wgpu::BindGroupLayout>,
    transient_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for SsgiTemporalNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.transient_bg = Some(
            crate::myth_bind_group!(ctx, self.layout, Some("SSGI Temporal BG"), [
                0 => self.raw_indirect,
                1 => self.indirect_history,
                2 => self.scene_depth,
                3 => self.scene_normals,
                4 => self.history_meta,
                5 => self.velocity,
                6 => CommonSampler::LinearClamp,
                7 => CommonSampler::NearestClamp,
                8 => self.uniforms,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let attachments = [
            ctx.get_color_attachment(self.output_indirect, RenderTargetOps::DontCare, None),
            ctx.get_color_attachment(self.output_meta, RenderTargetOps::DontCare, None),
        ];

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSGI Temporal Pass"),
            color_attachments: &attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(self.pipeline);
        rpass.set_bind_group(0, self.transient_bg.expect("SSGI temporal BG missing"), &[]);
        rpass.draw(0..3, 0..1);
    }
}

struct SsgiAtrousNode<'a> {
    input_indirect: TextureNodeId,
    scene_depth: TextureNodeId,
    scene_normals: TextureNodeId,
    variance_meta: TextureNodeId,
    uniforms: BufferNodeId,
    output_tex: TextureNodeId,
    pipeline: &'a wgpu::RenderPipeline,
    layout: &'a Tracked<wgpu::BindGroupLayout>,
    transient_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for SsgiAtrousNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.transient_bg = Some(
            crate::myth_bind_group!(ctx, self.layout, Some("SSGI A-Trous BG"), [
                0 => self.input_indirect,
                1 => self.scene_depth,
                2 => self.scene_normals,
                3 => self.variance_meta,
                4 => CommonSampler::LinearClamp,
                5 => CommonSampler::NearestClamp,
                6 => self.uniforms,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let output = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSGI A-Trous Pass"),
            color_attachments: &[output],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(self.pipeline);
        rpass.set_bind_group(0, self.transient_bg.expect("SSGI A-Trous BG missing"), &[]);
        rpass.draw(0..3, 0..1);
    }
}

struct SsgiMergeNode<'a> {
    current_color: TextureNodeId,
    clean_indirect: TextureNodeId,
    material_mrt: TextureNodeId,
    scene_depth: TextureNodeId,
    scene_normals: TextureNodeId,
    uniforms: BufferNodeId,
    output_tex: TextureNodeId,
    pipeline: &'a wgpu::RenderPipeline,
    layout: &'a Tracked<wgpu::BindGroupLayout>,
    transient_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for SsgiMergeNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.transient_bg = Some(
            crate::myth_bind_group!(ctx, self.layout, Some("SSGI Merge BG"), [
                0 => self.current_color,
                1 => self.clean_indirect,
                2 => self.material_mrt,
                3 => self.scene_depth,
                4 => self.scene_normals,
                5 => CommonSampler::LinearClamp,
                6 => CommonSampler::NearestClamp,
                7 => self.uniforms,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let output = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSGI Merge Pass"),
            color_attachments: &[output],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(self.pipeline);
        rpass.set_bind_group(0, self.transient_bg.expect("SSGI merge BG missing"), &[]);
        rpass.draw(0..3, 0..1);
    }
}

#[allow(dead_code)]
fn pyramid_mip_count(width: u32, height: u32) -> u32 {
    width.max(height).max(1).ilog2() + 1
}

fn persistent_texture_desc(width: u32, height: u32, format: wgpu::TextureFormat) -> TextureDesc {
    TextureDesc::new_2d(
        width,
        height,
        format,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    )
}
