//! Screen Space Reflections feature.
//!
//! SSR is implemented as a scene-level post process with four explicit stages:
//! raw Hi-Z tracing, temporal accumulation, roughness-aware spatial cleanup,
//! and a specular replacement merge back into the HDR scene color.

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
use myth_resources::ssr::SsrUniforms;
use myth_resources::uniforms::WgslStruct;

const SSR_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const SSR_HISTORY_FLAG_VALID: u32 = 1 << 0;

fn ssr_trace_resolution(width: u32, height: u32, half_resolution: bool) -> (u32, u32) {
    if half_resolution {
        ((width / 2).max(1), (height / 2).max(1))
    } else {
        (width.max(1), height.max(1))
    }
}

fn blue_noise_view_dimension() -> wgpu::TextureViewDimension {
    if cfg!(feature = "advanced_noise") {
        wgpu::TextureViewDimension::D2Array
    } else {
        wgpu::TextureViewDimension::D2
    }
}

#[must_use = "SSA Graph: consume the merged color output from SSR"]
pub struct SsrOutputs {
    pub merged_color: TextureNodeId,
    #[cfg(feature = "debug_view")]
    pub raw_reflection: TextureNodeId,
    #[cfg(feature = "debug_view")]
    pub clean_reflection: TextureNodeId,
    #[cfg(feature = "debug_view")]
    pub raw_diagnostic: Option<TextureNodeId>,
}

pub struct SsrFeature {
    raw_pipelines: FxHashMap<(u32, bool), RenderPipelineId>,
    temporal_pipeline: Option<RenderPipelineId>,
    spatial_pipeline: Option<RenderPipelineId>,
    merge_pipeline: Option<RenderPipelineId>,

    raw_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    temporal_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    spatial_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    merge_layout: Option<Tracked<wgpu::BindGroupLayout>>,

    uniforms_buffer: Option<Tracked<wgpu::Buffer>>,
    blue_noise_view: Option<Tracked<wgpu::TextureView>>,
    blue_noise_sampler: Option<Tracked<wgpu::Sampler>>,
    reflection_history_view: Option<Tracked<wgpu::TextureView>>,
    history_meta_view: Option<Tracked<wgpu::TextureView>>,

    prepared_max_steps: u32,
    prepared_trace_diagnostics: bool,
    trace_resolution: (u32, u32),
    history_valid: bool,
}

impl Default for SsrFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl SsrFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            raw_pipelines: FxHashMap::default(),
            temporal_pipeline: None,
            spatial_pipeline: None,
            merge_pipeline: None,
            raw_layout: None,
            temporal_layout: None,
            spatial_layout: None,
            merge_layout: None,
            uniforms_buffer: None,
            blue_noise_view: None,
            blue_noise_sampler: None,
            reflection_history_view: None,
            history_meta_view: None,
            prepared_max_steps: 24,
            prepared_trace_diagnostics: false,
            trace_resolution: (0, 0),
            history_valid: false,
        }
    }

    #[must_use]
    pub fn history_flags(&self) -> u32 {
        u32::from(self.history_valid)
    }

    pub fn invalidate_history(&mut self) {
        self.history_valid = false;
    }

    fn sync_history_flags(&self, ssr_uniforms: &CpuBuffer<SsrUniforms>) {
        let current_flags = ssr_uniforms.read().frame_params.w;
        let merged_flags = (current_flags & !SSR_HISTORY_FLAG_VALID) | self.history_flags();

        if current_flags != merged_flags {
            ssr_uniforms.write().frame_params.w = merged_flags;
        }
    }

    fn ensure_layouts(&mut self, device: &wgpu::Device) {
        if self.raw_layout.is_some() {
            return;
        }

        let raw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR Trace Layout"),
            entries: &[
                texture_entry(0, wgpu::TextureSampleType::Depth),
                texture_entry(1, wgpu::TextureSampleType::Float { filterable: true }),
                texture_entry(2, wgpu::TextureSampleType::Float { filterable: true }),
                texture_entry(3, wgpu::TextureSampleType::Float { filterable: false }),
                texture_entry(4, wgpu::TextureSampleType::Float { filterable: true }),
                texture_entry(5, wgpu::TextureSampleType::Float { filterable: true }),
                sampler_entry(6, wgpu::SamplerBindingType::Filtering),
                sampler_entry(7, wgpu::SamplerBindingType::NonFiltering),
                uniform_entry(8),
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: blue_noise_view_dimension(),
                        multisampled: false,
                    },
                    count: None,
                },
                sampler_entry(10, wgpu::SamplerBindingType::Filtering),
            ],
        });

        let temporal_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR Temporal Layout"),
            entries: &[
                texture_entry(0, wgpu::TextureSampleType::Float { filterable: true }),
                texture_entry(1, wgpu::TextureSampleType::Float { filterable: true }),
                texture_entry(2, wgpu::TextureSampleType::Depth),
                texture_entry(3, wgpu::TextureSampleType::Float { filterable: true }),
                texture_entry(4, wgpu::TextureSampleType::Float { filterable: true }),
                texture_entry(5, wgpu::TextureSampleType::Float { filterable: false }),
                texture_entry(6, wgpu::TextureSampleType::Float { filterable: true }),
                sampler_entry(7, wgpu::SamplerBindingType::Filtering),
                sampler_entry(8, wgpu::SamplerBindingType::NonFiltering),
                uniform_entry(9),
            ],
        });

        let spatial_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR Spatial Layout"),
            entries: &[
                texture_entry(0, wgpu::TextureSampleType::Float { filterable: true }),
                texture_entry(1, wgpu::TextureSampleType::Depth),
                texture_entry(2, wgpu::TextureSampleType::Float { filterable: true }),
                texture_entry(3, wgpu::TextureSampleType::Float { filterable: true }),
                sampler_entry(4, wgpu::SamplerBindingType::Filtering),
                sampler_entry(5, wgpu::SamplerBindingType::NonFiltering),
                uniform_entry(6),
            ],
        });

        let merge_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR Merge Layout"),
            entries: &[
                texture_entry(0, wgpu::TextureSampleType::Float { filterable: true }),
                texture_entry(1, wgpu::TextureSampleType::Float { filterable: true }),
                texture_entry(2, wgpu::TextureSampleType::Float { filterable: true }),
                texture_entry(3, wgpu::TextureSampleType::Float { filterable: true }),
                texture_entry(4, wgpu::TextureSampleType::Depth),
                texture_entry(5, wgpu::TextureSampleType::Float { filterable: true }),
                sampler_entry(6, wgpu::SamplerBindingType::Filtering),
                sampler_entry(7, wgpu::SamplerBindingType::NonFiltering),
                uniform_entry(8),
            ],
        });

        self.raw_layout = Some(Tracked::new(raw_layout));
        self.temporal_layout = Some(Tracked::new(temporal_layout));
        self.spatial_layout = Some(Tracked::new(spatial_layout));
        self.merge_layout = Some(Tracked::new(merge_layout));
    }

    fn ensure_history_buffers(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        half_resolution: bool,
    ) {
        let trace_resolution = ssr_trace_resolution(width, height, half_resolution);
        if self.trace_resolution == trace_resolution
            && self.reflection_history_view.is_some()
            && self.history_meta_view.is_some()
        {
            return;
        }

        let trace_extent = wgpu::Extent3d {
            width: trace_resolution.0,
            height: trace_resolution.1,
            depth_or_array_layers: 1,
        };

        let reflection_history = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSR History Reflection"),
            size: trace_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SSR_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let history_meta = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSR History Meta"),
            size: trace_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SSR_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.reflection_history_view = Some(Tracked::new(
            reflection_history.create_view(&wgpu::TextureViewDescriptor::default()),
        ));
        self.history_meta_view = Some(Tracked::new(
            history_meta.create_view(&wgpu::TextureViewDescriptor::default()),
        ));

        self.trace_resolution = trace_resolution;
        self.invalidate_history();
    }

    fn ensure_pipelines(
        &mut self,
        ctx: &mut ExtractContext,
        max_steps: u32,
        trace_diagnostics: bool,
    ) {
        let raw_key = (max_steps, trace_diagnostics);
        if !self.raw_pipelines.contains_key(&raw_key) {
            let global_state_key = (ctx.render_state.id, ctx.extracted_scene.scene_id);
            let gpu_world = ctx
                .resource_manager
                .get_global_state(global_state_key.0, global_state_key.1)
                .expect("SSR: GpuGlobalState must exist");

            let mut options = ShaderCompilationOptions::default();
            let max_steps_define = max_steps.to_string();
            options.add_define(
                "struct_definitions",
                SsrUniforms::wgsl_struct_def("SsrUniforms").as_str(),
            );
            options.inject_code("binding_code", &gpu_world.binding_wgsl);
            options.inject_code(
                "scene_lighting_structs",
                myth_resources::uniforms::scene_lighting_structs_wgsl(),
            );
            options.inject_code("ssr_max_steps", &max_steps_define);
            if trace_diagnostics {
                options.add_define("SSR_TRACE_DIAGNOSTICS", "1");
            }

            let (module, hash) = ctx.shader_manager.get_or_compile(
                ctx.device,
                ShaderSource::File("entry/features/ssr/trace"),
                &options,
            );

            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("SSR Trace Pipeline Layout"),
                        bind_group_layouts: &[Some(&gpu_world.layout), self.raw_layout.as_deref()],
                        immediate_size: 0,
                    });

            let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
                format: SSR_TEXTURE_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            });
            let key = if trace_diagnostics {
                let diagnostic_target = ColorTargetKey::from(wgpu::ColorTargetState {
                    format: SSR_TEXTURE_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                });

                FullscreenPipelineKey::fullscreen(
                    hash,
                    smallvec::smallvec![color_target, diagnostic_target],
                    None,
                )
            } else {
                FullscreenPipelineKey::fullscreen(hash, smallvec::smallvec![color_target], None)
            };

            let pipeline = ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                module,
                &pipeline_layout,
                &key,
                "SSR Trace Pipeline",
            );

            self.raw_pipelines.insert(raw_key, pipeline);
        }

        if self.temporal_pipeline.is_none() {
            let global_state_key = (ctx.render_state.id, ctx.extracted_scene.scene_id);
            let gpu_world = ctx
                .resource_manager
                .get_global_state(global_state_key.0, global_state_key.1)
                .expect("SSR: GpuGlobalState must exist");

            let mut options = ShaderCompilationOptions::default();
            options.add_define(
                "struct_definitions",
                SsrUniforms::wgsl_struct_def("SsrUniforms").as_str(),
            );
            options.inject_code("binding_code", &gpu_world.binding_wgsl);
            options.inject_code(
                "scene_lighting_structs",
                myth_resources::uniforms::scene_lighting_structs_wgsl(),
            );

            let (module, hash) = ctx.shader_manager.get_or_compile(
                ctx.device,
                ShaderSource::File("entry/features/ssr/temporal"),
                &options,
            );

            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("SSR Temporal Pipeline Layout"),
                        bind_group_layouts: &[
                            Some(&gpu_world.layout),
                            self.temporal_layout.as_deref(),
                        ],
                        immediate_size: 0,
                    });

            let reflection_target = ColorTargetKey::from(wgpu::ColorTargetState {
                format: SSR_TEXTURE_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            });
            let meta_target = ColorTargetKey::from(wgpu::ColorTargetState {
                format: SSR_TEXTURE_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            });

            let key = FullscreenPipelineKey::fullscreen(
                hash,
                smallvec::smallvec![reflection_target, meta_target],
                None,
            );

            self.temporal_pipeline = Some(ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                module,
                &pipeline_layout,
                &key,
                "SSR Temporal Pipeline",
            ));
        }

        if self.spatial_pipeline.is_none() {
            let mut options = ShaderCompilationOptions::default();
            options.add_define(
                "struct_definitions",
                SsrUniforms::wgsl_struct_def("SsrUniforms").as_str(),
            );

            let (module, hash) = ctx.shader_manager.get_or_compile(
                ctx.device,
                ShaderSource::File("entry/features/ssr/spatial_filter"),
                &options,
            );

            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("SSR Spatial Pipeline Layout"),
                        bind_group_layouts: &[self.spatial_layout.as_deref()],
                        immediate_size: 0,
                    });

            let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
                format: SSR_TEXTURE_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            });

            let key =
                FullscreenPipelineKey::fullscreen(hash, smallvec::smallvec![color_target], None);

            self.spatial_pipeline = Some(ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                module,
                &pipeline_layout,
                &key,
                "SSR Spatial Pipeline",
            ));
        }

        if self.merge_pipeline.is_none() {
            let global_state_key = (ctx.render_state.id, ctx.extracted_scene.scene_id);
            let gpu_world = ctx
                .resource_manager
                .get_global_state(global_state_key.0, global_state_key.1)
                .expect("SSR: GpuGlobalState must exist");

            let mut options = ShaderCompilationOptions::default();
            options.add_define(
                "struct_definitions",
                SsrUniforms::wgsl_struct_def("SsrUniforms").as_str(),
            );
            options.inject_code("binding_code", &gpu_world.binding_wgsl);
            options.inject_code(
                "scene_lighting_structs",
                myth_resources::uniforms::scene_lighting_structs_wgsl(),
            );

            let (module, hash) = ctx.shader_manager.get_or_compile(
                ctx.device,
                ShaderSource::File("entry/features/ssr/merge"),
                &options,
            );

            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("SSR Merge Pipeline Layout"),
                        bind_group_layouts: &[
                            Some(&gpu_world.layout),
                            self.merge_layout.as_deref(),
                        ],
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
                "SSR Merge Pipeline",
            ));
        }
    }

    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        ssr_uniforms: &CpuBuffer<SsrUniforms>,
        size: (u32, u32),
    ) {
        self.ensure_layouts(ctx.device);

        let uniforms = *ssr_uniforms.read();
        let half_resolution = uniforms.denoise_params.y != 0;
        self.ensure_history_buffers(ctx.device, size.0, size.1, half_resolution);
        self.sync_history_flags(ssr_uniforms);

        let max_steps = uniforms.frame_params.y;
        #[cfg(feature = "debug_view")]
        let trace_diagnostics = crate::graph::render_state::DebugViewTarget::from_mode(
            ctx.render_state.debug_view_mode,
        ) == crate::graph::render_state::DebugViewTarget::SsrTraceDebug;
        #[cfg(not(feature = "debug_view"))]
        let trace_diagnostics = false;

        self.ensure_pipelines(ctx, max_steps, trace_diagnostics);
        self.prepared_max_steps = max_steps;
        self.prepared_trace_diagnostics = trace_diagnostics;

        ctx.resource_manager.ensure_buffer(ssr_uniforms);

        self.blue_noise_view = Some(ctx.resource_manager.system_textures().blue_noise.clone());
        self.blue_noise_sampler = Some(
            ctx.resource_manager
                .system_textures()
                .blue_noise_sampler
                .clone(),
        );
        self.uniforms_buffer = ssr_uniforms.gpu_handle().and_then(|handle| {
            ctx.resource_manager
                .get_gpu_buffer(handle)
                .map(|gpu| Tracked::with_id(gpu.buffer.clone(), gpu.id))
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_to_graph<'a>(
        &'a mut self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        current_color: TextureNodeId,
        scene_depth: TextureNodeId,
        scene_hiz: TextureNodeId,
        scene_normals: TextureNodeId,
        velocity: TextureNodeId,
        material_mrt: TextureNodeId,
        specular_mrt: TextureNodeId,
    ) -> SsrOutputs {
        let trace_diagnostics_enabled = self.prepared_trace_diagnostics;
        let raw_pipeline = ctx.pipeline_cache.get_render_pipeline(
            self.raw_pipelines
                .get(&(self.prepared_max_steps, trace_diagnostics_enabled))
                .copied()
                .expect("SSR trace pipeline missing"),
        );
        let temporal_pipeline = ctx.pipeline_cache.get_render_pipeline(
            self.temporal_pipeline
                .expect("SSR temporal pipeline missing"),
        );
        let spatial_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.spatial_pipeline.expect("SSR spatial pipeline missing"));
        let merge_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.merge_pipeline.expect("SSR merge pipeline missing"));

        let raw_layout = self.raw_layout.as_ref().unwrap();
        let temporal_layout = self.temporal_layout.as_ref().unwrap();
        let spatial_layout = self.spatial_layout.as_ref().unwrap();
        let merge_layout = self.merge_layout.as_ref().unwrap();

        let uniforms_buffer = self
            .uniforms_buffer
            .as_ref()
            .expect("SSR uniforms buffer missing");
        let blue_noise_view = self.blue_noise_view.as_ref().unwrap();
        let blue_noise_sampler = self.blue_noise_sampler.as_ref().unwrap();
        let reflection_history_view = self.reflection_history_view.as_ref().unwrap();
        let history_meta_view = self.history_meta_view.as_ref().unwrap();
        let trace_width = reflection_history_view.texture().width();
        let trace_height = reflection_history_view.texture().height();

        let reflection_history_desc = persistent_texture_desc(
            reflection_history_view.texture().width(),
            reflection_history_view.texture().height(),
            SSR_TEXTURE_FORMAT,
        );
        let history_meta_desc = persistent_texture_desc(
            history_meta_view.texture().width(),
            history_meta_view.texture().height(),
            SSR_TEXTURE_FORMAT,
        );
        let uniforms_desc = BufferDesc::new(
            std::mem::size_of::<SsrUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let outputs = ctx.with_group("SSR_System", |ctx| {
            let raw_desc = TextureDesc::new_2d(
                trace_width,
                trace_height,
                SSR_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );

            let (raw_reflection, raw_diagnostic_opt): (TextureNodeId, Option<TextureNodeId>) =
                ctx.graph.add_pass("SSR_Trace", |builder| {
                    builder.read_texture(scene_depth);
                    builder.read_texture(scene_normals);
                    builder.read_texture(current_color);
                    builder.read_texture(scene_hiz);
                    builder.read_texture(material_mrt);
                    builder.read_texture(specular_mrt);

                    let uniforms = builder.read_external_buffer(
                        "SSR_Uniforms",
                        uniforms_desc,
                        uniforms_buffer,
                    );
                    let out = builder.create_texture("SSR_Raw_Reflection", raw_desc);
                    let diag_out = trace_diagnostics_enabled
                        .then(|| builder.create_texture("SSR_Raw_Diagnostic", raw_desc));
                    let node = SsrTraceNode {
                        scene_depth,
                        scene_normals,
                        source_color: current_color,
                        scene_hiz,
                        material_mrt,
                        specular_mrt,
                        uniforms,
                        blue_noise_view,
                        blue_noise_sampler,
                        output_reflection: out,
                        output_diagnostic: diag_out,
                        pipeline: raw_pipeline,
                        layout: raw_layout,
                        transient_bg: None,
                    };

                    (node, (out, diag_out))
                });

            let temporal_desc = TextureDesc::new_2d(
                trace_width,
                trace_height,
                SSR_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            );

            let (temporal_reflection, temporal_meta) =
                ctx.graph.add_pass("SSR_Temporal", |builder| {
                    let reflection_history = builder.read_external_texture(
                        "SSR_History_Reflection_Read",
                        reflection_history_desc,
                        reflection_history_view,
                    );
                    let history_meta = builder.read_external_texture(
                        "SSR_History_Meta_Read",
                        history_meta_desc,
                        history_meta_view,
                    );

                    builder.read_texture(raw_reflection);
                    builder.read_texture(scene_depth);
                    builder.read_texture(scene_normals);
                    builder.read_texture(velocity);
                    builder.read_texture(material_mrt);

                    let uniforms = builder.read_external_buffer(
                        "SSR_Uniforms",
                        uniforms_desc,
                        uniforms_buffer,
                    );

                    let temporal_reflection =
                        builder.create_texture("SSR_Temporal_Reflection", temporal_desc);
                    let temporal_meta = builder.create_texture("SSR_Temporal_Meta", temporal_desc);

                    let node = SsrTemporalNode {
                        raw_reflection,
                        reflection_history,
                        scene_depth,
                        scene_normals,
                        history_meta,
                        velocity,
                        material_mrt,
                        uniforms,
                        output_reflection: temporal_reflection,
                        output_meta: temporal_meta,
                        pipeline: temporal_pipeline,
                        layout: temporal_layout,
                        transient_bg: None,
                    };

                    (node, (temporal_reflection, temporal_meta))
                });

            let clean_desc = TextureDesc::new_2d(
                trace_width,
                trace_height,
                SSR_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            );

            let clean_reflection: TextureNodeId = ctx.graph.add_pass("SSR_Resolve", |builder| {
                builder.read_texture(temporal_reflection);
                builder.read_texture(scene_depth);
                builder.read_texture(scene_normals);
                builder.read_texture(material_mrt);

                let uniforms =
                    builder.read_external_buffer("SSR_Uniforms", uniforms_desc, uniforms_buffer);
                let out = builder.create_texture("SSR_Clean_Reflection", clean_desc);

                let node = SsrSpatialNode {
                    input_reflection: temporal_reflection,
                    scene_depth,
                    scene_normals,
                    material_mrt,
                    uniforms,
                    output_tex: out,
                    pipeline: spatial_pipeline,
                    layout: spatial_layout,
                    transient_bg: None,
                };

                (node, out)
            });

            let merged_desc = TextureDesc::new_2d(
                ctx.frame_config.width,
                ctx.frame_config.height,
                HDR_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            );

            let merged_color: TextureNodeId = ctx.graph.add_pass("SSR_Merge", |builder| {
                builder.read_texture(current_color);
                builder.read_texture(clean_reflection);
                builder.read_texture(material_mrt);
                builder.read_texture(specular_mrt);
                builder.read_texture(scene_depth);
                builder.read_texture(scene_normals);
                let uniforms =
                    builder.read_external_buffer("SSR_Uniforms", uniforms_desc, uniforms_buffer);

                let out = builder.create_texture("SSR_Merged_Color", merged_desc);
                let node = SsrMergeNode {
                    current_color,
                    clean_reflection,
                    material_mrt,
                    specular_mrt,
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

            ctx.graph
                .add_pass("SSR_Save_History_Reflection", |builder| {
                    builder.read_texture(temporal_reflection);
                    let history_out = builder.write_external_texture(
                        "SSR_History_Reflection_Write",
                        reflection_history_desc,
                        reflection_history_view,
                    );
                    (
                        CopyTextureNode {
                            src: temporal_reflection,
                            dst: history_out,
                        },
                        (),
                    )
                });

            ctx.graph.add_pass("SSR_Save_History_Meta", |builder| {
                builder.read_texture(temporal_meta);
                let history_out = builder.write_external_texture(
                    "SSR_History_Meta_Write",
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

            #[cfg(not(feature = "debug_view"))]
            let _ = raw_diagnostic_opt;

            SsrOutputs {
                merged_color,
                #[cfg(feature = "debug_view")]
                raw_reflection,
                #[cfg(feature = "debug_view")]
                clean_reflection,
                #[cfg(feature = "debug_view")]
                raw_diagnostic: raw_diagnostic_opt,
            }
        });

        self.history_valid = true;
        outputs
    }
}

#[cfg(test)]
mod tests {
    use super::{SSR_HISTORY_FLAG_VALID, SsrFeature, ssr_trace_resolution};
    use myth_resources::ssr::SsrSettings;

    const CAMERA_CUT_FLAG: u32 = 1 << 1;

    #[test]
    fn computes_trace_resolution_from_quality_budget() {
        assert_eq!(ssr_trace_resolution(1, 1, true), (1, 1));
        assert_eq!(ssr_trace_resolution(1920, 1080, true), (960, 540));
        assert_eq!(ssr_trace_resolution(2560, 1440, false), (2560, 1440));
    }

    #[test]
    fn sync_history_flags_clears_stale_history_bit() {
        let mut feature = SsrFeature::new();
        let mut settings = SsrSettings::default();

        settings.set_history_flags(SSR_HISTORY_FLAG_VALID | CAMERA_CUT_FLAG);
        feature.invalidate_history();
        feature.sync_history_flags(&settings.uniforms);

        assert_eq!(settings.uniforms.read().frame_params.w, CAMERA_CUT_FLAG);
    }

    #[test]
    fn sync_history_flags_preserves_external_runtime_bits() {
        let mut feature = SsrFeature::new();
        let mut settings = SsrSettings::default();

        settings.set_history_flags(CAMERA_CUT_FLAG);
        feature.history_valid = true;
        feature.sync_history_flags(&settings.uniforms);

        assert_eq!(
            settings.uniforms.read().frame_params.w,
            SSR_HISTORY_FLAG_VALID | CAMERA_CUT_FLAG
        );
    }
}

struct SsrTraceNode<'a> {
    scene_depth: TextureNodeId,
    scene_normals: TextureNodeId,
    source_color: TextureNodeId,
    scene_hiz: TextureNodeId,
    material_mrt: TextureNodeId,
    specular_mrt: TextureNodeId,
    uniforms: BufferNodeId,
    blue_noise_view: &'a Tracked<wgpu::TextureView>,
    blue_noise_sampler: &'a Tracked<wgpu::Sampler>,
    output_reflection: TextureNodeId,
    output_diagnostic: Option<TextureNodeId>,
    pipeline: &'a wgpu::RenderPipeline,
    layout: &'a Tracked<wgpu::BindGroupLayout>,
    transient_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for SsrTraceNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let builder = ctx
            .build_bind_group(self.layout, Some("SSR Trace BG"))
            .bind_texture(0, self.scene_depth)
            .bind_texture(1, self.scene_normals)
            .bind_texture(2, self.source_color)
            .bind_texture(3, self.scene_hiz)
            .bind_texture(4, self.material_mrt)
            .bind_texture(5, self.specular_mrt)
            .bind_common_sampler(6, CommonSampler::LinearClamp)
            .bind_common_sampler(7, CommonSampler::NearestClamp)
            .bind_buffer(8, self.uniforms)
            .bind_tracked_texture_view(9, self.blue_noise_view)
            .bind_tracked_sampler(10, self.blue_noise_sampler);

        self.transient_bg = Some(builder.build());
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let attachments = if let Some(output_diagnostic) = self.output_diagnostic {
            vec![
                ctx.get_color_attachment(self.output_reflection, RenderTargetOps::DontCare, None),
                ctx.get_color_attachment(output_diagnostic, RenderTargetOps::DontCare, None),
            ]
        } else {
            vec![ctx.get_color_attachment(self.output_reflection, RenderTargetOps::DontCare, None)]
        };
        let global_bg = ctx.baked_lists.global_bind_group;

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSR Trace Pass"),
            color_attachments: &attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(self.pipeline);
        rpass.set_bind_group(0, global_bg, &[]);
        rpass.set_bind_group(1, self.transient_bg.expect("SSR trace BG missing"), &[]);
        rpass.draw(0..3, 0..1);
    }
}

struct SsrTemporalNode<'a> {
    raw_reflection: TextureNodeId,
    reflection_history: TextureNodeId,
    scene_depth: TextureNodeId,
    scene_normals: TextureNodeId,
    history_meta: TextureNodeId,
    velocity: TextureNodeId,
    material_mrt: TextureNodeId,
    uniforms: BufferNodeId,
    output_reflection: TextureNodeId,
    output_meta: TextureNodeId,
    pipeline: &'a wgpu::RenderPipeline,
    layout: &'a Tracked<wgpu::BindGroupLayout>,
    transient_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for SsrTemporalNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.transient_bg = Some(
            crate::myth_bind_group!(ctx, self.layout, Some("SSR Temporal BG"), [
                0 => self.raw_reflection,
                1 => self.reflection_history,
                2 => self.scene_depth,
                3 => self.scene_normals,
                4 => self.history_meta,
                5 => self.velocity,
                6 => self.material_mrt,
                7 => CommonSampler::LinearClamp,
                8 => CommonSampler::NearestClamp,
                9 => self.uniforms,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let attachments = [
            ctx.get_color_attachment(self.output_reflection, RenderTargetOps::DontCare, None),
            ctx.get_color_attachment(self.output_meta, RenderTargetOps::DontCare, None),
        ];
        let global_bg = ctx.baked_lists.global_bind_group;

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSR Temporal Pass"),
            color_attachments: &attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(self.pipeline);
        rpass.set_bind_group(0, global_bg, &[]);
        rpass.set_bind_group(1, self.transient_bg.expect("SSR temporal BG missing"), &[]);
        rpass.draw(0..3, 0..1);
    }
}

struct SsrSpatialNode<'a> {
    input_reflection: TextureNodeId,
    scene_depth: TextureNodeId,
    scene_normals: TextureNodeId,
    material_mrt: TextureNodeId,
    uniforms: BufferNodeId,
    output_tex: TextureNodeId,
    pipeline: &'a wgpu::RenderPipeline,
    layout: &'a Tracked<wgpu::BindGroupLayout>,
    transient_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for SsrSpatialNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.transient_bg = Some(
            crate::myth_bind_group!(ctx, self.layout, Some("SSR Spatial BG"), [
                0 => self.input_reflection,
                1 => self.scene_depth,
                2 => self.scene_normals,
                3 => self.material_mrt,
                4 => CommonSampler::LinearClamp,
                5 => CommonSampler::NearestClamp,
                6 => self.uniforms,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let output = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSR Spatial Pass"),
            color_attachments: &[output],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(self.pipeline);
        rpass.set_bind_group(0, self.transient_bg.expect("SSR spatial BG missing"), &[]);
        rpass.draw(0..3, 0..1);
    }
}

struct SsrMergeNode<'a> {
    current_color: TextureNodeId,
    clean_reflection: TextureNodeId,
    material_mrt: TextureNodeId,
    specular_mrt: TextureNodeId,
    scene_depth: TextureNodeId,
    scene_normals: TextureNodeId,
    uniforms: BufferNodeId,
    output_tex: TextureNodeId,
    pipeline: &'a wgpu::RenderPipeline,
    layout: &'a Tracked<wgpu::BindGroupLayout>,
    transient_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for SsrMergeNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.transient_bg = Some(
            crate::myth_bind_group!(ctx, self.layout, Some("SSR Merge BG"), [
                0 => self.current_color,
                1 => self.clean_reflection,
                2 => self.material_mrt,
                3 => self.specular_mrt,
                4 => self.scene_depth,
                5 => self.scene_normals,
                6 => CommonSampler::LinearClamp,
                7 => CommonSampler::NearestClamp,
                8 => self.uniforms,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let output = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);
        let global_bg = ctx.baked_lists.global_bind_group;

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSR Merge Pass"),
            color_attachments: &[output],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(self.pipeline);
        rpass.set_bind_group(0, global_bg, &[]);
        rpass.set_bind_group(1, self.transient_bg.expect("SSR merge BG missing"), &[]);
        rpass.draw(0..3, 0..1);
    }
}

fn texture_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32, ty: wgpu::SamplerBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(ty),
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn persistent_texture_desc(width: u32, height: u32, format: wgpu::TextureFormat) -> TextureDesc {
    TextureDesc::new_2d(
        width,
        height,
        format,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    )
}
