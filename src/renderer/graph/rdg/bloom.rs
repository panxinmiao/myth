//! Bloom Feature & Transient Pass Node
//!
//! Physically-based bloom (COD: Advanced Warfare technique) split into the
//! **Feature + transient PassNode** architecture:
//!
//! - [`BloomFeature`] (persistent): owns pipelines, layouts, static karis
//!   uniform buffers. Stored in `RendererState`.
//! - [`RdgBloomPassNode`] (transient): per-frame mip chain management,
//!   bind groups, downsample / upsample / composite sub-pass execution.
//!
//! # Internal Sub-Passes
//!
//! 1. **Downsample**: scene HDR → bloom mip chain (Karis average on mip 0)
//! 2. **Upsample**: coarsest → finest mip (additive blend)
//! 3. **Composite**: original HDR + bloom mip0 → output

use crate::define_gpu_data_struct;
use crate::renderer::HDR_TEXTURE_FORMAT;
use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::resources::{CommonSampler, Tracked};
use crate::renderer::graph::rdg::builder::PassBuilder;
use crate::renderer::graph::rdg::context::{RdgExecuteContext, RdgPrepareContext};
use crate::renderer::graph::rdg::feature::ExtractContext;
use crate::renderer::graph::rdg::graph::RenderGraph;
use crate::renderer::graph::rdg::node::PassNode;
use crate::renderer::graph::rdg::types::{RdgTextureDesc, TextureNodeId};
use crate::renderer::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions,
};
use crate::resources::WgslType;
use crate::resources::bloom::{CompositeUniforms, UpsampleUniforms};
use crate::resources::uniforms::{UniformArray, WgslStruct};

define_gpu_data_struct!(
    /// Internal GPU uniform for the downsample shader (karis on/off flag).
    struct DownsampleUniforms {
        pub use_karis_average: u32,
        pub(crate) __pad: UniformArray<u32, 3>,
    }
);

// =============================================================================
// Configuration
// =============================================================================

/// Parameters passed to [`BloomFeature::add_to_graph`] each frame.
pub struct BloomParams {
    pub karis_average: bool,
    pub max_mip_levels: u32,
    pub upsample_uniforms_cpu_id: u64,
    pub composite_uniforms_cpu_id: u64,
}

// =============================================================================
// Persistent Feature
// =============================================================================

/// Persistent bloom Feature — owns pipelines, layouts, and static karis
/// uniform buffers that survive across frames.
pub struct BloomFeature {
    downsample_pipeline: Option<RenderPipelineId>,
    upsample_pipeline: Option<RenderPipelineId>,
    composite_pipeline: Option<RenderPipelineId>,

    downsample_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    upsample_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    composite_layout: Option<Tracked<wgpu::BindGroupLayout>>,

    /// GPU buffer with `use_karis_average = 1`, written once.
    karis_on_buffer: Option<Tracked<wgpu::Buffer>>,
    /// GPU buffer with `use_karis_average = 0`, written once.
    karis_off_buffer: Option<Tracked<wgpu::Buffer>>,
}

impl BloomFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            downsample_pipeline: None,
            upsample_pipeline: None,
            composite_pipeline: None,
            downsample_layout: None,
            upsample_layout: None,
            composite_layout: None,
            karis_on_buffer: None,
            karis_off_buffer: None,
        }
    }

    /// Prepare persistent GPU resources before graph assembly.
    pub fn extract_and_prepare(&mut self, ctx: &mut ExtractContext) {
        self.ensure_layouts(ctx.device);
        self.ensure_internal_buffers(ctx.device, ctx.queue);
        self.ensure_pipelines(ctx);
    }

    /// Build and inject a transient bloom node into the render graph.
    ///
    /// Returns the `TextureNodeId` of the composited HDR bloom output.
    pub fn add_to_graph(
        &self,
        rdg: &mut RenderGraph,
        input_color: TextureNodeId,
        params: BloomParams,
    ) -> TextureNodeId {
        let config = rdg.frame_config();
        let hdr_format = config.hdr_format;
        let (w, h) = (config.width, config.height);

        let hdr_desc = RdgTextureDesc::new_2d(
            w,
            h,
            hdr_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );
        let output_tex = rdg.register_resource("Bloom_Out", hdr_desc, false);

        let bloom_w = (w / 2).max(1);
        let bloom_h = (h / 2).max(1);
        let max_possible = ((bloom_w.max(bloom_h) as f32).log2().floor() as u32) + 1;
        let mip_count = params.max_mip_levels.min(max_possible).max(1);

        let bloom_chain_desc = RdgTextureDesc::new(
            bloom_w,
            bloom_h,
            1,
            mip_count,
            1,
            wgpu::TextureDimension::D2,
            hdr_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let bloom_texture = rdg.register_resource("Bloom_MipChain", bloom_chain_desc, false);

        let node = Box::new(RdgBloomPassNode {
            input_tex: input_color,
            output_tex,
            bloom_texture,
            karis_average: params.karis_average,
            _max_mip_levels: mip_count,
            upsample_uniforms_cpu_id: params.upsample_uniforms_cpu_id,
            composite_uniforms_cpu_id: params.composite_uniforms_cpu_id,
            downsample_pipeline: self.downsample_pipeline.unwrap(),
            upsample_pipeline: self.upsample_pipeline.unwrap(),
            composite_pipeline: self.composite_pipeline.unwrap(),
            downsample_layout: self.downsample_layout.clone().unwrap(),
            upsample_layout: self.upsample_layout.clone().unwrap(),
            composite_layout: self.composite_layout.clone().unwrap(),
            karis_on_buffer: self.karis_on_buffer.clone().unwrap(),
            karis_off_buffer: self.karis_off_buffer.clone().unwrap(),
            mip_views: Vec::new(),
            downsample_bind_groups: Vec::new(),
            upsample_bind_groups: Vec::new(),
            composite_bind_group: None,
        });
        rdg.add_pass_owned(node);
        output_tex
    }

    // =========================================================================
    // Lazy Initialization
    // =========================================================================

    fn ensure_layouts(&mut self, device: &wgpu::Device) {
        if self.downsample_layout.is_some() {
            return;
        }

        // Downsample: texture + sampler + uniforms
        let ds_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RDG Bloom Downsample Layout"),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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

        // Upsample: same signature
        let us_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RDG Bloom Upsample Layout"),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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

        // Composite: original + bloom + sampler + uniforms
        let comp_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RDG Bloom Composite Layout"),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        self.downsample_layout = Some(Tracked::new(ds_layout));
        self.upsample_layout = Some(Tracked::new(us_layout));
        self.composite_layout = Some(Tracked::new(comp_layout));
    }

    fn ensure_internal_buffers(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.karis_on_buffer.is_some() {
            return;
        }

        let karis_on_data = DownsampleUniforms {
            use_karis_average: 1,
            ..Default::default()
        };
        let karis_off_data = DownsampleUniforms::default();

        let buf_on = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RDG Bloom Karis On"),
            size: std::mem::size_of::<DownsampleUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf_on, 0, bytemuck::bytes_of(&karis_on_data));

        let buf_off = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RDG Bloom Karis Off"),
            size: std::mem::size_of::<DownsampleUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf_off, 0, bytemuck::bytes_of(&karis_off_data));

        self.karis_on_buffer = Some(Tracked::new(buf_on));
        self.karis_off_buffer = Some(Tracked::new(buf_off));
    }

    fn ensure_pipelines(&mut self, ctx: &mut ExtractContext) {
        if self.downsample_pipeline.is_some() {
            return;
        }

        let device = ctx.device;

        let color_target_replace = ColorTargetKey::from(wgpu::ColorTargetState {
            format: HDR_TEXTURE_FORMAT,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        });

        let color_target_additive = ColorTargetKey::from(wgpu::ColorTargetState {
            format: HDR_TEXTURE_FORMAT,
            blend: Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::OVER,
            }),
            write_mask: wgpu::ColorWrites::ALL,
        });

        let ds_layout = self.downsample_layout.as_ref().unwrap();
        let us_layout = self.upsample_layout.as_ref().unwrap();
        let comp_layout = self.composite_layout.as_ref().unwrap();

        // ─── Downsample Pipeline ───────────────────────────────────
        {
            let mut options = ShaderCompilationOptions::default();
            options.add_define(
                "struct_definitions",
                DownsampleUniforms::wgsl_struct_def("DownsampleUniforms").as_str(),
            );

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RDG Bloom Downsample Pipeline Layout"),
                bind_group_layouts: &[ds_layout],
                immediate_size: 0,
            });

            let (module, hash) = ctx.shader_manager.get_or_compile_template(
                device,
                "passes/bloom_downsample",
                &options,
                "",
                "",
            );

            let key = FullscreenPipelineKey::fullscreen(
                hash,
                smallvec::smallvec![color_target_replace.clone()],
                None,
            );

            self.downsample_pipeline = Some(ctx.pipeline_cache.get_or_create_fullscreen(
                device,
                module,
                &layout,
                &key,
                "RDG Bloom Downsample Pipeline",
            ));
        }

        // ─── Upsample Pipeline (additive blend) ───────────────────
        {
            let mut options = ShaderCompilationOptions::default();
            options.add_define(
                "struct_definitions",
                UpsampleUniforms::wgsl_struct_def("UpsampleUniforms").as_str(),
            );

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RDG Bloom Upsample Pipeline Layout"),
                bind_group_layouts: &[us_layout],
                immediate_size: 0,
            });

            let (module, hash) = ctx.shader_manager.get_or_compile_template(
                device,
                "passes/bloom_upsample",
                &options,
                "",
                "",
            );

            let key = FullscreenPipelineKey::fullscreen(
                hash,
                smallvec::smallvec![color_target_additive],
                None,
            );

            self.upsample_pipeline = Some(ctx.pipeline_cache.get_or_create_fullscreen(
                device,
                module,
                &layout,
                &key,
                "RDG Bloom Upsample Pipeline",
            ));
        }

        // ─── Composite Pipeline ────────────────────────────────────
        {
            let mut options = ShaderCompilationOptions::default();
            options.add_define(
                "struct_definitions",
                CompositeUniforms::wgsl_struct_def("CompositeUniforms").as_str(),
            );

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RDG Bloom Composite Pipeline Layout"),
                bind_group_layouts: &[comp_layout],
                immediate_size: 0,
            });

            let (module, hash) = ctx.shader_manager.get_or_compile_template(
                device,
                "passes/bloom_composite",
                &options,
                "",
                "",
            );

            let key = FullscreenPipelineKey::fullscreen(
                hash,
                smallvec::smallvec![color_target_replace],
                None,
            );

            self.composite_pipeline = Some(ctx.pipeline_cache.get_or_create_fullscreen(
                device,
                module,
                &layout,
                &key,
                "RDG Bloom Composite Pipeline",
            ));
        }
    }
}

// =============================================================================
// Transient Pass Node
// =============================================================================

/// Per-frame bloom pass node — created and discarded every frame.
///
/// Manages the mip chain views, per-mip bind groups for downsample / upsample,
/// and the final composite bind group. All bind groups reference RDG-allocated
/// transient textures and are rebuilt each frame.
struct RdgBloomPassNode {
    input_tex: TextureNodeId,
    output_tex: TextureNodeId,
    bloom_texture: TextureNodeId,

    karis_average: bool,
    _max_mip_levels: u32,
    upsample_uniforms_cpu_id: u64,
    composite_uniforms_cpu_id: u64,

    downsample_pipeline: RenderPipelineId,
    upsample_pipeline: RenderPipelineId,
    composite_pipeline: RenderPipelineId,
    downsample_layout: Tracked<wgpu::BindGroupLayout>,
    upsample_layout: Tracked<wgpu::BindGroupLayout>,
    composite_layout: Tracked<wgpu::BindGroupLayout>,
    karis_on_buffer: Tracked<wgpu::Buffer>,
    karis_off_buffer: Tracked<wgpu::Buffer>,

    mip_views: Vec<Tracked<wgpu::TextureView>>,
    downsample_bind_groups: Vec<wgpu::BindGroup>,
    upsample_bind_groups: Vec<wgpu::BindGroup>,
    composite_bind_group: Option<wgpu::BindGroup>,
}

impl RdgBloomPassNode {
    /// Create per-mip views into the RDG-allocated bloom texture.
    fn build_mip_views(&mut self, ctx: &mut RdgPrepareContext) {
        let texture = ctx.views.get_texture_view(self.bloom_texture).texture();
        let mip_count = texture.mip_level_count();

        self.mip_views.clear();
        for mip in 0..mip_count {
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("RDG Bloom Mip View"),
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..Default::default()
            });
            self.mip_views.push(Tracked::new(view));
        }
    }

    /// Build all downsample, upsample, and composite bind groups.
    fn build_bind_groups(&mut self, ctx: &mut RdgPrepareContext) {
        let sampler = ctx.sampler_registry.get_common(CommonSampler::LinearClamp);
        let mip_count = self.mip_views.len();
        if mip_count == 0 {
            return;
        }

        // ─── First Downsample BG (scene color → mip 0) ────────────
        let karis_buf = if self.karis_average {
            &self.karis_on_buffer
        } else {
            &self.karis_off_buffer
        };

        let input_view = ctx.views.get_texture_view(self.input_tex);
        let key = BindGroupKey::new(self.downsample_layout.id())
            .with_resource(input_view.id())
            .with_resource(sampler.id())
            .with_resource(karis_buf.id());

        let first_ds_bg = if let Some(cached) = ctx.global_bind_group_cache.get(&key) {
            cached.clone()
        } else {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bloom DS BG scene→0"),
                layout: &self.downsample_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: karis_buf.as_entire_binding(),
                    },
                ],
            });
            ctx.global_bind_group_cache.insert(key, bg.clone());
            bg
        };

        self.downsample_bind_groups.clear();
        self.downsample_bind_groups.push(first_ds_bg);

        // ─── Remaining Downsample BGs (mip[i] → mip[i+1]) ────────
        let karis_off_buf = &self.karis_off_buffer;
        for i in 0..(mip_count - 1) {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RDG Bloom DS BG mip→mip"),
                layout: &self.downsample_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&*self.mip_views[i]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&**sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: karis_off_buf.as_entire_binding(),
                    },
                ],
            });
            self.downsample_bind_groups.push(bg);
        }

        // ─── Upsample BGs (mip[i+1] → mip[i]) ───────────────────
        let upsample_gpu = ctx
            .resource_manager
            .gpu_buffers
            .get(&self.upsample_uniforms_cpu_id)
            .expect("RDG Bloom: upsample GPU buffer must exist");

        self.upsample_bind_groups.clear();
        for i in 0..(mip_count - 1) {
            let source_mip = i + 1;
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RDG Bloom US BG mip→mip"),
                layout: &self.upsample_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &*self.mip_views[source_mip],
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&**sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: upsample_gpu.buffer.as_entire_binding(),
                    },
                ],
            });
            self.upsample_bind_groups.push(bg);
        }

        // ─── Composite BG (original + bloom mip0 → output) ───────
        {
            let bloom_view = &self.mip_views[0];
            let composite_gpu = ctx
                .resource_manager
                .gpu_buffers
                .get(&self.composite_uniforms_cpu_id)
                .expect("Bloom composite GPU buffer must exist");

            let key = BindGroupKey::new(self.composite_layout.id())
                .with_resource(input_view.id())
                .with_resource(bloom_view.id())
                .with_resource(sampler.id())
                .with_resource(self.composite_uniforms_cpu_id);

            let bg = if let Some(cached) = ctx.global_bind_group_cache.get(&key) {
                cached.clone()
            } else {
                let new_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bloom Composite BG"),
                    layout: &self.composite_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(input_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(bloom_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: composite_gpu.buffer.as_entire_binding(),
                        },
                    ],
                });
                ctx.global_bind_group_cache.insert(key, new_bg.clone());
                new_bg
            };

            self.composite_bind_group = Some(bg);
        }
    }
}

impl PassNode for RdgBloomPassNode {
    fn name(&self) -> &'static str {
        "RDG_Bloom_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        builder.read_texture(self.input_tex);
        builder.write_texture(self.output_tex);
        builder.write_texture(self.bloom_texture);
    }

    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {
        self.build_mip_views(ctx);
        self.build_bind_groups(ctx);
    }

    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        if self.mip_views.is_empty() {
            return;
        }

        let ds_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.downsample_pipeline);
        let us_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.upsample_pipeline);
        let comp_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.composite_pipeline);

        let Some(comp_bg) = &self.composite_bind_group else {
            return;
        };

        // Phase 1: Downsample — Scene HDR → Bloom Mip Chain
        for i in 0..self.downsample_bind_groups.len() {
            let target_mip = i as u32;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("RDG Bloom Downsample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.mip_views[target_mip as usize],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(ds_pipeline);
            pass.set_bind_group(0, &self.downsample_bind_groups[i], &[]);
            pass.draw(0..3, 0..1);
        }

        // Phase 2: Upsample — Accumulate bloom from coarsest to finest
        for i in (0..self.upsample_bind_groups.len()).rev() {
            let target_mip = i as u32;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("RDG Bloom Upsample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.mip_views[target_mip as usize],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(us_pipeline);
            pass.set_bind_group(0, &self.upsample_bind_groups[i], &[]);
            pass.draw(0..3, 0..1);
        }

        // Phase 3: Composite — Original HDR + Bloom → Output
        let output_view = ctx.get_texture_view(self.output_tex);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RDG Bloom Composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(comp_pipeline);
        pass.set_bind_group(0, comp_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
