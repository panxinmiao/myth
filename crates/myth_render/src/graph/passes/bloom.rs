//! RDG Bloom Post-Processing — Flattened Micro-Pass Architecture.
//!
//! Implements the Call of Duty: Advanced Warfare physically-based bloom
//! technique within the RDG framework using fully flattened per-mip passes
//! and a strict two-layer binding separation:
//!
//! - **Group 0 (Static)**: Sampler + uniform buffer — built and owned by
//!   [`BloomFeature`], lifetime matches the engine.
//! - **Group 1 (Transient)**: RDG texture views — built by each micro-pass
//!   node during `prepare`, cached via [`GlobalBindGroupCache`].
//!
//! # Architecture
//!
//! - **`BloomFeature`** – persistent owner of GPU pipelines, bind-group
//!   layouts, static uniform buffers, and Group 0 bind groups.  Initialised
//!   lazily by [`extract_and_prepare`](BloomFeature::extract_and_prepare).
//!
//! - **Micro-Passes** – each downsample, upsample, and composite step is
//!   an independent [`PassNode`] with its own RDG resource declarations.
//!   This exposes fine-grained dependencies to the RDG compiler, enabling
//!   optimal barrier insertion and memory aliasing across mip levels.
//!
//! # RDG Topology (Flattened)
//!
//! ```text
//! Bloom_Extract (Scene HDR → Mip_0)
//!   → Bloom_Downsample_1 (Mip_0 → Mip_1)
//!   → Bloom_Downsample_2 (Mip_1 → Mip_2)
//!   → ...
//!   → Bloom_Upsample_N-2 (Mip_N-1 + Mip_N-2 → Up_N-2)
//!   → ...
//!   → Bloom_Upsample_0 (Up_1 + Mip_0 → Up_0)
//!   → Bloom_Composite (Scene HDR + Up_0 → Output)
//! ```

use crate::HDR_TEXTURE_FORMAT;
use crate::core::binding::BindGroupKey;
use crate::core::gpu::{CommonSampler, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    ExecuteContext, ExtractContext, PassNode, PrepareContext, RenderTargetOps, TextureDesc,
    TextureNodeId,
};
use crate::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions,
};
use myth_resources::WgslType;
use myth_resources::bloom::{CompositeUniforms, UpsampleUniforms};
use myth_resources::buffer::CpuBuffer;
use myth_resources::define_gpu_data_struct;
use myth_resources::uniforms::{UniformArray, WgslStruct};

define_gpu_data_struct!(
    /// Internal GPU uniform for the downsample shader (karis on/off flag).
    struct DownsampleUniforms {
        pub use_karis_average: u32,
        pub(crate) __pad: UniformArray<u32, 3>,
    }
);

// =============================================================================
// BloomFeature — persistent GPU resource owner
// =============================================================================

/// Persistent bloom feature owning GPU pipelines, layouts, and pre-built
/// static bind groups.
///
/// The Composer calls [`extract_and_prepare`](Self::extract_and_prepare)
/// once per frame before the render graph is built, then
/// [`add_to_graph`](Self::add_to_graph) to emit a chain of flattened
/// micro-passes (Extract → Downsample × N → Upsample × N → Composite)
/// into the RDG.
///
/// # Dual-Layer BindGroup Model
///
/// Static bind groups (Group 0) containing samplers and uniform buffers
/// are built here and shared across all micro-pass nodes as cheap clones.
/// Each micro-pass only assembles Group 1 (transient texture views).
pub struct BloomFeature {
    // ─── Pipelines ─────────────────────────────────────────────────
    downsample_pipeline: Option<RenderPipelineId>,
    upsample_pipeline: Option<RenderPipelineId>,
    composite_pipeline: Option<RenderPipelineId>,

    // ─── Bind Group Layouts ────────────────────────────────────────
    /// Group 0 layout for downsample: sampler + DownsampleUniforms.
    ds_static_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    /// Group 1 layout for downsample: single input texture.
    ds_transient_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    /// Group 0 layout for upsample: sampler + UpsampleUniforms.
    us_static_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    /// Group 1 layout for upsample: single source mip texture.
    us_transient_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    /// Group 0 layout for composite: sampler + CompositeUniforms.
    comp_static_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    /// Group 1 layout for composite: original + bloom textures.
    comp_transient_layout: Option<Tracked<wgpu::BindGroupLayout>>,

    // ─── Internal Static Uniform Buffers ───────────────────────────
    /// GPU buffer with `use_karis_average = 1`, written once.
    karis_on_buffer: Option<Tracked<wgpu::Buffer>>,
    /// GPU buffer with `use_karis_average = 0`, written once.
    karis_off_buffer: Option<Tracked<wgpu::Buffer>>,

    // ─── Pre-Built Static BindGroups (Group 0) ─────────────────────
    /// Downsample Group 0 with karis averaging enabled.
    karis_on_static_bg: Option<wgpu::BindGroup>,
    /// Downsample Group 0 with karis averaging disabled.
    karis_off_static_bg: Option<wgpu::BindGroup>,
    /// Upsample Group 0 (sampler + UpsampleUniforms buffer).
    upsample_static_bg: Option<wgpu::BindGroup>,
    /// Composite Group 0 (sampler + CompositeUniforms buffer).
    composite_static_bg: Option<wgpu::BindGroup>,

    // ─── Staleness tracking for externally-managed GPU buffers ─────
    last_upsample_buffer_id: u64,
    last_composite_buffer_id: u64,
}

impl Default for BloomFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl BloomFeature {
    /// Creates a new bloom feature. All GPU resources are lazily allocated.
    #[must_use]
    pub fn new() -> Self {
        Self {
            downsample_pipeline: None,
            upsample_pipeline: None,
            composite_pipeline: None,

            ds_static_layout: None,
            ds_transient_layout: None,
            us_static_layout: None,
            us_transient_layout: None,
            comp_static_layout: None,
            comp_transient_layout: None,

            karis_on_buffer: None,
            karis_off_buffer: None,

            karis_on_static_bg: None,
            karis_off_static_bg: None,
            upsample_static_bg: None,
            composite_static_bg: None,

            last_upsample_buffer_id: 0,
            last_composite_buffer_id: 0,
        }
    }

    // =========================================================================
    // Extract & Prepare (called before RDG build)
    // =========================================================================

    /// Ensure all persistent GPU resources (layouts, static buffers, pipelines)
    /// are initialised. Build or rebuild static bind groups (Group 0) when the
    /// underlying GPU buffer identity changes.
    ///
    /// `upsample_uniform` and `composite_uniform` are the CpuBuffer IDs
    /// whose GPU mirrors have already been uploaded via `ensure_buffer()`.
    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        upsample_uniform: &CpuBuffer<UpsampleUniforms>,
        composite_uniform: &CpuBuffer<CompositeUniforms>,
    ) {
        self.ensure_layouts(ctx.device);
        self.ensure_internal_buffers(ctx.device, ctx.queue);
        self.ensure_pipelines(ctx);

        ctx.resource_manager.ensure_buffer(upsample_uniform);
        ctx.resource_manager.ensure_buffer(composite_uniform);

        self.build_static_bind_groups(ctx, upsample_uniform, composite_uniform);
    }

    // =========================================================================
    // Lazy Initialization
    // =========================================================================

    fn ensure_layouts(&mut self, device: &wgpu::Device) {
        if self.ds_static_layout.is_some() {
            return;
        }

        // ─── Shared entry helpers ──────────────────────────────────
        let sampler_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };

        // ─── Downsample ───────────────────────────────────────────
        self.ds_static_layout = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Bloom DS Static Layout (G0)"),
                entries: &[sampler_entry, uniform_entry],
            },
        )));
        self.ds_transient_layout = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Bloom DS Transient Layout (G1)"),
                entries: &[texture_entry(0)],
            },
        )));

        // ─── Upsample ────────────────────────────────────────────
        self.us_static_layout = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Bloom US Static Layout (G0)"),
                entries: &[sampler_entry, uniform_entry],
            },
        )));
        self.us_transient_layout = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Bloom US Transient Layout (G1)"),
                entries: &[texture_entry(0)],
            },
        )));

        // ─── Composite ───────────────────────────────────────────
        self.comp_static_layout = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Bloom Comp Static Layout (G0)"),
                entries: &[sampler_entry, uniform_entry],
            },
        )));
        self.comp_transient_layout = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Bloom Comp Transient Layout (G1)"),
                entries: &[texture_entry(0), texture_entry(1)],
            },
        )));
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
            label: Some("Bloom Karis On"),
            size: std::mem::size_of::<DownsampleUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf_on, 0, bytemuck::bytes_of(&karis_on_data));

        let buf_off = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bloom Karis Off"),
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

        let ds_static = self.ds_static_layout.as_ref().unwrap();
        let ds_trans = self.ds_transient_layout.as_ref().unwrap();
        let us_static = self.us_static_layout.as_ref().unwrap();
        let us_trans = self.us_transient_layout.as_ref().unwrap();
        let comp_static = self.comp_static_layout.as_ref().unwrap();
        let comp_trans = self.comp_transient_layout.as_ref().unwrap();

        // ─── Downsample Pipeline ───────────────────────────────────
        {
            let mut options = ShaderCompilationOptions::default();
            options.add_define(
                "struct_definitions",
                DownsampleUniforms::wgsl_struct_def("DownsampleUniforms").as_str(),
            );

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bloom DS Pipeline Layout"),
                bind_group_layouts: &[ds_static, ds_trans],
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
                "Bloom DS Pipeline",
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
                label: Some("Bloom US Pipeline Layout"),
                bind_group_layouts: &[us_static, us_trans],
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
                "Bloom US Pipeline",
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
                label: Some("Bloom Comp Pipeline Layout"),
                bind_group_layouts: &[comp_static, comp_trans],
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
                "Bloom Comp Pipeline",
            ));
        }
    }

    /// Build all static bind groups (Group 0) that pair samplers with
    /// persistent uniform buffers. Karis BGs are built once; upsample
    /// and composite BGs are rebuilt only when the underlying GPU buffer
    /// identity changes (e.g. after an `ensure_buffer` resize).
    fn build_static_bind_groups(
        &mut self,
        ctx: &mut ExtractContext,
        upsample_uniform: &CpuBuffer<UpsampleUniforms>,
        composite_uniform: &CpuBuffer<CompositeUniforms>,
    ) {
        let sampler = ctx
            .resource_manager
            .sampler_registry
            .get_common(CommonSampler::LinearClamp);
        let ds_layout = self.ds_static_layout.as_ref().unwrap();

        // ─── Karis BGs (eternal, built once) ───────────────────────
        if self.karis_on_static_bg.is_none() {
            let karis_on = self.karis_on_buffer.as_ref().unwrap();
            let karis_off = self.karis_off_buffer.as_ref().unwrap();

            self.karis_on_static_bg =
                Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bloom DS G0 (karis on)"),
                    layout: ds_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: karis_on.as_entire_binding(),
                        },
                    ],
                }));

            self.karis_off_static_bg =
                Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bloom DS G0 (karis off)"),
                    layout: ds_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: karis_off.as_entire_binding(),
                        },
                    ],
                }));
        }

        // ─── Upsample static BG (rebuild on buffer identity change) ──
        if let Some(handle) = upsample_uniform.gpu_handle()
            && let Some(g) = ctx.resource_manager.gpu_buffers.get(handle)
            && (self.upsample_static_bg.is_none() || self.last_upsample_buffer_id != g.id)
        {
            let us_layout = self.us_static_layout.as_ref().unwrap();
            self.upsample_static_bg =
                Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bloom US G0"),
                    layout: us_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: g.buffer.as_entire_binding(),
                        },
                    ],
                }));
            self.last_upsample_buffer_id = g.id;
        }

        // ─── Composite static BG (rebuild on buffer identity change) ──
        if let Some(handle) = composite_uniform.gpu_handle()
            && let Some(g) = ctx.resource_manager.gpu_buffers.get(handle)
            && (self.composite_static_bg.is_none() || self.last_composite_buffer_id != g.id)
        {
            let comp_layout = self.comp_static_layout.as_ref().unwrap();
            self.composite_static_bg =
                Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bloom Comp G0"),
                    layout: comp_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: g.buffer.as_entire_binding(),
                        },
                    ],
                }));
            self.last_composite_buffer_id = g.id;
        }
    }

    // =========================================================================
    // Graph Integration — Flattened Micro-Pass Architecture
    // =========================================================================

    /// Emit the full bloom pipeline as a chain of independent RDG passes.
    ///
    /// Each downsample and upsample step is a separate pass operating on
    /// independent 2D transient textures with progressively halved
    /// resolution. The RDG compiler sees fine-grained dependencies between
    /// every mip level, enabling:
    ///
    /// - **Optimal barrier/transition insertion** between individual steps.
    /// - **Aggressive memory aliasing** — textures whose lifetimes don't
    ///   overlap are automatically mapped to the same physical GPU memory.
    ///
    /// Returns the composited output [`TextureNodeId`] (scene HDR with
    /// bloom applied) for downstream wiring (e.g. ToneMapping).
    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        input_color: TextureNodeId,
        karis_average: bool,
        max_mip_levels: u32,
    ) -> TextureNodeId {
        let fc = ctx.frame_config;
        let hdr_format = fc.hdr_format;
        let bloom_w = (fc.width / 2).max(1);
        let bloom_h = (fc.height / 2).max(1);
        let max_possible = ((bloom_w.max(bloom_h) as f32).log2().floor() as u32) + 1;
        let mip_count = max_mip_levels.min(max_possible).max(1) as usize;

        let ds_pipeline = ctx.pipeline_cache.get_render_pipeline(
            self.downsample_pipeline
                .expect("BloomFeature: downsample pipeline not initialised"),
        );
        let us_pipeline = ctx.pipeline_cache.get_render_pipeline(
            self.upsample_pipeline
                .expect("BloomFeature: upsample pipeline not initialised"),
        );
        let comp_pipeline = ctx.pipeline_cache.get_render_pipeline(
            self.composite_pipeline
                .expect("BloomFeature: composite pipeline not initialised"),
        );

        let karis_on_bg = self
            .karis_on_static_bg
            .as_ref()
            .expect("BloomFeature: karis_on static BG not built");
        let karis_off_bg = self
            .karis_off_static_bg
            .as_ref()
            .expect("BloomFeature: karis_off static BG not built");
        let upsample_bg = self
            .upsample_static_bg
            .as_ref()
            .expect("BloomFeature: upsample static BG not built");
        let composite_bg = self
            .composite_static_bg
            .as_ref()
            .expect("BloomFeature: composite static BG not built");

        let ds_transient_layout = self.ds_transient_layout.as_ref().unwrap();
        let us_transient_layout = self.us_transient_layout.as_ref().unwrap();
        let comp_transient_layout = self.comp_transient_layout.as_ref().unwrap();

        ctx.with_group("Bloom_System", |ctx| {
            // ─── 1. Extract: Scene HDR → Bloom Mip 0 ──────────────
            let mip0_w = bloom_w;
            let mip0_h = bloom_h;
            let mip0_desc = TextureDesc::new_2d(
                mip0_w,
                mip0_h,
                hdr_format,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );

            let static_bg = if karis_average {
                karis_on_bg
            } else {
                karis_off_bg
            };

            let mut current_mip: TextureNodeId = ctx.graph.add_pass("Bloom_Extract", |builder| {
                builder.read_texture(input_color);
                let out = builder.create_texture("Bloom_Mip_0", mip0_desc);
                let node = BloomDownsampleNode {
                    input_tex: input_color,
                    output_tex: out,
                    pipeline: ds_pipeline,
                    static_bg,
                    transient_layout: ds_transient_layout,
                    transient_bg: None,
                };
                (node, out)
            });

            // Collect downsample chain for upsample phase references.
            let mut downsample_chain = Vec::with_capacity(mip_count);
            downsample_chain.push(current_mip);

            // ─── 2. Downsample chain: Mip N → Mip N+1 ─────────────
            let mut w = mip0_w;
            let mut h = mip0_h;
            for i in 1..mip_count {
                w = (w / 2).max(1);
                h = (h / 2).max(1);

                let desc = TextureDesc::new_2d(
                    w,
                    h,
                    hdr_format,
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                );

                let input = current_mip;
                let label: &'static str = DOWNSAMPLE_PASS_NAMES[i];
                let out_label: &'static str = DOWNSAMPLE_MIP_NAMES[i];

                current_mip = ctx.graph.add_pass(label, |builder| {
                    builder.read_texture(input);
                    let out = builder.create_texture(out_label, desc);
                    let node = BloomDownsampleNode {
                        input_tex: input,
                        output_tex: out,
                        pipeline: ds_pipeline,
                        static_bg: karis_off_bg,
                        transient_layout: ds_transient_layout,
                        transient_bg: None,
                    };
                    (node, out)
                });

                downsample_chain.push(current_mip);
            }

            // ─── 3. Upsample chain: coarse → fine with additive blend ─
            for i in (0..(mip_count - 1)).rev() {
                let coarser = current_mip;
                let finer = downsample_chain[i];

                let label: &'static str = UPSAMPLE_PASS_NAMES[i];
                let out_label: &'static str = UPSAMPLE_MIP_NAMES[i];

                current_mip = ctx.graph.add_pass(label, |builder| {
                    builder.read_texture(coarser);
                    let out = builder.mutate_texture(finer, out_label);
                    let node = BloomUpsampleNode {
                        coarser_tex: coarser,
                        output_tex: out,
                        pipeline: us_pipeline,
                        static_bg: upsample_bg,
                        transient_layout: us_transient_layout,
                        transient_bg: None,
                    };
                    (node, out)
                });
            }

            // ─── 4. Composite: Scene HDR + Bloom → Output ─────────
            let bloom_result = current_mip;
            ctx.graph.add_pass("Bloom_Composite", |builder| {
                builder.read_texture(input_color);
                builder.read_texture(bloom_result);

                let out_desc = TextureDesc::new_2d(
                    fc.width,
                    fc.height,
                    hdr_format,
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                );

                let out = builder.create_texture("Scene_Color_Bloom", out_desc);

                let node = BloomCompositeNode {
                    original_tex: input_color,
                    bloom_tex: bloom_result,
                    output_tex: out,
                    pipeline: comp_pipeline,
                    static_bg: composite_bg,
                    transient_layout: comp_transient_layout,
                    transient_bg: None,
                };
                (node, out)
            })
        })
    }
}

// =============================================================================
// Pre-allocated static pass/resource name tables
// =============================================================================
//
// RDG pass and resource names must be `&'static str`. We pre-allocate up to
// 16 mip levels (covers 65536×65536 max resolution) as compile-time constants
// to avoid runtime string formatting and heap allocation.

/// Pass names for each downsample step (index 0 is the extract step).
static DOWNSAMPLE_PASS_NAMES: [&str; 16] = [
    "Bloom_Extract",
    "Bloom_Downsample_1",
    "Bloom_Downsample_2",
    "Bloom_Downsample_3",
    "Bloom_Downsample_4",
    "Bloom_Downsample_5",
    "Bloom_Downsample_6",
    "Bloom_Downsample_7",
    "Bloom_Downsample_8",
    "Bloom_Downsample_9",
    "Bloom_Downsample_10",
    "Bloom_Downsample_11",
    "Bloom_Downsample_12",
    "Bloom_Downsample_13",
    "Bloom_Downsample_14",
    "Bloom_Downsample_15",
];

/// Resource names for each downsample mip output.
static DOWNSAMPLE_MIP_NAMES: [&str; 16] = [
    "Bloom_Mip_0",
    "Bloom_Mip_1",
    "Bloom_Mip_2",
    "Bloom_Mip_3",
    "Bloom_Mip_4",
    "Bloom_Mip_5",
    "Bloom_Mip_6",
    "Bloom_Mip_7",
    "Bloom_Mip_8",
    "Bloom_Mip_9",
    "Bloom_Mip_10",
    "Bloom_Mip_11",
    "Bloom_Mip_12",
    "Bloom_Mip_13",
    "Bloom_Mip_14",
    "Bloom_Mip_15",
];

/// Pass names for each upsample step.
static UPSAMPLE_PASS_NAMES: [&str; 16] = [
    "Bloom_Upsample_0",
    "Bloom_Upsample_1",
    "Bloom_Upsample_2",
    "Bloom_Upsample_3",
    "Bloom_Upsample_4",
    "Bloom_Upsample_5",
    "Bloom_Upsample_6",
    "Bloom_Upsample_7",
    "Bloom_Upsample_8",
    "Bloom_Upsample_9",
    "Bloom_Upsample_10",
    "Bloom_Upsample_11",
    "Bloom_Upsample_12",
    "Bloom_Upsample_13",
    "Bloom_Upsample_14",
    "Bloom_Upsample_15",
];

/// Resource names for each upsample output.
static UPSAMPLE_MIP_NAMES: [&str; 16] = [
    "Bloom_Up_0",
    "Bloom_Up_1",
    "Bloom_Up_2",
    "Bloom_Up_3",
    "Bloom_Up_4",
    "Bloom_Up_5",
    "Bloom_Up_6",
    "Bloom_Up_7",
    "Bloom_Up_8",
    "Bloom_Up_9",
    "Bloom_Up_10",
    "Bloom_Up_11",
    "Bloom_Up_12",
    "Bloom_Up_13",
    "Bloom_Up_14",
    "Bloom_Up_15",
];

// =============================================================================
// Micro-Pass Nodes
// =============================================================================

/// Downsample micro-pass: reads one input texture, writes one output texture
/// at half resolution using a 13-tap box filter.
///
/// Used for both the initial extract (scene HDR → mip 0) and subsequent
/// mip-chain downsamples. The karis averaging flag is controlled by which
/// static bind group (Group 0) is provided.
struct BloomDownsampleNode<'a> {
    input_tex: TextureNodeId,
    output_tex: TextureNodeId,
    pipeline: &'a wgpu::RenderPipeline,
    static_bg: &'a wgpu::BindGroup,
    transient_layout: &'a Tracked<wgpu::BindGroupLayout>,
    transient_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for BloomDownsampleNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let PrepareContext {
            views,
            global_bind_group_cache: cache,
            device,
            ..
        } = ctx;
        let device = *device;
        let input_view = views.get_texture_view(self.input_tex);
        let key = BindGroupKey::new(self.transient_layout.id()).with_resource(input_view.id());
        let layout = &**self.transient_layout;

        let bg = cache.get_or_create_bg(key, || {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bloom DS G1"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                }],
            })
        });
        self.transient_bg = Some(bg);
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let transient_bg = self
            .transient_bg
            .expect("Bloom DS transient BG not prepared");

        let rtt = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Bloom Downsample"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(self.pipeline);
        rpass.set_bind_group(0, self.static_bg, &[]);
        rpass.set_bind_group(1, transient_bg, &[]);
        rpass.draw(0..3, 0..1);
    }
}

struct BloomUpsampleNode<'a> {
    coarser_tex: TextureNodeId,
    output_tex: TextureNodeId,
    pipeline: &'a wgpu::RenderPipeline,
    static_bg: &'a wgpu::BindGroup,
    transient_layout: &'a Tracked<wgpu::BindGroupLayout>,
    transient_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for BloomUpsampleNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let PrepareContext {
            views,
            global_bind_group_cache: cache,
            device,
            ..
        } = ctx;
        let device = *device;
        let coarser_view = views.get_texture_view(self.coarser_tex);
        let key = BindGroupKey::new(self.transient_layout.id()).with_resource(coarser_view.id());
        let layout = &**self.transient_layout;

        let bg = cache.get_or_create_bg(key, || {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bloom US G1"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(coarser_view),
                }],
            })
        });
        self.transient_bg = Some(bg);
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let transient_bg = self
            .transient_bg
            .expect("Bloom US transient BG not prepared");

        let rtt = ctx.get_color_attachment(self.output_tex, RenderTargetOps::Load, None);
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Bloom Upsample"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(self.pipeline);
        rpass.set_bind_group(0, self.static_bg, &[]);
        rpass.set_bind_group(1, transient_bg, &[]);
        rpass.draw(0..3, 0..1);
    }
}

struct BloomCompositeNode<'a> {
    original_tex: TextureNodeId,
    bloom_tex: TextureNodeId,
    output_tex: TextureNodeId,
    pipeline: &'a wgpu::RenderPipeline,
    static_bg: &'a wgpu::BindGroup,
    transient_layout: &'a Tracked<wgpu::BindGroupLayout>,
    transient_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for BloomCompositeNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let PrepareContext {
            views,
            global_bind_group_cache: cache,
            device,
            ..
        } = ctx;
        let device = *device;
        let original_view = views.get_texture_view(self.original_tex);
        let bloom_view = views.get_texture_view(self.bloom_tex);
        let key = BindGroupKey::new(self.transient_layout.id())
            .with_resource(original_view.id())
            .with_resource(bloom_view.id());
        let layout = &**self.transient_layout;

        let bg = cache.get_or_create_bg(key, || {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bloom Comp G1"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(original_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(bloom_view),
                    },
                ],
            })
        });
        self.transient_bg = Some(bg);
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let transient_bg = self
            .transient_bg
            .expect("Bloom Comp transient BG not prepared");

        let rtt = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Bloom Composite"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(self.pipeline);
        rpass.set_bind_group(0, self.static_bg, &[]);
        rpass.set_bind_group(1, transient_bg, &[]);
        rpass.draw(0..3, 0..1);
    }
}
