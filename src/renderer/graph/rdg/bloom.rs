//! RDG Bloom Post-Processing — Dual-Layer BindGroup Architecture.
//!
//! Implements the Call of Duty: Advanced Warfare physically-based bloom
//! technique within the RDG framework using a strict two-layer binding
//! separation:
//!
//! - **Group 0 (Static)**: Sampler + uniform buffer — built and owned by
//!   [`BloomFeature`], lifetime matches the engine.
//! - **Group 1 (Transient)**: RDG texture views — built by
//!   [`BloomPassNode`] each frame, cached via [`GlobalBindGroupCache`].
//!
//! # Architecture
//!
//! - **`BloomFeature`** – persistent owner of GPU pipelines, bind-group
//!   layouts, static uniform buffers, and Group 0 bind groups.  Initialised
//!   lazily by [`extract_and_prepare`](BloomFeature::extract_and_prepare).
//!
//! - **`BloomPassNode`** – ephemeral per-frame node added to the render
//!   graph by [`add_to_graph`](BloomFeature::add_to_graph).  Receives
//!   pre-built Group 0 bind groups and only assembles Group 1 (transient
//!   texture) bind groups during `prepare`.
//!
//! # RDG Slots (explicit wiring)
//!
//! - `input_tex`     – HDR scene color (input)
//! - `output_tex`    – HDR scene color with bloom composited (output)
//! - `bloom_texture` – internal mip chain

use crate::define_gpu_data_struct;
use crate::renderer::HDR_TEXTURE_FORMAT;
use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::resources::{CommonSampler, Tracked};
use crate::renderer::graph::rdg::allocator::SubViewKey;
use crate::renderer::graph::rdg::builder::PassBuilder;
use crate::renderer::graph::rdg::context::{ExtractContext, RdgExecuteContext, RdgPrepareContext};
use crate::renderer::graph::rdg::node::PassNode;
use crate::renderer::graph::rdg::types::{RdgTextureDesc, TextureNodeId};
use crate::renderer::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions,
};
use crate::resources::WgslType;
use crate::resources::bloom::{CompositeUniforms, UpsampleUniforms};
use crate::resources::buffer::CpuBuffer;
use crate::resources::uniforms::{UniformArray, WgslStruct};

use super::graph::RenderGraph;

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
/// [`add_to_graph`](Self::add_to_graph) to inject an ephemeral
/// [`BloomPassNode`] into the RDG.
///
/// # Dual-Layer BindGroup Model
///
/// Static bind groups (Group 0) containing samplers and uniform buffers
/// are built here and handed to the PassNode as cheap `Arc` clones.
/// The PassNode only assembles Group 1 (transient texture views).
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

        self.build_static_bind_groups(ctx, upsample_uniform.id(), composite_uniform.id());
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
        upsample_cpu_id: u64,
        composite_cpu_id: u64,
    ) {
        let sampler = ctx.sampler_registry.get_common(CommonSampler::LinearClamp);
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
        if let Some(g) = ctx.resource_manager.gpu_buffers.get(&upsample_cpu_id) {
            if self.upsample_static_bg.is_none() || self.last_upsample_buffer_id != g.id {
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
        }

        // ─── Composite static BG (rebuild on buffer identity change) ──
        if let Some(g) = ctx.resource_manager.gpu_buffers.get(&composite_cpu_id) {
            if self.composite_static_bg.is_none() || self.last_composite_buffer_id != g.id {
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
    }

    // =========================================================================
    // Graph Integration
    // =========================================================================

    /// Create an ephemeral [`BloomPassNode`] and add it to the render graph.
    ///
    /// The output texture (`Bloom_Out`) is registered here and returned as
    /// a [`TextureNodeId`] for explicit downstream wiring (ToneMap).
    /// The PassNode receives pre-built Group 0 static bind groups from the
    /// Feature and only assembles Group 1 (transient texture) bind groups
    /// during its prepare phase.
    pub fn add_to_graph(
        &self,
        rdg: &mut RenderGraph,
        input_color: TextureNodeId,
        karis_average: bool,
        max_mip_levels: u32,
    ) -> TextureNodeId {
        let fc = *rdg.frame_config();
        let output_desc = RdgTextureDesc::new_2d(
            fc.width,
            fc.height,
            fc.hdr_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );
        let output_tex = rdg.register_resource("Bloom_Out", output_desc, false);

        let node = BloomPassNode {
            input_tex: input_color,
            output_tex,
            bloom_texture: TextureNodeId(0),

            karis_average,
            max_mip_levels,

            downsample_pipeline: self
                .downsample_pipeline
                .expect("BloomFeature: downsample pipeline not initialised"),
            upsample_pipeline: self
                .upsample_pipeline
                .expect("BloomFeature: upsample pipeline not initialised"),
            composite_pipeline: self
                .composite_pipeline
                .expect("BloomFeature: composite pipeline not initialised"),

            karis_on_static_bg: self
                .karis_on_static_bg
                .clone()
                .expect("BloomFeature: karis_on static BG not built"),
            karis_off_static_bg: self
                .karis_off_static_bg
                .clone()
                .expect("BloomFeature: karis_off static BG not built"),
            upsample_static_bg: self
                .upsample_static_bg
                .clone()
                .expect("BloomFeature: upsample static BG not built"),
            composite_static_bg: self
                .composite_static_bg
                .clone()
                .expect("BloomFeature: composite static BG not built"),

            ds_transient_layout: self.ds_transient_layout.clone().unwrap(),
            us_transient_layout: self.us_transient_layout.clone().unwrap(),
            comp_transient_layout: self.comp_transient_layout.clone().unwrap(),

            mip_render_views: Vec::new(),
            ds_transient_bgs: Vec::new(),
            us_transient_bgs: Vec::new(),
            comp_transient_bg: None,
            last_input_view_id: 0,
        };

        rdg.add_pass(Box::new(node));
        output_tex
    }
}

// =============================================================================
// BloomPassNode — ephemeral per-frame render graph node
// =============================================================================

/// Ephemeral bloom pass node inserted into the RDG each frame.
///
/// Receives pre-built **Group 0** (static) bind groups from [`BloomFeature`]
/// as cheap `Arc` clones. Only assembles **Group 1** (transient texture)
/// bind groups during [`prepare`](PassNode::prepare), achieving clean
/// separation between persistent and per-frame GPU resources.
pub struct BloomPassNode {
    // ─── RDG Resource Slots ────────────────────────────────────────
    input_tex: TextureNodeId,
    output_tex: TextureNodeId,
    bloom_texture: TextureNodeId,

    // ─── Push Parameters ───────────────────────────────────────────
    karis_average: bool,
    max_mip_levels: u32,

    // ─── Pipeline IDs ──────────────────────────────────────────────
    downsample_pipeline: RenderPipelineId,
    upsample_pipeline: RenderPipelineId,
    composite_pipeline: RenderPipelineId,

    // ─── Static BindGroups (Group 0, from Feature) ─────────────────
    /// Downsample Group 0 with karis on (sampler + karis_on uniforms).
    karis_on_static_bg: wgpu::BindGroup,
    /// Downsample Group 0 with karis off (sampler + karis_off uniforms).
    karis_off_static_bg: wgpu::BindGroup,
    /// Upsample Group 0 (sampler + UpsampleUniforms).
    upsample_static_bg: wgpu::BindGroup,
    /// Composite Group 0 (sampler + CompositeUniforms).
    composite_static_bg: wgpu::BindGroup,

    // ─── Transient Layouts (Group 1) ───────────────────────────────
    ds_transient_layout: Tracked<wgpu::BindGroupLayout>,
    us_transient_layout: Tracked<wgpu::BindGroupLayout>,
    comp_transient_layout: Tracked<wgpu::BindGroupLayout>,

    // ─── Per-Mip Render Views (resolved in prepare) ────────────────
    mip_render_views: Vec<Tracked<wgpu::TextureView>>,

    // ─── Transient BindGroups (Group 1, built in prepare) ──────────
    /// `ds_transient_bgs[i]` binds the input view for downsample step i.
    ds_transient_bgs: Vec<wgpu::BindGroup>,
    /// `us_transient_bgs[i]` binds the source mip view for upsample step i.
    us_transient_bgs: Vec<wgpu::BindGroup>,
    /// Composite Group 1: original + bloom mip 0 textures.
    comp_transient_bg: Option<wgpu::BindGroup>,

    /// Tracked ID of the input texture view for staleness detection.
    last_input_view_id: u64,
}

impl BloomPassNode {
    // =========================================================================
    // Mip Chain Management via RdgViewResolver
    // =========================================================================

    /// Populate per-mip render views from the pool's sub-view cache.
    fn resolve_mip_views(&mut self, ctx: &mut RdgPrepareContext) {
        let mip_count = ctx.views.get_texture(self.bloom_texture).mip_level_count();

        self.mip_render_views.clear();
        // self.mip_view_ids.clear();

        for mip in 0..mip_count {
            let key = SubViewKey {
                base_mip: mip,
                mip_count: Some(1),
                ..Default::default()
            };
            let tracked = ctx.views.get_or_create_sub_view(self.bloom_texture, key);
            // self.mip_view_ids.push(tracked.id());
            self.mip_render_views.push(tracked.clone());
        }
    }

    /// Build all transient (Group 1) bind groups.
    ///
    /// Group 1 only contains RDG texture views — no samplers or uniform
    /// buffers. Cache keys are trivially small, yielding high hit rates.
    fn rebuild_transient_bind_groups(&mut self, ctx: &mut RdgPrepareContext) {
        let input_view = ctx.views.get_texture_view(self.input_tex);
        let input_view_id = input_view.id();
        let mip_count = self.mip_render_views.len();

        let needs_full_rebuild =
            self.ds_transient_bgs.len() != mip_count || self.last_input_view_id != input_view_id;

        // ─── First DS transient BG (scene → mip 0) ────────────────
        let first_ds_key =
            BindGroupKey::new(self.ds_transient_layout.id()).with_resource(input_view_id);

        let first_ds_bg = ctx
            .global_bind_group_cache
            .get_or_create(first_ds_key, || {
                ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bloom DS G1 scene→0"),
                    layout: &self.ds_transient_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_view),
                    }],
                })
            })
            .clone();

        if needs_full_rebuild {
            // ─── Downsample transient BGs ──────────────────────────
            self.ds_transient_bgs.clear();
            self.ds_transient_bgs.push(first_ds_bg);

            for i in 0..(mip_count - 1) {
                let mip_view = &self.mip_render_views[i];
                let layout = &self.ds_transient_layout;

                let key = BindGroupKey::new(layout.id()).with_resource(mip_view.id());
                let bg = ctx
                    .global_bind_group_cache
                    .get_or_create(key, || {
                        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("Bloom DS G1 mip→mip"),
                            layout,
                            entries: &[wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(mip_view),
                            }],
                        })
                    })
                    .clone();
                self.ds_transient_bgs.push(bg);
            }

            // ─── Upsample transient BGs ────────────────────────────
            self.us_transient_bgs.clear();
            for i in 0..(mip_count - 1) {
                let source_mip = i + 1;
                let mip_view = &self.mip_render_views[source_mip];
                let layout = &self.us_transient_layout;

                let key = BindGroupKey::new(layout.id()).with_resource(mip_view.id());
                let bg = ctx
                    .global_bind_group_cache
                    .get_or_create(key, || {
                        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("Bloom US G1 mip→mip"),
                            layout,
                            entries: &[wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(mip_view),
                            }],
                        })
                    })
                    .clone();
                self.us_transient_bgs.push(bg);
            }

            self.last_input_view_id = input_view_id;
        } else {
            // Only the first DS BG needs updating (input view may differ).
            self.ds_transient_bgs[0] = first_ds_bg;
        }

        // ─── Composite transient BG ───────────────────────────────
        if needs_full_rebuild || self.comp_transient_bg.is_none() {
            let bloom_view = &self.mip_render_views[0];
            let layout = &self.comp_transient_layout;

            let key = BindGroupKey::new(layout.id())
                .with_resource(input_view_id)
                .with_resource(bloom_view.id());

            let bg = ctx
                .global_bind_group_cache
                .get_or_create(key, || {
                    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Bloom Comp G1"),
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(input_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(bloom_view),
                            },
                        ],
                    })
                })
                .clone();

            self.comp_transient_bg = Some(bg);
        }
    }
}

// =============================================================================
// PassNode implementation
// =============================================================================

impl PassNode for BloomPassNode {
    fn name(&self) -> &'static str {
        "RDG_Bloom_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        let (w, h) = builder.global_resolution();
        let hdr_format = builder.frame_config().hdr_format;

        // Output (pre-registered in add_to_graph).
        builder.write_texture(self.output_tex);

        // Internal mip chain.
        let bloom_w = (w / 2).max(1);
        let bloom_h = (h / 2).max(1);
        let max_possible = ((bloom_w.max(bloom_h) as f32).log2().floor() as u32) + 1;
        let mip_count = self.max_mip_levels.min(max_possible).max(1);

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
        self.bloom_texture = builder.create_texture("Bloom_MipChain", bloom_chain_desc);

        // Input: scene color (explicit wiring).
        builder.read_texture(self.input_tex);
    }

    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {
        self.resolve_mip_views(ctx);
        self.rebuild_transient_bind_groups(ctx);
    }

    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        if self.mip_render_views.is_empty() {
            return;
        }

        let Some(comp_bg) = &self.comp_transient_bg else {
            return;
        };

        let ds_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.downsample_pipeline);
        let us_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.upsample_pipeline);
        let comp_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.composite_pipeline);

        // =====================================================================
        // Phase 1: Downsample — Scene HDR → Bloom Mip Chain
        // =====================================================================
        for i in 0..self.ds_transient_bgs.len() {
            // First mip uses karis-aware static BG; subsequent mips use karis-off.
            let static_bg = if i == 0 && self.karis_average {
                &self.karis_on_static_bg
            } else {
                &self.karis_off_static_bg
            };

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Downsample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.mip_render_views[i],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::DontCare(wgpu::LoadOpDontCare::default()),
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
            pass.set_bind_group(0, static_bg, &[]);
            pass.set_bind_group(1, &self.ds_transient_bgs[i], &[]);
            pass.draw(0..3, 0..1);
        }

        // =====================================================================
        // Phase 2: Upsample — Accumulate bloom from coarsest to finest
        // =====================================================================
        for i in (0..self.us_transient_bgs.len()).rev() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Upsample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.mip_render_views[i],
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
            pass.set_bind_group(0, &self.upsample_static_bg, &[]);
            pass.set_bind_group(1, &self.us_transient_bgs[i], &[]);
            pass.draw(0..3, 0..1);
        }

        // =====================================================================
        // Phase 3: Composite — Original HDR + Bloom → Output
        // =====================================================================
        // let output_view = ctx.get_texture_view(self.output_tex);

        let rtt = ctx.get_color_attachment(self.output_tex, None, None);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Bloom Composite"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(comp_pipeline);
        pass.set_bind_group(0, &self.composite_static_bg, &[]);
        pass.set_bind_group(1, comp_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
