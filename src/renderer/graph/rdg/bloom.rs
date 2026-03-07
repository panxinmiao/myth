//! RDG Bloom Post-Processing — Feature + PassNode split.
//!
//! Implements the Call of Duty: Advanced Warfare physically-based bloom
//! technique within the RDG framework.
//!
//! # Architecture
//!
//! - **`BloomFeature`** – persistent owner of GPU pipelines, bind-group
//!   layouts, and static uniform buffers.  Initialised lazily by
//!   [`extract_and_prepare`](BloomFeature::extract_and_prepare) (called
//!   once per frame *before* the render graph is built).
//!
//! - **`BloomPassNode`** – ephemeral per-frame node added to the render
//!   graph by [`add_to_graph`](BloomFeature::add_to_graph).  Carries
//!   cloned handles (pipeline IDs, layout/buffer clones, push params)
//!   and owns transient mip-chain state.
//!
//! # RDG Slots (declared by BloomPassNode::setup)
//!
//! - `input_tex`     – HDR scene color (read via blackboard)
//! - `output_tex`    – HDR scene color with bloom composited
//! - `bloom_texture` – internal mip chain
//!
//! # Push Model
//!
//! All scene-level parameters (`karis_average`, `max_mip_levels`,
//! uniform buffer IDs) are passed into `add_to_graph` by the Composer.

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

/// Persistent bloom feature that owns GPU pipelines, layouts, and buffers.
///
/// The Composer calls [`extract_and_prepare`](Self::extract_and_prepare)
/// once per frame before the render graph is built, then
/// [`add_to_graph`](Self::add_to_graph) to inject an ephemeral
/// [`BloomPassNode`] into the RDG.
pub struct BloomFeature {
    // ─── Pipelines ─────────────────────────────────────────────────
    downsample_pipeline: Option<RenderPipelineId>,
    upsample_pipeline: Option<RenderPipelineId>,
    composite_pipeline: Option<RenderPipelineId>,

    // ─── Bind Group Layouts ────────────────────────────────────────
    downsample_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    upsample_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    composite_layout: Option<Tracked<wgpu::BindGroupLayout>>,

    // ─── Internal Static Uniform Buffers ───────────────────────────
    /// GPU buffer with `use_karis_average = 1`, written once.
    karis_on_buffer: Option<Tracked<wgpu::Buffer>>,
    /// GPU buffer with `use_karis_average = 0`, written once.
    karis_off_buffer: Option<Tracked<wgpu::Buffer>>,

    // ─── Resolved GPU Buffers (populated per-frame in extract_and_prepare) ──
    /// Resolved GPU buffer for upsample uniforms.
    upsample_gpu_buffer: Option<wgpu::Buffer>,
    /// Stable resource ID for the upsample GPU buffer (for bind-group cache key).
    upsample_gpu_buffer_id: u64,
    /// Resolved GPU buffer for composite uniforms.
    composite_gpu_buffer: Option<wgpu::Buffer>,
    /// Stable resource ID for the composite GPU buffer (for bind-group cache key).
    composite_gpu_buffer_id: u64,
}

impl BloomFeature {
    /// Creates a new bloom feature. All GPU resources are lazily allocated.
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

            upsample_gpu_buffer: None,
            upsample_gpu_buffer_id: 0,
            composite_gpu_buffer: None,
            composite_gpu_buffer_id: 0,
        }
    }

    // =========================================================================
    // Extract & Prepare (called before RDG build)
    // =========================================================================

    /// Ensure all persistent GPU resources (layouts, static buffers, pipelines)
    /// are initialised.  Called once per frame by the Composer before the
    /// render graph is constructed.
    /// Pre-RDG resource preparation: create layouts, compile pipelines,
    /// resolve GPU buffers for upsample/composite uniforms.
    ///
    /// `upsample_cpu_id` and `composite_cpu_id` are the CpuBuffer IDs
    /// whose GPU mirrors have already been uploaded via `ensure_buffer()`.
    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        upsample_cpu_id: u64,
        composite_cpu_id: u64,
    ) {
        self.ensure_layouts(ctx.device);
        self.ensure_internal_buffers(ctx.device, ctx.queue);
        self.ensure_pipelines(ctx);

        // Pre-resolve GPU buffers so the PassNode does not need ResourceManager.
        self.upsample_gpu_buffer = ctx
            .resource_manager
            .gpu_buffers
            .get(&upsample_cpu_id)
            .map(|g| {
                self.upsample_gpu_buffer_id = g.id;
                g.buffer.clone()
            });
        self.composite_gpu_buffer = ctx
            .resource_manager
            .gpu_buffers
            .get(&composite_cpu_id)
            .map(|g| {
                self.composite_gpu_buffer_id = g.id;
                g.buffer.clone()
            });
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

    // =========================================================================
    // Graph Integration
    // =========================================================================

    /// Create an ephemeral [`BloomPassNode`] and add it to the render graph.
    ///
    /// Returns the [`TextureNodeId`] of the `"Bloom_Out"` resource created
    /// by the node during setup.
    pub fn add_to_graph(
        &self,
        rdg: &mut RenderGraph,
        karis_average: bool,
        max_mip_levels: u32,
    ) -> TextureNodeId {
        let node = BloomPassNode {
            input_tex: TextureNodeId(0),
            output_tex: TextureNodeId(0),
            bloom_texture: TextureNodeId(0),

            karis_average,
            max_mip_levels,

            upsample_gpu_buffer: self
                .upsample_gpu_buffer
                .clone()
                .expect("BloomFeature: upsample GPU buffer not resolved"),
            _upsample_gpu_buffer_id: self.upsample_gpu_buffer_id,
            composite_gpu_buffer: self
                .composite_gpu_buffer
                .clone()
                .expect("BloomFeature: composite GPU buffer not resolved"),
            composite_gpu_buffer_id: self.composite_gpu_buffer_id,

            downsample_pipeline: self
                .downsample_pipeline
                .expect("BloomFeature: downsample pipeline not initialised"),
            upsample_pipeline: self
                .upsample_pipeline
                .expect("BloomFeature: upsample pipeline not initialised"),
            composite_pipeline: self
                .composite_pipeline
                .expect("BloomFeature: composite pipeline not initialised"),

            downsample_layout: self
                .downsample_layout
                .clone()
                .expect("BloomFeature: downsample layout not initialised"),
            upsample_layout: self
                .upsample_layout
                .clone()
                .expect("BloomFeature: upsample layout not initialised"),
            composite_layout: self
                .composite_layout
                .clone()
                .expect("BloomFeature: composite layout not initialised"),

            karis_on_buffer: self
                .karis_on_buffer
                .clone()
                .expect("BloomFeature: karis_on buffer not initialised"),
            karis_off_buffer: self
                .karis_off_buffer
                .clone()
                .expect("BloomFeature: karis_off buffer not initialised"),

            mip_render_views: Vec::new(),
            mip_view_ids: Vec::new(),
            downsample_bind_groups: Vec::new(),
            upsample_bind_groups: Vec::new(),
            composite_bind_group: None,
            last_input_view_id: 0,
        };

        rdg.add_pass(Box::new(node));
        rdg.find_resource("Bloom_Out")
            .expect("Bloom_Out must be registered by BloomPassNode::setup")
    }
}

// =============================================================================
// BloomPassNode — ephemeral per-frame render graph node
// =============================================================================

/// Ephemeral bloom pass node inserted into the RDG each frame.
///
/// All persistent GPU handles are **cloned** from [`BloomFeature`] at
/// creation time; transient mip-chain views and bind groups are built
/// during [`prepare`](PassNode::prepare).
pub struct BloomPassNode {
    // ─── RDG Resource Slots (filled in setup) ──────────────────────
    input_tex: TextureNodeId,
    output_tex: TextureNodeId,
    bloom_texture: TextureNodeId,

    // ─── Push Parameters (from Composer) ───────────────────────────
    karis_average: bool,
    max_mip_levels: u32,

    // ─── Resolved GPU Buffers (from Feature, no ResourceManager needed) ──
    /// Resolved GPU buffer for upsample uniforms.
    upsample_gpu_buffer: wgpu::Buffer,
    /// Stable resource ID for the upsample GPU buffer.
    _upsample_gpu_buffer_id: u64,
    /// Resolved GPU buffer for composite uniforms.
    composite_gpu_buffer: wgpu::Buffer,
    /// Stable resource ID for the composite GPU buffer.
    composite_gpu_buffer_id: u64,

    // ─── Pipeline IDs (cloned from Feature, non-Option) ────────────
    downsample_pipeline: RenderPipelineId,
    upsample_pipeline: RenderPipelineId,
    composite_pipeline: RenderPipelineId,

    // ─── Bind Group Layout Clones ──────────────────────────────────
    downsample_layout: Tracked<wgpu::BindGroupLayout>,
    upsample_layout: Tracked<wgpu::BindGroupLayout>,
    composite_layout: Tracked<wgpu::BindGroupLayout>,

    // ─── Karis Buffer Clones ───────────────────────────────────────
    /// GPU buffer with `use_karis_average = 1`.
    karis_on_buffer: Tracked<wgpu::Buffer>,
    /// GPU buffer with `use_karis_average = 0`.
    karis_off_buffer: Tracked<wgpu::Buffer>,

    // ─── Per-Mip Render Views (populated during prepare) ───────────
    /// Cloned `TextureView` handles for use as render targets in execute.
    /// Indexed by mip level.  Populated from the pool's sub-view cache
    /// via [`RdgViewResolver::get_or_create_sub_view`].
    mip_render_views: Vec<wgpu::TextureView>,
    /// Stable Tracked IDs for each mip sub-view (parallel to `mip_render_views`).
    /// Used to detect changes and construct bind group cache keys.
    mip_view_ids: Vec<u64>,

    // ─── Cached BindGroups ─────────────────────────────────────────
    /// `downsample_bind_groups[i]` binds mip_views[i] → mip_views[i+1].
    /// Index 0 binds the scene color input → mip_views[0].
    downsample_bind_groups: Vec<wgpu::BindGroup>,
    /// `upsample_bind_groups[i]` binds mip_views[i+1] → mip_views[i].
    upsample_bind_groups: Vec<wgpu::BindGroup>,
    /// Composite BindGroup: original + bloom mip0 → output.
    composite_bind_group: Option<wgpu::BindGroup>,

    /// Tracked ID of the input texture view; when it changes, first DS BG must rebuild.
    last_input_view_id: u64,
}

impl BloomPassNode {
    // =========================================================================
    // Mip Chain Management via RdgViewResolver
    // =========================================================================

    /// Populate per-mip render views from the pool's sub-view cache.
    ///
    /// Uses [`RdgViewResolver::get_or_create_sub_view`] so that the same
    /// physical texture always maps to the same `Tracked` ID, giving near
    /// 100 % bind-group cache hit rate across frames.
    fn resolve_mip_views(&mut self, ctx: &mut RdgPrepareContext) {
        let mip_count = ctx.views.get_texture(self.bloom_texture).mip_level_count();

        self.mip_render_views.clear();
        self.mip_view_ids.clear();

        for mip in 0..mip_count {
            let key = SubViewKey {
                base_mip: mip,
                mip_count: Some(1),
                ..Default::default()
            };
            // Create-or-reuse in the pool; the returned ID is stable.
            let tracked = ctx.views.get_or_create_sub_view(self.bloom_texture, key);
            self.mip_view_ids.push(tracked.id());
            self.mip_render_views.push((**tracked).clone());
        }
    }

    /// Build the first downsample bind group (scene colour → mip 0).
    fn build_first_downsample_bg(&self, ctx: &mut RdgPrepareContext) -> wgpu::BindGroup {
        let karis_buf = if self.karis_average {
            &self.karis_on_buffer
        } else {
            &self.karis_off_buffer
        };

        let input_view = ctx.views.get_texture_view(self.input_tex);
        let input_view_id = input_view.id();
        let sampler = ctx.sampler_registry.get_common(CommonSampler::LinearClamp);

        let key = BindGroupKey::new(self.downsample_layout.id())
            .with_resource(input_view_id)
            .with_resource(sampler.id())
            .with_resource(karis_buf.id());

        if let Some(cached) = ctx.global_bind_group_cache.get(&key) {
            return cached.clone();
        }

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
    }

    /// Build the composite bind group (original + bloom mip 0 → output).
    fn build_composite_bg(&self, ctx: &mut RdgPrepareContext) -> wgpu::BindGroup {
        let input_view = ctx.views.get_texture_view(self.input_tex);
        let input_view_id = input_view.id();
        let bloom_view_id = self.mip_view_ids[0];
        let sampler = ctx.sampler_registry.get_common(CommonSampler::LinearClamp);

        let key = BindGroupKey::new(self.composite_layout.id())
            .with_resource(input_view_id)
            .with_resource(bloom_view_id)
            .with_resource(sampler.id())
            .with_resource(self.composite_gpu_buffer_id);

        if let Some(cached) = ctx.global_bind_group_cache.get(&key) {
            return cached.clone();
        }

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bloom Composite BG"),
            layout: &self.composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.mip_render_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.composite_gpu_buffer.as_entire_binding(),
                },
            ],
        });
        ctx.global_bind_group_cache.insert(key, bg.clone());
        bg
    }

    /// Build all internal bind groups.
    ///
    /// Checks whether the input view or mip-view IDs have changed and
    /// performs a full or partial rebuild accordingly.
    fn rebuild_bind_groups(&mut self, ctx: &mut RdgPrepareContext) {
        let input_view_id = ctx.views.get_texture_view(self.input_tex).id();
        let mip_count = self.mip_render_views.len();

        let needs_full_rebuild = self.downsample_bind_groups.len() != mip_count
            || self.last_input_view_id != input_view_id;

        let first_ds_bg = self.build_first_downsample_bg(ctx);

        if needs_full_rebuild {
            let sampler = ctx.sampler_registry.get_common(CommonSampler::LinearClamp);

            // ─── Downsample BGs ────────────────────────────────────
            self.downsample_bind_groups.clear();
            self.downsample_bind_groups.push(first_ds_bg);

            for i in 0..(mip_count - 1) {
                let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RDG Bloom DS BG mip→mip"),
                    layout: &self.downsample_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &self.mip_render_views[i],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&**sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.karis_off_buffer.as_entire_binding(),
                        },
                    ],
                });
                self.downsample_bind_groups.push(bg);
            }

            // ─── Upsample BGs ─────────────────────────────────────
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
                                &self.mip_render_views[source_mip],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&**sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.upsample_gpu_buffer.as_entire_binding(),
                        },
                    ],
                });
                self.upsample_bind_groups.push(bg);
            }

            self.last_input_view_id = input_view_id;
        } else {
            self.downsample_bind_groups[0] = first_ds_bg;
        }

        if needs_full_rebuild || self.composite_bind_group.is_none() {
            self.composite_bind_group = Some(self.build_composite_bg(ctx));
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
        let hdr_desc = RdgTextureDesc::new_2d(
            w,
            h,
            hdr_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );

        // Producer: create the bloom output and mip chain textures.
        self.output_tex = builder.create_texture("Bloom_Out", hdr_desc);

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

        // Consumer: read scene color input.
        self.input_tex = builder.read_blackboard("Scene_Color_HDR");
    }

    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {
        // Resolve mip views from the pool's sub-view cache (stable Tracked IDs).
        self.resolve_mip_views(ctx);
        // Assemble transient bind groups using resolved views.
        self.rebuild_bind_groups(ctx);
    }

    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        if self.mip_render_views.is_empty() {
            return;
        }

        let Some(comp_bg) = &self.composite_bind_group else {
            return;
        };

        let ds_pipeline = ctx.pipeline_cache.get_render_pipeline(self.downsample_pipeline);
        let us_pipeline = ctx.pipeline_cache.get_render_pipeline(self.upsample_pipeline);
        let comp_pipeline = ctx.pipeline_cache.get_render_pipeline(self.composite_pipeline);

        // =====================================================================
        // Phase 1: Downsample — Scene HDR → Bloom Mip Chain
        // =====================================================================
        for i in 0..self.downsample_bind_groups.len() {
            let target_mip = i as u32;

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("RDG Bloom Downsample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.mip_render_views[target_mip as usize],
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

        // =====================================================================
        // Phase 2: Upsample — Accumulate bloom from coarsest to finest
        // =====================================================================
        for i in (0..self.upsample_bind_groups.len()).rev() {
            let target_mip = i as u32;

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("RDG Bloom Upsample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.mip_render_views[target_mip as usize],
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

        // =====================================================================
        // Phase 3: Composite — Original HDR + Bloom → Output
        // =====================================================================
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
