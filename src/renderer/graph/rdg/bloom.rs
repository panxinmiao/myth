//! RDG Bloom Post-Processing Pass (Macro Node)
//!
//! Implements the Call of Duty: Advanced Warfare physically-based bloom
//! technique within the new RDG framework. This pass is a **macro node**:
//! the RDG only sees a single (input → output) edge while the internal
//! downsample-upsample-composite pipeline is managed entirely inside the
//! pass.
//!
//! # RDG Slots
//!
//! - `input_tex`: HDR scene color
//! - `output_tex`: HDR scene color with bloom composited
//!
//! # Internal Resources (not registered in RDG)
//!
//! - A multi-mip bloom texture for the downsample/upsample chain
//! - Three pipelines: downsample, upsample (additive), composite
//! - Per-mip bind groups (cached, rebuilt only on resolution change)
//!
//! # Push Model
//!
//! All scene-level parameters (`enabled`, `karis_average`, `max_mip_levels`,
//! uniform buffer IDs) are set externally by the Composer.

use crate::define_gpu_data_struct;
use crate::renderer::core::resources::Tracked;
use crate::renderer::graph::rdg::builder::PassBuilder;
use crate::renderer::graph::rdg::context::{RdgExecuteContext, RdgPrepareContext};
use crate::renderer::graph::rdg::node::PassNode;
use crate::renderer::graph::rdg::types::TextureNodeId;
use crate::renderer::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions,
};
use crate::renderer::HDR_TEXTURE_FORMAT;
use crate::resources::WgslType;
use crate::resources::bloom::{CompositeUniforms, UpsampleUniforms};
use crate::resources::uniforms::{UniformArray, WgslStruct};

use std::sync::atomic::{AtomicU64, Ordering};

/// Generate unique IDs for internal GPU buffers (not Tracked resources).
fn next_internal_id() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(1_000_000);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

define_gpu_data_struct!(
    /// Internal GPU uniform for the downsample shader (karis on/off flag).
    struct DownsampleUniforms {
        pub use_karis_average: u32,
        pub(crate) __pad: UniformArray<u32, 3>,
    }
);

/// Physically-based bloom pass (macro node).
///
/// The Composer pushes scene-level bloom parameters before the RDG prepare
/// loop. The pass internally manages its own mip chain texture and bind
/// groups, completely independent of the RDG resource graph.
pub struct RdgBloomPass {
    // ─── RDG Resource Slots (set by Composer) ──────────────────────
    pub input_tex: TextureNodeId,
    pub output_tex: TextureNodeId,

    // ─── Push Parameters (set by Composer from Scene) ──────────────
    pub karis_average: bool,
    pub max_mip_levels: u32,
    /// CPU-side buffer ID for `CpuBuffer<UpsampleUniforms>`.
    pub upsample_uniforms_cpu_id: u64,
    /// CPU-side buffer ID for `CpuBuffer<CompositeUniforms>`.
    pub composite_uniforms_cpu_id: u64,

    // ─── Pipelines ─────────────────────────────────────────────────
    downsample_pipeline: Option<RenderPipelineId>,
    upsample_pipeline: Option<RenderPipelineId>,
    composite_pipeline: Option<RenderPipelineId>,

    // ─── Bind Group Layouts ────────────────────────────────────────
    downsample_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    upsample_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    composite_layout: Option<Tracked<wgpu::BindGroupLayout>>,

    // ─── Shared Sampler ────────────────────────────────────────────
    sampler: Option<Tracked<wgpu::Sampler>>,

    // ─── Internal Static Uniform Buffers ───────────────────────────
    /// GPU buffer with `use_karis_average = 1`, written once.
    karis_on_buffer: Option<(wgpu::Buffer, u64)>,
    /// GPU buffer with `use_karis_average = 0`, written once.
    karis_off_buffer: Option<(wgpu::Buffer, u64)>,

    // ─── Internal Mip Chain ────────────────────────────────────────
    /// The bloom mip chain texture (owned by this pass, NOT by RDG).
    bloom_texture: Option<wgpu::Texture>,
    /// Per-mip views into the bloom texture.
    mip_views: Vec<Tracked<wgpu::TextureView>>,
    /// Actual number of mip levels in the current allocation.
    current_mip_count: u32,
    /// Size of mip 0 (half of source resolution).
    current_bloom_size: (u32, u32),

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

impl RdgBloomPass {
    /// Creates a new bloom pass. All GPU resources are lazily allocated.
    #[must_use]
    pub fn new() -> Self {
        Self {
            input_tex: TextureNodeId(0),
            output_tex: TextureNodeId(0),

            karis_average: true,
            max_mip_levels: 6,
            upsample_uniforms_cpu_id: 0,
            composite_uniforms_cpu_id: 0,

            downsample_pipeline: None,
            upsample_pipeline: None,
            composite_pipeline: None,

            downsample_layout: None,
            upsample_layout: None,
            composite_layout: None,

            sampler: None,
            karis_on_buffer: None,
            karis_off_buffer: None,

            bloom_texture: None,
            mip_views: Vec::new(),
            current_mip_count: 0,
            current_bloom_size: (0, 0),

            downsample_bind_groups: Vec::new(),
            upsample_bind_groups: Vec::new(),
            composite_bind_group: None,

            last_input_view_id: 0,
        }
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

        // Sampler — shared between all internal sub-passes
        self.sampler = Some(Tracked::new(device.create_sampler(
            &wgpu::SamplerDescriptor {
                label: Some("RDG Bloom Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
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
        let id_on = next_internal_id();

        let buf_off = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RDG Bloom Karis Off"),
            size: std::mem::size_of::<DownsampleUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf_off, 0, bytemuck::bytes_of(&karis_off_data));
        let id_off = next_internal_id();

        self.karis_on_buffer = Some((buf_on, id_on));
        self.karis_off_buffer = Some((buf_off, id_off));
    }

    fn ensure_pipelines(&mut self, ctx: &mut RdgPrepareContext) {
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
    // Mip Chain Management
    // =========================================================================

    /// Allocate the bloom mip chain texture and rebuild internal bind groups
    /// when the resolution or input view changes.
    fn ensure_mip_chain(&mut self, ctx: &mut RdgPrepareContext) {
        let input_desc = &ctx.graph.resources[self.input_tex.0 as usize].desc;
        let source_w = input_desc.size.width;
        let source_h = input_desc.size.height;

        let bloom_w = (source_w / 2).max(1);
        let bloom_h = (source_h / 2).max(1);

        let max_possible = ((bloom_w.max(bloom_h) as f32).log2().floor() as u32) + 1;
        let mip_count = self.max_mip_levels.min(max_possible).max(1);

        // Check if recreation is needed
        let size_changed = self.current_bloom_size != (bloom_w, bloom_h)
            || self.current_mip_count != mip_count;

        if size_changed {
            // Recreate bloom texture
            let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("RDG Bloom Mip Chain"),
                size: wgpu::Extent3d {
                    width: bloom_w,
                    height: bloom_h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: mip_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HDR_TEXTURE_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            // Create per-mip views
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

            self.bloom_texture = Some(texture);
            self.current_mip_count = mip_count;
            self.current_bloom_size = (bloom_w, bloom_h);

            // Force bind group rebuild
            self.last_input_view_id = 0;
            self.downsample_bind_groups.clear();
            self.upsample_bind_groups.clear();
            self.composite_bind_group = None;
        }
    }

    /// Build all internal bind groups. Called every frame since the first
    /// downsample BG and composite BG depend on the input scene color view,
    /// which may change due to RDG memory aliasing.
    fn rebuild_bind_groups(&mut self, ctx: &mut RdgPrepareContext) {
        let input_view = ctx.get_physical_texture(self.input_tex);
        let input_view_id = input_view.id();

        let mip_count = self.current_mip_count;
        let sampler = self.sampler.as_ref().unwrap();
        let ds_layout = self.downsample_layout.as_ref().unwrap();
        let us_layout = self.upsample_layout.as_ref().unwrap();
        let comp_layout = self.composite_layout.as_ref().unwrap();

        // Check if the input view or mip0 changed
        let _mip0_view_id = self.mip_views[0].id();
        let needs_full_rebuild = self.downsample_bind_groups.len() != mip_count as usize
            || self.last_input_view_id != input_view_id;

        // ─── First downsample BG (scene color → mip0) ─────────────
        // Always rebuild since input_tex may alias to a different physical texture.
        let karis_buf = if self.karis_average {
            self.karis_on_buffer.as_ref().unwrap()
        } else {
            self.karis_off_buffer.as_ref().unwrap()
        };

        let first_ds_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RDG Bloom DS BG scene→mip0"),
            layout: ds_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&**input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&**sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: karis_buf.0.as_entire_binding(),
                },
            ],
        });

        if needs_full_rebuild {
            self.downsample_bind_groups.clear();
            self.downsample_bind_groups.push(first_ds_bg);

            let karis_off_buf = self.karis_off_buffer.as_ref().unwrap();

            // Remaining downsample BGs: mip[i] → mip[i+1]
            for i in 0..(mip_count - 1) as usize {
                let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RDG Bloom DS BG mip→mip"),
                    layout: ds_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &*self.mip_views[i],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&**sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: karis_off_buf.0.as_entire_binding(),
                        },
                    ],
                });
                self.downsample_bind_groups.push(bg);
            }

            // Upsample BGs: mip[i+1] → mip[i]
            let upsample_gpu = ctx
                .resource_manager
                .gpu_buffers
                .get(&self.upsample_uniforms_cpu_id)
                .expect("RDG Bloom: upsample GPU buffer must exist");

            self.upsample_bind_groups.clear();
            for i in 0..(mip_count - 1) as usize {
                let source_mip = i + 1;
                let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RDG Bloom US BG mip→mip"),
                    layout: us_layout,
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

            self.last_input_view_id = input_view_id;
        } else {
            // Only rebuild the first DS BG (input may have changed)
            self.downsample_bind_groups[0] = first_ds_bg;
        }

        // ─── Composite BG (original + bloom mip0 → output) ────────
        // Always rebuild since input_tex view may alias differently each frame.
        let composite_gpu = ctx
            .resource_manager
            .gpu_buffers
            .get(&self.composite_uniforms_cpu_id)
            .expect("RDG Bloom: composite GPU buffer must exist");

        self.composite_bind_group = Some(ctx.device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("RDG Bloom Composite BG"),
                layout: comp_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&**input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &*self.mip_views[0],
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&**sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: composite_gpu.buffer.as_entire_binding(),
                    },
                ],
            },
        ));
    }
}

impl PassNode for RdgBloomPass {
    fn name(&self) -> &'static str {
        "RDG_Bloom_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        builder.read_texture(self.input_tex);
        builder.write_texture(self.output_tex);
    }

    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {
        // 1. Lazy init
        self.ensure_layouts(ctx.device);
        self.ensure_internal_buffers(ctx.device, ctx.queue);
        self.ensure_pipelines(ctx);

        // 2. Mip chain
        self.ensure_mip_chain(ctx);

        // 3. Bind groups
        self.rebuild_bind_groups(ctx);
    }

    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        if self.current_mip_count == 0 {
            return;
        }

        let Some(ds_pipeline_id) = self.downsample_pipeline else {
            return;
        };
        let Some(us_pipeline_id) = self.upsample_pipeline else {
            return;
        };
        let Some(comp_pipeline_id) = self.composite_pipeline else {
            return;
        };
        let Some(comp_bg) = &self.composite_bind_group else {
            return;
        };

        let ds_pipeline = ctx.pipeline_cache.get_render_pipeline(ds_pipeline_id);
        let us_pipeline = ctx.pipeline_cache.get_render_pipeline(us_pipeline_id);
        let comp_pipeline = ctx.pipeline_cache.get_render_pipeline(comp_pipeline_id);

        // =====================================================================
        // Phase 1: Downsample — Scene HDR → Bloom Mip Chain
        // =====================================================================
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

        // =====================================================================
        // Phase 2: Upsample — Accumulate bloom from coarsest to finest
        // =====================================================================
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
