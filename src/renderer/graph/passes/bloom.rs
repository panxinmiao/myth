//! Physically-Based Bloom Post-Processing Pass
//!
//! Implements the bloom technique from *Call of Duty: Advanced Warfare*
//! (SIGGRAPH 2014), adapted for the Myth engine's render graph.
//!
//! # Algorithm
//!
//! 1. **Downsample**: Progressive 13-tap filter from the HDR scene buffer
//!    into a mip chain (half resolution at each level). The first level
//!    optionally applies Karis averaging to suppress firefly artifacts.
//!
//! 2. **Upsample**: 3×3 tent filter from the coarsest mip back up,
//!    additively blending each level into the next finer one.
//!
//! 3. **Composite**: The accumulated bloom result at mip 0 is combined
//!    with the original HDR scene color using additive blending controlled
//!    by `BloomSettings.strength`.
//!
//! # Data Flow
//!
//! ```text
//! Scene.bloom (BloomSettings — Source of Truth)
//!        │
//!        ▼  (version check in prepare())
//! BloomPass (owns mip chain texture + 3 pipelines)
//!        │
//!        ▼
//! Ping-Pong Color Buffer (consumed by ToneMapPass)
//! ```
//!
//! # Performance
//!
//! - Mip chain texture is only recreated on resolution change
//! - Pipelines are created once and cached
//! - Internal BindGroups are cached and rebuilt only when mip chain changes
//! - Two static uniform buffers (Karis on/off) avoid per-frame `write_buffer` calls
//! - Version-based dirty checking avoids redundant uniform uploads
//! - Uses `textureSampleLevel` for explicit LOD control (no mip auto-select overhead)
//! - Only 2 BindGroups created per frame (first downsample + composite, both
//!   depend on the ping-pong scene color view)

use crate::define_gpu_data_struct;
use crate::render::RenderNode;
use crate::renderer::HDR_TEXTURE_FORMAT;
use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::resources::Tracked;
use crate::renderer::graph::context::{ExecuteContext, GraphResource, PrepareContext};
use crate::renderer::graph::transient_pool::{TransientTextureDesc, TransientTextureId};
use crate::renderer::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions,
};
use crate::resources::WgslType;
use crate::resources::bloom::{CompositeUniforms, UpsampleUniforms};
use crate::resources::buffer::CpuBuffer;
use crate::resources::uniforms::{UniformArray, WgslStruct};

define_gpu_data_struct!(
    /// GPU uniform data for the downsample shader.
    ///
    /// This struct stays in the pass because the two static buffers (karis on/off)
    /// are pass-internal implementation details, not user-facing settings.
    struct DownsampleUniforms {
        pub use_karis_average: u32,
        pub(crate) __pad: UniformArray<u32, 3>,
    }
);

/// Physically-based bloom post-processing pass.
///
/// Pulls configuration from `Scene.bloom` each frame. Manages its own
/// mip-chain texture and three internal sub-pipelines.
///
/// # BindGroup caching strategy
///
/// Internal mip-to-mip BindGroups are deterministic once the mip chain is
/// allocated, so they are pre-built in [`ensure_mip_chain`] and reused every
/// frame. Only the **first downsample** and **composite** BindGroups must be
/// created per-frame because they reference the ping-pong scene color view
/// which alternates each frame.
pub struct BloomPass {
    // === Pipelines (cached as IDs in the global PipelineCache) ===
    downsample_pipeline: Option<RenderPipelineId>,
    upsample_pipeline: Option<RenderPipelineId>,
    composite_pipeline: Option<RenderPipelineId>,

    // === Bind Group Layouts ===
    downsample_layout: Tracked<wgpu::BindGroupLayout>,
    upsample_layout: Tracked<wgpu::BindGroupLayout>,
    composite_layout: Tracked<wgpu::BindGroupLayout>,

    // === Shared Resources ===
    sampler: Tracked<wgpu::Sampler>,

    /// Static uniform buffer with `use_karis_average = 1`. Written once via
    /// `CpuBuffer`, synced automatically by `ensure_buffer_id`.
    buffer_karis_on: CpuBuffer<DownsampleUniforms>,
    /// Static uniform buffer with `use_karis_average = 0`. Written once via
    /// `CpuBuffer`, synced automatically by `ensure_buffer_id`.
    buffer_karis_off: CpuBuffer<DownsampleUniforms>,

    // === Mip Chain (allocated from TransientTexturePool each frame) ===
    /// Handle to the bloom mip chain texture in the transient pool.
    bloom_texture_id: Option<TransientTextureId>,
    /// Actual number of mip levels in the current allocation.
    current_mip_count: u32,
    /// Tracked ID of the pool's mip-0 view from the previous frame.
    /// Used to detect when the pool returned a different physical texture
    /// (e.g. after a resolution change), which requires rebuilding internal
    /// mip-to-mip BindGroups.
    last_bloom_view_id: u64,

    // === Cached BindGroups (rebuilt when mip chain changes) ===
    /// `downsample_bind_groups[i]` binds `mip_views[i]` → `mip_views[i+1]`
    /// using `buffer_karis_off`. Length = `mip_count - 1`.
    downsample_bind_groups: Vec<wgpu::BindGroup>,
    /// `upsample_bind_groups[i]` binds `mip_views[i+1]` as source for
    /// upsampling into `mip_views[i]`. Length = `mip_count - 1`.
    upsample_bind_groups: Vec<wgpu::BindGroup>,

    composite_bind_group: Option<wgpu::BindGroup>,

    output_view: Option<Tracked<wgpu::TextureView>>,

    // === Runtime State (set during prepare, used during run) ===
    enabled: bool,
}

impl BloomPass {
    /// Creates a new bloom pass, allocating GPU layouts, sampler, and uniform buffers.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        // --- Bind Group Layouts ---

        // Downsample / Upsample share the same layout: texture + sampler + uniforms
        let downsample_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bloom Downsample Layout"),
            entries: &[
                // Binding 0: Source texture
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
                // Binding 1: Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Binding 2: Uniforms
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

        let upsample_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bloom Upsample Layout"),
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

        // Composite layout: original_texture + bloom_texture + sampler + uniforms
        let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bloom Composite Layout"),
            entries: &[
                // Binding 0: Original HDR texture
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
                // Binding 1: Bloom texture
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
                // Binding 2: Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Binding 3: Uniforms
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

        // --- Sampler ---
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Bloom Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // --- Uniform Buffers via CpuBuffer<T> (auto-managed version tracking + GPU sync) ---
        let karis_on_data = DownsampleUniforms {
            use_karis_average: 1,
            ..Default::default()
        };
        let buffer_karis_on = CpuBuffer::new(
            karis_on_data,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some("Bloom Karis On"),
        );

        let buffer_karis_off = CpuBuffer::new(
            DownsampleUniforms::default(), // use_karis_average = 0 by default
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some("Bloom Karis Off"),
        );

        Self {
            downsample_pipeline: None,
            upsample_pipeline: None,
            composite_pipeline: None,

            downsample_layout: Tracked::new(downsample_layout),
            upsample_layout: Tracked::new(upsample_layout),
            composite_layout: Tracked::new(composite_layout),

            sampler: Tracked::new(sampler),
            buffer_karis_on,
            buffer_karis_off,

            bloom_texture_id: None,
            current_mip_count: 0,
            last_bloom_view_id: 0,

            downsample_bind_groups: Vec::new(),
            upsample_bind_groups: Vec::new(),
            composite_bind_group: None,

            output_view: None,

            enabled: false,
        }
    }

    // =========================================================================
    // Pipeline Creation
    // =========================================================================

    fn ensure_pipelines(&mut self, ctx: &mut PrepareContext) {
        if self.downsample_pipeline.is_some() {
            return;
        }

        let device = &ctx.wgpu_ctx.device;

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

        // --- Downsample pipeline ---
        let mut ds_options = ShaderCompilationOptions::default();
        ds_options.add_define(
            "struct_definitions",
            DownsampleUniforms::wgsl_struct_def("DownsampleUniforms").as_str(),
        );

        let downsample_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bloom Downsample Pipeline Layout"),
                bind_group_layouts: &[&self.downsample_layout],
                immediate_size: 0,
            });

        let (ds_module, ds_hash) = ctx.shader_manager.get_or_compile_template(
            device,
            "passes/bloom_downsample",
            &ds_options,
            "",
            "",
        );

        let ds_key = FullscreenPipelineKey::fullscreen(
            ds_hash,
            smallvec::smallvec![color_target_replace.clone()],
            None,
        );

        self.downsample_pipeline = Some(ctx.pipeline_cache.get_or_create_fullscreen(
            device,
            ds_module,
            &downsample_pipeline_layout,
            &ds_key,
            "Bloom Downsample Pipeline",
        ));

        // --- Upsample pipeline (additive blend) ---
        let mut us_options = ShaderCompilationOptions::default();
        us_options.add_define(
            "struct_definitions",
            UpsampleUniforms::wgsl_struct_def("UpsampleUniforms").as_str(),
        );

        let upsample_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bloom Upsample Pipeline Layout"),
                bind_group_layouts: &[&self.upsample_layout],
                immediate_size: 0,
            });

        let (us_module, us_hash) = ctx.shader_manager.get_or_compile_template(
            device,
            "passes/bloom_upsample",
            &us_options,
            "",
            "",
        );

        let us_key = FullscreenPipelineKey::fullscreen(
            us_hash,
            smallvec::smallvec![color_target_additive],
            None,
        );

        self.upsample_pipeline = Some(ctx.pipeline_cache.get_or_create_fullscreen(
            device,
            us_module,
            &upsample_pipeline_layout,
            &us_key,
            "Bloom Upsample Pipeline",
        ));

        // --- Composite pipeline ---
        let mut comp_options = ShaderCompilationOptions::default();
        comp_options.add_define(
            "struct_definitions",
            CompositeUniforms::wgsl_struct_def("CompositeUniforms").as_str(),
        );

        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bloom Composite Pipeline Layout"),
                bind_group_layouts: &[&self.composite_layout],
                immediate_size: 0,
            });

        let (comp_module, comp_hash) = ctx.shader_manager.get_or_compile_template(
            device,
            "passes/bloom_composite",
            &comp_options,
            "",
            "",
        );

        let comp_key = FullscreenPipelineKey::fullscreen(
            comp_hash,
            smallvec::smallvec![color_target_replace],
            None,
        );

        self.composite_pipeline = Some(ctx.pipeline_cache.get_or_create_fullscreen(
            device,
            comp_module,
            &composite_pipeline_layout,
            &comp_key,
            "Bloom Composite Pipeline",
        ));
    }

    // =========================================================================
    // Mip Chain Management
    // =========================================================================

    /// Allocate the bloom mip chain from the transient texture pool and
    /// rebuild internal BindGroups when the underlying texture changes.
    fn allocate_bloom_texture(
        &mut self,
        ctx: &mut PrepareContext,
        source_width: u32,
        source_height: u32,
        max_mip_levels: u32,
    ) {
        // Bloom works at half resolution
        let bloom_w = (source_width / 2).max(1);
        let bloom_h = (source_height / 2).max(1);

        // Calculate actual mip count
        let max_possible = ((bloom_w.max(bloom_h) as f32).log2().floor() as u32) + 1;
        let mip_count = max_mip_levels.min(max_possible).max(1);
        self.current_mip_count = mip_count;

        // Allocate from transient pool (pool handles recycling transparently)
        let bloom_id = ctx.transient_pool.allocate(
            &ctx.wgpu_ctx.device,
            &TransientTextureDesc {
                width: bloom_w,
                height: bloom_h,
                format: HDR_TEXTURE_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                mip_level_count: mip_count,
                label: "Bloom Mip Chain",
            },
        );
        self.bloom_texture_id = Some(bloom_id);

        // ── Always rebuild the first downsample BG (depends on scene color) ──
        let first_ds_bg = self.get_first_mip_bind_group(ctx);
        if self.downsample_bind_groups.is_empty() {
            self.downsample_bind_groups.push(first_ds_bg);
        } else {
            self.downsample_bind_groups[0] = first_ds_bg;
        }

        // ── Check if the pool returned a different texture ────────────────
        let mip0_view_id = ctx.transient_pool.get_mip_view(bloom_id, 0).id();
        if self.last_bloom_view_id == mip0_view_id
            && self.downsample_bind_groups.len() == mip_count as usize
        {
            // Pool reused the same physical texture → keep cached BGs
            return;
        }
        self.last_bloom_view_id = mip0_view_id;

        // ── Rebuild internal mip-to-mip BindGroups ────────────────────────
        // Downsample: mip[i] → mip[i+1], always with Karis OFF
        self.downsample_bind_groups.truncate(1); // keep [0] (first DS, already set)

        // Look up GPU buffer for karis_off (ensure_buffer_id already called in prepare)
        let karis_off_cpu_id = self.buffer_karis_off.id();
        let karis_off_gpu = ctx
            .resource_manager
            .gpu_buffers
            .get(&karis_off_cpu_id)
            .expect("Bloom karis_off GPU buffer must exist after ensure");

        for i in 0..(mip_count - 1) as usize {
            let source_mip = i;
            let bg = ctx
                .wgpu_ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bloom DS BG mip→mip"),
                    layout: &self.downsample_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                ctx.transient_pool.get_mip_view(bloom_id, source_mip as u32),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: karis_off_gpu.buffer.as_entire_binding(),
                        },
                    ],
                });
            self.downsample_bind_groups.push(bg);
        }

        // Upsample: mip[i+1] → mip[i] (additive blend into target)
        let upsample_cpu_id = ctx.scene.bloom.upsample_uniforms.id();
        let upsample_gpu = ctx
            .resource_manager
            .gpu_buffers
            .get(&upsample_cpu_id)
            .expect("Bloom upsample GPU buffer must exist after ensure");

        self.upsample_bind_groups.clear();
        for i in 0..(mip_count - 1) as usize {
            let source_mip = i + 1;
            let bg = ctx
                .wgpu_ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bloom US BG mip→mip"),
                    layout: &self.upsample_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                ctx.transient_pool.get_mip_view(bloom_id, source_mip as u32),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: upsample_gpu.buffer.as_entire_binding(),
                        },
                    ],
                });
            self.upsample_bind_groups.push(bg);
        }

        log::debug!(
            "Bloom mip chain allocated: {}×{}, {} levels (pool), {} cached bind groups",
            bloom_w,
            bloom_h,
            mip_count,
            self.downsample_bind_groups.len() + self.upsample_bind_groups.len(),
        );
    }

    fn get_first_mip_bind_group(&self, ctx: &mut PrepareContext) -> wgpu::BindGroup {
        // Select the appropriate static Karis buffer for the first downsample
        let karis_buffer = if ctx.scene.bloom.karis_average {
            &self.buffer_karis_on
        } else {
            &self.buffer_karis_off
        };

        let input_view = ctx.get_resource_view(GraphResource::SceneColorInput);

        // 1. Prepare Cache Key IDs (GPU buffer IDs from resource manager)
        let layout_id = self.downsample_layout.id();
        let input_view_id = input_view.id();
        let sampler_id = self.sampler.id();

        let karis_cpu_id = karis_buffer.id();
        let karis_gpu_buffer_id = ctx
            .resource_manager
            .gpu_buffers
            .get(&karis_cpu_id)
            .expect("Bloom karis GPU buffer must exist after ensure")
            .id;

        // 2. Build Key
        let key = BindGroupKey::new(layout_id)
            .with_resource(input_view_id)
            .with_resource(sampler_id)
            .with_resource(karis_gpu_buffer_id);

        // 3. Get from cache or create
        if let Some(cached) = ctx.global_bind_group_cache.get(&key) {
            cached.clone()
        } else {
            let karis_gpu = ctx
                .resource_manager
                .gpu_buffers
                .get(&karis_cpu_id)
                .expect("Bloom karis GPU buffer must exist");

            let new_bg = ctx
                .wgpu_ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bloom DS BG scene→0"),
                    layout: &self.downsample_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(input_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: karis_gpu.buffer.as_entire_binding(),
                        },
                    ],
                });

            ctx.global_bind_group_cache.insert(key, new_bg.clone());
            new_bg
        }
    }

    fn get_composite_bind_group(&self, ctx: &mut PrepareContext) -> wgpu::BindGroup {
        let input_view = ctx.get_resource_view(GraphResource::SceneColorInput);
        let bloom_id = self
            .bloom_texture_id
            .expect("bloom_texture_id must be set before composite");
        let bloom_view = ctx.transient_pool.get_mip_view(bloom_id, 0);

        // 1. Prepare Cache Key IDs
        let layout_id = self.composite_layout.id();
        let input_view_id = input_view.id();
        let bloom_view_id = bloom_view.id();
        let sampler_id = self.sampler.id();

        let composite_cpu_id = ctx.scene.bloom.composite_uniforms.id();
        let composite_gpu_buffer_id = ctx
            .resource_manager
            .gpu_buffers
            .get(&composite_cpu_id)
            .expect("Bloom composite GPU buffer must exist after ensure")
            .id;

        // 2. Build Key
        let key = BindGroupKey::new(layout_id)
            .with_resource(input_view_id)
            .with_resource(bloom_view_id)
            .with_resource(sampler_id)
            .with_resource(composite_gpu_buffer_id);

        // 3. Get from cache or create
        if let Some(cached) = ctx.global_bind_group_cache.get(&key) {
            cached.clone()
        } else {
            let composite_gpu = ctx
                .resource_manager
                .gpu_buffers
                .get(&composite_cpu_id)
                .expect("Bloom composite GPU buffer must exist");

            let new_bg = ctx
                .wgpu_ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
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
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: composite_gpu.buffer.as_entire_binding(),
                        },
                    ],
                });

            ctx.global_bind_group_cache.insert(key, new_bg.clone());
            new_bg
        }
    }
}

impl RenderNode for BloomPass {
    fn name(&self) -> &'static str {
        "Bloom Pass"
    }

    fn prepare(&mut self, ctx: &mut PrepareContext) {
        // Read bloom settings from the scene
        let settings = &ctx.scene.bloom;

        // If bloom is disabled, skip everything
        if !settings.enabled {
            self.enabled = false;
            return;
        }
        self.enabled = true;

        // 1. Ensure pipelines exist
        self.ensure_pipelines(ctx);

        // 2. Ensure all GPU buffers are synced
        //    - Static karis buffers: written once at creation, only synced on first frame
        //    - Upsample/composite: owned by BloomSettings, version auto-tracked by CpuBuffer.
        //      User setters (set_radius, set_strength) call write() which bumps the version;
        //      ensure_buffer_id only uploads to GPU when the version has changed.
        ctx.resource_manager.ensure_buffer_id(&self.buffer_karis_on);
        ctx.resource_manager
            .ensure_buffer_id(&self.buffer_karis_off);
        ctx.resource_manager
            .ensure_buffer_id(&ctx.scene.bloom.upsample_uniforms);
        ctx.resource_manager
            .ensure_buffer_id(&ctx.scene.bloom.composite_uniforms);

        // 4. Allocate mip chain from transient pool and rebuild bind groups
        let (source_w, source_h) = ctx.wgpu_ctx.size();
        let max_mip_levels = ctx.scene.bloom.max_mip_levels;
        self.allocate_bloom_texture(ctx, source_w, source_h, max_mip_levels);

        self.composite_bind_group = Some(self.get_composite_bind_group(ctx));

        self.output_view = Some(
            ctx.get_resource_view(GraphResource::SceneColorOutput)
                .clone(),
        );

        // Flip ping-pong so downstream passes see our output as their input
        ctx.flip_scene_color();
    }

    fn run(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        if !self.enabled || self.current_mip_count == 0 {
            return;
        }

        let Some(downsample_pipeline_id) = self.downsample_pipeline else {
            return;
        };
        let Some(upsample_pipeline_id) = self.upsample_pipeline else {
            return;
        };
        let Some(composite_pipeline_id) = self.composite_pipeline else {
            return;
        };
        let Some(composite_bind_group) = &self.composite_bind_group else {
            return;
        };
        let Some(output_view) = &self.output_view else {
            return;
        };
        let Some(bloom_id) = self.bloom_texture_id else {
            return;
        };

        let downsample_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(downsample_pipeline_id);
        let upsample_pipeline = ctx.pipeline_cache.get_render_pipeline(upsample_pipeline_id);
        let composite_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(composite_pipeline_id);

        // =====================================================================
        // Phase 1: Downsample — Scene HDR → Bloom Mip Chain
        // =====================================================================
        for i in 0..self.downsample_bind_groups.len() {
            let target_mip = i as u32;

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Downsample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ctx.transient_pool.get_mip_view(bloom_id, target_mip),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });

            pass.set_pipeline(downsample_pipeline);
            pass.set_bind_group(0, &self.downsample_bind_groups[i], &[]);
            pass.draw(0..3, 0..1);
        }

        // =====================================================================
        // Phase 2: Upsample — Accumulate bloom from coarsest to finest
        // =====================================================================

        // Walk from the coarsest mip upward, additively blending into each
        // finer level. The blend state on the pipeline handles the additive
        // accumulation. All BindGroups are cached.
        for i in (0..self.upsample_bind_groups.len()).rev() {
            let target_mip = i as u32;

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Upsample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ctx.transient_pool.get_mip_view(bloom_id, target_mip),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Load existing content (the downsample result), then additive blend
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });

            pass.set_pipeline(upsample_pipeline);
            pass.set_bind_group(0, &self.upsample_bind_groups[i], &[]);
            pass.draw(0..3, 0..1);
        }

        // =====================================================================
        // Phase 3: Composite — Original HDR + Bloom → Output
        // =====================================================================

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Bloom Composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            ..Default::default()
        });

        pass.set_pipeline(composite_pipeline);
        pass.set_bind_group(0, composite_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
