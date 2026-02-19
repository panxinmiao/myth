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

use std::borrow::Cow;

use crate::render::{RenderContext, RenderNode};
use crate::renderer::HDR_TEXTURE_FORMAT;
use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::resources::Tracked;
use crate::renderer::pipeline::ShaderCompilationOptions;
use crate::renderer::pipeline::shader_gen::ShaderGenerator;

/// GPU uniform data for the downsample shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DownsampleUniforms {
    use_karis_average: u32,
    _pad: [u32; 3],
}

/// GPU uniform data for the upsample shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UpsampleUniforms {
    filter_radius: f32,
    _pad: [u32; 3],
}

/// GPU uniform data for the composite shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeUniforms {
    bloom_strength: f32,
    _pad: [u32; 3],
}

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
    // === Pipelines (created once, cached) ===
    downsample_pipeline: Option<wgpu::RenderPipeline>,
    upsample_pipeline: Option<wgpu::RenderPipeline>,
    composite_pipeline: Option<wgpu::RenderPipeline>,

    // === Bind Group Layouts ===
    downsample_layout: Tracked<wgpu::BindGroupLayout>,
    upsample_layout: Tracked<wgpu::BindGroupLayout>,
    composite_layout: Tracked<wgpu::BindGroupLayout>,

    // === Shared Resources ===
    sampler: Tracked<wgpu::Sampler>,

    /// Static uniform buffer with `use_karis_average = 1`. Created once,
    /// never written again after initialization.
    buffer_karis_on: Tracked<wgpu::Buffer>,
    /// Static uniform buffer with `use_karis_average = 0`. Created once,
    /// never written again after initialization.
    buffer_karis_off: Tracked<wgpu::Buffer>,
    /// Dynamic uniform buffer for upsample `filter_radius`. Written only
    /// when `BloomSettings.version` changes.
    upsample_uniform_buffer: Tracked<wgpu::Buffer>,
    /// Dynamic uniform buffer for composite `bloom_strength`. Written only
    /// when `BloomSettings.version` changes.
    composite_uniform_buffer: Tracked<wgpu::Buffer>,

    // === Mip Chain ===
    /// The bloom mip chain texture (half-res at level 0, quarter at level 1, etc.)
    bloom_texture: Option<wgpu::Texture>,
    /// Per-mip views for the bloom chain.
    bloom_mip_views: Vec<Tracked<wgpu::TextureView>>,
    /// Current mip chain dimensions (width, height at mip 0).
    bloom_size: (u32, u32),
    /// Actual number of mip levels in the current chain.
    mip_count: u32,

    // === Cached BindGroups (rebuilt when mip chain changes) ===
    /// `downsample_bind_groups[i]` binds `mip_views[i]` → `mip_views[i+1]`
    /// using `buffer_karis_off`. Length = `mip_count - 1`.
    downsample_bind_groups: Vec<wgpu::BindGroup>,
    /// `upsample_bind_groups[i]` binds `mip_views[i+1]` as source for
    /// upsampling into `mip_views[i]`. Length = `mip_count - 1`.
    upsample_bind_groups: Vec<wgpu::BindGroup>,

    composite_bind_group: Option<wgpu::BindGroup>,

    output_view: Option<Tracked<wgpu::TextureView>>,

    // === Version Tracking ===
    last_settings_version: u64,

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

        // --- Static Karis uniform buffers (written once at creation) ---
        let buffer_karis_on = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bloom Karis On"),
            size: std::mem::size_of::<DownsampleUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buffer_karis_off = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bloom Karis Off"),
            size: std::mem::size_of::<DownsampleUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Dynamic Uniform Buffers ---
        let upsample_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bloom Upsample Uniforms"),
            size: std::mem::size_of::<UpsampleUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let composite_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bloom Composite Uniforms"),
            size: std::mem::size_of::<CompositeUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            downsample_pipeline: None,
            upsample_pipeline: None,
            composite_pipeline: None,

            downsample_layout: Tracked::new(downsample_layout),
            upsample_layout: Tracked::new(upsample_layout),
            composite_layout: Tracked::new(composite_layout),

            sampler: Tracked::new(sampler),
            buffer_karis_on: Tracked::new(buffer_karis_on),
            buffer_karis_off: Tracked::new(buffer_karis_off),
            upsample_uniform_buffer: Tracked::new(upsample_uniform_buffer),
            composite_uniform_buffer: Tracked::new(composite_uniform_buffer),

            bloom_texture: None,
            bloom_mip_views: Vec::new(),
            bloom_size: (0, 0),
            mip_count: 0,

            downsample_bind_groups: Vec::new(),
            upsample_bind_groups: Vec::new(),
            composite_bind_group: None,

            output_view: None,

            last_settings_version: u64::MAX,
            enabled: false,
        }
    }

    // =========================================================================
    // One-time Initialization (called once in first prepare)
    // =========================================================================

    /// Writes the two static Karis uniform buffers. Called exactly once during
    /// the first `prepare` when we have access to the queue.
    fn init_static_buffers(&self, queue: &wgpu::Queue) {
        let on = DownsampleUniforms {
            use_karis_average: 1,
            _pad: [0; 3],
        };
        queue.write_buffer(&self.buffer_karis_on, 0, bytemuck::bytes_of(&on));

        let off = DownsampleUniforms {
            use_karis_average: 0,
            _pad: [0; 3],
        };
        queue.write_buffer(&self.buffer_karis_off, 0, bytemuck::bytes_of(&off));
    }

    // =========================================================================
    // Pipeline Creation
    // =========================================================================

    fn ensure_pipelines(&mut self, device: &wgpu::Device) {
        if self.downsample_pipeline.is_some() {
            return;
        }

        let options = ShaderCompilationOptions::default();

        // --- Downsample pipeline ---
        let downsample_shader_code =
            ShaderGenerator::generate_shader("", "", "passes/bloom_downsample", &options);
        let downsample_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom Downsample Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(downsample_shader_code)),
        });

        let downsample_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bloom Downsample Pipeline Layout"),
                bind_group_layouts: &[&self.downsample_layout],
                immediate_size: 0,
            });

        self.downsample_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Bloom Downsample Pipeline"),
                layout: Some(&downsample_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &downsample_module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &downsample_module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HDR_TEXTURE_FORMAT,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        ));

        // --- Upsample pipeline (additive blend) ---
        let upsample_shader_code =
            ShaderGenerator::generate_shader("", "", "passes/bloom_upsample", &options);
        let upsample_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom Upsample Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(upsample_shader_code)),
        });

        let upsample_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bloom Upsample Pipeline Layout"),
                bind_group_layouts: &[&self.upsample_layout],
                immediate_size: 0,
            });

        self.upsample_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Bloom Upsample Pipeline"),
                layout: Some(&upsample_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &upsample_module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &upsample_module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HDR_TEXTURE_FORMAT,
                        // Additive blend: src + dst
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::OVER,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        ));

        // --- Composite pipeline ---
        let composite_shader_code =
            ShaderGenerator::generate_shader("", "", "passes/bloom_composite", &options);
        let composite_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(composite_shader_code)),
        });

        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bloom Composite Pipeline Layout"),
                bind_group_layouts: &[&self.composite_layout],
                immediate_size: 0,
            });

        self.composite_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Bloom Composite Pipeline"),
                layout: Some(&composite_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &composite_module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &composite_module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HDR_TEXTURE_FORMAT,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        ));
    }

    // =========================================================================
    // Mip Chain Management
    // =========================================================================

    /// Recreates the bloom mip chain texture and pre-builds all internal
    /// BindGroups when the render target size changes.
    fn ensure_mip_chain(
        &mut self,
        ctx: &mut RenderContext,
        source_width: u32,
        source_height: u32,
        max_mip_levels: u32,
    ) {
        // The first downsample BindGroup is the only one that depends on the ping-pong scene color view,
        if self.downsample_bind_groups.is_empty() {
            self.downsample_bind_groups
                .push(self.get_first_mip_bind_group(ctx));
        } else {
            self.downsample_bind_groups[0] = self.get_first_mip_bind_group(ctx);
        }

        // Bloom works at half resolution
        let bloom_w = (source_width / 2).max(1);
        let bloom_h = (source_height / 2).max(1);

        if self.bloom_size == (bloom_w, bloom_h) {
            return;
        }

        self.bloom_size = (bloom_w, bloom_h);

        // Calculate actual mip count
        let max_possible = ((bloom_w.max(bloom_h) as f32).log2().floor() as u32) + 1;
        self.mip_count = max_mip_levels.min(max_possible).max(1);

        // Create the mip chain texture
        let texture = ctx
            .wgpu_ctx
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Bloom Mip Chain"),
                size: wgpu::Extent3d {
                    width: bloom_w,
                    height: bloom_h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: self.mip_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HDR_TEXTURE_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

        // Create per-mip views
        self.bloom_mip_views.clear();
        for mip in 0..self.mip_count {
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("Bloom Mip {mip}")),
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..Default::default()
            });
            self.bloom_mip_views.push(Tracked::new(view));
        }

        self.bloom_texture = Some(texture);

        // -----------------------------------------------------------------
        // Pre-build internal BindGroups (deterministic, no ping-pong input)
        // -----------------------------------------------------------------

        // Downsample: mip[i] → mip[i+1], always with Karis OFF
        // The first downsample BindGroup (scene → mip 0) is created per-frame in _get_first_mip_bind_group()
        // Only keep the first one and rebuild the rest based on the new mip views
        self.downsample_bind_groups.truncate(1);

        for i in 0..(self.mip_count - 1) as usize {
            let source_mip = i;
            let bg = ctx
                .wgpu_ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("Bloom DS BG {}->{}", source_mip, source_mip + 1)),
                    layout: &self.downsample_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &self.bloom_mip_views[source_mip],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.buffer_karis_off.as_entire_binding(),
                        },
                    ],
                });
            self.downsample_bind_groups.push(bg);
        }

        // Upsample: mip[i+1] → mip[i] (additive blend into target)
        self.upsample_bind_groups.clear();
        for i in 0..(self.mip_count - 1) as usize {
            let target_mip = i;
            let source_mip = i + 1;
            let bg = ctx
                .wgpu_ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("Bloom US BG {source_mip}→{target_mip}")),
                    layout: &self.upsample_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &self.bloom_mip_views[source_mip],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.upsample_uniform_buffer.as_entire_binding(),
                        },
                    ],
                });
            self.upsample_bind_groups.push(bg);
        }

        log::debug!(
            "Bloom mip chain created: {}×{}, {} levels, {} cached bind groups",
            bloom_w,
            bloom_h,
            self.mip_count,
            self.downsample_bind_groups.len() + self.upsample_bind_groups.len(),
        );
    }

    fn get_first_mip_bind_group(&self, ctx: &mut RenderContext) -> wgpu::BindGroup {
        // Select the appropriate static Karis buffer for the first downsample
        let karis_buffer = if ctx.scene.bloom.karis_average {
            &self.buffer_karis_on
        } else {
            &self.buffer_karis_off
        };

        let input_view = ctx.get_scene_color_input();

        // 1. 准备 Cache Key 所需的 ID
        let layout_id = self.downsample_layout.id();

        let input_view_id = input_view.id();
        let sampler_id = self.sampler.id();
        let buffer_id = karis_buffer.id();

        // 2. 构建 Key
        let key = BindGroupKey::new(layout_id)
            .with_resource(input_view_id)
            .with_resource(sampler_id)
            .with_resource(buffer_id);

        // 3. 从缓存获取或创建

        if let Some(cached) = ctx.global_bind_group_cache.get(&key) {
            cached.clone()
        } else {
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
                            resource: karis_buffer.as_entire_binding(), // 这个来自 self，不冲突
                        },
                    ],
                });

            ctx.global_bind_group_cache.insert(key, new_bg.clone());
            new_bg
        }
    }

    fn get_composite_bind_group(&self, ctx: &mut RenderContext) -> wgpu::BindGroup {
        let input_view = ctx.get_scene_color_input();
        let bloom_view = &self.bloom_mip_views[0];

        // 1. 准备 Cache Key 所需的 ID
        let layout_id = self.composite_layout.id();

        let input_view_id = input_view.id();
        let bloom_view_id = bloom_view.id();
        let sampler_id = self.sampler.id();
        let buffer_id = self.composite_uniform_buffer.id();

        // 2. 构建 Key
        let key = BindGroupKey::new(layout_id)
            .with_resource(input_view_id)
            .with_resource(bloom_view_id)
            .with_resource(sampler_id)
            .with_resource(buffer_id);

        // 3. 从缓存获取或创建

        if let Some(cached) = ctx.global_bind_group_cache.get(&key) {
            cached.clone()
        } else {
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
                            resource: self.composite_uniform_buffer.as_entire_binding(),
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

    fn should_flip_ping_pong(&self) -> bool {
        // 如果 Bloom 没启用，就不翻转
        self.enabled && self.mip_count > 0
    }

    fn prepare(&mut self, ctx: &mut RenderContext) {
        // Read bloom settings from the scene
        let settings = &ctx.scene.bloom;

        // If bloom is disabled, skip everything
        if !settings.enabled {
            self.enabled = false;
            return;
        }
        self.enabled = true;

        let device = &ctx.wgpu_ctx.device;
        let queue = &ctx.wgpu_ctx.queue;

        // 1. Ensure pipelines exist
        self.ensure_pipelines(device);

        // 2. One-time: initialize static Karis buffers (safe to call every
        //    frame — the guard is the version sentinel at u64::MAX)
        if self.last_settings_version == u64::MAX {
            self.init_static_buffers(queue);
        }

        // 3. Ensure mip chain exists and matches current resolution
        let (source_w, source_h) = ctx.wgpu_ctx.size();
        self.ensure_mip_chain(ctx, source_w, source_h, settings.max_mip_levels);

        self.composite_bind_group = Some(self.get_composite_bind_group(ctx));

        self.output_view = Some(ctx.get_scene_color_output().clone());

        let settings = &ctx.scene.bloom;
        // 4. Upload dynamic uniforms if settings changed
        if self.last_settings_version != settings.version() {
            self.last_settings_version = settings.version();

            // Upsample uniforms
            let upsample = UpsampleUniforms {
                filter_radius: settings.radius,
                _pad: [0; 3],
            };
            queue.write_buffer(
                &self.upsample_uniform_buffer,
                0,
                bytemuck::bytes_of(&upsample),
            );

            // Composite uniforms
            let composite = CompositeUniforms {
                bloom_strength: settings.strength,
                _pad: [0; 3],
            };
            queue.write_buffer(
                &self.composite_uniform_buffer,
                0,
                bytemuck::bytes_of(&composite),
            );
        }
    }

    fn run(&self, _ctx: &mut RenderContext, encoder: &mut wgpu::CommandEncoder) {
        if !self.enabled || self.mip_count == 0 {
            return;
        }

        let Some(downsample_pipeline) = &self.downsample_pipeline else {
            return;
        };
        let Some(upsample_pipeline) = &self.upsample_pipeline else {
            return;
        };
        let Some(composite_pipeline) = &self.composite_pipeline else {
            return;
        };
        let Some(composite_bind_group) = &self.composite_bind_group else {
            return;
        };
        let Some(output_view) = &self.output_view else {
            return;
        };

        // =====================================================================
        // Phase 1: Downsample — Scene HDR → Bloom Mip Chain
        // =====================================================================
        for i in 0..self.downsample_bind_groups.len() {
            let target_mip = i;

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Downsample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_mip_views[target_mip],
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
            let target_mip = i; // blend into this mip

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Upsample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_mip_views[target_mip],
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
