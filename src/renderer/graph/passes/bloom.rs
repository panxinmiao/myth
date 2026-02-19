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
//! - Version-based dirty checking avoids redundant uniform uploads
//! - Uses `textureSampleLevel` for explicit LOD control (no mip auto-select overhead)

use std::borrow::Cow;

use crate::render::{RenderContext, RenderNode};
use crate::renderer::HDR_TEXTURE_FORMAT;
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
pub struct BloomPass {
    // === Pipelines (created once, cached) ===
    downsample_pipeline: Option<wgpu::RenderPipeline>,
    downsample_pipeline_first: Option<wgpu::RenderPipeline>,
    upsample_pipeline: Option<wgpu::RenderPipeline>,
    composite_pipeline: Option<wgpu::RenderPipeline>,

    // === Bind Group Layouts ===
    downsample_layout: wgpu::BindGroupLayout,
    upsample_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,

    // === Shared Resources ===
    sampler: wgpu::Sampler,
    downsample_uniform_buffer: wgpu::Buffer,
    upsample_uniform_buffer: wgpu::Buffer,
    composite_uniform_buffer: wgpu::Buffer,

    // === Mip Chain ===
    /// The bloom mip chain texture (half-res at level 0, quarter at level 1, etc.)
    bloom_texture: Option<wgpu::Texture>,
    /// Per-mip views for the bloom chain.
    bloom_mip_views: Vec<wgpu::TextureView>,
    /// Current mip chain dimensions (width, height at mip 0).
    bloom_size: (u32, u32),
    /// Actual number of mip levels in the current chain.
    mip_count: u32,

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
        let downsample_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let upsample_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let composite_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        // --- Uniform Buffers ---
        let downsample_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bloom Downsample Uniforms"),
            size: std::mem::size_of::<DownsampleUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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
            downsample_pipeline_first: None,
            upsample_pipeline: None,
            composite_pipeline: None,

            downsample_layout,
            upsample_layout,
            composite_layout,

            sampler,
            downsample_uniform_buffer,
            upsample_uniform_buffer,
            composite_uniform_buffer,

            bloom_texture: None,
            bloom_mip_views: Vec::new(),
            bloom_size: (0, 0),
            mip_count: 0,

            last_settings_version: u64::MAX,
            enabled: false,
        }
    }

    // =========================================================================
    // Pipeline Creation
    // =========================================================================

    fn ensure_pipelines(&mut self, device: &wgpu::Device) {
        if self.downsample_pipeline.is_some() {
            return;
        }

        let options = ShaderCompilationOptions::default();

        // --- Downsample pipelines ---
        let downsample_shader_code = ShaderGenerator::generate_shader(
            "",
            "",
            "passes/bloom_downsample",
            &options,
        );
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

        // Standard downsample pipeline (no Karis)
        self.downsample_pipeline =
            Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
            }));

        // First downsample pipeline is the same shader — the Karis toggle is via uniform
        self.downsample_pipeline_first = self.downsample_pipeline.clone();

        // --- Upsample pipeline (additive blend) ---
        let upsample_shader_code = ShaderGenerator::generate_shader(
            "",
            "",
            "passes/bloom_upsample",
            &options,
        );
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

        self.upsample_pipeline =
            Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
            }));

        // --- Composite pipeline ---
        let composite_shader_code = ShaderGenerator::generate_shader(
            "",
            "",
            "passes/bloom_composite",
            &options,
        );
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

        self.composite_pipeline =
            Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
            }));
    }

    // =========================================================================
    // Mip Chain Management
    // =========================================================================

    /// Recreates the bloom mip chain texture when the render target size changes.
    fn ensure_mip_chain(&mut self, device: &wgpu::Device, source_width: u32, source_height: u32, max_mip_levels: u32) {
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
        let texture = device.create_texture(&wgpu::TextureDescriptor {
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
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
            self.bloom_mip_views.push(view);
        }

        self.bloom_texture = Some(texture);

        log::debug!(
            "Bloom mip chain created: {}×{}, {} levels",
            bloom_w,
            bloom_h,
            self.mip_count
        );
    }
}

impl RenderNode for BloomPass {
    fn name(&self) -> &str {
        "Bloom Pass"
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

        // 1. Ensure pipelines exist
        self.ensure_pipelines(device);

        // 2. Ensure mip chain exists and matches current resolution
        let (source_w, source_h) = ctx.wgpu_ctx.size();
        self.ensure_mip_chain(device, source_w, source_h, settings.max_mip_levels);

        // 3. Upload uniforms if settings changed
        if self.last_settings_version != settings.version() {
            self.last_settings_version = settings.version();

            // Downsample uniforms (Karis toggle is set per-pass in run())
            // Upsample uniforms
            let upsample = UpsampleUniforms {
                filter_radius: settings.radius,
                _pad: [0; 3],
            };
            ctx.wgpu_ctx.queue.write_buffer(
                &self.upsample_uniform_buffer,
                0,
                bytemuck::bytes_of(&upsample),
            );

            // Composite uniforms
            let composite = CompositeUniforms {
                bloom_strength: settings.strength,
                _pad: [0; 3],
            };
            ctx.wgpu_ctx.queue.write_buffer(
                &self.composite_uniform_buffer,
                0,
                bytemuck::bytes_of(&composite),
            );
        }
    }

    fn run(&self, ctx: &mut RenderContext, encoder: &mut wgpu::CommandEncoder) {
        if !self.enabled || self.mip_count == 0 {
            return;
        }

        let device = &ctx.wgpu_ctx.device;
        let queue = &ctx.wgpu_ctx.queue;

        let downsample_pipeline = match &self.downsample_pipeline {
            Some(p) => p,
            None => return,
        };
        let upsample_pipeline = match &self.upsample_pipeline {
            Some(p) => p,
            None => return,
        };
        let composite_pipeline = match &self.composite_pipeline {
            Some(p) => p,
            None => return,
        };

        let karis_enabled = ctx.scene.bloom.karis_average;

        // =====================================================================
        // Phase 1: Downsample — Scene HDR → Bloom Mip Chain
        // =====================================================================

        // Current scene color (input for bloom)
        let current_idx = ctx.color_view_flip_flop;
        let scene_color_view = &ctx.frame_resources.scene_color_view[current_idx];

        // First downsample: scene → mip 0 (optionally with Karis average)
        {
            let ds_uniforms = DownsampleUniforms {
                use_karis_average: if karis_enabled { 1 } else { 0 },
                _pad: [0; 3],
            };
            queue.write_buffer(
                &self.downsample_uniform_buffer,
                0,
                bytemuck::bytes_of(&ds_uniforms),
            );

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bloom DS BG 0"),
                layout: &self.downsample_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(scene_color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.downsample_uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Downsample 0"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_mip_views[0],
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
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Subsequent downsamples: mip[i-1] → mip[i] (no Karis)
        for i in 1..self.mip_count as usize {
            let ds_uniforms = DownsampleUniforms {
                use_karis_average: 0,
                _pad: [0; 3],
            };
            queue.write_buffer(
                &self.downsample_uniform_buffer,
                0,
                bytemuck::bytes_of(&ds_uniforms),
            );

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bloom DS BG"),
                layout: &self.downsample_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.bloom_mip_views[i - 1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.downsample_uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Downsample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_mip_views[i],
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
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // =====================================================================
        // Phase 2: Upsample — Accumulate bloom from coarsest to finest
        // =====================================================================

        // Walk from the second-to-last mip upward, additively blending into
        // each coarser level. The blend state on the pipeline handles the
        // additive accumulation.
        for i in (0..(self.mip_count as usize).saturating_sub(1)).rev() {
            let source_mip = i + 1; // read from finer mip
            let target_mip = i; // blend into coarser mip

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bloom US BG"),
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
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // =====================================================================
        // Phase 3: Composite — Original HDR + Bloom → Output
        // =====================================================================

        // Use the ping-pong mechanism: read from current, write to the other
        let (input_view, output_view) = {
            let current_idx = ctx.color_view_flip_flop;
            let input = &ctx.frame_resources.scene_color_view[current_idx];
            let output = &ctx.frame_resources.scene_color_view[1 - current_idx];
            // Flip the flip-flop for the next pass
            ctx.color_view_flip_flop = 1 - current_idx;
            (input, output)
        };

        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bloom Composite BG"),
            layout: &self.composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_mip_views[0]),
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

        {
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
            pass.set_bind_group(0, &composite_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
