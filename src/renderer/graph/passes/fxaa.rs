//! FXAA (Fast Approximate Anti-Aliasing) Post-Processing Pass
//!
//! Applies screen-space anti-aliasing to a LDR image by detecting aliased edges
//! via luma contrast and blending along them at sub-pixel precision.
//!
//! FXAA operates on **post-tone-mapped** (LDR) data, so it must be placed
//! **after** `ToneMapPass` in the render graph.
//!
//! # Data Flow
//!
//! ```text
//! ToneMapPass → Transient LDR Texture → FxaaPass → Surface
//! ```
//!
//! When FXAA is disabled, `ToneMapPass` writes directly to the surface and
//! this pass is not added to the graph.
//!
//! # Quality Presets
//!
//! The quality preset is pulled from `Scene.fxaa.quality()` each frame.
//! When the quality changes, the shader is recompiled (edge exploration
//! iteration count is a compile-time constant for maximum performance).
//!
//! # Performance
//!
//! - Single pipeline per quality level, cached across frames
//! - BindGroup rebuilt only when the input texture view changes
//! - Zero uniform buffers — all parameters are compile-time constants
//! - Uses `textureSampleLevel` with integer offsets for cache-friendly sampling

use std::borrow::Cow;

use rustc_hash::FxHashMap;

use crate::render::RenderNode;
use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::resources::Tracked;
use crate::renderer::graph::context::{ExecuteContext, PrepareContext};
use crate::renderer::graph::transient_pool::TransientTextureId;
use crate::renderer::pipeline::ShaderCompilationOptions;
use crate::renderer::pipeline::shader_gen::ShaderGenerator;
use crate::resources::fxaa::FxaaQuality;

/// Pipeline cache key: (quality, output_format).
type PipelineCacheKey = (FxaaQuality, wgpu::TextureFormat);

/// FXAA post-processing render pass.
///
/// This pass reads from a transient LDR texture (written by `ToneMapPass`)
/// and writes to the final surface. It has no uniforms — quality is controlled
/// entirely via shader defines (compile-time constants).
///
/// # Pipeline Caching
///
/// Pipelines are cached by `(quality, output_format)`. Typical usage produces
/// only 1 cached pipeline (the active quality level). Switching quality at
/// runtime triggers a one-time recompilation.
pub struct FxaaPass {
    // === GPU Resources ===
    /// Bind group layout: [texture_2d, sampler]
    layout: Tracked<wgpu::BindGroupLayout>,
    /// Linear filtering sampler for sub-pixel blending
    sampler: Tracked<wgpu::Sampler>,

    // === Cache State ===
    /// Current quality level (mirrors `Scene.fxaa.quality()`)
    current_quality: FxaaQuality,
    /// Cached pipelines by (quality, format) — typically 1 entry
    pipeline_cache: FxHashMap<PipelineCacheKey, wgpu::RenderPipeline>,

    // === Runtime State (set during prepare, used during run) ===
    /// Current frame's bind group
    current_bind_group: Option<wgpu::BindGroup>,
    /// Current frame's pipeline
    current_pipeline: Option<wgpu::RenderPipeline>,

    /// Input transient texture ID (set by the graph wiring code before prepare)
    pub input_texture_id: Option<TransientTextureId>,
}

impl FxaaPass {
    /// Creates a new FXAA pass with the given device.
    ///
    /// Only allocates the bind group layout and sampler. Pipelines are
    /// lazily created on first use (or when quality changes).
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let entries = [
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
        ];

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FXAA BindGroup Layout"),
            entries: &entries,
        });

        // Linear sampler is critical for FXAA's sub-pixel blending quality
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("FXAA Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        Self {
            layout: Tracked::new(layout),
            sampler: Tracked::new(sampler),
            current_quality: FxaaQuality::default(),
            pipeline_cache: FxHashMap::default(),
            current_pipeline: None,
            current_bind_group: None,
            input_texture_id: None,
        }
    }

    /// Gets or creates a pipeline for the current quality and format.
    fn get_or_create_pipeline(&mut self, ctx: &PrepareContext) -> wgpu::RenderPipeline {
        let cache_key = (self.current_quality, ctx.wgpu_ctx.surface_view_format);

        if let Some(pipeline) = self.pipeline_cache.get(&cache_key) {
            return pipeline.clone();
        }

        log::debug!(
            "Compiling FXAA pipeline for quality {:?}, format {:?}",
            self.current_quality,
            ctx.wgpu_ctx.surface_view_format,
        );

        let mut options = ShaderCompilationOptions::default();

        // Only Low and High need explicit defines; Medium is the default in the shader
        match self.current_quality {
            FxaaQuality::Low => options.add_define("FXAA_QUALITY_LOW", "1"),
            FxaaQuality::High => options.add_define("FXAA_QUALITY_HIGH", "1"),
            FxaaQuality::Medium => {} // Shader default
        }

        let shader_code = ShaderGenerator::generate_shader("", "", "passes/fxaa", &options);

        let shader = ctx
            .wgpu_ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("FXAA Shader {:?}", self.current_quality)),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_code)),
            });

        let pipeline_layout =
            ctx.wgpu_ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("FXAA Pipeline Layout"),
                    bind_group_layouts: &[&self.layout],
                    immediate_size: 0,
                });

        let pipeline =
            ctx.wgpu_ctx
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(&format!("FXAA Pipeline {:?}", self.current_quality)),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: ctx.wgpu_ctx.surface_view_format,
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
                });

        self.pipeline_cache.insert(cache_key, pipeline.clone());
        pipeline
    }
}

impl RenderNode for FxaaPass {
    fn prepare(&mut self, ctx: &mut PrepareContext) {
        // =====================================================================
        // 1. Sync quality from Scene
        // =====================================================================
        let quality = ctx.scene.fxaa.quality();
        if self.current_quality != quality {
            self.current_quality = quality;
            self.current_pipeline = None; // Force pipeline rebuild
        }

        // =====================================================================
        // 2. Build BindGroup from transient input texture
        // =====================================================================
        let input_id = self
            .input_texture_id
            .expect("FxaaPass requires input_texture_id to be set before prepare()");

        let input_view = ctx.transient_pool.get_view(input_id);
        let input_view_id = input_view.id();

        let key = BindGroupKey::new(self.layout.id())
            .with_resource(input_view_id)
            .with_resource(self.sampler.id());

        self.current_bind_group = Some(
            if let Some(cached) = ctx.global_bind_group_cache.get(&key) {
                cached.clone()
            } else {
                let new_bg = ctx
                    .wgpu_ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("FXAA BindGroup"),
                        layout: &self.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(input_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&self.sampler),
                            },
                        ],
                    });

                ctx.global_bind_group_cache.insert(key, new_bg.clone());
                new_bg
            },
        );

        // =====================================================================
        // 3. Ensure pipeline exists
        // =====================================================================
        if self.current_pipeline.is_none() {
            self.current_pipeline = Some(self.get_or_create_pipeline(ctx));
        }
    }

    fn run(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let Some(pipeline) = &self.current_pipeline else {
            return;
        };
        let Some(bind_group) = &self.current_bind_group else {
            return;
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("FXAA Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.surface_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::DontCare(wgpu::LoadOpDontCare::default()),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            ..Default::default()
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    fn name(&self) -> &'static str {
        "FXAA Pass"
    }
}
