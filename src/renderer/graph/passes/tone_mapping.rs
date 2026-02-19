//! Tone Mapping Post-Processing Pass
//!
//! This pass converts HDR (High Dynamic Range) rendering output to LDR (Low Dynamic Range)
//! for display on standard monitors. It reads configuration from `Scene.tone_mapping` using
//! a data-driven approach with version tracking for efficient GPU synchronization.
//!
//! # Data Flow
//!
//! ```text
//! Scene.tone_mapping (Source of Truth)
//!        │
//!        ▼  (version check in prepare())
//! ToneMapPass (cached state + GPU resources)
//!        │
//!        ▼
//! Final Surface Output
//! ```

use std::borrow::Cow;

use rustc_hash::FxHashMap;

use crate::ShaderDefines;
use crate::render::{RenderContext, RenderNode};
use crate::renderer::core::{binding::BindGroupKey, resources::Tracked};
use crate::renderer::pipeline::{ShaderCompilationOptions, shader_gen::ShaderGenerator};
use crate::resources::buffer::{CpuBuffer, GpuData};
use crate::resources::tone_mapping::ToneMappingMode;

/// GPU uniform data for tone mapping shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ToneMapUniforms {
    pub exposure: f32,
    pub _pad: [u32; 3],
}

impl Default for ToneMapUniforms {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            _pad: [0; 3],
        }
    }
}

impl GpuData for ToneMapUniforms {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    fn byte_size(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

/// Tone mapping post-processing pass.
///
/// This pass is a pure "executor" that pulls configuration from `Scene.tone_mapping`.
/// It uses version tracking to minimize GPU updates:
///
/// - Uniform buffer is only updated when `ToneMappingSettings.version` changes
/// - Pipeline is only rebuilt when the tone mapping mode changes
///
/// # Performance
///
/// - Pipeline caching by mode (typically 1-2 pipelines active)
/// - Version-based dirty checking (O(1) comparison)
/// - `BindGroup` caching via global cache
pub struct ToneMapPass {
    // === GPU Resources ===
    /// Bind group layout for tone mapping
    layout: Tracked<wgpu::BindGroupLayout>,
    /// Linear sampler for input texture
    sampler: Tracked<wgpu::Sampler>,
    /// Uniform buffer (exposure, etc.)
    uniforms: CpuBuffer<ToneMapUniforms>,

    // === Cache State ===
    /// Currently active tone mapping mode (mirrors `Scene.tone_mapping.mode`)
    current_mode: ToneMappingMode,
    /// Cached pipelines by (mode, `output_format`) to handle HDR toggle correctly
    pipeline_cache: FxHashMap<(ToneMappingMode, wgpu::TextureFormat), wgpu::RenderPipeline>,

    // === Runtime State (set during prepare, used during run) ===
    /// Current frame's bind group
    current_bind_group: Option<wgpu::BindGroup>,
    /// Current frame's pipeline
    current_pipeline: Option<wgpu::RenderPipeline>,

    // === Version Tracking ===
    /// Last known version from `Scene.tone_mapping`
    last_settings_version: u64,
}

impl ToneMapPass {
    /// Creates a new tone mapping pass.
    ///
    /// This initializes GPU resources but does not configure any settings.
    /// Settings are pulled from `Scene.tone_mapping` during each frame's prepare phase.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ToneMap Layout"),
            entries: &[
                // Binding 0: Input HDR texture
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

        let tracked_layout = Tracked::new(bind_group_layout);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ToneMap Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniforms = CpuBuffer::new(
            ToneMapUniforms::default(),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some("ToneMap Uniforms"),
        );

        Self {
            layout: tracked_layout,
            sampler: Tracked::new(sampler),
            uniforms,
            current_mode: ToneMappingMode::default(),
            pipeline_cache: FxHashMap::default(),
            current_pipeline: None,
            current_bind_group: None,
            // Set to MAX to ensure first frame always triggers update
            last_settings_version: u64::MAX,
        }
    }

    /// Gets or creates a pipeline for the current tone mapping mode.
    fn get_or_create_pipeline(
        &mut self,
        device: &wgpu::Device,
        view_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let cache_key = (self.current_mode, view_format);

        // Check cache first
        if let Some(pipeline) = self.pipeline_cache.get(&cache_key) {
            return pipeline.clone();
        }

        // Cache miss - compile new pipeline
        log::debug!(
            "Compiling ToneMap pipeline for mode {:?}, format {:?}",
            self.current_mode,
            view_format
        );

        // 1. Prepare shader defines
        let mut defines = ShaderDefines::new();
        self.current_mode.apply_to_defines(&mut defines);

        // 2. Generate shader code
        let options = ShaderCompilationOptions { defines };

        let shader_code = ShaderGenerator::generate_shader(
            "",                    // vertex snippet (use default)
            "",                    // binding code (use default)
            "passes/tone_mapping", // template name
            &options,
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("ToneMap Shader {:?}", self.current_mode)),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_code)),
        });

        // 3. Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ToneMap Pipeline Layout"),
            bind_group_layouts: &[&self.layout],
            immediate_size: 0,
        });

        // 4. Create render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("ToneMap Pipeline {:?}", self.current_mode)),
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
                    format: view_format,
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

        // Store in cache
        self.pipeline_cache.insert(cache_key, pipeline.clone());
        pipeline
    }
}

impl RenderNode for ToneMapPass {
    fn prepare(&mut self, ctx: &mut RenderContext) {
        // =====================================================================
        // 1. Sync settings from Scene (data-driven approach)
        // =====================================================================
        let settings = &ctx.scene.tone_mapping;
        let current_version = settings.version();

        // Version check: only update when Scene settings have changed
        if self.last_settings_version != current_version {
            // A. Sync uniform data (exposure)
            {
                let mut data = self.uniforms.write();
                data.exposure = settings.exposure;
            }

            // B. Handle mode change (triggers pipeline rebuild)
            if self.current_mode != settings.mode {
                self.current_mode = settings.mode;
                self.current_pipeline = None;
            }

            // C. Update local version
            self.last_settings_version = current_version;
        }

        // =====================================================================
        // 2. Prepare GPU resources
        // =====================================================================

        let input_view_tracked = &ctx.get_scene_color_input();
        let input_view_id = input_view_tracked.id();

        // Ensure buffer is ready
        let gpu_buffer_id = ctx.resource_manager.ensure_buffer_id(&self.uniforms);
        let cpu_buffer_id = self.uniforms.id();

        // Build BindGroup cache key
        let key = BindGroupKey::new(self.layout.id())
            .with_resource(input_view_id)
            .with_resource(self.sampler.id())
            .with_resource(gpu_buffer_id);

        // Get or create BindGroup
        let bind_group = if let Some(cached) = ctx.global_bind_group_cache.get(&key) {
            cached.clone()
        } else {
            let gpu_buffer = ctx
                .resource_manager
                .gpu_buffers
                .get(&cpu_buffer_id)
                .expect("GpuBuffer must exist after ensure_buffer_id");

            let input_view = &ctx.get_scene_color_input();

            let new_bind_group =
                ctx.wgpu_ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("ToneMap BindGroup"),
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
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: gpu_buffer.buffer.as_entire_binding(),
                            },
                        ],
                    });

            ctx.global_bind_group_cache
                .insert(key, new_bind_group.clone());
            new_bind_group
        };

        self.current_bind_group = Some(bind_group);

        // =====================================================================
        // 3. Ensure pipeline exists
        // =====================================================================
        if self.current_pipeline.is_none() {
            self.current_pipeline = Some(
                self.get_or_create_pipeline(&ctx.wgpu_ctx.device, ctx.wgpu_ctx.surface_view_format),
            );
        }
    }

    fn run(&self, ctx: &mut RenderContext, encoder: &mut wgpu::CommandEncoder) {
        let pass_desc = wgpu::RenderPassDescriptor {
            label: Some("ToneMap Pass"),
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
        };

        let mut pass = encoder.begin_render_pass(&pass_desc);

        if let Some(pipeline) = &self.current_pipeline {
            pass.set_pipeline(pipeline);
            if let Some(bg) = &self.current_bind_group {
                pass.set_bind_group(0, bg, &[]);
            }
        }

        // Draw fullscreen triangle
        pass.draw(0..3, 0..1);
    }

    fn name(&self) -> &'static str {
        "Tone Mapping Pass"
    }
}
