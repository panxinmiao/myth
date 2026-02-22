//! Tone Mapping Post-Processing Pass
//!
//! This pass converts HDR (High Dynamic Range) rendering output to LDR (Low Dynamic Range)
//! for display on standard monitors. It reads configuration from `Scene.tone_mapping` using
//! a data-driven approach with version tracking for efficient GPU synchronization.
//!
//! # Features
//!
//! - **Tone Mapping**: Multiple algorithms (Linear, Neutral, Reinhard, Cineon, ACES, AgX)
//! - **Color Grading**: Optional 3D LUT texture for color manipulation
//! - **Vignette**: Edge darkening effect with configurable intensity and smoothness
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
use crate::render::RenderNode;
// use crate::render::core::ResourceBuilder;
use crate::renderer::core::{binding::BindGroupKey, resources::Tracked};
use crate::renderer::graph::context::{ExecuteContext, GraphResource, PrepareContext};
use crate::renderer::graph::transient_pool::TransientTextureId;
use crate::renderer::pipeline::{ShaderCompilationOptions, shader_gen::ShaderGenerator};
use crate::resources::texture::TextureSource;
use crate::resources::tone_mapping::{ToneMappingMode, ToneMappingUniforms};
use crate::resources::uniforms::WgslStruct;

/// Pipeline cache key: (mode, output_format, has_lut).
///
/// The `has_lut` flag affects shader compilation (USE_LUT macro),
/// so it must be part of the cache key.
type PipelineCacheKey = (ToneMappingMode, wgpu::TextureFormat, bool);

/// Tone mapping post-processing pass.
///
/// This pass is a pure "executor" that pulls configuration from `Scene.tone_mapping`.
/// It uses version tracking to minimize GPU updates:
///
/// - Uniform buffer is only updated when `ToneMappingSettings.version` changes
/// - Pipeline is only rebuilt when the tone mapping mode or LUT state changes
///
/// # Performance
///
/// - Pipeline caching by (mode, format, has_lut) — typically 1-2 pipelines active
/// - Version-based dirty checking (O(1) comparison)
/// - Dynamic `BindGroupLayout` based on LUT presence (avoids unused bindings)
/// - `BindGroup` caching via global cache
pub struct ToneMapPass {
    // === GPU Resources ===
    /// Base bind group layout (without LUT bindings)
    layout_base: Tracked<wgpu::BindGroupLayout>,
    /// Extended bind group layout (with LUT bindings)
    layout_with_lut: Tracked<wgpu::BindGroupLayout>,
    /// Linear sampler for input texture
    sampler: Tracked<wgpu::Sampler>,
    /// Linear sampler dedicated to 3D LUT texture (ClampToEdge on all axes)
    lut_sampler: Tracked<wgpu::Sampler>,
    /// Uniform buffer (exposure, vignette, lut_contribution)
    // uniforms: CpuBuffer<ToneMapUniforms>,

    // === Cache State ===
    /// Currently active tone mapping mode (mirrors `Scene.tone_mapping.mode`)
    current_mode: ToneMappingMode,
    /// Whether the current configuration has a LUT texture
    current_has_lut: bool,
    /// Cached pipelines by (mode, `output_format`, has_lut) to handle HDR toggle and LUT changes
    pipeline_cache: FxHashMap<PipelineCacheKey, wgpu::RenderPipeline>,

    // === Runtime State (set during prepare, used during run) ===
    /// Current frame's bind group
    current_bind_group: Option<wgpu::BindGroup>,
    /// Current frame's pipeline
    current_pipeline: Option<wgpu::RenderPipeline>,

    // === Output Routing ===
    /// If `Some`, output to a transient texture (for downstream FXAA).
    /// If `None`, output directly to the surface.
    pub output_texture_id: Option<TransientTextureId>,
}

impl ToneMapPass {
    /// Creates a new tone mapping pass.
    ///
    /// This initializes GPU resources but does not configure any settings.
    /// Settings are pulled from `Scene.tone_mapping` during each frame's prepare phase.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        // Base entries shared by both layouts
        let base_entries = [
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
        ];

        let layout_base = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ToneMap Layout"),
            entries: &base_entries,
        });

        // Extended layout: base entries + LUT 3D Texture + LUT Sampler
        let mut lut_entries = base_entries.to_vec();
        lut_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        });
        lut_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });

        let layout_with_lut = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ToneMap Layout (LUT)"),
            entries: &lut_entries,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ToneMap Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // LUT sampler: ClampToEdge on all axes, Linear filtering for trilinear interpolation
        let lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ToneMap LUT Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        Self {
            layout_base: Tracked::new(layout_base),
            layout_with_lut: Tracked::new(layout_with_lut),
            sampler: Tracked::new(sampler),
            lut_sampler: Tracked::new(lut_sampler),
            // uniforms,
            current_mode: ToneMappingMode::default(),
            current_has_lut: false,
            pipeline_cache: FxHashMap::default(),
            current_pipeline: None,
            current_bind_group: None,
            output_texture_id: None,
        }
    }

    /// Returns the appropriate layout for the current LUT state.
    #[inline]
    fn current_layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        if self.current_has_lut {
            &self.layout_with_lut
        } else {
            &self.layout_base
        }
    }

    /// Gets or creates a pipeline for the current tone mapping configuration.
    fn get_or_create_pipeline(&mut self, ctx: &PrepareContext) -> wgpu::RenderPipeline {
        let cache_key = (
            self.current_mode,
            ctx.wgpu_ctx.surface_view_format,
            self.current_has_lut,
        );

        // Check cache first
        if let Some(pipeline) = self.pipeline_cache.get(&cache_key) {
            return pipeline.clone();
        }

        // Cache miss - compile new pipeline
        log::debug!(
            "Compiling ToneMap pipeline for mode {:?}, format {:?}, has_lut: {}",
            self.current_mode,
            ctx.wgpu_ctx.surface_view_format,
            self.current_has_lut,
        );

        // 1. Prepare shader defines
        let mut defines = ShaderDefines::new();
        self.current_mode.apply_to_defines(&mut defines);
        if self.current_has_lut {
            defines.set("USE_LUT", "1");
        }

        let gpu_world = ctx
            .resource_manager
            .get_global_state(ctx.render_state.id, ctx.scene.id)
            .expect("Global state must exist");

        // 2. Generate shader code
        let mut options = ShaderCompilationOptions { defines };

        options.add_define(
            "struct_definitions",
            ToneMappingUniforms::wgsl_struct_def("Uniforms").as_str(),
        );

        let shader_code = ShaderGenerator::generate_shader(
            "",                      // vertex snippet (use default)
            &gpu_world.binding_wgsl, // binding code (use default)
            "passes/tone_mapping",   // template name
            &options,
        );

        let shader = ctx
            .wgpu_ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!(
                    "ToneMap Shader {:?} lut={}",
                    self.current_mode, self.current_has_lut
                )),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_code)),
            });

        // 3. Create pipeline layout with the appropriate bind group layout
        let layout = self.current_layout();

        let pipeline_layout =
            ctx.wgpu_ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ToneMap Pipeline Layout"),
                    bind_group_layouts: &[
                        &gpu_world.layout, // Global bind group (frame-level resources)
                        layout,
                    ],
                    immediate_size: 0,
                });

        // 4. Create render pipeline
        let pipeline =
            ctx.wgpu_ctx
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(&format!(
                        "ToneMap Pipeline {:?} lut={}",
                        self.current_mode, self.current_has_lut
                    )),
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

        // Store in cache
        self.pipeline_cache.insert(cache_key, pipeline.clone());
        pipeline
    }
}

impl RenderNode for ToneMapPass {
    fn prepare(&mut self, ctx: &mut PrepareContext) {
        // =====================================================================
        // 1. Sync settings from Scene (data-driven approach)
        // =====================================================================
        let settings = &ctx.scene.tone_mapping;

        let uniforms = &settings.uniforms;
        // let current_version = settings.version();
        let has_lut = settings.has_lut();

        // B. Handle mode or LUT state change (triggers pipeline rebuild)
        if self.current_mode != settings.mode || self.current_has_lut != has_lut {
            self.current_mode = settings.mode;
            self.current_has_lut = has_lut;
            self.current_pipeline = None;
        }
        // =====================================================================
        // 2. Prepare GPU resources
        // =====================================================================

        // Ensure LUT texture is uploaded if present
        let lut_view_id = if let Some(lut_handle) = settings.lut_texture {
            ctx.resource_manager.prepare_texture(ctx.assets, lut_handle);
            ctx.resource_manager
                .get_texture_binding(lut_handle)
                .map(|b| b.view_id)
        } else {
            None
        };

        let input_view_tracked = &ctx.get_resource_view(GraphResource::SceneColorInput);
        let input_view_id = input_view_tracked.id();

        // Ensure buffer is ready
        let gpu_buffer_id = ctx.resource_manager.ensure_buffer_id(uniforms);
        let cpu_buffer_id = uniforms.id();

        // Build BindGroup cache key (includes LUT resources when present)
        let layout = self.current_layout();
        let mut key = BindGroupKey::new(layout.id())
            .with_resource(input_view_id)
            .with_resource(self.sampler.id())
            .with_resource(gpu_buffer_id);

        if let Some(lut_id) = lut_view_id {
            key = key
                .with_resource(lut_id)
                .with_resource(self.lut_sampler.id());
        }

        // Get or create BindGroup
        let bind_group = if let Some(cached) = ctx.global_bind_group_cache.get(&key) {
            cached.clone()
        } else {
            let gpu_buffer = ctx
                .resource_manager
                .gpu_buffers
                .get(&cpu_buffer_id)
                .expect("GpuBuffer must exist after ensure_buffer_id");

            let input_view = &ctx.get_resource_view(GraphResource::SceneColorInput);

            let mut entries = vec![
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
            ];

            // Add LUT bindings if present
            if let Some(lut_handle) = settings.lut_texture {
                let lut_view = ctx
                    .resource_manager
                    .get_texture_view(&TextureSource::Asset(lut_handle));
                entries.push(wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(lut_view),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.lut_sampler),
                });
            }

            let new_bind_group =
                ctx.wgpu_ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("ToneMap BindGroup"),
                        layout,
                        entries: &entries,
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
            self.current_pipeline = Some(self.get_or_create_pipeline(ctx));
        }
    }

    fn run(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let render_lists = ctx.render_lists;
        let Some(gpu_global_bind_group) = &render_lists.gpu_global_bind_group else {
            return;
        };

        // Output to transient texture (for downstream FXAA) or directly to surface
        let target_view: &wgpu::TextureView = if let Some(id) = self.output_texture_id {
            ctx.transient_pool.get_view(id)
        } else {
            ctx.surface_view
        };

        let pass_desc = wgpu::RenderPassDescriptor {
            label: Some("ToneMap Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
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
            pass.set_bind_group(0, gpu_global_bind_group, &[]);
            if let Some(bg) = &self.current_bind_group {
                pass.set_bind_group(1, bg, &[]);
            }
        }

        // Draw fullscreen triangle
        pass.draw(0..3, 0..1);
    }

    fn name(&self) -> &'static str {
        "Tone Mapping Pass"
    }
}
