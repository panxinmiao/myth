//! RDG Tone Mapping Feature + Ephemeral PassNode
//!
//! - **`ToneMapFeature`** (long-lived): owns pipeline cache, bind group layouts.
//!   `extract_and_prepare()` compiles pipelines for the current mode/format.
//! - **`ToneMapPassNode`** (ephemeral per-frame): carries lightweight IDs and
//!   a transient bind-group slot.  Created by `ToneMapFeature::add_to_graph()`.
//!
//! # RDG Slots
//!
//! - `input_tex`: HDR scene color (after Bloom, if enabled)
//! - `output_tex`: LDR output (fed to FXAA or directly to surface)
//!
//! # Features
//!
//! - Multiple tone mapping algorithms (Linear, Neutral, Reinhard, Cineon, ACES, AgX)
//! - Vignette, color grading (3D LUT), film grain, chromatic aberration
//! - Version-tracked uniform buffer via `CpuBuffer<ToneMappingUniforms>`
//! - L1 pipeline cache with (mode, format, has_lut) key

use rustc_hash::FxHashMap;

use crate::ShaderDefines;
use crate::assets::TextureHandle;
use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::resources::{CommonSampler, Tracked};
use crate::renderer::graph::rdg::builder::PassBuilder;
use crate::renderer::graph::rdg::context::{ExtractContext, RdgExecuteContext, RdgPrepareContext};
use crate::renderer::graph::rdg::graph::RenderGraph;
use crate::renderer::graph::rdg::node::PassNode;
use crate::renderer::graph::rdg::types::TextureNodeId;
use crate::renderer::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions,
};
use crate::resources::texture::TextureSource;
use crate::resources::tone_mapping::{ToneMappingMode, ToneMappingUniforms};
use crate::resources::uniforms::WgslStruct;

/// Pipeline cache key: (mode, output_format, has_lut).
type PipelineCacheKey = (ToneMappingMode, wgpu::TextureFormat, bool);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Feature (long-lived, stored in RenderFeatures)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Long-lived tone mapping feature.
///
/// Owns bind group layouts and pipeline cache. Scene-level parameters
/// are passed in via `extract_and_prepare()` and `add_to_graph()`.
///
/// # GPU Resource Lifecycle
///
/// - `BindGroupLayout`s and samplers are created lazily on first use.
/// - Pipelines are cached by `(mode, format, has_lut)`.
/// - `BindGroup`s are cached via `GlobalBindGroupCache` with content-addressed keys.
pub struct ToneMapFeature {
    // ─── Persistent Cache ──────────────────────────────────────────
    /// Base bind group layout (without LUT bindings).
    layout_base: Option<Tracked<wgpu::BindGroupLayout>>,
    /// Extended bind group layout (with LUT bindings).
    layout_with_lut: Option<Tracked<wgpu::BindGroupLayout>>,

    /// Cached pipeline IDs by (mode, output_format, has_lut).
    local_cache: FxHashMap<PipelineCacheKey, RenderPipelineId>,
    /// Pipeline ID for the current frame.
    current_pipeline: Option<RenderPipelineId>,
    /// Output texture format — set during `extract_and_prepare()`.
    pub output_format: wgpu::TextureFormat,
    // ─── Resolved GPU Resources (populated per-frame) ───────────────
    /// Resolved GPU buffer for `ToneMappingUniforms`.
    uniforms_gpu_buffer: Option<wgpu::Buffer>,
    /// Stable resource ID for the uniforms GPU buffer.
    uniforms_gpu_buffer_id: u64,
    /// Resolved LUT texture view (if LUT is enabled).
    lut_texture_view: Option<wgpu::TextureView>,
    /// Tracked ID of the LUT view (for bind-group cache key).
    lut_view_id: Option<u64>,}

impl ToneMapFeature {
    /// Creates a new tone mapping feature.
    ///
    /// All GPU resources are lazily initialized on first `extract_and_prepare()` call.
    #[must_use]
    pub fn new() -> Self {
        Self {
            layout_base: None,
            layout_with_lut: None,
            local_cache: FxHashMap::default(),
            current_pipeline: None,
            output_format: wgpu::TextureFormat::Bgra8UnormSrgb,

            uniforms_gpu_buffer: None,
            uniforms_gpu_buffer_id: 0,
            lut_texture_view: None,
            lut_view_id: None,
        }
    }

    // ─── Lazy Initialization ───────────────────────────────────────

    fn ensure_layouts(&mut self, device: &wgpu::Device) {
        if self.layout_base.is_some() {
            return;
        }

        // Base entries: input texture + sampler + uniforms
        let base_entries = [
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
        ];

        self.layout_base = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("RDG ToneMap Layout (base)"),
                entries: &base_entries,
            },
        )));

        // Extended layout: base + LUT 3D texture + LUT sampler
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

        self.layout_with_lut = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("RDG ToneMap Layout (LUT)"),
                entries: &lut_entries,
            },
        )));
    }

    // ─── Helpers ───────────────────────────────────────────────────

    #[inline]
    fn current_layout(&self, has_lut: bool) -> &Tracked<wgpu::BindGroupLayout> {
        if has_lut {
            self.layout_with_lut.as_ref().unwrap()
        } else {
            self.layout_base.as_ref().unwrap()
        }
    }

    /// Pre-RDG resource preparation: create layouts, compile pipeline,
    /// resolve GPU buffer and optional LUT texture view.
    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        mode: ToneMappingMode,
        output_format: wgpu::TextureFormat,
        has_lut: bool,
        global_state_key: (u32, u32),
        uniforms_cpu_id: u64,
        lut_handle: Option<TextureHandle>,
    ) {
        // ─── 1. Lazy initialization ────────────────────────────────
        self.ensure_layouts(ctx.device);
        self.output_format = output_format;

        // ─── 2. Pipeline (re)creation ──────────────────────────────
        let cache_key = (mode, output_format, has_lut);

        if self.current_pipeline.is_none() || !self.local_cache.contains_key(&cache_key) {
            self.current_pipeline =
                Some(self.get_or_create_pipeline(ctx, mode, has_lut, global_state_key));
        } else {
            self.current_pipeline = self.local_cache.get(&cache_key).copied();
        }

        // ─── 3. Resolve GPU resources for the PassNode ─────────────
        self.uniforms_gpu_buffer = ctx
            .resource_manager
            .gpu_buffers
            .get(&uniforms_cpu_id)
            .map(|g| {
                self.uniforms_gpu_buffer_id = g.id;
                g.buffer.clone()
            });

        // Resolve LUT texture view if present
        self.lut_texture_view = None;
        self.lut_view_id = None;
        if has_lut {
            if let Some(handle) = lut_handle {
                let binding = ctx.resource_manager.get_texture_binding(handle);
                if let Some(b) = binding {
                    self.lut_view_id = Some(b.view_id);
                    let view = ctx
                        .resource_manager
                        .get_texture_view(&TextureSource::Asset(handle));
                    self.lut_texture_view = Some(view.clone());
                }
            }
        }
    }

    /// Gets or creates a pipeline for the given (mode, format, has_lut) triple.
    fn get_or_create_pipeline(
        &mut self,
        ctx: &mut ExtractContext,
        mode: ToneMappingMode,
        has_lut: bool,
        global_state_key: (u32, u32),
    ) -> RenderPipelineId {
        let output_format = self.output_format;
        let cache_key = (mode, output_format, has_lut);

        if let Some(&id) = self.local_cache.get(&cache_key) {
            return id;
        }

        log::debug!(
            "RDG ToneMap: compiling pipeline for {:?}, fmt={:?}, lut={}",
            mode,
            output_format,
            has_lut,
        );

        let device = ctx.device;

        // Shader defines
        let mut defines = ShaderDefines::new();
        mode.apply_to_defines(&mut defines);
        if has_lut {
            defines.set("USE_LUT", "1");
        }

        let gpu_world = ctx
            .resource_manager
            .get_global_state(global_state_key.0, global_state_key.1)
            .expect("RDG ToneMap: GpuGlobalState must exist");

        let mut options = ShaderCompilationOptions { defines };
        options.add_define(
            "struct_definitions",
            ToneMappingUniforms::wgsl_struct_def("Uniforms").as_str(),
        );

        let (shader_module, shader_hash) = ctx.shader_manager.get_or_compile_template(
            device,
            "passes/tone_mapping",
            &options,
            "",
            &gpu_world.binding_wgsl,
        );

        let layout = self.current_layout(has_lut);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RDG ToneMap Pipeline Layout"),
            bind_group_layouts: &[&gpu_world.layout, layout],
            immediate_size: 0,
        });

        let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
            format: output_format,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        });

        let key =
            FullscreenPipelineKey::fullscreen(shader_hash, smallvec::smallvec![color_target], None);

        let id = ctx.pipeline_cache.get_or_create_fullscreen(
            device,
            shader_module,
            &pipeline_layout,
            &key,
            &format!("RDG ToneMap Pipeline {:?} lut={}", mode, has_lut),
        );

        self.local_cache.insert(cache_key, id);
        id
    }

    /// Build the ephemeral pass node and insert it into the graph.
    pub fn add_to_graph(
        &self,
        rdg: &mut RenderGraph,
        input_tex: TextureNodeId,
        output_tex: TextureNodeId,
        has_lut: bool,
    ) {
        let layout = if has_lut {
            self.layout_with_lut.clone().unwrap()
        } else {
            self.layout_base.clone().unwrap()
        };

        let node = ToneMapPassNode {
            input_tex,
            output_tex,
            pipeline_id: self.current_pipeline.expect("ToneMapFeature not prepared"),
            layout,
            has_lut,
            uniforms_gpu_buffer: self
                .uniforms_gpu_buffer
                .clone()
                .expect("ToneMapFeature: uniforms GPU buffer not resolved"),
            uniforms_gpu_buffer_id: self.uniforms_gpu_buffer_id,
            lut_view: self.lut_texture_view.clone(),
            lut_view_id: self.lut_view_id,
            current_bind_group_key: None,
        };
        rdg.add_pass(Box::new(node));
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PassNode (ephemeral, created per frame)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct ToneMapPassNode {
    input_tex: TextureNodeId,
    output_tex: TextureNodeId,
    pipeline_id: RenderPipelineId,
    layout: Tracked<wgpu::BindGroupLayout>,

    // ─── Push Parameters ───────────────────────────────────────────
    has_lut: bool,

    // ─── Resolved GPU Resources (from Feature, no ResourceManager needed) ──
    /// Resolved GPU buffer for `ToneMappingUniforms`.
    uniforms_gpu_buffer: wgpu::Buffer,
    /// Stable resource ID for the uniforms GPU buffer.
    uniforms_gpu_buffer_id: u64,
    /// Resolved LUT texture view (if LUT is enabled).
    lut_view: Option<wgpu::TextureView>,
    /// Tracked ID of the LUT view (for bind-group cache key).
    lut_view_id: Option<u64>,

    // ─── Transient State ───────────────────────────────────────────
    current_bind_group_key: Option<BindGroupKey>,
}

impl PassNode for ToneMapPassNode {
    fn name(&self) -> &'static str {
        "RDG_ToneMap_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        builder.read_texture(self.input_tex);
        builder.write_texture(self.output_tex);
    }

    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {
        // ─── Transient BindGroup assembly ──────────────────────────
        let input_view = ctx.views.get_texture_view(self.input_tex);
        let sampler = ctx.sampler_registry.get_common(CommonSampler::LinearClamp);
        let layout = &self.layout;

        let mut key = BindGroupKey::new(layout.id())
            .with_resource(input_view.id())
            .with_resource(sampler.id())
            .with_resource(self.uniforms_gpu_buffer_id);

        // LUT resources
        if self.has_lut {
            if let Some(lut_id) = self.lut_view_id {
                key = key.with_resource(lut_id).with_resource(sampler.id());
            }
        }

        if self.current_bind_group_key.as_ref() != Some(&key) {
            if ctx.global_bind_group_cache.get(&key).is_none() {
                let mut entries = vec![
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
                        resource: self.uniforms_gpu_buffer.as_entire_binding(),
                    },
                ];

                if self.has_lut {
                    if let Some(lut_view) = &self.lut_view {
                        entries.push(wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(lut_view),
                        });
                        entries.push(wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        });
                    }
                }

                let new_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RDG ToneMap BindGroup"),
                    layout: &**layout,
                    entries: &entries,
                });

                ctx.global_bind_group_cache.insert(key.clone(), new_bg);
            }

            self.current_bind_group_key = Some(key);
        }
    }

    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let Some(global_bind_group) = ctx.global_bind_group else {
            log::warn!("RDG ToneMap: global_bind_group missing, skipping");
            return;
        };

        let output_view = ctx.get_texture_view(self.output_tex);

        let pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.pipeline_id);

        let bind_group_key = self
            .current_bind_group_key
            .as_ref()
            .expect("BindGroupKey should have been set in prepare!");
        let bind_group = ctx
            .global_bind_group_cache
            .get(bind_group_key)
            .expect("BindGroup should have been prepared!");

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RDG ToneMap Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
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

        rpass.set_pipeline(pipeline);
        rpass.set_bind_group(0, global_bind_group, &[]);
        rpass.set_bind_group(1, bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }
}
