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
/// # Dual-Layer BindGroup Model
///
/// - Group 0: global scene (from Composer)
/// - Group 1 (static): sampler + uniforms + optional LUT — Feature-owned
/// - Group 2 (transient): input scene color texture — PassNode-owned
pub struct ToneMapFeature {
    // ─── Persistent Cache ──────────────────────────────────────────
    /// Group 1 static layout (base): sampler + uniforms.
    static_layout_base: Option<Tracked<wgpu::BindGroupLayout>>,
    /// Group 1 static layout (LUT): sampler + uniforms + LUT texture + LUT sampler.
    static_layout_lut: Option<Tracked<wgpu::BindGroupLayout>>,
    /// Group 2 transient layout: single input texture.
    transient_layout: Option<Tracked<wgpu::BindGroupLayout>>,

    /// Cached pipeline IDs by (mode, output_format, has_lut).
    local_cache: FxHashMap<PipelineCacheKey, RenderPipelineId>,
    /// Pipeline ID for the current frame.
    current_pipeline: Option<RenderPipelineId>,
    /// Output texture format — set during `extract_and_prepare()`.
    pub output_format: wgpu::TextureFormat,

    // ─── Pre-Built Static BindGroup (Group 1) ──────────────────────
    /// Feature-owned static bind group (sampler + uniforms + LUT if present).
    static_bg: Option<wgpu::BindGroup>,
    /// Whether the current static BG was built with LUT.
    static_bg_has_lut: bool,
    /// Staleness tracking for uniforms buffer identity.
    last_uniforms_buffer_id: u64,
    /// Staleness tracking for LUT view identity.
    last_lut_view_id: u64,
}

impl ToneMapFeature {
    /// Creates a new tone mapping feature.
    ///
    /// All GPU resources are lazily initialized on first `extract_and_prepare()` call.
    #[must_use]
    pub fn new() -> Self {
        Self {
            static_layout_base: None,
            static_layout_lut: None,
            transient_layout: None,
            local_cache: FxHashMap::default(),
            current_pipeline: None,
            output_format: wgpu::TextureFormat::Bgra8UnormSrgb,

            static_bg: None,
            static_bg_has_lut: false,
            last_uniforms_buffer_id: 0,
            last_lut_view_id: 0,
        }
    }

    // ─── Lazy Initialization ───────────────────────────────────────

    fn ensure_layouts(&mut self, device: &wgpu::Device) {
        if self.static_layout_base.is_some() {
            return;
        }

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

        // Base static layout (Group 1): sampler + uniforms
        self.static_layout_base = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("ToneMap Static Layout (base, G1)"),
                entries: &[sampler_entry, uniform_entry],
            },
        )));

        // LUT static layout (Group 1): sampler + uniforms + LUT texture + LUT sampler
        let lut_entries = [
            sampler_entry,
            uniform_entry,
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ];

        self.static_layout_lut = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("ToneMap Static Layout (LUT, G1)"),
                entries: &lut_entries,
            },
        )));

        // Transient layout (Group 2): single input texture
        self.transient_layout = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("ToneMap Transient Layout (G2)"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            },
        )));
    }

    // ─── Helpers ───────────────────────────────────────────────────

    /// Returns the static layout matching the current LUT mode.
    #[inline]
    fn current_static_layout(&self, has_lut: bool) -> &Tracked<wgpu::BindGroupLayout> {
        if has_lut {
            self.static_layout_lut.as_ref().unwrap()
        } else {
            self.static_layout_base.as_ref().unwrap()
        }
    }

    /// Pre-RDG resource preparation: create layouts, compile pipeline,
    /// build static bind group (Group 1) with sampler + uniforms + optional LUT.
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

        // ─── 3. Build static bind group (Group 1) ─────────────────
        // Resolve GPU buffer for uniforms
        let gpu_buf = ctx
            .resource_manager
            .gpu_buffers
            .get(&uniforms_cpu_id);
        let gpu_buf = match gpu_buf {
            Some(g) => g,
            None => return,
        };
        let buf_id = gpu_buf.id;

        // Resolve LUT view if present
        let (lut_view, lut_view_id) = if has_lut {
            if let Some(handle) = lut_handle {
                let binding = ctx.resource_manager.get_texture_binding(handle);
                if let Some(b) = binding {
                    let view = ctx
                        .resource_manager
                        .get_texture_view(&TextureSource::Asset(handle));
                    (Some(view.clone()), b.view_id)
                } else {
                    (None, 0)
                }
            } else {
                (None, 0)
            }
        } else {
            (None, 0)
        };

        // Check staleness — rebuild only when buffer or LUT identity changes
        let needs_rebuild = self.static_bg.is_none()
            || buf_id != self.last_uniforms_buffer_id
            || has_lut != self.static_bg_has_lut
            || (has_lut && lut_view_id != self.last_lut_view_id);

        if needs_rebuild {
            let sampler = ctx
                .sampler_registry
                .get_common(CommonSampler::LinearClamp);
            let layout = self.current_static_layout(has_lut);

            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gpu_buf.buffer.as_entire_binding(),
                },
            ];

            if has_lut {
                if let Some(ref view) = lut_view {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(view),
                    });
                    entries.push(wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    });
                }
            }

            self.static_bg = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ToneMap Static BG (G1)"),
                layout: &**layout,
                entries: &entries,
            }));
            self.static_bg_has_lut = has_lut;
            self.last_uniforms_buffer_id = buf_id;
            self.last_lut_view_id = lut_view_id;
        }
    }

    /// Gets or creates a pipeline for the given (mode, format, has_lut) triple.
    ///
    /// Pipeline layout uses 3 bind-group layouts:
    /// - Group 0: global scene
    /// - Group 1: static (sampler + uniforms + optional LUT)
    /// - Group 2: transient (input texture)
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

        let static_layout = self.current_static_layout(has_lut);
        let transient_layout = self.transient_layout.as_ref().unwrap();
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RDG ToneMap Pipeline Layout"),
            bind_group_layouts: &[&gpu_world.layout, static_layout, transient_layout],
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
    ///
    /// PassNode carries only pipeline ID, pre-built static BG (Arc clone),
    /// and transient layout — no GPU buffers or raw texture views.
    pub fn add_to_graph(
        &self,
        rdg: &mut RenderGraph,
        input_tex: TextureNodeId,
        output_tex: TextureNodeId,
    ) {
        let node = ToneMapPassNode {
            input_tex,
            output_tex,
            pipeline_id: self.current_pipeline.expect("ToneMapFeature not prepared"),
            static_bg: self
                .static_bg
                .clone()
                .expect("ToneMapFeature: static BG not built"),
            transient_layout: self.transient_layout.clone().unwrap(),
            current_bind_group_key: None,
        };
        rdg.add_pass(Box::new(node));
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PassNode (ephemeral, created per frame)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Ephemeral tone mapping pass node.
///
/// Carries only:
/// - Pipeline ID (from Feature's L1 cache)
/// - Pre-built static BG (Group 1, Arc clone from Feature)
/// - Transient layout for building Group 2 with the RDG input texture
struct ToneMapPassNode {
    input_tex: TextureNodeId,
    output_tex: TextureNodeId,
    pipeline_id: RenderPipelineId,

    /// Feature-owned static bind group (Group 1): sampler + uniforms + optional LUT.
    static_bg: wgpu::BindGroup,
    /// Layout for transient bind group (Group 2).
    transient_layout: Tracked<wgpu::BindGroupLayout>,

    /// Cache key for the transient bind group built in `prepare()`.
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
        // ─── Transient BindGroup (Group 2): input texture only ─────
        let input_view = ctx.views.get_texture_view(self.input_tex);
        let layout = &self.transient_layout;

        let key = BindGroupKey::new(layout.id()).with_resource(input_view.id());

        if self.current_bind_group_key.as_ref() != Some(&key) {
            if ctx.global_bind_group_cache.get(&key).is_none() {
                let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ToneMap Transient BG (G2)"),
                    layout: &**layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_view),
                    }],
                });
                ctx.global_bind_group_cache.insert(key.clone(), bg);
            }
            self.current_bind_group_key = Some(key);
        }
    }

    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let global_bind_group = ctx.baked_lists.global_bind_group;

        let output_view = ctx.get_texture_view(self.output_tex);

        let pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.pipeline_id);

        let transient_bg_key = self
            .current_bind_group_key
            .as_ref()
            .expect("BindGroupKey should have been set in prepare!");
        let transient_bg = ctx
            .global_bind_group_cache
            .get(transient_bg_key)
            .expect("Transient BindGroup should have been prepared!");

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
        rpass.set_bind_group(1, &self.static_bg, &[]);
        rpass.set_bind_group(2, transient_bg, &[]);
        rpass.draw(0..3, 0..1);
    }
}
