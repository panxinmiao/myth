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

use crate::core::binding::BindGroupKey;
use crate::core::gpu::{CommonSampler, ResourceState, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    ExecuteContext, ExtractContext, PassNode, PrepareContext, RenderTargetOps, TextureNodeId,
};
use crate::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions, ShaderSource,
};
use myth_assets::TextureHandle;
use myth_resources::ShaderDefines;
use myth_resources::buffer::CpuBuffer;
use myth_resources::texture::TextureSource;
use myth_resources::tone_mapping::{ToneMappingMode, ToneMappingUniforms};
use myth_resources::uniforms::WgslStruct;

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
pub struct ToneMappingFeature {
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

impl Default for ToneMappingFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl ToneMappingFeature {
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
        global_state_key: (u32, u32),
        uniforms: &CpuBuffer<ToneMappingUniforms>,
        lut_handle: Option<TextureHandle>,
    ) {
        // ─── 1. Lazy initialization ────────────────────────────────
        self.ensure_layouts(ctx.device);
        self.output_format = output_format;

        let mut has_lut = false;

        if let Some(handle) = lut_handle {
            let state = ctx.resource_manager.prepare_texture(ctx.assets, handle);
            match state {
                ResourceState::Ready => {
                    has_lut = true;
                }
                ResourceState::Pending => {
                    if self.current_pipeline.is_some() {
                        // LUT is pending but we already have a pipeline (from a previous frame)
                        // keep using it until the LUT is ready to avoid stalling the GPU.
                        return;
                    }
                }
                ResourceState::Unknown => {
                    // ResourceState::Failed or missing texture — treat as no LUT (fallback to default pipeline if needed)
                }
            }
        }

        // ─── 2. Pipeline (re)creation ──────────────────────────────
        self.current_pipeline =
            Some(self.get_or_create_pipeline(ctx, mode, has_lut, global_state_key));

        // ─── 3. Build static bind group (Group 1) ─────────────────
        // Resolve GPU buffer for uniforms

        let (buf_handle, _) = ctx.resource_manager.ensure_buffer(uniforms);

        let gpu_buf = ctx.resource_manager.gpu_buffers.get(buf_handle);

        let Some(gpu_buf) = gpu_buf else { return };
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
                .resource_manager
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

            if has_lut && let Some(ref view) = lut_view {
                entries.push(wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(view),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(sampler),
                });
            }

            self.static_bg = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ToneMap Static BG (G1)"),
                layout,
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
            "ToneMap: compiling pipeline for {mode:?}, fmt={output_format:?}, lut={has_lut}",
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
            .expect("ToneMap: GpuGlobalState must exist");

        let mut options = ShaderCompilationOptions {
            defines,
            ..Default::default()
        };
        options.add_define(
            "struct_definitions",
            ToneMappingUniforms::wgsl_struct_def("Uniforms").as_str(),
        );
        options.inject_code("binding_code", &gpu_world.binding_wgsl);

        let (shader_module, shader_hash) = ctx.shader_manager.get_or_compile(
            device,
            ShaderSource::File("passes/tone_mapping"),
            &options,
        );

        let static_layout = self.current_static_layout(has_lut);
        let transient_layout = self.transient_layout.as_ref().unwrap();
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ToneMap Pipeline Layout"),
            bind_group_layouts: &[
                Some(&gpu_world.layout),
                Some(static_layout),
                Some(transient_layout),
            ],
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
            &format!("ToneMap Pipeline {mode:?} lut={has_lut}"),
        );

        self.local_cache.insert(cache_key, id);
        id
    }

    /// Build the ephemeral pass node and insert it into the graph.
    ///
    /// Accepts the HDR input and the target LDR texture, performs an SSA
    /// relay on `target_ldr` (via `mutate_texture`), and returns the
    /// updated target handle. This enforces a pure dataflow chain where
    /// every Feature explicitly produces a new resource version.
    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        // graph: &mut RenderGraph<'a>,
        // pipeline_cache: &'a PipelineCache,
        input_hdr: TextureNodeId,
        target_ldr: TextureNodeId,
    ) -> TextureNodeId {
        let pipeline_id = self.current_pipeline.expect("ToneMapFeature not prepared");
        let pipeline = ctx.pipeline_cache.get_render_pipeline(pipeline_id);
        let static_bg = self
            .static_bg
            .as_ref()
            .expect("ToneMapFeature: static BG not built");
        let transient_layout = self.transient_layout.as_ref().unwrap();

        ctx.graph.add_pass("ToneMap_Pass", |builder| {
            builder.read_texture(input_hdr);

            let output = builder.write_texture(target_ldr);

            let node = ToneMapPassNode {
                input_tex: input_hdr,
                output_tex: output,
                pipeline,
                static_bg,
                transient_layout,
                transient_bg: None,
            };
            (node, output)
        })
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PassNode (ephemeral, created per frame)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Ephemeral tone mapping pass node — zero-drop POD.
///
/// All fields are either `Copy` types or borrowed references with
/// frame lifetime `'a`.  No hash lookups occur during execute.
struct ToneMapPassNode<'a> {
    input_tex: TextureNodeId,
    output_tex: TextureNodeId,
    pipeline: &'a wgpu::RenderPipeline,

    /// Feature-owned static bind group (Group 1): sampler + uniforms + optional LUT.
    static_bg: &'a wgpu::BindGroup,
    /// Layout for transient bind group (Group 2).
    transient_layout: &'a Tracked<wgpu::BindGroupLayout>,

    /// Pointer-stable transient bind group built in `prepare()`.
    transient_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for ToneMapPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let PrepareContext {
            views,
            global_bind_group_cache: cache,
            device,
            ..
        } = ctx;
        let device = *device;
        let input_view = views.get_texture_view(self.input_tex);
        let layout = self.transient_layout;

        let key = BindGroupKey::new(layout.id()).with_resource(input_view.id());

        let bg = cache.get_or_create_bg(key, || {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ToneMap Transient BG (G2)"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                }],
            })
        });
        self.transient_bg = Some(bg);
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let global_bind_group = ctx.baked_lists.global_bind_group;

        let transient_bg = self.transient_bg.expect("Transient BG not prepared!");

        let rtt = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ToneMap Pass"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(self.pipeline);
        rpass.set_bind_group(0, global_bind_group, &[]);
        rpass.set_bind_group(1, self.static_bg, &[]);
        rpass.set_bind_group(2, transient_bg, &[]);
        rpass.draw(0..3, 0..1);
    }
}
