//! CAS (Contrast Adaptive Sharpening) Feature + Ephemeral PassNode
//!
//! - **`CasFeature`** (long-lived): owns pipeline cache, bind group layout,
//!   and a small uniform buffer for sharpening intensity.
//!   `extract_and_prepare()` compiles the pipeline once (format-keyed).
//! - **`CasPassNode`** (ephemeral per-frame): carries lightweight IDs and
//!   a transient bind-group slot.  Created by `CasFeature::add_to_graph()`.
//!
//! CAS is placed immediately after TAA resolve in the HDR pipeline to
//! recover fine detail lost to temporal filtering, before bloom/tone-mapping.
//!
//! # Binding Layout (Group 0)
//!
//! | Binding | Type          | Content                 |
//! |---------|---------------|-------------------------|
//! | 0       | `texture_2d`  | Source HDR colour       |
//! | 1       | `sampler`     | Nearest clamp sampler   |
//! | 2       | `uniform`     | CasParams { sharpness } |

use crate::core::gpu::{CommonSampler, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    BufferDesc, BufferNodeId, ExecuteContext, ExtractContext, PassNode, PrepareContext,
    RenderTargetOps, TextureDesc, TextureNodeId,
};
use crate::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions, ShaderSource,
};
use wgpu::CommandEncoder;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Feature (long-lived, stored in RendererState)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub struct CasFeature {
    // ─── Pipeline ──────────────────────────────────────────────────
    cached_format: Option<wgpu::TextureFormat>,
    pipeline_id: Option<RenderPipelineId>,
    bind_group_layout: Option<Tracked<wgpu::BindGroupLayout>>,

    // ─── Uniform Buffer ────────────────────────────────────────────
    params_buffer: Option<Tracked<wgpu::Buffer>>,
    last_sharpness: f32,
}

impl Default for CasFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl CasFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            cached_format: None,
            pipeline_id: None,
            bind_group_layout: None,
            params_buffer: None,
            last_sharpness: -1.0, // force first upload
        }
    }

    /// Pre-RDG resource preparation: create layout, compile pipeline,
    /// upload uniform if sharpness changed.
    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        sharpness: f32,
        output_format: wgpu::TextureFormat,
    ) {
        // ── 1. Bind group layout (once) ────────────────────────────────
        if self.bind_group_layout.is_none() {
            let layout = ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("CAS BindGroup Layout"),
                    entries: &[
                        // binding 0: source texture
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
                        // binding 1: nearest sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        // binding 2: CasParams uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(16),
                            },
                            count: None,
                        },
                    ],
                });
            self.bind_group_layout = Some(Tracked::new(layout));
        }

        // ── 2. Uniform buffer (create once, update on change) ──────────
        if self.params_buffer.is_none() {
            let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("CAS Params"),
                size: 16, // vec4 padding
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.params_buffer = Some(Tracked::new(buffer));
        }

        if (self.last_sharpness - sharpness).abs() > f32::EPSILON {
            let data: [f32; 4] = [sharpness, 0.0, 0.0, 0.0];
            ctx.queue.write_buffer(
                self.params_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&data),
            );
            self.last_sharpness = sharpness;
        }

        // ── 3. Pipeline (compile on format change) ─────────────────────
        if self.cached_format != Some(output_format) {
            let options = ShaderCompilationOptions::default();

            let (shader_module, shader_hash) = ctx.shader_manager.get_or_compile(
                ctx.device,
                ShaderSource::File("entry/post_process/cas"),
                &options,
            );

            let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
                format: output_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            });

            let key = FullscreenPipelineKey::fullscreen(
                shader_hash,
                smallvec::smallvec![color_target],
                None,
            );

            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("CAS Pipeline Layout"),
                        bind_group_layouts: &[self.bind_group_layout.as_deref()],
                        immediate_size: 0,
                    });

            let id = ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                shader_module,
                &pipeline_layout,
                &key,
                "CAS Pipeline",
            );
            self.pipeline_id = Some(id);
            self.cached_format = Some(output_format);
        }
    }

    /// Insert the CAS pass into the RDG.
    ///
    /// Reads from `input_color` (the TAA-resolved HDR texture), creates a
    /// new `CAS_Output` texture, and returns its `TextureNodeId` for
    /// downstream passes (bloom, tone-mapping).
    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        input_color: TextureNodeId,
    ) -> TextureNodeId {
        let pipeline_id = self.pipeline_id.expect("CasFeature not prepared");
        let pipeline = ctx.pipeline_cache.get_render_pipeline(pipeline_id);
        let layout = self.bind_group_layout.as_ref().unwrap();
        let params_buffer = self.params_buffer.as_ref().unwrap();

        let cas_desc = TextureDesc::new_2d(
            ctx.frame_config.width,
            ctx.frame_config.height,
            crate::HDR_TEXTURE_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );

        ctx.graph.add_pass("CAS_Pass", |builder| {
            builder.read_texture(input_color);
            let params_buffer = builder.read_external_buffer(
                "CAS_Params",
                BufferDesc::new(
                    16,
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                ),
                params_buffer,
            );
            let output = builder.create_texture("CAS_Output", cas_desc);

            let node = CasPassNode {
                input_tex: input_color,
                output_tex: output,
                pipeline,
                layout,
                params_buffer,
                transient_bg: None,
            };
            (node, output)
        })
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PassNode (ephemeral, created per frame)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct CasPassNode<'a> {
    input_tex: TextureNodeId,
    output_tex: TextureNodeId,
    pipeline: &'a wgpu::RenderPipeline,
    layout: &'a Tracked<wgpu::BindGroupLayout>,
    params_buffer: BufferNodeId,
    transient_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for CasPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.transient_bg = Some(crate::myth_bind_group!(ctx, self.layout, Some("CAS BindGroup"), [
            0 => self.input_tex,
            1 => CommonSampler::NearestClamp,
            2 => self.params_buffer,
        ]));
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut CommandEncoder) {
        let bind_group = self.transient_bg.expect("CAS BG not prepared!");

        let rtt = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("CAS Pass"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(self.pipeline);
        rpass.set_bind_group(0, bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }
}
