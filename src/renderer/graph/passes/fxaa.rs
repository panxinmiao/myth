//! FXAA Feature + Ephemeral PassNode
//!
//! - **`FxaaFeature`** (long-lived): owns pipeline cache, bind group layout.
//!   `extract_and_prepare()` compiles pipelines for the target quality/format.
//! - **`FxaaPassNode`** (ephemeral per-frame): carries lightweight IDs and
//!   a transient bind-group slot.  Created by `FxaaFeature::add_to_graph()`.

use crate::FxaaQuality;
use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::gpu::{CommonSampler, Tracked};
use crate::renderer::graph::core::{
    ExecuteContext, ExtractContext, PassNode, PrepareContext, RenderGraph, RenderTargetOps,
    TextureNodeId,
};
use crate::renderer::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions,
};
use wgpu::CommandEncoder;

type FxaaL1CacheKey = (FxaaQuality, wgpu::TextureFormat);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Feature (long-lived, stored in RenderFeatures)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub struct FxaaFeature {
    /// Target FXAA quality — set by the caller before extract_and_prepare.
    pub target_quality: FxaaQuality,

    // ─── Persistent Cache ──────────────────────────────────────────
    l1_cache_key: Option<FxaaL1CacheKey>,
    pipeline_id: Option<RenderPipelineId>,
    bind_group_layout: Option<Tracked<wgpu::BindGroupLayout>>,
}

impl Default for FxaaFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl FxaaFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            target_quality: FxaaQuality::High,
            l1_cache_key: None,
            pipeline_id: None,
            bind_group_layout: None,
        }
    }

    /// Pre-RDG resource preparation: create layout, compile pipeline.
    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        output_format: wgpu::TextureFormat,
    ) {
        // ── 1. Lazy-create BindGroupLayout (once) ──────────────────
        if self.bind_group_layout.is_none() {
            let layout = ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("FXAA BindGroup Layout"),
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
                    ],
                });
            self.bind_group_layout = Some(Tracked::new(layout));
        }

        // ── 2. L1 Cache: compile pipeline on quality/format change ─
        let current_key = (self.target_quality, output_format);

        if self.l1_cache_key != Some(current_key) {
            let mut options = ShaderCompilationOptions::default();
            if self.target_quality != FxaaQuality::Medium {
                options.add_define(self.target_quality.define_key(), "1");
            }

            let (shader_module, shader_hash) = ctx.shader_manager.get_or_compile_template(
                ctx.device,
                "passes/fxaa",
                &options,
                "",
                "",
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
                        label: Some("FXAA Pipeline Layout"),
                        bind_group_layouts: &[self.bind_group_layout.as_ref().unwrap()],
                        immediate_size: 0,
                    });

            let id = ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                shader_module,
                &pipeline_layout,
                &key,
                &format!("FXAA Pipeline {:?}", self.target_quality),
            );
            self.pipeline_id = Some(id);
            self.l1_cache_key = Some(current_key);
        }
    }

    /// Build the ephemeral pass node and insert it into the graph.
    ///
    /// Accepts the LDR input and the target surface, performs an SSA relay
    /// on `target_surface` (via `mutate_and_export`), and returns the
    /// updated surface handle. This enforces a pure dataflow chain where
    /// every Feature explicitly produces a new resource version.
    pub fn add_to_graph<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        input_ldr: TextureNodeId,
        target_surface: TextureNodeId,
    ) -> TextureNodeId {
        let pipeline_id = self.pipeline_id.expect("FxaaFeature not prepared");
        let pipeline = graph.pipeline_cache.get_render_pipeline(pipeline_id);
        let layout = self.bind_group_layout.as_ref().unwrap();

        graph.add_pass("FXAA_Pass", |builder| {
            builder.read_texture(input_ldr);
            let output = builder.mutate_and_export(target_surface, "Surface_FXAA");

            let node = FxaaPassNode {
                input_tex: input_ldr,
                output_tex: output,
                pipeline,
                layout,
                transient_bg: None,
            };
            (node, output)
        })
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PassNode (ephemeral, created per frame)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct FxaaPassNode<'a> {
    input_tex: TextureNodeId,
    output_tex: TextureNodeId,
    pipeline: &'a wgpu::RenderPipeline,
    layout: &'a Tracked<wgpu::BindGroupLayout>,
    transient_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for FxaaPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let PrepareContext { views, global_bind_group_cache: cache, device, sampler_registry, .. } = ctx;
        let device = *device;
        let input_view = views.get_texture_view(self.input_tex);
        let sampler = sampler_registry.get_common(CommonSampler::LinearClamp);

        let key = BindGroupKey::new(self.layout.id())
            .with_resource(input_view.id())
            .with_resource(sampler.id());

        let layout = self.layout;
        let bg = cache.get_or_create_bg(key, || {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("FXAA BindGroup"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            })
        });
        self.transient_bg = Some(bg);
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut CommandEncoder) {
        let bind_group = self.transient_bg.expect("FXAA BG not prepared!");

        let rtt = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("FXAA Pass"),
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
