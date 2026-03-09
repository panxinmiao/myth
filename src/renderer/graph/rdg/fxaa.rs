//! FXAA Feature + Ephemeral PassNode
//!
//! - **`FxaaFeature`** (long-lived): owns pipeline cache, bind group layout.
//!   `extract_and_prepare()` compiles pipelines for the target quality/format.
//! - **`FxaaPassNode`** (ephemeral per-frame): carries lightweight IDs and
//!   a transient bind-group slot.  Created by `FxaaFeature::add_to_graph()`.

use super::builder::PassBuilder;
use super::context::{ExtractContext, RdgExecuteContext, RdgPrepareContext};
use super::graph::RenderGraph;
use super::node::PassNode;
use super::types::TextureNodeId;
use crate::FxaaQuality;
use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::resources::{CommonSampler, Tracked};
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
                        bind_group_layouts: &[&self.bind_group_layout.as_ref().unwrap()],
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
    pub fn add_to_graph(
        &self,
        rdg: &mut RenderGraph,
        input_tex: TextureNodeId,
        output_tex: TextureNodeId,
    ) {
        let node = FxaaPassNode {
            input_tex,
            output_tex,
            pipeline_id: self.pipeline_id.expect("FxaaFeature not prepared"),
            layout: self.bind_group_layout.clone().unwrap(),
            current_bind_group_key: None,
        };
        rdg.add_pass(Box::new(node));
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PassNode (ephemeral, created per frame)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct FxaaPassNode {
    input_tex: TextureNodeId,
    output_tex: TextureNodeId,
    pipeline_id: RenderPipelineId,
    layout: Tracked<wgpu::BindGroupLayout>,
    current_bind_group_key: Option<BindGroupKey>,
}

impl PassNode for FxaaPassNode {
    fn name(&self) -> &'static str {
        "RDG_FXAA_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        builder.read_texture(self.input_tex);
        builder.write_texture(self.output_tex);
    }

    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {
        let input_view = ctx.views.get_texture_view(self.input_tex);
        let sampler = ctx.sampler_registry.get_common(CommonSampler::LinearClamp);

        let current_key = BindGroupKey::new(self.layout.id())
            .with_resource(input_view.id())
            .with_resource(sampler.id());

        if self.current_bind_group_key.as_ref() != Some(&current_key) {
            if ctx.global_bind_group_cache.get(&current_key).is_none() {
                let new_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("FXAA BindGroup"),
                    layout: &*self.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&**input_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&**sampler),
                        },
                    ],
                });
                ctx.global_bind_group_cache
                    .insert(current_key.clone(), new_bg);
            }
            self.current_bind_group_key = Some(current_key);
        }
    }

    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut CommandEncoder) {
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

        let rtt = ctx.get_color_attachment(self.output_tex, None, None);

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RDG FXAA Pass"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(pipeline);
        rpass.set_bind_group(0, bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }
}
