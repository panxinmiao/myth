//! FXAA Feature & Transient Pass Node
//!
//! - [`FxaaFeature`] (persistent): pipeline cache by (quality, format), layout.
//! - [`RdgFxaaPassNode`] (transient): per-frame bind group and execution.

use crate::FxaaQuality;
use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::resources::{CommonSampler, Tracked};
use crate::renderer::graph::rdg::builder::PassBuilder;
use crate::renderer::graph::rdg::context::{RdgExecuteContext, RdgPrepareContext};
use crate::renderer::graph::rdg::feature::ExtractContext;
use crate::renderer::graph::rdg::graph::RenderGraph;
use crate::renderer::graph::rdg::node::PassNode;
use crate::renderer::graph::rdg::types::TextureNodeId;
use crate::renderer::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions,
};
use wgpu::CommandEncoder;

type FxaaL1CacheKey = (FxaaQuality, wgpu::TextureFormat);

// =============================================================================
// Persistent Feature
// =============================================================================

/// Persistent FXAA Feature — owns bind group layout and a pipeline cache
/// keyed by `(quality, output_format)`.
pub struct FxaaFeature {
    bind_group_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    l1_cache_key: Option<FxaaL1CacheKey>,
    pipeline_id: Option<RenderPipelineId>,
}

impl FxaaFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            bind_group_layout: None,
            l1_cache_key: None,
            pipeline_id: None,
        }
    }

    /// Prepare persistent GPU resources and compile the pipeline for the
    /// current frame's quality / format.
    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        quality: FxaaQuality,
        output_format: wgpu::TextureFormat,
    ) {
        self.ensure_layout(ctx.device);

        let current_key = (quality, output_format);
        if self.l1_cache_key != Some(current_key) {
            let mut options = ShaderCompilationOptions::default();
            if quality != FxaaQuality::Medium {
                options.add_define(quality.define_key(), "1");
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

            self.pipeline_id = Some(ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                shader_module,
                &pipeline_layout,
                &key,
                &format!("FXAA Pipeline {:?}", quality),
            ));
            self.l1_cache_key = Some(current_key);
        }
    }

    /// Build and inject a transient FXAA node into the render graph.
    pub fn add_to_graph(
        &self,
        rdg: &mut RenderGraph,
        input_tex: TextureNodeId,
        output_tex: TextureNodeId,
    ) {
        let node = Box::new(RdgFxaaPassNode {
            input_tex,
            output_tex,
            pipeline_id: self.pipeline_id.unwrap(),
            bind_group_layout: self.bind_group_layout.clone().unwrap(),
            current_bind_group_key: None,
        });
        rdg.add_pass_owned(node);
    }

    fn ensure_layout(&mut self, device: &wgpu::Device) {
        if self.bind_group_layout.is_some() {
            return;
        }

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
}

// =============================================================================
// Transient Pass Node
// =============================================================================

struct RdgFxaaPassNode {
    input_tex: TextureNodeId,
    output_tex: TextureNodeId,
    pipeline_id: RenderPipelineId,
    bind_group_layout: Tracked<wgpu::BindGroupLayout>,
    current_bind_group_key: Option<BindGroupKey>,
}

impl PassNode for RdgFxaaPassNode {
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

        let key = BindGroupKey::new(self.bind_group_layout.id())
            .with_resource(input_view.id())
            .with_resource(sampler.id());

        if self.current_bind_group_key.as_ref() != Some(&key) {
            if ctx.global_bind_group_cache.get(&key).is_none() {
                let new_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("FXAA BindGroup"),
                    layout: &*self.bind_group_layout,
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
                ctx.global_bind_group_cache.insert(key.clone(), new_bg);
            }
            self.current_bind_group_key = Some(key);
        }
    }

    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut CommandEncoder) {
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
            label: Some("RDG FXAA Pass"),
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
        rpass.set_bind_group(0, bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }
}
