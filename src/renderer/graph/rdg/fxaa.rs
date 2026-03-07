use super::builder::PassBuilder;
use super::context::RdgExecuteContext;
use super::node::PassNode;
use super::types::TextureNodeId;
use crate::FxaaQuality;
use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::resources::{CommonSampler, Tracked};
use crate::renderer::graph::rdg::context::{PassPrepareContext, RdgPrepareContext};
use crate::renderer::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions,
};
use wgpu::CommandEncoder;

type FxaaL1CacheKey = (FxaaQuality, wgpu::TextureFormat);

pub struct RdgFxaaPass {
    // ─── RDG Slots (set by Composer) ───────────────────────────────
    pub input_tex: TextureNodeId,
    pub output_tex: TextureNodeId,

    // ─── Push Parameters ───────────────────────────────────────────
    pub target_quality: FxaaQuality,
    /// Output texture format — pushed by the Composer before
    /// `prepare_resources()` so the pipeline can be compiled without
    /// access to the render graph.
    pub output_format: wgpu::TextureFormat,

    // ─── Persistent Cache (populated in prepare_resources) ─────────
    l1_cache_key: Option<FxaaL1CacheKey>,
    pipeline_id: Option<RenderPipelineId>,
    bind_group_layout: Option<Tracked<wgpu::BindGroupLayout>>,

    current_bind_group_key: Option<BindGroupKey>,
}

impl RdgFxaaPass {
    #[must_use]
    pub fn new() -> Self {
        Self {
            input_tex: TextureNodeId(0),
            output_tex: TextureNodeId(0),
            target_quality: FxaaQuality::High,
            output_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            l1_cache_key: None,
            pipeline_id: None,
            bind_group_layout: None,
            current_bind_group_key: None,
        }
    }
}

impl PassNode for RdgFxaaPass {
    fn name(&self) -> &'static str {
        "RDG_FXAA_Pass"
    }

    fn prepare_resources(&mut self, ctx: &mut PassPrepareContext) {
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
        let current_key = (self.target_quality, self.output_format);

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
                format: self.output_format,
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

            // Pipeline changed — invalidate cached BindGroup key
            self.current_bind_group_key = None;
        }
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        self.input_tex = builder.read_blackboard("LDR_Intermediate");
        self.output_tex = builder.write_blackboard("Surface_Out");
    }

    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {
        // Assemble transient BindGroup (references RDG-allocated input view)
        let input_view = ctx.views.get_texture_view(self.input_tex);
        let sampler = ctx.sampler_registry.get_common(CommonSampler::LinearClamp);
        let bind_group_layout = self.bind_group_layout.as_ref().unwrap();

        let current_key = BindGroupKey::new(bind_group_layout.id())
            .with_resource(input_view.id())
            .with_resource(sampler.id());

        if self.current_bind_group_key.as_ref() != Some(&current_key) {
            if ctx.global_bind_group_cache.get(&current_key).is_none() {
                let new_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("FXAA BindGroup"),
                    layout: &**bind_group_layout,
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
        let output_view = ctx.get_texture_view(self.output_tex);

        let pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.pipeline_id.expect("Pipeline not initialized!"));

        let bind_group_key = self
            .current_bind_group_key
            .as_ref()
            .expect("BindGroupKey should have been set in prepare!");
        let bind_group = ctx
            .global_bind_group_cache
            .get(&bind_group_key)
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
