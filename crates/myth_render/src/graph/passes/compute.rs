//! RDG compute utilities.
//!
//! This module currently contains only the BRDF LUT pre-integration pass.
//! Environment source conversion and PMREM generation live in dedicated pass
//! modules so the RenderGraph can model their dependencies explicitly.

use crate::core::gpu::BRDF_LUT_SIZE;
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::context::{ExecuteContext, ExtractContext};
use crate::graph::core::node::PassNode;
use crate::pipeline::{
    ComputePipelineId, ComputePipelineKey, ShaderCompilationOptions, ShaderSource,
};

/// BRDF LUT compute feature.
///
/// Dispatches a single compute shader that fills a 128x128 `Rgba16Float`
/// storage texture with pre-integrated BRDF data. The texture is owned by the
/// `ResourceManager` and consumed through the global bind group.
pub struct BrdfLutFeature {
    pipeline_id: Option<ComputePipelineId>,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    active: bool,
}

impl BrdfLutFeature {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BRDF LUT BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });

        Self {
            pipeline_id: None,
            bind_group_layout,
            bind_group: None,
            active: false,
        }
    }

    fn ensure_pipeline(&mut self, ctx: &mut ExtractContext) {
        if self.pipeline_id.is_some() {
            return;
        }

        let compilation_options = wgpu::PipelineCompilationOptions::default();

        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/utility/brdf_lut"),
            &ShaderCompilationOptions::default(),
        );

        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("BRDF LUT Pipeline Layout"),
                bind_group_layouts: &[Some(&self.bind_group_layout)],
                immediate_size: 0,
            });

        self.pipeline_id = Some(ctx.pipeline_cache.get_or_create_compute(
            ctx.device,
            module,
            &layout,
            &ComputePipelineKey::new(shader_hash).with_compilation_options(&compilation_options),
            &compilation_options,
            "BRDF LUT Pipeline",
        ));
    }

    pub fn extract_and_prepare(&mut self, ctx: &mut ExtractContext) {
        if !ctx.resource_manager.needs_brdf_compute {
            self.active = false;
            return;
        }

        self.ensure_pipeline(ctx);

        let texture = ctx
            .resource_manager
            .brdf_lut_texture
            .as_ref()
            .expect("BRDF LUT texture must exist before compute");

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BRDF LUT BG"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        }));

        self.active = true;
        ctx.resource_manager.needs_brdf_compute = false;
    }

    pub fn add_to_graph<'a>(&'a self, ctx: &mut GraphBuilderContext<'a, '_>) {
        let pipeline = self
            .pipeline_id
            .map(|id| ctx.pipeline_cache.get_compute_pipeline(id));
        let bind_group = self.bind_group.as_ref();
        let active = self.active;

        ctx.graph.add_pass("BRDF_LUT", |builder| {
            builder.mark_side_effect();
            (
                BrdfLutPassNode {
                    pipeline,
                    bind_group,
                    active,
                },
                (),
            )
        });
    }
}

struct BrdfLutPassNode<'a> {
    pipeline: Option<&'a wgpu::ComputePipeline>,
    bind_group: Option<&'a wgpu::BindGroup>,
    active: bool,
}

impl PassNode<'_> for BrdfLutPassNode<'_> {
    fn execute(&self, _ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        if !self.active {
            return;
        }

        let Some(bind_group) = self.bind_group else {
            return;
        };

        let pipeline = self.pipeline.expect("BRDF LUT pipeline must exist");

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("BRDF LUT"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.dispatch_workgroups(BRDF_LUT_SIZE / 8, BRDF_LUT_SIZE / 8, 1);
    }
}
