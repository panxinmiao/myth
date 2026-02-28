use crate::renderer::graph::RenderNode;
use crate::renderer::graph::context::{ExecuteContext, PrepareContext};
use crate::renderer::pipeline::{ComputePipelineId, ComputePipelineKey};

pub struct BRDFLutComputePass {
    pipeline_id: Option<ComputePipelineId>,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl BRDFLutComputePass {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BRDF LUT Layout"),
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
        }
    }

    /// Ensures the compute pipeline is created via the global `PipelineCache`.
    fn ensure_pipeline(&mut self, ctx: &mut PrepareContext) {
        if self.pipeline_id.is_some() {
            return;
        }

        let source = include_str!("../../pipeline/shaders/program/brdf_lut.wgsl");
        let (module, shader_hash) =
            ctx.shader_manager
                .get_or_compile_raw(&ctx.wgpu_ctx.device, "BRDF LUT Shader", source);

        let layout = ctx
            .wgpu_ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("BRDF LUT Pipeline Layout"),
                bind_group_layouts: &[&self.bind_group_layout],
                immediate_size: 0,
            });

        let key = ComputePipelineKey { shader_hash };
        self.pipeline_id = Some(ctx.pipeline_cache.get_or_create_compute(
            &ctx.wgpu_ctx.device,
            module,
            &layout,
            &key,
            "BRDF LUT Pipeline",
        ));
    }
}

impl RenderNode for BRDFLutComputePass {
    fn name(&self) -> &'static str {
        "BRDF LUT Gen"
    }

    fn prepare(&mut self, ctx: &mut PrepareContext) {
        if !ctx.resource_manager.needs_brdf_compute {
            return;
        }

        // Ensure pipeline exists (lazy creation via PipelineCache)
        self.ensure_pipeline(ctx);

        let pipeline = ctx
            .pipeline_cache
            .get_compute_pipeline(self.pipeline_id.expect("Pipeline must exist after ensure"));

        let texture = ctx
            .resource_manager
            .brdf_lut_texture
            .as_ref()
            .expect("BRDF LUT texture must be created by ensure_brdf_lut");

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = ctx
            .wgpu_ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("BRDF LUT BindGroup"),
                layout: &self.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                }],
            });

        let size = crate::renderer::core::resources::BRDF_LUT_SIZE;

        let mut encoder =
            ctx.wgpu_ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("BRDF LUT Compute Encoder"),
                });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BRDF LUT Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(size / 8, size / 8, 1);
        }

        ctx.wgpu_ctx.queue.submit(std::iter::once(encoder.finish()));
        ctx.resource_manager.needs_brdf_compute = false;
    }

    fn run(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {
        // Compute work is done in prepare()
    }
}
