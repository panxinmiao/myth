use crate::renderer::graph::RenderNode;
use crate::renderer::graph::context::{ExecuteContext, PrepareContext};
use std::borrow::Cow;

pub struct BRDFLutComputePass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl BRDFLutComputePass {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("BRDF LUT Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../../pipeline/shaders/program/brdf_lut.wgsl"
            ))),
        });

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

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("BRDF LUT Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BRDF LUT Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
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
            cpass.set_pipeline(&self.pipeline);
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
