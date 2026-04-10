//! PMREM generation pass.
//!
//! Reads the persistent scene-owned base cubemap and writes the persistent
//! scene-owned PMREM cubemap. Source conversion and procedural sky baking are
//! handled by separate nodes so the RenderGraph can express precise ordering.

use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::TextureNodeId;
use crate::graph::core::context::{ExecuteContext, ExtractContext};
use crate::graph::core::node::PassNode;
use crate::pipeline::{
    ComputePipelineId, ComputePipelineKey, ShaderCompilationOptions, ShaderSource,
};

pub struct IblComputeFeature {
    pipeline_id: Option<ComputePipelineId>,
    source_layout: wgpu::BindGroupLayout,
    dest_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl IblComputeFeature {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let source_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("IBL Source BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let dest_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("IBL Dest BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                },
                count: None,
            }],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("IBL PMREM Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        Self {
            pipeline_id: None,
            source_layout,
            dest_layout,
            sampler,
        }
    }

    pub fn extract_and_prepare(&mut self, ctx: &mut ExtractContext) {
        self.ensure_pipeline(ctx);
    }

    fn ensure_pipeline(&mut self, ctx: &mut ExtractContext) {
        if self.pipeline_id.is_some() {
            return;
        }

        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/utility/ibl"),
            &ShaderCompilationOptions::default(),
        );

        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("IBL Compute PL"),
                bind_group_layouts: &[Some(&self.source_layout), Some(&self.dest_layout)],
                immediate_size: 0,
            });

        self.pipeline_id = Some(ctx.pipeline_cache.get_or_create_compute(
            ctx.device,
            module,
            &layout,
            &ComputePipelineKey { shader_hash },
            "IBL Compute Pipeline",
        ));
    }

    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        base_cube: TextureNodeId,
        pmrem: TextureNodeId,
        base_cube_texture: &'a wgpu::Texture,
        pmrem_texture: &'a wgpu::Texture,
    ) {
        let pipeline = self
            .pipeline_id
            .map(|id| ctx.pipeline_cache.get_compute_pipeline(id));

        ctx.graph.add_pass("IBL_Compute", |builder| {
            builder.read_texture(base_cube);
            builder.write_texture(pmrem);
            let node = IblComputePassNode {
                base_cube_texture,
                pmrem_texture,
                pipeline,
                source_layout: &self.source_layout,
                dest_layout: &self.dest_layout,
                sampler: &self.sampler,
            };
            (node, ())
        });
    }
}

struct IblComputePassNode<'a> {
    base_cube_texture: &'a wgpu::Texture,
    pmrem_texture: &'a wgpu::Texture,
    pipeline: Option<&'a wgpu::ComputePipeline>,
    source_layout: &'a wgpu::BindGroupLayout,
    dest_layout: &'a wgpu::BindGroupLayout,
    sampler: &'a wgpu::Sampler,
}

impl IblComputePassNode<'_> {
    fn create_mapped_buffer(device: &wgpu::Device, label: &str, params: [f32; 4]) -> wgpu::Buffer {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: std::mem::size_of::<[f32; 4]>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&params));
        buffer.unmap();
        buffer
    }
}

impl PassNode<'_> for IblComputePassNode<'_> {
    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let pipeline = self.pipeline.expect("IBL pipeline must exist");

        let source_view = self
            .base_cube_texture
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("IBL Source Cube"),
                dimension: Some(wgpu::TextureViewDimension::Cube),
                ..Default::default()
            });

        let mip_levels = self.pmrem_texture.mip_level_count();
        let pmrem_size = self.pmrem_texture.width();
        let roughness_denominator = (mip_levels.saturating_sub(1)).max(1) as f32;

        for mip in 0..mip_levels {
            let mip_size = (pmrem_size >> mip).max(1);
            let params = [mip as f32 / roughness_denominator, mip_size as f32, 0.0, 0.0];
            let param_buffer = Self::create_mapped_buffer(ctx.device, "IBL Params", params);

            let source_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("IBL Source BG"),
                layout: self.source_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&source_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: param_buffer.as_entire_binding(),
                    },
                ],
            });

            let dest_view = self
                .pmrem_texture
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some("IBL PMREM Mip"),
                    format: Some(wgpu::TextureFormat::Rgba16Float),
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(6),
                    usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                });

            let dest_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("IBL Dest BG"),
                layout: self.dest_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&dest_view),
                }],
            });

            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("IBL Compute"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &source_bg, &[]);
            cpass.set_bind_group(1, &dest_bg, &[]);
            let group_count = mip_size.div_ceil(8);
            cpass.dispatch_workgroups(group_count, group_count, 6);
        }
    }
}