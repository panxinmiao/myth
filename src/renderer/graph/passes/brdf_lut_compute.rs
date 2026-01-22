use std::{borrow::Cow, cell::RefCell};
use crate::{renderer::graph::{RenderContext, RenderNode}, resources::texture::TextureSource};

pub struct BRDFLutComputePass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    brdf_lut_texture_handle: RefCell<Option<u64>>,
}

impl BRDFLutComputePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("BRDF LUT Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../../pipeline/shaders/program/brdf_lut.wgsl"))),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BRDF LUT Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
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
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            brdf_lut_texture_handle: RefCell::new(None),
        }
    }
}

impl RenderNode for BRDFLutComputePass {
    fn name(&self) -> &str { "BRDF LUT Gen" }

    fn run(&self, ctx: &mut RenderContext, encoder: &mut wgpu::CommandEncoder) {
        // 检查是否已经生成过 BRDF LUT
        if ctx.scene.environment.brdf_lut.is_some() {
            return;
        }
        if self.brdf_lut_texture_handle.borrow().is_some() {
            ctx.scene.environment.brdf_lut = Some(TextureSource::Attachment(
                self.brdf_lut_texture_handle.borrow().unwrap()
            ));
            return;
        }

        // 1. 创建 LUT 纹理 (512x512 Rgba16Float)
        let size = 512;
        let texture_desc = wgpu::TextureDescriptor {
            label: Some("BRDF LUT"),
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        
        // 使用 ResourceManager 分配 (假设你有这个 API，或者直接用 device 创建并注册)
        let texture = ctx.wgpu_ctx.device.create_texture(&texture_desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 2. 创建 BindGroup
        let bind_group = ctx.wgpu_ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BRDF LUT BindGroup"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });

        // 3. Dispatch
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { 
                label: Some("BRDF LUT Pass"), 
                timestamp_writes: None 
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(size / 8, size / 8, 1);
        }

        // 4. 将生成的纹理注册到 Asset 系统并保存 Handle 到 Environment
        let handle = ctx.resource_manager.register_internal_texture_by_name("BRDF_LUT", view);
        ctx.scene.environment.brdf_lut = Some(TextureSource::Attachment(handle));

        self.brdf_lut_texture_handle.replace(Some(handle));

        // println!("BRDF LUT generated and registered with handle {}", handle);
    }
}