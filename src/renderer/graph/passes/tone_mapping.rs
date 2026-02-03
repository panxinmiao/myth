use std::borrow::Cow;

use crate::{render::{RenderContext, RenderNode}, renderer::{core::{binding::BindGroupKey, resources::Tracked}, pipeline::shader_gen::ShaderGenerator}, resources::buffer::{CpuBuffer, GpuData}};


// 定义 Uniform 数据
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ToneMapUniforms {
    pub exposure: f32,
    // 0: Linear, 1: Reinhard, 2: Cineon, 3: ACESFilmic
    pub tone_mapping_mode: u32, 
    pub _pad: [u32; 2],
}


impl Default for ToneMapUniforms {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            tone_mapping_mode: 3,
            _pad: [0; 2],
        }
    }
}

impl GpuData for ToneMapUniforms {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    fn byte_size(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

pub struct ToneMapPass {
    // 资源
    layout: Tracked<wgpu::BindGroupLayout>,
    pipeline: wgpu::RenderPipeline,
    sampler: Tracked<wgpu::Sampler>,

    pub uniforms: CpuBuffer<ToneMapUniforms>,

    // 运行时状态 (Prepare 阶段生成，Run 阶段使用)
    current_bind_group: Option<wgpu::BindGroup>,
}

impl ToneMapPass {
    pub fn new(device: &wgpu::Device) -> Self {

        let shader_code = ShaderGenerator::generate_shader(
            "",
            "",
            "passes/tone_mapping",
            &Default::default(),
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tone Map Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader_code)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tone Map Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { 
                        sample_type: wgpu::TextureSampleType::Float { filterable: true }, 
                        view_dimension: wgpu::TextureViewDimension::D2, 
                        multisampled: false 
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },

                // Binding 2: Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let tracked_layout = Tracked::new(bind_group_layout);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tone Map Pipeline Layout"),
            bind_group_layouts: &[&tracked_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Tone Map Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // 4. 创建 Sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ToneMap Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniforms = CpuBuffer::new(ToneMapUniforms::default(), wgpu::BufferUsages::UNIFORM, Some("ToneMap Uniforms"));

        Self {
            layout: tracked_layout,
            pipeline,
            sampler: Tracked::new(sampler),
            uniforms,
            // uniform_version: 0,
            current_bind_group: None,
        }

    }

    pub fn set_exposure(&mut self, exposure: f32) {
        let data = self.uniforms.read();
        if (data.exposure - exposure).abs() > 1e-5 {
            drop(data); // 释放锁
            let mut data = self.uniforms.write();
            data.exposure = exposure;
            // self.uniform_version += 1; // 标记变化
        }
    }


}

impl RenderNode for ToneMapPass {

    fn prepare(&mut self, ctx: &mut RenderContext) {
        // let (input, output) = ctx.acquire_post_process_io();

        let input_view = ctx.current_post_process_output();

        let gpu_buffer_id = ctx.resource_manager.ensure_buffer_id(&self.uniforms);

        let key = BindGroupKey::new(self.layout.id())
            .with_resource(input_view.id())
            .with_resource(self.sampler.id())
            .with_resource(gpu_buffer_id);

        let bind_group = ctx.global_bind_group_cache.get_or_create(key, || {

            let gpu_buffer = ctx.resource_manager.gpu_buffers.get(&gpu_buffer_id)
                .expect("GpuBuffer must exist after ensure_buffer_id");

            ctx.wgpu_ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ToneMap BindGroup"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: gpu_buffer.buffer.as_entire_binding(),
                    },
                ],
            })
        });

        self.current_bind_group = Some(bind_group.clone());

    }



    fn run(&self, ctx: &mut RenderContext, encoder: &mut wgpu::CommandEncoder) {
        //todo: 分离两个阶段。
        // self.prepare(ctx);

        // 输出到 Surface (屏幕)
        let pass_desc = wgpu::RenderPassDescriptor {
            label: Some("Final ToneMap Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.surface_view, 
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            ..Default::default()
        };

        let mut pass = encoder.begin_render_pass(&pass_desc);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &ctx.frame_resources.screen_bind_group, &[]);
        pass.draw(0..3, 0..1); // 全屏三角形

    }
    
    fn name(&self) -> &str {
        "Tone Mapping Pass"
    }
}