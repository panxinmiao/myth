use std::borrow::Cow;

use rustc_hash::FxHashMap;

use crate::{ShaderDefines, render::{RenderContext, RenderNode}, renderer::{core::{binding::BindGroupKey, resources::Tracked}, pipeline::{ShaderCompilationOptions, shader_gen::ShaderGenerator}}, resources::buffer::{CpuBuffer, GpuData}};



#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToneMappingMode {
    Linear,
    NEUTRAL,
    Reinhard,
    Cineon,
    ACESFilmic,
    AGXX,
}

impl ToneMappingMode {
    pub fn apply_to_defines(&self, defines: &mut ShaderDefines) {
        match self {
            Self::Linear => defines.set("TONE_MAPPING_MODE", "LINEAR"),
            Self::Reinhard => defines.set("TONE_MAPPING_MODE", "REINHARD"),
            Self::Cineon => defines.set("TONE_MAPPING_MODE", "CINEON"),
            Self::ACESFilmic => defines.set("TONE_MAPPING_MODE", "ACES_FILMIC"),
            Self::NEUTRAL => defines.set("TONE_MAPPING_MODE", "NEUTRAL"),
            Self::AGXX => defines.set("TONE_MAPPING_MODE", "AGXX"),
        }
        
    }
}


// 定义 Uniform 数据
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ToneMapUniforms {
    pub exposure: f32,
    pub _pad: [u32; 3],
}


impl Default for ToneMapUniforms {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            _pad: [0; 3],
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
    sampler: Tracked<wgpu::Sampler>,
    
    pub uniforms: CpuBuffer<ToneMapUniforms>,

    current_mode: ToneMappingMode,
    
    pipeline_cache: FxHashMap<ToneMappingMode, wgpu::RenderPipeline>,

    // 运行时状态 (Prepare 阶段生成，Run 阶段使用)
    current_bind_group: Option<wgpu::BindGroup>,
    current_pipeline: Option<wgpu::RenderPipeline>,

    // target_view: Option<wgpu::TextureView>,

}

impl ToneMapPass {
    pub fn new(device: &wgpu::Device) -> Self {
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

        // 创建 Sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ToneMap Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniforms = CpuBuffer::new(ToneMapUniforms::default(), wgpu::BufferUsages::UNIFORM, Some("ToneMap Uniforms"));

        Self {
            layout: tracked_layout,
            sampler: Tracked::new(sampler),
            uniforms,
            current_mode: ToneMappingMode::NEUTRAL,
            pipeline_cache: FxHashMap::default(),
            current_pipeline: None,
            current_bind_group: None,
            // target_view: None,
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

    pub fn set_mode(&mut self, mode: ToneMappingMode) {
        if self.current_mode != mode {
            self.current_mode = mode;
            self.current_pipeline = None; // 标记需要更新 Pipeline
        }
    }


    fn get_or_create_pipeline(&mut self, device: &wgpu::Device, view_format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
        if let Some(pipeline) = self.pipeline_cache.get(&self.current_mode) {
            return pipeline.clone();
        }

        // 缓存未命中，开始编译
        
        // 1. 准备宏定义
        let mut defines = ShaderDefines::new();
        self.current_mode.apply_to_defines(&mut defines);

        // 2. 生成 Shader 代码
        let options = ShaderCompilationOptions {
             defines
        };

        let shader_code = ShaderGenerator::generate_shader(
            "", // vertex code snippet (empty implies full file or default)
            "", // binding code (empty)
            "passes/tone_mapping", // template name
            &options,
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("ToneMap Shader {:?}", self.current_mode)),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_code)),
        });

        // 3.  创建 Pipeline Layout (复用 self.layout)
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tone Map Pipeline Layout"),
            bind_group_layouts: &[&self.layout],
            immediate_size: 0,
        });

        // 4. 创建 Pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("ToneMap Pipeline {:?}", self.current_mode)),
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
                    format: view_format,
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

        // 存入缓存
        self.pipeline_cache.insert(self.current_mode, pipeline.clone());
        pipeline
    }


}

impl RenderNode for ToneMapPass {

    // #[inline]
    // fn output_to_screen(&self) -> bool {
    //     true
    // }

    fn prepare(&mut self, ctx: &mut RenderContext) {

        let (input_view, _out_view) = ctx.acquire_pass_io();

        let gpu_buffer_id = ctx.resource_manager.ensure_buffer_id(&self.uniforms);

        let cpu_buffer_id = self.uniforms.id();

        let key = BindGroupKey::new(self.layout.id())
            .with_resource(input_view.id())
            .with_resource(self.sampler.id())
            .with_resource(gpu_buffer_id);

        let bind_group = ctx.global_bind_group_cache.get_or_create(key, || {

            let gpu_buffer = ctx.resource_manager.gpu_buffers.get(&cpu_buffer_id)
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

        if self.current_pipeline.is_none() {
            self.current_pipeline = Some(self.get_or_create_pipeline(&ctx.wgpu_ctx.device, ctx.wgpu_ctx.surface_view_format));
        }

    }


    fn run(&self, ctx: &mut RenderContext, encoder: &mut wgpu::CommandEncoder) {

        // 输出到 Surface (屏幕)
        let pass_desc = wgpu::RenderPassDescriptor {
            label: Some("Final ToneMap Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.surface_view,    // We assume tone mapping always outputs to screen
                resolve_target: None,   
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::DontCare(wgpu::LoadOpDontCare::default()),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            ..Default::default()
        };

        let mut pass = encoder.begin_render_pass(&pass_desc);

        if let Some(pipeline) = &self.current_pipeline {
            pass.set_pipeline(pipeline);
            if let Some(bg) = &self.current_bind_group {
                pass.set_bind_group(0, bg, &[]);
            }
        } 
        pass.draw(0..3, 0..1); // 全屏三角形

    }
    
    fn name(&self) -> &str {
        "Tone Mapping Pass"
    }
}