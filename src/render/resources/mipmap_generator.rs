use rustc_hash::FxHashMap;
use std::borrow::Cow;

const BLIT_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var output : VertexOutput;
    output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
    output.uv = pos[vertexIndex] * 0.5 + 0.5;
    output.uv.y = 1.0 - output.uv.y; // WGPU/Vulkan UV 翻转
    return output;
}

@group(0) @binding(0) var t_diffuse : texture_2d<f32>;
@group(0) @binding(1) var s_diffuse : sampler;

@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.uv);
}
"#;

pub struct MipmapGenerator {
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    shader: wgpu::ShaderModule,
    // [缓存] 针对不同纹理格式的 Pipeline
    pipelines: FxHashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
}

impl MipmapGenerator {
    pub fn new(device: &wgpu::Device) -> Self {
        // 1. 创建 Shader Module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mipmap Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(BLIT_WGSL)),
        });

        // 2. 创建 BindGroupLayout (Slot 0: Texture, Slot 1: Sampler)
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mipmap Layout"),
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

        // 3. 创建通用线性采样器
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Mipmap Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear, // 这里是采样上一级时的过滤
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        Self {
            layout,
            sampler,
            shader,
            pipelines: FxHashMap::default(),
        }
    }

    fn get_pipeline(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
        self.pipelines.entry(format).or_insert_with(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("Mipmap Pipeline {:?}", format)),
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Mipmap Pipeline Layout"),
                    bind_group_layouts: &[&self.layout],
                    immediate_size: 0,
                })),
                vertex: wgpu::VertexState {
                    module: &self.shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &self.shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        }).clone()
    }

    pub fn generate(
        &mut self, 
        device: &wgpu::Device, 
        encoder: &mut wgpu::CommandEncoder, 
        texture: &wgpu::Texture, 
        mip_count: u32
    ) {
        if mip_count < 2 { return; }

        let format = texture.format();
        
        let pipeline = self.get_pipeline(device, format);

        // 2. 逐层生成
        for i in 0..mip_count - 1 {
            let src_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Mipmap Src"),
                format: None,
                dimension: None,
                aspect: wgpu::TextureAspect::All,
                base_mip_level: i,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: None,
                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            });

            let dst_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Mipmap Dst"),
                format: None,
                dimension: None,
                aspect: wgpu::TextureAspect::All,
                base_mip_level: i + 1,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: None,
                usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Mipmap BG"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Mipmap Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                ..Default::default()
            });

            rpass.set_pipeline(&pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }
    }
}