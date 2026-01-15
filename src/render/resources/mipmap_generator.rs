use std::collections::HashMap;
use std::borrow::Cow;


const BLIT: &str = r#"
// 全屏三角形 Vertex Shader
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
    // 将 Clip Space (-1, 1) 映射到 UV (0, 1)
    // 注意 WGPU/Vulkan 的 UV 原点在左上角，Y轴向下
    output.uv = pos[vertexIndex] * 0.5 + 0.5;
    output.uv.y = 1.0 - output.uv.y; 
    return output;
}

// Fragment Shader
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
    // 缓存针对不同 TextureFormat 的 Pipeline
    pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
}

impl MipmapGenerator {
    pub fn new(device: &wgpu::Device) -> Self {
        // 1. 创建通用 Shader (Blit)
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mipmap Generator Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(BLIT)),
        });

        // 2. 创建 BindGroupLayout (Slot 0: Texture, Slot 1: Sampler)
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mipmap Generator Layout"),
            entries: &[
                // Texture Input
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
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // 3. 创建通用 Sampler (Linear Filter)
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Mipmap Generator Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        Self {
            layout,
            sampler,
            shader,
            pipelines: HashMap::new(),
        }
    }

    /// 获取或创建针对特定格式的 Pipeline
    fn get_pipeline(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
        self.pipelines.entry(format).or_insert_with(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("Mipmap Gen Pipeline {:?}", format)),
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&self.layout],
                    immediate_size: 0u32,
                })),
                vertex: wgpu::VertexState {
                    module: &self.shader,
                    entry_point: Some("vs_main"), // 假设 shader 中入口叫 vs_main
                    buffers: &[], // 全屏 Quad 不需要 Vertex Buffer
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &self.shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format, // 关键：Pipeline 必须匹配纹理格式
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

    /// 生成 Mipmaps
    /// 注意：需要传入 &device 来创建 View 和 BindGroup
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

        // 逐层生成：Level i -> Level i+1
        for i in 0..mip_count - 1 {
            // Source: Level i
            let src_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Mipmap Src View"),
                format: None,
                dimension: None,
                aspect: wgpu::TextureAspect::All,
                base_mip_level: i,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: None,
                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            });

            // Destination: Level i+1
            let dst_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Mipmap Dst View"),
                format: None,
                dimension: None,
                aspect: wgpu::TextureAspect::All,
                base_mip_level: i + 1,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: None,
                usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
            });

            // 创建临时 BindGroup
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Mipmap BindGroup"),
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

            // 渲染 Pass
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Mipmap Gen Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), // 其实无所谓，因为是全覆盖
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(&pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1); // 绘制 3 个顶点 (全屏三角形)
        }
    }
}