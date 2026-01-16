use rustc_hash::FxHashMap;
use std::borrow::Cow;

// Shader 不需要变动，因为我们将每一层都视为单独的 2D 纹理处理
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
    pipelines: FxHashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
}

impl MipmapGenerator {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mipmap Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(BLIT_WGSL)),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mipmap Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        // 关键点：这里强制声明 Shader 接受的是 D2，
                        // 即使原图是 Cube/Array，我们也会以 View 的形式作为 D2 传入
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

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Mipmap Sampler"),
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
            pipelines: FxHashMap::default(),
        }
    }

    fn get_pipeline(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
        // Pipeline 创建逻辑不变
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

        // 1. 获取数组层数 (普通2D=1, Cube=6, Array=N)
        let layer_count = texture.depth_or_array_layers();

        // 2. 双重循环：遍历所有层 -> 遍历所有 Mip
        for layer in 0..layer_count {
            for i in 0..mip_count - 1 {
                // 创建源视图 (上一级 Mip)
                let src_view = texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("Mipmap Src"),
                    format: None,
                    // 强制指定为 D2，这样 Shader 才能将其作为普通 texture_2d 采样
                    dimension: Some(wgpu::TextureViewDimension::D2), 
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: i,
                    mip_level_count: Some(1),
                    base_array_layer: layer, // 指定当前层
                    array_layer_count: Some(1), // 每次只看 1 层
                    usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                });

                // 创建目标视图 (下一级 Mip)
                let dst_view = texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("Mipmap Dst"),
                    format: None,
                    dimension: Some(wgpu::TextureViewDimension::D2), // 同样强制 D2
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: i + 1,
                    mip_level_count: Some(1),
                    base_array_layer: layer, // 指定当前层
                    array_layer_count: Some(1),
                    usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
                });

                // 创建 BindGroup
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

                // 渲染 Pass
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Mipmap Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &dst_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), // 或者 Load，如果不需要清空背景
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
}