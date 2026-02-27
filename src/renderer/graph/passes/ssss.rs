use crate::renderer::HDR_TEXTURE_FORMAT;
use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::resources::Tracked;
use crate::renderer::graph::context::{ExecuteContext, GraphResource, PrepareContext};
use crate::renderer::graph::{RenderNode, TransientTextureDesc};
use crate::resources::screen_space::{STENCIL_FEATURE_SSS, SssProfileData};
use std::mem::size_of;
use wgpu::util::DeviceExt;

pub struct SssssPass {
    enabled: bool,
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group_layout: Option<Tracked<wgpu::BindGroupLayout>>,

    // 全局配置缓冲
    profiles_buffer: Tracked<wgpu::Buffer>,
    last_registry_version: u64,

    // 模糊方向 Uniform (0: 水平, 1: 垂直)
    dir_buffers: [wgpu::Buffer; 2],
    dir_bind_groups: [Option<wgpu::BindGroup>; 2],

    // 动态 BindGroups
    horizontal_bind_group: Option<wgpu::BindGroup>,
    vertical_bind_group: Option<wgpu::BindGroup>,

    output_view: Option<Tracked<wgpu::TextureView>>,
    sampler: Option<Tracked<wgpu::Sampler>>,
}

impl SssssPass {
    pub fn new(device: &wgpu::Device) -> Self {
        // 创建 256 个元素的 Storage Buffer
        let profiles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SSS Profiles Buffer"),
            size: (256 * size_of::<SssProfileData>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 创建模糊方向 Buffer
        let dir_buffers = [
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SSS Horizontal Dir"),
                contents: bytemuck::cast_slice(&[1.0f32, 0.0f32]),
                usage: wgpu::BufferUsages::UNIFORM,
            }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SSS Vertical Dir"),
                contents: bytemuck::cast_slice(&[0.0f32, 1.0f32]),
                usage: wgpu::BufferUsages::UNIFORM,
            }),
        ];

        Self {
            enabled: false,
            pipeline: None,
            bind_group_layout: None,
            profiles_buffer: Tracked::new(profiles_buffer),
            last_registry_version: 0,
            dir_buffers,
            dir_bind_groups: [None, None],
            horizontal_bind_group: None,
            vertical_bind_group: None,
            output_view: None,
            sampler: None,
        }
    }
}

impl RenderNode for SssssPass {
    fn name(&self) -> &'static str {
        "SSSSS Pass"
    }

    fn prepare(&mut self, ctx: &mut PrepareContext) {
        self.enabled = ctx.scene.screen_space.enable_sss;
        if !self.enabled {
            return;
        }

        // 1. 极致 Diff-Sync 同步
        let registry = ctx.assets.sss_registry.read();
        if self.last_registry_version != registry.version {
            ctx.wgpu_ctx.queue.write_buffer(
                &self.profiles_buffer,
                0,
                bytemuck::cast_slice(&registry.buffer_data),
            );
            self.last_registry_version = registry.version;
        }

        // 2. 初始化 Pipeline 与 Layout (延迟初始化)
        if self.pipeline.is_none() {
            let device = &ctx.wgpu_ctx.device;

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("SSSSS Bind Group Layout"),
                    entries: &[
                        // t_color
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
                        // t_normal
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // t_depth
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Depth,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // u_profiles
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // s_sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        // t_feature_id
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Uint,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // t_specular
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                    ],
                });

            let dir_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("SSSSS Dir Bind Group Layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

            self.dir_bind_groups[0] = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SSSSS Horizontal Dir Bind Group"),
                layout: &dir_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.dir_buffers[0].as_entire_binding(),
                }],
            }));

            self.dir_bind_groups[1] = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SSSSS Vertical Dir Bind Group"),
                layout: &dir_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.dir_buffers[1].as_entire_binding(),
                }],
            }));

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SSSSS Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout, &dir_bind_group_layout],
                immediate_size: 0,
            });

            let shader_code =
                crate::renderer::pipeline::shader_gen::ShaderGenerator::generate_shader(
                    "",
                    "",
                    "passes/ssss",
                    &crate::renderer::pipeline::ShaderCompilationOptions::default(),
                );

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("SSSSS Shader"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(shader_code)),
            });

            let stencil_face = wgpu::StencilFaceState {
                compare: wgpu::CompareFunction::Equal,
                fail_op: wgpu::StencilOperation::Keep,
                depth_fail_op: wgpu::StencilOperation::Keep,
                pass_op: wgpu::StencilOperation::Keep,
            };

            let depth_stencil = Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState {
                    front: stencil_face,
                    back: stencil_face,
                    read_mask: STENCIL_FEATURE_SSS,
                    write_mask: 0x00,
                },
                bias: Default::default(),
            });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("SSSSS Pipeline"),
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
                        format: HDR_TEXTURE_FORMAT,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

            self.pipeline = Some(pipeline);
            self.bind_group_layout = Some(Tracked::new(bind_group_layout));
        }

        if self.sampler.is_none() {
            let sampler = ctx
                .wgpu_ctx
                .device
                .create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("SSSSS Sampler"),
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                });
            self.sampler = Some(Tracked::new(sampler));
        }

        // 3. 申请 PingPong 中间渲染目标 (复用 Transient Pool)
        let color_format = HDR_TEXTURE_FORMAT;
        let (w, h) = ctx.wgpu_ctx.size();
        let pingpong_tex = ctx.transient_pool.allocate(
            &ctx.wgpu_ctx.device,
            &TransientTextureDesc {
                width: w,
                height: h,
                format: color_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                mip_level_count: 1,
                label: "Transient SSSS PingPong",
            },
        );

        let scene_color_view = &ctx.frame_resources.scene_color_view[ctx.color_view_flip_flop];
        let scene_normal_view = ctx.transient_pool.get_view(
            ctx.blackboard
                .scene_normal_texture_id
                .expect("SceneNormal must exist for SSSSS"),
        );
        let scene_depth_view = &ctx.frame_resources.depth_only_view;
        let feature_id_view = ctx.transient_pool.get_view(
            ctx.blackboard
                .feature_id_texture_id
                .expect("FeatureId must exist for SSSSS"),
        );

        let pingpong_view = ctx.transient_pool.get_view(pingpong_tex);
        let specular_view = ctx.transient_pool.get_view(
            ctx.blackboard
                .specular_texture_id
                .expect("Specular texture must exist for SSSSS"),
        );

        // 4. 构建 Horizontal 和 Vertical 两次 Draw 的 BindGroup
        let layout = self.bind_group_layout.as_ref().unwrap();
        let sampler = self.sampler.as_ref().unwrap();

        // Horizontal: 从 SceneColor 读取，渲染到 pingpong_tex
        let horizontal_key = BindGroupKey::new(layout.id())
            .with_resource(scene_color_view.id())
            .with_resource(scene_normal_view.id())
            .with_resource(scene_depth_view.id())
            .with_resource(self.profiles_buffer.id())
            .with_resource(sampler.id())
            .with_resource(feature_id_view.id())
            .with_resource(specular_view.id());

        self.horizontal_bind_group = Some(
            ctx.global_bind_group_cache
                .get_or_create(horizontal_key, || {
                    ctx.wgpu_ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("SSSSS Horizontal Bind Group"),
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(scene_color_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(scene_normal_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(scene_depth_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: self.profiles_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::Sampler(sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::TextureView(feature_id_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: wgpu::BindingResource::TextureView(specular_view),
                            },
                        ],
                    })
                })
                .clone(),
        );

        // Vertical: 从 pingpong_tex 读取，渲染到 SceneColor
        let vertical_key = BindGroupKey::new(layout.id())
            .with_resource(pingpong_view.id())
            .with_resource(scene_normal_view.id())
            .with_resource(scene_depth_view.id())
            .with_resource(self.profiles_buffer.id())
            .with_resource(sampler.id())
            .with_resource(feature_id_view.id())
            .with_resource(specular_view.id());

        self.vertical_bind_group = Some(
            ctx.global_bind_group_cache
                .get_or_create(vertical_key, || {
                    ctx.wgpu_ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("SSSSS Vertical Bind Group"),
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(pingpong_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(scene_normal_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(scene_depth_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: self.profiles_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::Sampler(sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::TextureView(feature_id_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: wgpu::BindingResource::TextureView(specular_view),
                            },
                        ],
                    })
                })
                .clone(),
        );

        // Store pingpong_tex in blackboard so we can use it in run
        ctx.blackboard.sssss_pingpong_texture_id = Some(pingpong_tex);

        // 5. 输出绑定到 SceneColorInput (PingPong 作为中间目标，最终结果写回 SceneColorInput)
        self.output_view = Some(
            ctx.get_resource_view(GraphResource::SceneColorInput)
                .clone(),
        );
    }

    fn run(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        if !self.enabled {
            return;
        }

        let depth_stencil_view = ctx.get_resource_view(GraphResource::DepthStencil);
        let scene_color_output_view = self.output_view.as_ref().unwrap();

        let pingpong_tex = ctx.blackboard.sssss_pingpong_texture_id.unwrap();
        let pingpong_view = ctx.transient_pool.get_view(pingpong_tex);

        // 第一次绘制：水平模糊 (Horizontal)
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSSSS Horizontal"),
                // 渲染到 PingPong 纹理
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: pingpong_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                // 必须挂载带 Stencil 的 DepthBuffer！
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_stencil_view,
                    depth_ops: None,
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            rpass.set_pipeline(self.pipeline.as_ref().unwrap());
            rpass.set_stencil_reference(STENCIL_FEATURE_SSS); // 触发硬件剔除！
            rpass.set_bind_group(0, self.horizontal_bind_group.as_ref().unwrap(), &[]);
            rpass.set_bind_group(1, self.dir_bind_groups[0].as_ref().unwrap(), &[]);
            rpass.draw(0..3, 0..1);
        }

        // 第二次绘制：垂直模糊 (Vertical)
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSSSS Vertical"),
                // 渲染回 SceneColorInput
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: scene_color_output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_stencil_view,
                    depth_ops: None,
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            rpass.set_pipeline(self.pipeline.as_ref().unwrap());
            rpass.set_stencil_reference(STENCIL_FEATURE_SSS); // 触发硬件剔除！
            rpass.set_bind_group(0, self.vertical_bind_group.as_ref().unwrap(), &[]);
            rpass.set_bind_group(1, self.dir_bind_groups[1].as_ref().unwrap(), &[]);
            rpass.draw(0..3, 0..1);
        }
    }
}
