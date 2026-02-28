//! Screen-Space Sub-Surface Scattering (SSSSS) Post-Processing Pass
//!
//! Implements a separable Gaussian blur for SSS materials identified via the
//! **Thin G-Buffer** Normal.a channel.  Two sequential render passes are
//! executed (H then V) to reconstruct a 2-D scatter.
//!
//! # Data Flow
//!
//! ```text
//!  DepthNormalPrepass          SssssPass
//!       │              ┌────────────────────────────┐
//! SceneColor ─────────►│  H Sub-Pass: Horizontal blur │──► "ssss_ping" (transient)
//! SceneNormal ────┬───►│                              │         │
//! SceneDepth  ────┤    │  V Sub-Pass: Vertical blur   │◄────────┘
//!                 └───►│   (overwrites SceneColor)    │──► SceneColor (in-place)
//!                      └────────────────────────────┘
//! ```
//!
//! # Integration
//!
//! Must come **after** `TransparentPass` and **before** `BloomPass` in the
//! `HighFidelity` render path.  Zero cost when disabled.
//!
//! # GPU Resources
//!
//! - **Profiles StorageBuffer**: 256 × `ScreenSpaceMaterialData` (12 KB).
//!   Uploaded from `ExtractedScene.current_screen_space_profiles`.
//! - **H/V Uniforms UniformBuffer**: 2 × `SsssUniforms` (64 bytes total).
//! - **Ping TransientTexture**: `Rgba16Float`, same size as scene colour.

use crate::renderer::HDR_TEXTURE_FORMAT;
use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::resources::Tracked;
use crate::renderer::graph::context::{ExecuteContext, GraphResource, PrepareContext};
use crate::renderer::graph::{RenderNode, TransientTextureDesc};
use crate::renderer::pipeline::{
    ColorTargetKey, DepthStencilKey, FullscreenPipelineKey, RenderPipelineId,
    ShaderCompilationOptions,
};
use crate::resources::screen_space::{STENCIL_FEATURE_SSS, SssProfileData};
use std::mem::size_of;

pub struct SssssPass {
    enabled: bool,
    horizontal_pipeline: Option<RenderPipelineId>,
    vertical_pipeline: Option<RenderPipelineId>,
    bind_group_layout: Option<Tracked<wgpu::BindGroupLayout>>,

    // Global configuration buffer
    profiles_buffer: Tracked<wgpu::Buffer>,
    last_registry_version: u64,

    // Dynamic BindGroups
    horizontal_bind_group: Option<wgpu::BindGroup>,
    vertical_bind_group: Option<wgpu::BindGroup>,

    output_view: Option<Tracked<wgpu::TextureView>>,
    sampler: Option<Tracked<wgpu::Sampler>>,
}

impl SssssPass {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        // Create a Storage Buffer for 256 elements
        let profiles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SSS Profiles Buffer"),
            size: (256 * size_of::<SssProfileData>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            enabled: false,
            horizontal_pipeline: None,
            vertical_pipeline: None,
            bind_group_layout: None,
            profiles_buffer: Tracked::new(profiles_buffer),
            last_registry_version: 0,
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

        // 1. Minimal diff-sync synchronization
        let registry = ctx.assets.sss_registry.read();
        if self.last_registry_version != registry.version {
            ctx.wgpu_ctx.queue.write_buffer(
                &self.profiles_buffer,
                0,
                bytemuck::cast_slice(&registry.buffer_data),
            );
            self.last_registry_version = registry.version;
        }

        // 2. Initialize Pipeline and Layout (lazy initialization)
        if self.horizontal_pipeline.is_none() {
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

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SSSSS Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });

            // ── Shader compilation via ShaderManager ───────────────────
            let mut shader_defines = ShaderCompilationOptions::default();

            let (hor_shader, hor_hash) = ctx.shader_manager.get_or_compile_template(
                device,
                "passes/ssss",
                &shader_defines,
                "",
                "",
            );

            // ── Build the shared key fragments ─────────────────────────
            let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
                format: HDR_TEXTURE_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            });

            let stencil_face = wgpu::StencilFaceState {
                compare: wgpu::CompareFunction::Equal,
                fail_op: wgpu::StencilOperation::Keep,
                depth_fail_op: wgpu::StencilOperation::Keep,
                pass_op: wgpu::StencilOperation::Keep,
            };

            let depth_stencil = DepthStencilKey::from(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState {
                    front: stencil_face,
                    back: stencil_face,
                    read_mask: STENCIL_FEATURE_SSS,
                    write_mask: 0x00,
                },
                bias: wgpu::DepthBiasState::default(),
            });

            // ── Horizontal pipeline via PipelineCache ──────────────────
            let hor_key = FullscreenPipelineKey::fullscreen(
                hor_hash,
                smallvec::smallvec![color_target.clone()],
                Some(depth_stencil),
            );

            self.horizontal_pipeline = Some(ctx.pipeline_cache.get_or_create_fullscreen(
                device,
                hor_shader,
                &pipeline_layout,
                &hor_key,
                "SSSSS Horizontal Pipeline",
            ));

            // ── Vertical pipeline via PipelineCache ────────────────────
            shader_defines.add_define("SSSSS_VERTICAL_PASS", "1");

            let (vert_shader, vert_hash) = ctx.shader_manager.get_or_compile_template(
                device,
                "passes/ssss",
                &shader_defines,
                "",
                "",
            );

            let vert_key = FullscreenPipelineKey::fullscreen(
                vert_hash,
                smallvec::smallvec![color_target],
                Some(depth_stencil),
            );

            self.vertical_pipeline = Some(ctx.pipeline_cache.get_or_create_fullscreen(
                device,
                vert_shader,
                &pipeline_layout,
                &vert_key,
                "SSSSS Vertical Pipeline",
            ));

            self.bind_group_layout = Some(Tracked::new(bind_group_layout));
            // self.pipeline_layout = Some(pipeline_layout);
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

        // 3. Allocate PingPong intermediate render target (reusing Transient Pool)
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

        // 4. Build BindGroups for Horizontal and Vertical draw passes
        let layout = self.bind_group_layout.as_ref().unwrap();
        let sampler = self.sampler.as_ref().unwrap();

        // Horizontal: read from SceneColor, render to pingpong_tex
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
                    ctx.wgpu_ctx
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
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

        // Vertical: read from pingpong_tex, render to SceneColor
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
                    ctx.wgpu_ctx
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
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

        // 5. Bind output to SceneColorInput (PingPong as intermediate target, final result written back to SceneColorInput)
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

        // Resolve pipeline handles to actual GPU pipeline references (O(1))
        let hor_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.horizontal_pipeline.unwrap());
        let vert_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.vertical_pipeline.unwrap());

        // First draw: Horizontal blur
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSSSS Horizontal"),
                // Render to PingPong texture
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: pingpong_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                // Must attach a DepthBuffer with Stencil!
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_stencil_view,
                    depth_ops: None,
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            rpass.set_pipeline(hor_pipeline);
            rpass.set_stencil_reference(STENCIL_FEATURE_SSS); // Trigger hardware stencil culling!
            rpass.set_bind_group(0, self.horizontal_bind_group.as_ref().unwrap(), &[]);
            rpass.draw(0..3, 0..1);
        }

        // Second draw: Vertical blur
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSSSS Vertical"),
                // Render back to SceneColorInput
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

            rpass.set_pipeline(vert_pipeline);
            rpass.set_stencil_reference(STENCIL_FEATURE_SSS); // Trigger hardware stencil culling!
            rpass.set_bind_group(0, self.vertical_bind_group.as_ref().unwrap(), &[]);
            rpass.draw(0..3, 0..1);
        }
    }
}
