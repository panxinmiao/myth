//! TAA Resolve Feature + Ephemeral PassNode
//!
//! Industrial-grade Temporal Anti-Aliasing implementing:
//! - Velocity Dilation (3×3 closest-depth)
//! - Catmull-Rom 5-Tap history sampling
//! - Reversible Tonemapping (Reinhard)
//! - Variance Clipping in YCoCg space
//! - Depth Rejection (disocclusion detection)
//!
//! # Architecture
//!
//! **Persistent History Buffers** — `TaaFeature` owns independent history
//! colour and depth textures that survive across frames.  After TAA resolve,
//! the clean results are copied into these buffers via `CopyTextureNode`,
//! preventing transparent-pass pollution of the history.
//!
//! # Binding Layout (Group 0)
//!
//! | Binding | Type                      | Content                      |
//! |---------|---------------------------|------------------------------|
//! | 0       | `texture_2d<f32>`         | Current frame colour (HDR)   |
//! | 1       | `texture_2d<f32>`         | History colour (HDR)         |
//! | 2       | `texture_2d<f32>`         | Velocity buffer (Rg16Float)  |
//! | 3       | `texture_2d<f32>`         | Current scene depth          |
//! | 4       | `texture_2d<f32>`         | History depth                |
//! | 5       | `sampler`                 | Linear clamp sampler         |
//! | 6       | `sampler`                 | Nearest clamp sampler        |
//! | 7       | `uniform`                 | TaaParams                    |

use crate::HDR_TEXTURE_FORMAT;
use crate::core::binding::BindGroupKey;
use crate::core::gpu::{CommonSampler, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    ExecuteContext, ExtractContext, PassNode, PrepareContext, RenderTargetOps, TextureDesc,
    TextureNodeId,
};
use crate::graph::passes::utils::CopyTextureNode;
use crate::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions,
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Feature (long-lived, stored in RendererState)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Persistent TAA feature owning history colour/depth buffers, the resolve
/// pipeline, bind-group layout, and a small uniform buffer.
pub struct TaaFeature {
    // ─── History Buffers (persistent, single-copy archiving) ───────
    history_view: Option<Tracked<wgpu::TextureView>>,
    history_depth_view: Option<Tracked<wgpu::TextureView>>,

    /// Cached dimensions for resize detection.
    history_size: (u32, u32),
    /// Depth format matching `Scene_Depth`.
    depth_format: wgpu::TextureFormat,

    // ─── Pipeline ──────────────────────────────────────────────────
    pipeline_id: Option<RenderPipelineId>,
    bind_group_layout: Option<Tracked<wgpu::BindGroupLayout>>,

    // ─── Uniform Buffer ────────────────────────────────────────────
    params_buffer: Option<Tracked<wgpu::Buffer>>,

    last_feedback_weight: f32,
    last_camera_near: f32,
}

impl Default for TaaFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl TaaFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            history_view: None,
            history_depth_view: None,
            history_size: (0, 0),
            depth_format: wgpu::TextureFormat::Depth32Float,
            pipeline_id: None,
            bind_group_layout: None,
            params_buffer: None,
            last_feedback_weight: -1.0, // invalid default to ensure first update
            last_camera_near: -1.0,     // invalid default to ensure first update
        }
    }

    // ─── History Buffer Management ─────────────────────────────────────

    /// Ensures history colour and depth buffers exist and match the given
    /// dimensions.  Must be called before `add_to_graph` each frame.
    pub fn ensure_history_buffers(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth_format: wgpu::TextureFormat,
    ) {
        self.depth_format = depth_format;

        if self.history_size == (width, height)
            && self.history_view.is_some()
            && self.history_depth_view.is_some()
        {
            return;
        }

        // History colour (HDR)
        let color_desc = wgpu::TextureDescriptor {
            label: Some("TAA History Color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };
        let color_tex = device.create_texture(&color_desc);
        self.history_view = Some(Tracked::new(
            color_tex.create_view(&wgpu::TextureViewDescriptor::default()),
        ));

        // History depth (same format as scene depth)
        let depth_desc = wgpu::TextureDescriptor {
            label: Some("TAA History Depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: depth_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };
        let depth_tex = device.create_texture(&depth_desc);
        self.history_depth_view = Some(Tracked::new(
            depth_tex.create_view(&wgpu::TextureViewDescriptor::default()),
        ));

        self.history_size = (width, height);
    }

    // ─── Extract & Prepare (pre-RDG) ───────────────────────────────────

    /// Compile the TAA resolve pipeline (lazy, cached).
    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        feedback_weight: f32,
        size: (u32, u32),
        output_format: wgpu::TextureFormat,
    ) {
        let depth_format = ctx.wgpu_ctx.depth_format;

        // ── 1. Bind group layout (once) ────────────────────────────────
        if self.bind_group_layout.is_none() {
            let layout = ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("TAA BindGroup Layout"),
                    entries: &[
                        // binding 0: current colour
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
                        // binding 1: history colour
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // binding 2: velocity
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // binding 3: current scene depth (unfilterable)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Depth,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // binding 4: history depth (unfilterable)
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Depth,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // binding 5: linear sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        // binding 6: nearest sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                            count: None,
                        },
                        // binding 7: TaaParams uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 7,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(16),
                            },
                            count: None,
                        },
                    ],
                });
            self.bind_group_layout = Some(Tracked::new(layout));
        }

        // ── 2. Uniform buffer (create once, update on change) ──────────
        if self.params_buffer.is_none() {
            let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("TAA Params"),
                size: 16, // vec4 padding
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.params_buffer = Some(Tracked::new(buffer));
        }

        if (self.last_feedback_weight - feedback_weight).abs() > f32::EPSILON
            || (ctx.render_camera.near - self.last_camera_near).abs() > f32::EPSILON
        {
            let data: [f32; 4] = [feedback_weight, ctx.render_camera.near, 0.0, 0.0];
            ctx.queue.write_buffer(
                self.params_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&data),
            );
            self.last_feedback_weight = feedback_weight;
            self.last_camera_near = ctx.render_camera.near;
        }

        self.ensure_history_buffers(ctx.device, size.0, size.1, depth_format);

        // ── 3. Pipeline (compile on format change) ─────────────────────
        if self.pipeline_id.is_none() {
            let options = ShaderCompilationOptions::default();

            let (shader_module, shader_hash) = ctx.shader_manager.get_or_compile_template(
                ctx.device,
                "passes/taa_resolve",
                &options,
                "",
                "",
            );

            let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
                format: output_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            });

            let key = FullscreenPipelineKey::fullscreen(
                shader_hash,
                smallvec::smallvec![color_target],
                None,
            );

            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("TAA Pipeline Layout"),
                        bind_group_layouts: &[self.bind_group_layout.as_ref().unwrap()],
                        immediate_size: 0,
                    });

            let id = ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                shader_module,
                &pipeline_layout,
                &key,
                "TAA Resolve Pipeline",
            );
            self.pipeline_id = Some(id);
        }
    }

    // ─── Graph Integration ─────────────────────────────────────────────

    /// Insert the TAA resolve pass and history-save copy nodes into the RDG.
    ///
    /// Returns the `TextureNodeId` of the resolved HDR colour that
    /// downstream passes (bloom, tone-mapping) should consume.
    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        active_color: TextureNodeId,
        velocity_buffer: TextureNodeId,
        scene_depth: TextureNodeId,
    ) -> TextureNodeId {
        let pipeline_id = self.pipeline_id.expect("TaaFeature not prepared");
        let pipeline = ctx.pipeline_cache.get_render_pipeline(pipeline_id);
        let layout = self.bind_group_layout.as_ref().unwrap();
        let params_buffer = self.params_buffer.as_ref().unwrap();

        let history_view = self
            .history_view
            .as_ref()
            .expect("TAA history colour view not initialized");
        let history_depth_view = self
            .history_depth_view
            .as_ref()
            .expect("TAA history depth view not initialized");

        let depth_desc = TextureDesc::new(
            history_depth_view.texture().width(),
            history_depth_view.texture().height(),
            1,
            1,
            1,
            wgpu::TextureDimension::D2,
            self.depth_format,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );

        let resolved_color: TextureNodeId = ctx.graph.add_pass("TAA_Resolve", |builder| {
            let history_color_desc = TextureDesc::new_2d(
                history_view.texture().width(),
                history_view.texture().height(),
                HDR_TEXTURE_FORMAT,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            );
            builder.read_external_texture(
                "TAA_History_Color_Read",
                history_color_desc,
                history_view,
            );
            builder.read_external_texture("TAA_History_Depth_Read", depth_desc, history_depth_view);

            builder.read_texture(active_color);
            builder.read_texture(velocity_buffer);
            builder.read_texture(scene_depth);

            let color_desc = TextureDesc::new_2d(
                history_view.texture().width(),
                history_view.texture().height(),
                HDR_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            );

            let resolved_color = builder.create_texture("TAA_Resolved", color_desc);

            let node = TaaPassNode {
                current_color: active_color,
                velocity: velocity_buffer,
                scene_depth,
                output: resolved_color,
                history_view,
                history_depth_view,
                pipeline,
                layout,
                params_buffer,
                transient_bg: None,
            };
            (node, resolved_color)
        });

        // Archive resolved colour → persistent history colour buffer.
        ctx.graph.add_pass("TAA_Save_History_Color", |builder| {
            builder.read_texture(resolved_color);
            let history_color_desc = TextureDesc::new_2d(
                history_view.texture().width(),
                history_view.texture().height(),
                HDR_TEXTURE_FORMAT,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            );
            let history_out = builder.write_external_texture(
                "TAA_History_Color_Write",
                history_color_desc,
                history_view,
            );
            (
                CopyTextureNode {
                    src: resolved_color,
                    dst: history_out,
                },
                (),
            )
        });

        // Archive scene depth → persistent history depth buffer.
        ctx.graph.add_pass("TAA_Save_History_Depth", |builder| {
            builder.read_texture(scene_depth);
            let depth_out = builder.write_external_texture(
                "TAA_History_Depth_Write",
                depth_desc,
                history_depth_view,
            );
            (
                CopyTextureNode {
                    src: scene_depth,
                    dst: depth_out,
                },
                (),
            )
        });

        resolved_color
    }

    /// Returns `true` if the TAA history buffers have been allocated.
    #[must_use]
    pub fn has_history(&self) -> bool {
        self.history_view.is_some()
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PassNode (ephemeral, created per frame)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct TaaPassNode<'a> {
    current_color: TextureNodeId,
    velocity: TextureNodeId,
    scene_depth: TextureNodeId,
    output: TextureNodeId,
    history_view: &'a Tracked<wgpu::TextureView>,
    history_depth_view: &'a Tracked<wgpu::TextureView>,
    pipeline: &'a wgpu::RenderPipeline,
    layout: &'a Tracked<wgpu::BindGroupLayout>,
    params_buffer: &'a Tracked<wgpu::Buffer>,
    transient_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for TaaPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let PrepareContext {
            views,
            global_bind_group_cache: cache,
            device,
            sampler_registry,
            ..
        } = ctx;
        let device = *device;

        let current_view = views.get_texture_view(self.current_color);
        let velocity_view = views.get_texture_view(self.velocity);
        let depth_view = views.get_texture_view(self.scene_depth);
        let linear_sampler = sampler_registry.get_common(CommonSampler::LinearClamp);
        let nearest_sampler = sampler_registry.get_common(CommonSampler::NearestClamp);

        let key = BindGroupKey::new(self.layout.id())
            .with_resource(current_view.id())
            .with_resource(self.history_view.id())
            .with_resource(velocity_view.id())
            .with_resource(depth_view.id())
            .with_resource(self.history_depth_view.id())
            // .with_resource(CommonSampler::LinearClamp as u64)
            // .with_resource(CommonSampler::NearestClamp as u64)
            .with_resource(self.params_buffer.id());

        let layout = self.layout;
        let history_view = self.history_view;
        let history_depth_view = self.history_depth_view;
        let params_buf = self.params_buffer;

        let bg = cache.get_or_create_bg(key, || {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("TAA BindGroup"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(current_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(history_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(velocity_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(history_depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(nearest_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            })
        });
        self.transient_bg = Some(bg);
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let bind_group = self.transient_bg.expect("TAA BG not prepared!");

        let rtt = ctx.get_color_attachment(self.output, RenderTargetOps::DontCare, None);

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("TAA Resolve Pass"),
                color_attachments: &[rtt],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(self.pipeline);
            rpass.set_bind_group(0, bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }
    }
}
