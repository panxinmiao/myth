//! TAA Resolve Feature + Ephemeral PassNode
//!
//! Implements Temporal Anti-Aliasing as a full-screen post-process pass within
//! the Declarative Render Graph (RDG).
//!
//! # Architecture
//!
//! **Ping-Pong History Buffers** — `TaaFeature` owns two persistent HDR
//! textures that survive across frames.  Each frame one buffer is the
//! "history read" (previous result) and the other is the "history write"
//! (current output).  After rendering, the roles swap.
//!
//! The write buffer is registered as an **external** RDG resource so that
//! downstream passes (bloom, tone-mapping) can read it through the normal
//! `TextureNodeId` pipeline.  The read buffer is passed directly to the
//! `TaaPassNode` as a `&TextureView` reference — it never enters the RDG
//! resource table.
//!
//! # Binding Layout (Group 0)
//!
//! | Binding | Type              | Content                    |
//! |---------|-------------------|----------------------------|
//! | 0       | texture_2d<f32>   | Current frame colour (HDR) |
//! | 1       | texture_2d<f32>   | History colour (HDR)       |
//! | 2       | texture_2d<f32>   | Velocity buffer (Rg16Float)|
//! | 3       | sampler           | Linear clamp sampler       |
//! | 4       | sampler           | Nearest clamp sampler      |
//! | 5       | uniform           | TaaParams (feedback_weight)|

use crate::renderer::HDR_TEXTURE_FORMAT;
use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::gpu::{CommonSampler, Tracked};
use crate::renderer::graph::composer::GraphBuilderContext;
use crate::renderer::graph::core::{
    ExecuteContext, ExtractContext, PassNode, PrepareContext, RenderTargetOps, TextureDesc,
    TextureNodeId,
};
use crate::renderer::graph::passes::utils::CopyTextureNode;
use crate::renderer::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions,
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Feature (long-lived, stored in RendererState)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Persistent TAA feature owning ping-pong history buffers, the resolve
/// pipeline, bind-group layout, and a small uniform buffer.
pub struct TaaFeature {
    // ─── History Ping-Pong ─────────────────────────────────────────
    // history_textures: Option<[wgpu::Texture; 2]>,
    history_view: Option<Tracked<wgpu::TextureView>>,

    /// Cached dimensions for resize detection.
    history_size: (u32, u32),

    // ─── Pipeline ──────────────────────────────────────────────────
    pipeline_id: Option<RenderPipelineId>,
    bind_group_layout: Option<Tracked<wgpu::BindGroupLayout>>,

    // ─── Uniform Buffer ────────────────────────────────────────────
    params_buffer: Option<Tracked<wgpu::Buffer>>,
    last_feedback_weight: f32,
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
            history_size: (0, 0),
            pipeline_id: None,
            bind_group_layout: None,
            params_buffer: None,
            last_feedback_weight: -1.0, // force first upload
        }
    }

    // ─── History Buffer Management ─────────────────────────────────────

    /// Ensures history buffers exist and match the given dimensions.
    /// Must be called before `add_to_graph` each frame.
    pub fn ensure_history_buffers(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.history_size == (width, height) && self.history_view.is_some() {
            return;
        }

        let desc = wgpu::TextureDescriptor {
            label: Some("TAA History"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };

        let tex = device.create_texture(&desc);
        let view = Tracked::new(tex.create_view(&wgpu::TextureViewDescriptor::default()));

        self.history_view = Some(view);
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
                        // binding 3: linear sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        // binding 4: nearest sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                            count: None,
                        },
                        // binding 5: TaaParams uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
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

        if (self.last_feedback_weight - feedback_weight).abs() > f32::EPSILON {
            let data: [f32; 4] = [feedback_weight, 0.0, 0.0, 0.0];
            ctx.queue.write_buffer(
                self.params_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&data),
            );
            self.last_feedback_weight = feedback_weight;
        }

        self.ensure_history_buffers(ctx.device, size.0, size.1);

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

    /// Insert the TAA resolve pass into the RDG.
    ///
    /// Returns the `TextureNodeId` of the resolved HDR colour that
    /// downstream passes (bloom, tone-mapping) should consume instead
    /// of the raw opaque output.
    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        active_color: TextureNodeId,
        velocity_buffer: TextureNodeId,
    ) -> TextureNodeId {
        let pipeline_id = self.pipeline_id.expect("TaaFeature not prepared");
        let pipeline = ctx.pipeline_cache.get_render_pipeline(pipeline_id);
        let layout = self.bind_group_layout.as_ref().unwrap();
        let params_buffer = self.params_buffer.as_ref().unwrap();

        let history_view = self
            .history_view
            .as_ref()
            .expect("TAA history view not initialized");

        // Register the write-side history buffer as an external RDG resource
        // so the graph compiler handles barriers and downstream reads.
        // let fc = ctx.frame_config;
        let desc = TextureDesc::new_2d(
            history_view.texture().width(),
            history_view.texture().height(),
            HDR_TEXTURE_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );

        ctx.with_group("TAA_System", |ctx| {
            let resolved_color = ctx.graph.add_pass("TAA_Resolve", |builder| {
                builder.read_external_texture("TAA_History_Read", desc, history_view);

                builder.read_texture(active_color);
                builder.read_texture(velocity_buffer);

                let resolved_color = builder.create_texture("TAA_Resolved", desc);

                let node = TaaPassNode {
                    current_color: active_color,
                    velocity: velocity_buffer,
                    output: resolved_color,
                    history_view,
                    pipeline,
                    layout,
                    params_buffer,
                    transient_bg: None,
                };
                (node, resolved_color)
            });

            // data diversion
            ctx.graph.add_pass("TAA_Save_History", |builder| {
                builder.read_texture(resolved_color);
                let history_out =
                    builder.write_external_texture("TAA_History_Write", desc, history_view);

                (
                    CopyTextureNode {
                        src: resolved_color,
                        dst: history_out,
                    },
                    (),
                )
            });

            resolved_color
        })
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
    output: TextureNodeId,
    /// Direct reference to the history read view (not in the RDG pool).
    history_view: &'a Tracked<wgpu::TextureView>,
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
        let linear_sampler = sampler_registry.get_common(CommonSampler::LinearClamp);
        let nearest_sampler = sampler_registry.get_common(CommonSampler::NearestClamp);

        let key = BindGroupKey::new(self.layout.id())
            .with_resource(current_view.id())
            .with_resource(self.history_view.id())
            .with_resource(velocity_view.id())
            .with_resource(linear_sampler.id())
            .with_resource(nearest_sampler.id())
            .with_resource(self.params_buffer.id());

        let layout = self.layout;
        let history_view = self.history_view;
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
                        resource: wgpu::BindingResource::Sampler(linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(nearest_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
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

        // let view = ctx.get_texture_view(self.output);

        // encoder.copy_texture_to_texture(
        //     wgpu::TexelCopyTextureInfo {
        //         texture: view.texture(),
        //         mip_level: 0,
        //         origin: wgpu::Origin3d::ZERO,
        //         aspect: wgpu::TextureAspect::All,
        //     },
        //     wgpu::TexelCopyTextureInfo {
        //         texture: self.history_view.texture(),
        //         mip_level: 0,
        //         origin: wgpu::Origin3d::ZERO,
        //         aspect: wgpu::TextureAspect::All,
        //     },
        //     wgpu::Extent3d {
        //         width: view.texture().width(),
        //         height: view.texture().height(),
        //         depth_or_array_layers: 1,
        //     },
        // );
    }
}
