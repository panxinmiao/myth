//! SSAO Feature & Transient Pass Node
//!
//! Production-grade Screen Space Ambient Occlusion split into the modern
//! **Feature + transient PassNode** architecture:
//!
//! - [`SsaoFeature`] (persistent): owns layouts, noise texture, pipelines.
//!   Stored in `RendererState` and survives across frames.
//! - [`RdgSsaoPassNode`] (transient): lightweight per-frame data carrier
//!   holding texture node IDs and cached bind-group keys. Created by
//!   [`SsaoFeature::add_to_graph`] and destroyed after execution.
//!
//! # Internal Sub-Passes
//!
//! 1. **Raw SSAO**: Hemisphere sampling with kernel → noisy R8Unorm
//! 2. **Cross-Bilateral Blur**: Depth/normal-aware spatial filter → clean AO

use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::resources::{CommonSampler, Tracked};
use crate::renderer::graph::rdg::allocator::SubViewKey;
use crate::renderer::graph::rdg::builder::PassBuilder;
use crate::renderer::graph::rdg::context::{RdgExecuteContext, RdgPrepareContext};
use crate::renderer::graph::rdg::feature::{ExtractContext, PrepassOutput};
use crate::renderer::graph::rdg::graph::RenderGraph;
use crate::renderer::graph::rdg::node::PassNode;
use crate::renderer::graph::rdg::types::{RdgTextureDesc, TextureNodeId};
use crate::renderer::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions,
};
use crate::resources::ssao::{SsaoUniforms, generate_ssao_noise};
use crate::resources::uniforms::WgslStruct;

const SSAO_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

// =============================================================================
// Persistent Feature
// =============================================================================

/// Persistent SSAO Feature — owns GPU resources that survive across frames.
///
/// Stored in `RendererState`. The Composer calls [`extract_and_prepare`] before
/// graph assembly, then [`add_to_graph`] to inject a transient pass node.
pub struct SsaoFeature {
    raw_pipeline: Option<RenderPipelineId>,
    blur_pipeline: Option<RenderPipelineId>,
    raw_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    raw_uniforms_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    blur_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    noise_texture_view: Option<Tracked<wgpu::TextureView>>,
}

impl SsaoFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            raw_pipeline: None,
            blur_pipeline: None,
            raw_layout: None,
            raw_uniforms_layout: None,
            blur_layout: None,
            noise_texture_view: None,
        }
    }

    /// Prepare persistent GPU resources before graph assembly.
    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        _uniforms_cpu_id: u64,
        global_state_key: (u32, u32),
    ) {
        self.ensure_layouts(ctx.device);
        self.ensure_noise_texture(ctx.device, ctx.queue);
        self.ensure_pipelines(ctx, global_state_key);
    }

    /// Build and inject a transient SSAO pass node into the render graph.
    ///
    /// Returns the `TextureNodeId` of the blurred AO output (half-res R8Unorm).
    pub fn add_to_graph(
        &self,
        rdg: &mut RenderGraph,
        prepass_out: &PrepassOutput,
        uniforms_cpu_id: u64,
    ) -> TextureNodeId {
        let normal = match prepass_out.normal {
            Some(n) => n,
            None => {
                log::warn!("SSAO requires Normal Prepass! Bypassing SSAO.");
                return prepass_out.depth;
            }
        };

        let config = rdg.frame_config();
        let half_desc = RdgTextureDesc::new_2d(
            config.width / 2,
            config.height / 2,
            SSAO_TEXTURE_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let output_tex = rdg.register_resource("SSAO_Output", half_desc.clone(), false);
        let internal_raw_tex = rdg.register_resource("SSAO_Raw_Internal", half_desc, false);

        let node = Box::new(RdgSsaoPassNode {
            depth_tex: prepass_out.depth,
            normal_tex: normal,
            output_tex,
            internal_raw_tex,
            uniforms_cpu_id,
            raw_pipeline: self.raw_pipeline.unwrap(),
            blur_pipeline: self.blur_pipeline.unwrap(),
            raw_layout: self.raw_layout.clone().unwrap(),
            raw_uniforms_layout: self.raw_uniforms_layout.clone().unwrap(),
            blur_layout: self.blur_layout.clone().unwrap(),
            noise_texture_view: self.noise_texture_view.clone().unwrap(),
            raw_bind_group_key: None,
            raw_uniforms_bind_group_key: None,
            blur_bind_group_key: None,
        });
        rdg.add_pass_owned(node);
        output_tex
    }

    // =========================================================================
    // Lazy Initialization
    // =========================================================================

    fn ensure_layouts(&mut self, device: &wgpu::Device) {
        if self.raw_layout.is_some() {
            return;
        }

        let raw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RDG SSAO Raw Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let raw_uniforms_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RDG SSAO Uniforms Layout"),
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

        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RDG SSAO Blur Layout"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        self.raw_layout = Some(Tracked::new(raw_layout));
        self.raw_uniforms_layout = Some(Tracked::new(raw_uniforms_layout));
        self.blur_layout = Some(Tracked::new(blur_layout));
    }

    fn ensure_noise_texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.noise_texture_view.is_some() {
            return;
        }

        let noise_data = generate_ssao_noise();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RDG SSAO Noise 4x4"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let flat: Vec<u8> = noise_data.iter().flat_map(|p| p.iter().copied()).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &flat,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * 4),
                rows_per_image: Some(4),
            },
            wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.noise_texture_view = Some(Tracked::new(view));
    }

    fn ensure_pipelines(&mut self, ctx: &mut ExtractContext, global_state_key: (u32, u32)) {
        if self.raw_pipeline.is_some() {
            return;
        }

        let device = ctx.device;
        let raw_layout = self.raw_layout.as_ref().unwrap();
        let uniforms_layout = self.raw_uniforms_layout.as_ref().unwrap();
        let blur_layout = self.blur_layout.as_ref().unwrap();

        let gpu_world = ctx
            .resource_manager
            .get_global_state(global_state_key.0, global_state_key.1)
            .expect("RDG SSAO: GpuGlobalState must exist");

        let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
            format: SSAO_TEXTURE_FORMAT,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        });

        // ─── Raw SSAO Pipeline ─────────────────────────────────────
        {
            let mut options = ShaderCompilationOptions::default();
            options.add_define(
                "struct_definitions",
                SsaoUniforms::wgsl_struct_def("SsaoUniforms").as_str(),
            );

            let (module, hash) = ctx.shader_manager.get_or_compile_template(
                device,
                "passes/ssao_raw",
                &options,
                "",
                &gpu_world.binding_wgsl,
            );

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RDG SSAO Raw Pipeline Layout"),
                bind_group_layouts: &[&gpu_world.layout, raw_layout, uniforms_layout],
                immediate_size: 0,
            });

            let key = FullscreenPipelineKey::fullscreen(
                hash,
                smallvec::smallvec![color_target.clone()],
                None,
            );

            self.raw_pipeline = Some(ctx.pipeline_cache.get_or_create_fullscreen(
                device,
                module,
                &pipeline_layout,
                &key,
                "RDG SSAO Raw Pipeline",
            ));
        }

        // ─── Blur Pipeline ─────────────────────────────────────────
        {
            let (module, hash) = ctx.shader_manager.get_or_compile_template(
                device,
                "passes/ssao_blur",
                &ShaderCompilationOptions::default(),
                "",
                "",
            );

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RDG SSAO Blur Pipeline Layout"),
                bind_group_layouts: &[blur_layout],
                immediate_size: 0,
            });

            let key =
                FullscreenPipelineKey::fullscreen(hash, smallvec::smallvec![color_target], None);

            self.blur_pipeline = Some(ctx.pipeline_cache.get_or_create_fullscreen(
                device,
                module,
                &pipeline_layout,
                &key,
                "RDG SSAO Blur Pipeline",
            ));
        }
    }
}

// =============================================================================
// Transient Pass Node
// =============================================================================

/// Per-frame SSAO pass node — created and discarded every frame.
struct RdgSsaoPassNode {
    depth_tex: TextureNodeId,
    normal_tex: TextureNodeId,
    output_tex: TextureNodeId,
    internal_raw_tex: TextureNodeId,

    uniforms_cpu_id: u64,
    raw_pipeline: RenderPipelineId,
    blur_pipeline: RenderPipelineId,
    raw_layout: Tracked<wgpu::BindGroupLayout>,
    raw_uniforms_layout: Tracked<wgpu::BindGroupLayout>,
    blur_layout: Tracked<wgpu::BindGroupLayout>,
    noise_texture_view: Tracked<wgpu::TextureView>,

    raw_bind_group_key: Option<BindGroupKey>,
    raw_uniforms_bind_group_key: Option<BindGroupKey>,
    blur_bind_group_key: Option<BindGroupKey>,
}

impl RdgSsaoPassNode {
    fn build_bind_groups(&mut self, ctx: &mut RdgPrepareContext) {
        let device = ctx.device;

        let depth_key = SubViewKey {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        };
        ctx.views
            .get_or_create_sub_view(self.depth_tex, depth_key.clone());
        let depth_only_view = ctx
            .views
            .get_sub_view(self.depth_tex, &depth_key)
            .expect("RDG SSAO: depth-only view must exist");

        let normal_view = ctx.views.get_texture_view(self.normal_tex);
        let noise_view = &self.noise_texture_view;

        let linear_sampler = ctx.sampler_registry.get_common(CommonSampler::LinearClamp);
        let noise_sampler = ctx
            .sampler_registry
            .get_common(CommonSampler::NearestRepeat);
        let point_sampler = ctx.sampler_registry.get_common(CommonSampler::NearestClamp);

        // Raw SSAO BindGroup (Group 1)
        {
            let key = BindGroupKey::new(self.raw_layout.id())
                .with_resource(depth_only_view.id())
                .with_resource(normal_view.id())
                .with_resource(noise_view.id())
                .with_resource(linear_sampler.id())
                .with_resource(noise_sampler.id())
                .with_resource(point_sampler.id());

            if ctx.global_bind_group_cache.get(&key).is_none() {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RDG SSAO Raw BG"),
                    layout: &self.raw_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(depth_only_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(normal_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(noise_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(linear_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(noise_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::Sampler(point_sampler),
                        },
                    ],
                });
                ctx.global_bind_group_cache.insert(key.clone(), bg);
            }
            self.raw_bind_group_key = Some(key);
        }

        // Uniforms BindGroup (Group 2)
        {
            let gpu_buffer = ctx
                .resource_manager
                .gpu_buffers
                .get(&self.uniforms_cpu_id)
                .expect("RDG SSAO: uniforms GPU buffer must exist");

            let key =
                BindGroupKey::new(self.raw_uniforms_layout.id()).with_resource(gpu_buffer.id);

            if ctx.global_bind_group_cache.get(&key).is_none() {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RDG SSAO Uniforms BG"),
                    layout: &self.raw_uniforms_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: gpu_buffer.buffer.as_entire_binding(),
                    }],
                });
                ctx.global_bind_group_cache.insert(key.clone(), bg);
            }
            self.raw_uniforms_bind_group_key = Some(key);
        }

        // Blur BindGroup (Group 0)
        {
            let raw_view = ctx.views.get_texture_view(self.internal_raw_tex);

            let key = BindGroupKey::new(self.blur_layout.id())
                .with_resource(raw_view.id())
                .with_resource(depth_only_view.id())
                .with_resource(normal_view.id())
                .with_resource(linear_sampler.id())
                .with_resource(point_sampler.id());

            if ctx.global_bind_group_cache.get(&key).is_none() {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RDG SSAO Blur BG"),
                    layout: &self.blur_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(raw_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(depth_only_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(normal_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(linear_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(point_sampler),
                        },
                    ],
                });
                ctx.global_bind_group_cache.insert(key.clone(), bg);
            }
            self.blur_bind_group_key = Some(key);
        }
    }
}

impl PassNode for RdgSsaoPassNode {
    fn name(&self) -> &'static str {
        "RDG_SSAO_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        builder.read_texture(self.depth_tex);
        builder.read_texture(self.normal_tex);
        builder.write_texture(self.output_tex);
        builder.write_texture(self.internal_raw_tex);
    }

    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {
        self.build_bind_groups(ctx);
    }

    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let Some(raw_bg_key) = &self.raw_bind_group_key else {
            return;
        };
        let Some(uniforms_bg_key) = &self.raw_uniforms_bind_group_key else {
            return;
        };
        let Some(blur_bg_key) = &self.blur_bind_group_key else {
            return;
        };
        let Some(global_bg) = ctx.global_bind_group else {
            log::warn!("RDG SSAO: global_bind_group missing, skipping");
            return;
        };

        let raw_bg = ctx
            .global_bind_group_cache
            .get(raw_bg_key)
            .expect("SSAO raw BG should exist");
        let uniforms_bg = ctx
            .global_bind_group_cache
            .get(uniforms_bg_key)
            .expect("SSAO uniforms BG should exist");
        let blur_bg = ctx
            .global_bind_group_cache
            .get(blur_bg_key)
            .expect("SSAO blur BG should exist");

        let raw_pipeline = ctx.pipeline_cache.get_render_pipeline(self.raw_pipeline);
        let blur_pipeline = ctx.pipeline_cache.get_render_pipeline(self.blur_pipeline);
        let raw_view = ctx.get_texture_view(self.internal_raw_tex);

        // Sub-Pass 1: Raw SSAO
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("RDG SSAO Raw Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: raw_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::DontCare(wgpu::LoadOpDontCare::default()),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_pipeline(raw_pipeline);
            pass.set_bind_group(0, global_bg, &[]);
            pass.set_bind_group(1, raw_bg, &[]);
            pass.set_bind_group(2, uniforms_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Sub-Pass 2: Cross-Bilateral Blur
        {
            let output_view = ctx.get_texture_view(self.output_tex);

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("RDG SSAO Blur Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::DontCare(wgpu::LoadOpDontCare::default()),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_pipeline(blur_pipeline);
            pass.set_bind_group(0, blur_bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
