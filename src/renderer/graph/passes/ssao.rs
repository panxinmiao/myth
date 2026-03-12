//! SSAO Feature + Ephemeral PassNodes (Flattened)
//!
//! - **`SsaoFeature`** (long-lived): owns pipelines, bind group layouts,
//!   noise texture.  `extract_and_prepare()` compiles pipelines and uploads
//!   persistent GPU data.
//! - **`SsaoRawNode`** / **`SsaoBlurNode`** (ephemeral per-frame):
//!   two independent RDG passes created by `SsaoFeature::add_to_graph()`.
//!
//! Implements production-grade SSAO within the RDG framework.
//! The output texture is registered by `add_to_graph()` and returned
//! as a [`TextureNodeId`] for explicit downstream wiring.
//!
//! # RDG Slots (explicit wiring)
//!
//! - `depth_tex`: Scene depth buffer (input, from Prepass)
//! - `normal_tex`: Scene normal buffer (input, from Prepass)
//! - `output_tex`: Blurred AO texture (output, half-res R8Unorm)
//!
//! # Internal Sub-Passes
//!
//! 1. **Raw SSAO**: Hemisphere sampling with kernel, produces noisy R8Unorm
//! 2. **Cross-Bilateral Blur**: Depth/normal-aware spatial filter
//!
//! # Push Model
//!
//! All parameters (uniform buffer ID, global state key) are pushed by the
//! Composer via `add_to_graph()`.  The pass never accesses Scene directly.
//! Samplers are obtained from the global [`SamplerRegistry`].

use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::gpu::{CommonSampler, Tracked};
use crate::renderer::graph::core::{
    ExecuteContext, ExtractContext, PassNode, PrepareContext, RenderGraph, RenderTargetOps,
    SubViewKey, TextureDesc, TextureNodeId,
};
use crate::renderer::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions,
};
use crate::resources::buffer::CpuBuffer;
use crate::resources::ssao::{SsaoUniforms, generate_ssao_noise};
use crate::resources::uniforms::WgslStruct;

/// The SSAO output texture format: single-channel unsigned normalized.
const SSAO_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Feature (long-lived, stored in RendererState)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Long-lived SSAO feature — owns persistent GPU resources (pipelines,
/// bind group layouts, noise texture).
///
/// Produces an ephemeral [`SsaoPassNode`] each frame via [`Self::add_to_graph`].
#[derive(Default)]
pub struct SsaoFeature {
    // ─── Pipelines ─────────────────────────────────────────────────
    raw_pipeline: Option<RenderPipelineId>,
    blur_pipeline: Option<RenderPipelineId>,

    // ─── Bind Group Layouts ────────────────────────────────────────
    raw_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    raw_uniforms_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    blur_layout: Option<Tracked<wgpu::BindGroupLayout>>,

    // ─── Persistent Resources ──────────────────────────────────────
    noise_texture_view: Option<Tracked<wgpu::TextureView>>,

    // ─── Pre-Built Static BindGroup (Group 2: uniforms) ────────────
    /// Feature-owned uniform bind group — eliminates GPU buffer leak to PassNode.
    uniforms_static_bg: Option<wgpu::BindGroup>,
    /// Tracked buffer identity for staleness detection.
    last_uniforms_buffer_id: u64,
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

            uniforms_static_bg: None,
            last_uniforms_buffer_id: 0,
        }
    }

    // =========================================================================
    // Lazy Initialization
    // =========================================================================

    fn ensure_layouts(&mut self, device: &wgpu::Device) {
        if self.raw_layout.is_some() {
            return;
        }

        // ─── Raw SSAO Layout (Group 1): depth, normal, noise + samplers ───
        let raw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSAO Raw Layout"),
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

        // ─── Uniforms Layout (Group 2) ─────────────────────────────
        let raw_uniforms_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SSAO Uniforms Layout"),
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

        // ─── Blur Layout (Group 0): raw AO + depth + normal + samplers ────
        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSAO Blur Layout"),
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
            label: Some("SSAO Noise 4x4"),
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

    fn ensure_pipelines(&mut self, ctx: &mut ExtractContext) {
        if self.raw_pipeline.is_some() {
            return;
        }

        let device = ctx.device;
        let raw_layout = self.raw_layout.as_ref().unwrap();
        let uniforms_layout = self.raw_uniforms_layout.as_ref().unwrap();
        let blur_layout = self.blur_layout.as_ref().unwrap();

        let global_state_key = (ctx.render_state.id, ctx.extracted_scene.scene_id);
        let gpu_world = ctx
            .resource_manager
            .get_global_state(global_state_key.0, global_state_key.1)
            .expect("SSAO: GpuGlobalState must exist");

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
                label: Some("SSAO Raw Pipeline Layout"),
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
                "SSAO Raw Pipeline",
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
                label: Some("SSAO Blur Pipeline Layout"),
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
                "SSAO Blur Pipeline",
            ));
        }
    }

    /// Pre-RDG resource preparation: create layouts, noise texture, compile pipelines,
    /// build the static uniforms bind group (Group 2).
    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        ssao_uniforms: &CpuBuffer<SsaoUniforms>,
    ) {
        // Persistent GPU resources: layouts, noise texture, pipelines.
        self.ensure_layouts(ctx.device);
        self.ensure_noise_texture(ctx.device, ctx.queue);
        self.ensure_pipelines(ctx);

        ctx.resource_manager.ensure_buffer(ssao_uniforms);

        // Build Group 2 static BG (uniforms only) — rebuild on buffer identity change.
        if let Some(g) = ctx.resource_manager.gpu_buffers.get(&ssao_uniforms.id())
            && (self.uniforms_static_bg.is_none() || self.last_uniforms_buffer_id != g.id)
        {
            let layout = self.raw_uniforms_layout.as_ref().unwrap();
            self.uniforms_static_bg =
                Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SSAO Uniforms G2 (static)"),
                    layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: g.buffer.as_entire_binding(),
                    }],
                }));
            self.last_uniforms_buffer_id = g.id;
        }
    }

    /// Build the ephemeral pass nodes and insert them into the graph as
    /// two independent RDG passes within an `"SSAO_System"` group.
    ///
    /// Returns the [`TextureNodeId`] of the half-resolution AO output for
    /// explicit downstream wiring (Opaque, Transparent).
    ///
    /// # Flattened Pass Chain
    ///
    /// 1. **SSAO_Raw** — hemisphere sampling → noisy R8Unorm
    /// 2. **SSAO_Blur** — cross-bilateral blur → clean AO output
    pub fn add_to_graph(
        &self,
        graph: &mut RenderGraph,
        scene_depth: TextureNodeId,
        scene_normals: TextureNodeId,
    ) -> TextureNodeId {
        let fc = *graph.frame_config();
        let half_w = (fc.width / 2).max(1);
        let half_h = (fc.height / 2).max(1);

        let raw_pipeline = self.raw_pipeline.expect("SsaoFeature not prepared");
        let blur_pipeline = self.blur_pipeline.expect("SsaoFeature not prepared");
        let raw_layout = self.raw_layout.clone().unwrap();
        let blur_layout = self.blur_layout.clone().unwrap();
        let noise_texture_view = self.noise_texture_view.clone().unwrap();
        let uniforms_static_bg = self
            .uniforms_static_bg
            .clone()
            .expect("SsaoFeature: uniforms static BG not built");

        graph.with_group("SSAO_System", |g| {
            // ─── Pass 1: Raw SSAO ──────────────────────────────────
            let raw_desc = TextureDesc::new_2d(
                half_w,
                half_h,
                SSAO_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );

            let raw_tex: TextureNodeId = g.add_pass("SSAO_Raw", |builder| {
                builder.read_texture(scene_depth);
                builder.read_texture(scene_normals);
                let out = builder.create_and_export("SSAO_Raw_Tex", raw_desc);
                let node = SsaoRawNode {
                    depth_tex: scene_depth,
                    normal_tex: scene_normals,
                    output_tex: out,
                    uniforms_static_bg: uniforms_static_bg.clone(),
                    raw_pipeline,
                    raw_layout: raw_layout.clone(),
                    noise_texture_view: noise_texture_view.clone(),
                    bind_group_key: None,
                };
                (node, out)
            });

            // ─── Pass 2: Cross-Bilateral Blur ──────────────────────
            let output_desc = TextureDesc::new_2d(
                half_w,
                half_h,
                SSAO_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );

            let blur_out: TextureNodeId = g.add_pass("SSAO_Blur", |builder| {
                builder.read_texture(raw_tex);
                builder.read_texture(scene_depth);
                builder.read_texture(scene_normals);
                let out = builder.create_and_export("SSAO_Output", output_desc);
                let node = SsaoBlurNode {
                    raw_tex,
                    depth_tex: scene_depth,
                    normal_tex: scene_normals,
                    output_tex: out,
                    blur_pipeline,
                    blur_layout: blur_layout.clone(),
                    bind_group_key: None,
                };
                (node, out)
            });

            blur_out
        })
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Pass 1: SsaoRawNode (ephemeral, created per frame)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Ephemeral per-frame node for the raw SSAO hemisphere sampling pass.
///
/// Binds depth, normals, noise texture, and uniforms to produce a noisy
/// single-channel AO texture at half resolution.
struct SsaoRawNode {
    depth_tex: TextureNodeId,
    normal_tex: TextureNodeId,
    output_tex: TextureNodeId,

    uniforms_static_bg: wgpu::BindGroup,
    raw_pipeline: RenderPipelineId,
    raw_layout: Tracked<wgpu::BindGroupLayout>,
    noise_texture_view: Tracked<wgpu::TextureView>,

    bind_group_key: Option<BindGroupKey>,
}

impl PassNode for SsaoRawNode {
    fn prepare(&mut self, ctx: &mut PrepareContext) {
        let device = ctx.device;

        let depth_key = SubViewKey {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        };
        ctx.views.get_or_create_sub_view(self.depth_tex, &depth_key);
        let depth_only_view = ctx
            .views
            .get_sub_view(self.depth_tex, &depth_key)
            .expect("SSAO Raw: depth-only view must exist");

        let normal_view = ctx.views.get_texture_view(self.normal_tex);
        let noise_view = &self.noise_texture_view;

        let linear_sampler = ctx.sampler_registry.get_common(CommonSampler::LinearClamp);
        let noise_sampler = ctx
            .sampler_registry
            .get_common(CommonSampler::NearestRepeat);
        let point_sampler = ctx.sampler_registry.get_common(CommonSampler::NearestClamp);

        let layout = &self.raw_layout;

        let key = BindGroupKey::new(layout.id())
            .with_resource(depth_only_view.id())
            .with_resource(normal_view.id())
            .with_resource(noise_view.id())
            .with_resource(linear_sampler.id())
            .with_resource(noise_sampler.id())
            .with_resource(point_sampler.id());

        if self.bind_group_key.as_ref() != Some(&key) {
            if ctx.global_bind_group_cache.get(&key).is_none() {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SSAO Raw BG (G1)"),
                    layout,
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
            self.bind_group_key = Some(key);
        }
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let Some(bg_key) = &self.bind_group_key else {
            return;
        };

        let global_bg = ctx.baked_lists.global_bind_group;
        let raw_bg = ctx
            .global_bind_group_cache
            .get(bg_key)
            .expect("SSAO raw BG should exist");
        let pipeline = ctx.pipeline_cache.get_render_pipeline(self.raw_pipeline);

        let rtt = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSAO Raw Pass"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, global_bg, &[]);
        pass.set_bind_group(1, raw_bg, &[]);
        pass.set_bind_group(2, &self.uniforms_static_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Pass 2: SsaoBlurNode (ephemeral, created per frame)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Ephemeral per-frame node for the cross-bilateral blur pass.
///
/// Reads the noisy raw AO texture and produces the final clean AO output
/// using depth/normal-aware spatial filtering.
struct SsaoBlurNode {
    raw_tex: TextureNodeId,
    depth_tex: TextureNodeId,
    normal_tex: TextureNodeId,
    output_tex: TextureNodeId,

    blur_pipeline: RenderPipelineId,
    blur_layout: Tracked<wgpu::BindGroupLayout>,

    bind_group_key: Option<BindGroupKey>,
}

impl PassNode for SsaoBlurNode {
    fn prepare(&mut self, ctx: &mut PrepareContext) {
        let device = ctx.device;

        let depth_key = SubViewKey {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        };
        ctx.views.get_or_create_sub_view(self.depth_tex, &depth_key);
        let depth_only_view = ctx
            .views
            .get_sub_view(self.depth_tex, &depth_key)
            .expect("SSAO Blur: depth-only view must exist");

        let raw_view = ctx.views.get_texture_view(self.raw_tex);
        let normal_view = ctx.views.get_texture_view(self.normal_tex);

        let linear_sampler = ctx.sampler_registry.get_common(CommonSampler::LinearClamp);
        let point_sampler = ctx.sampler_registry.get_common(CommonSampler::NearestClamp);

        let layout = &self.blur_layout;

        let key = BindGroupKey::new(layout.id())
            .with_resource(raw_view.id())
            .with_resource(depth_only_view.id())
            .with_resource(normal_view.id())
            .with_resource(linear_sampler.id())
            .with_resource(point_sampler.id());

        if self.bind_group_key.as_ref() != Some(&key) {
            if ctx.global_bind_group_cache.get(&key).is_none() {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SSAO Blur BG (G0)"),
                    layout,
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
            self.bind_group_key = Some(key);
        }
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let Some(bg_key) = &self.bind_group_key else {
            return;
        };

        let blur_bg = ctx
            .global_bind_group_cache
            .get(bg_key)
            .expect("SSAO blur BG should exist");
        let pipeline = ctx.pipeline_cache.get_render_pipeline(self.blur_pipeline);

        let rtt = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSAO Blur Pass"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, blur_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
