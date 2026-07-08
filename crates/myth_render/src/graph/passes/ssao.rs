//! SSAO Feature + Ephemeral PassNodes (Flattened)
//!
//! - **`SsaoFeature`** (long-lived): owns pipelines, bind group layouts,
//!   and system blue-noise handles. `extract_and_prepare()` compiles pipelines and uploads
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

use crate::core::gpu::{CommonSampler, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    ExecuteContext, ExtractContext, PassNode, PrepareContext, RenderTargetOps, TextureDesc,
    TextureNodeId,
};
use crate::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions, ShaderSource,
};
use myth_resources::buffer::CpuBuffer;
use myth_resources::ssao::SsaoUniforms;
use myth_resources::uniforms::WgslStruct;

/// The SSAO output texture format: single-channel unsigned normalized.
const SSAO_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Feature (long-lived, stored in RendererState)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Long-lived SSAO feature — owns persistent GPU resources (pipelines,
/// bind group layouts, system blue-noise handles).
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
    blue_noise_view: Option<Tracked<wgpu::TextureView>>,
    blue_noise_sampler: Option<Tracked<wgpu::Sampler>>,

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

            blue_noise_view: None,
            blue_noise_sampler: None,

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

        // ─── Raw SSAO Layout (Group 1): depth, normal, blue noise + samplers ───
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
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: blue_noise_view_dimension(),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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
            options.inject_code("binding_code", &gpu_world.binding_wgsl);
            options.inject_code(
                "scene_lighting_structs",
                myth_resources::uniforms::scene_lighting_structs_wgsl(),
            );

            let (module, hash) = ctx.shader_manager.get_or_compile(
                device,
                ShaderSource::File("entry/features/ssao/raw"),
                &options,
            );

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SSAO Raw Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&gpu_world.layout),
                    Some(raw_layout),
                    Some(uniforms_layout),
                ],
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
            let (module, hash) = ctx.shader_manager.get_or_compile(
                device,
                ShaderSource::File("entry/features/ssao/blur"),
                &ShaderCompilationOptions::default(),
            );

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SSAO Blur Pipeline Layout"),
                bind_group_layouts: &[Some(blur_layout)],
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

    /// Pre-RDG resource preparation: create layouts, sync blue-noise state, compile pipelines,
    /// build the static uniforms bind group (Group 2).
    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        ssao_uniforms: &CpuBuffer<SsaoUniforms>,
    ) {
        // Persistent GPU resources: layouts and pipelines.
        self.ensure_layouts(ctx.device);
        self.ensure_pipelines(ctx);

        {
            let viewport = ctx.render_state.uniforms().read().viewport;
            let mut uniforms = ssao_uniforms.write();
            uniforms.noise_scale =
                glam::Vec2::new((viewport.x * 0.5).max(1.0), (viewport.y * 0.5).max(1.0));
            uniforms.frame_index = ctx.resource_manager.frame_index() as u32;
        }

        ctx.resource_manager.ensure_buffer(ssao_uniforms);
        self.blue_noise_view = Some(ctx.resource_manager.system_textures().blue_noise.clone());
        self.blue_noise_sampler = Some(
            ctx.resource_manager
                .system_textures()
                .blue_noise_sampler
                .clone(),
        );

        // Build Group 2 static BG (uniforms only) — rebuild on buffer identity change.
        if let Some(handle) = ssao_uniforms.gpu_handle()
            && let Some(g) = ctx.resource_manager.get_gpu_buffer(handle)
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
    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        scene_depth: TextureNodeId,
        scene_normals: TextureNodeId,
    ) -> TextureNodeId {
        let fc = ctx.frame_config;
        let half_w = (fc.width / 2).max(1);
        let half_h = (fc.height / 2).max(1);

        let raw_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.raw_pipeline.expect("SsaoFeature not prepared"));
        let blur_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.blur_pipeline.expect("SsaoFeature not prepared"));
        let raw_layout = self.raw_layout.as_ref().unwrap();
        let blur_layout = self.blur_layout.as_ref().unwrap();
        let blue_noise_view = self.blue_noise_view.as_ref().unwrap();
        let blue_noise_sampler = self.blue_noise_sampler.as_ref().unwrap();
        let uniforms_static_bg = self
            .uniforms_static_bg
            .as_ref()
            .expect("SsaoFeature: uniforms static BG not built");

        ctx.with_group("SSAO_System", |ctx| {
            // ─── Pass 1: Raw SSAO ──────────────────────────────────
            let raw_desc = TextureDesc::new_2d(
                half_w,
                half_h,
                SSAO_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );

            let raw_tex: TextureNodeId = ctx.graph.add_pass("SSAO_Raw", |builder| {
                builder.read_texture(scene_depth);
                builder.read_texture(scene_normals);
                let out = builder.create_texture("SSAO_Raw_Tex", raw_desc);
                let node = SsaoRawNode {
                    depth_tex: scene_depth,
                    normal_tex: scene_normals,
                    output_tex: out,
                    uniforms_static_bg,
                    pipeline: raw_pipeline,
                    raw_layout,
                    blue_noise_view,
                    blue_noise_sampler,
                    transient_bg: None,
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

            let blur_out: TextureNodeId = ctx.graph.add_pass("SSAO_Blur", |builder| {
                builder.read_texture(raw_tex);
                builder.read_texture(scene_depth);
                builder.read_texture(scene_normals);
                let out = builder.create_texture("SSAO_Output", output_desc);
                let node = SsaoBlurNode {
                    raw_tex,
                    depth_tex: scene_depth,
                    normal_tex: scene_normals,
                    output_tex: out,
                    pipeline: blur_pipeline,
                    blur_layout,
                    transient_bg: None,
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
/// Binds depth, normals, blue noise, and uniforms to produce a noisy
/// single-channel AO texture at half resolution.
struct SsaoRawNode<'a> {
    depth_tex: TextureNodeId,
    normal_tex: TextureNodeId,
    output_tex: TextureNodeId,

    uniforms_static_bg: &'a wgpu::BindGroup,
    pipeline: &'a wgpu::RenderPipeline,
    raw_layout: &'a Tracked<wgpu::BindGroupLayout>,
    blue_noise_view: &'a Tracked<wgpu::TextureView>,
    blue_noise_sampler: &'a Tracked<wgpu::Sampler>,

    transient_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for SsaoRawNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.transient_bg = Some(
            crate::myth_bind_group!(ctx, self.raw_layout, Some("SSAO Raw BG (G1)"), [
                0 => self.depth_tex,
                1 => self.normal_tex,
                2 => self.blue_noise_view,
                3 => CommonSampler::LinearClamp,
                4 => self.blue_noise_sampler,
                5 => CommonSampler::NearestClamp,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let global_bg = ctx.baked_lists.global_bind_group;
        let raw_bg = self.transient_bg.expect("SSAO raw BG not prepared");

        let rtt = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSAO Raw Pass"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(self.pipeline);
        pass.set_bind_group(0, global_bg, &[]);
        pass.set_bind_group(1, raw_bg, &[]);
        pass.set_bind_group(2, self.uniforms_static_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn blue_noise_view_dimension() -> wgpu::TextureViewDimension {
    if cfg!(feature = "advanced_noise") {
        wgpu::TextureViewDimension::D2Array
    } else {
        wgpu::TextureViewDimension::D2
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Pass 2: SsaoBlurNode (ephemeral, created per frame)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Ephemeral per-frame node for the cross-bilateral blur pass.
///
/// Reads the noisy raw AO texture and produces the final clean AO output
/// using depth/normal-aware spatial filtering.
struct SsaoBlurNode<'a> {
    raw_tex: TextureNodeId,
    depth_tex: TextureNodeId,
    normal_tex: TextureNodeId,
    output_tex: TextureNodeId,

    pipeline: &'a wgpu::RenderPipeline,
    blur_layout: &'a Tracked<wgpu::BindGroupLayout>,

    transient_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for SsaoBlurNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.transient_bg = Some(
            crate::myth_bind_group!(ctx, self.blur_layout, Some("SSAO Blur BG (G0)"), [
                0 => self.raw_tex,
                1 => self.depth_tex,
                2 => self.normal_tex,
                3 => CommonSampler::LinearClamp,
                4 => CommonSampler::NearestClamp,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let blur_bg = self.transient_bg.expect("SSAO blur BG not prepared");

        let rtt = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSAO Blur Pass"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(self.pipeline);
        pass.set_bind_group(0, blur_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
