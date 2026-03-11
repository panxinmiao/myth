//! SSSS Feature + Ephemeral PassNode
//!
//! - **`SsssFeature`** (long-lived): owns pipeline cache, bind group layout,
//!   profiles storage buffer.  `extract_and_prepare()` compiles pipelines and
//!   uploads SSS profile data.
//! - **`SsssPassNode`** (ephemeral per-frame): carries lightweight IDs and
//!   transient bind-group slots.  Created by `SsssFeature::add_to_graph()`.
//!
//! Implements a separable Gaussian blur for SSS materials identified via the
//! **Thin G-Buffer** stencil channel.  Two sequential render passes are
//! executed (H then V) to reconstruct a 2-D scatter kernel.
//!
//! # Data Flow (explicit wiring)
//!
//! ```text
//!  Opaque                            SsssPassNode
//!       |                 +-----------------------------------+
//! color_in  ------------>|  H Sub-Pass: Horizontal blur      |---> temp_blur
//! normal_in ----+------->|                                   |         |
//! depth_in  ----|        |  V Sub-Pass: Vertical blur        |<--------+
//! feature_id ---|        |  (writes back to color_in         |
//! specular_tex -+        |   in-place)                       |---> color_in
//!                        +-----------------------------------+
//! ```
//!
//! # Integration
//!
//! Must come **after** `OpaquePass` and **before** the Skybox/MSAA-Sync
//! stage in the `HighFidelity` render path.  The Feature only calls
//! `add_to_graph` when SSS is enabled — zero cost when disabled.
//!
//! # GPU Resources
//!
//! - **Profiles StorageBuffer**: 256 x `SssProfileData` (12 KB).
//!   Uploaded from `AssetServer.sss_registry`.
//! - **temp_blur TransientTexture**: `Rgba16Float`, same size as scene colour.

use crate::renderer::HDR_TEXTURE_FORMAT;
use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::gpu::{CommonSampler, Tracked};
use crate::renderer::graph::core::{
    ExecuteContext, ExtractContext, PassBuilder, PassNode, PrepareContext, RenderGraph, SubViewKey,
    TextureDesc, TextureNodeId,
};
use crate::renderer::pipeline::{
    ColorTargetKey, DepthStencilKey, FullscreenPipelineKey, RenderPipelineId,
    ShaderCompilationOptions,
};
use crate::resources::screen_space::{STENCIL_FEATURE_SSS, SssProfileData};
use std::mem::size_of;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Feature (long-lived, stored in RenderFeatures)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Long-lived SSSS feature — owns persistent GPU resources (pipelines,
/// bind group layout, profiles buffer).
///
/// Produces an ephemeral [`SsssPassNode`] each frame via [`Self::add_to_graph`].
pub struct SsssFeature {
    // --- Pipelines -------------------------------------------------
    horizontal_pipeline: Option<RenderPipelineId>,
    vertical_pipeline: Option<RenderPipelineId>,
    bind_group_layout: Option<Tracked<wgpu::BindGroupLayout>>,

    // --- Persistent GPU Resources ----------------------------------
    profiles_buffer: Option<Tracked<wgpu::Buffer>>,
    last_registry_version: u64,
}

impl Default for SsssFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl SsssFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            horizontal_pipeline: None,
            vertical_pipeline: None,
            bind_group_layout: None,
            profiles_buffer: None,
            last_registry_version: 0,
        }
    }

    /// Pre-RDG resource preparation: create layout, compile pipelines,
    /// upload SSS profile data.
    pub fn extract_and_prepare(&mut self, ctx: &mut ExtractContext) {
        let device = ctx.device;

        // -- 1. Lazy-create profiles storage buffer ---------------------
        if self.profiles_buffer.is_none() {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SSS Profiles Buffer"),
                size: (256 * size_of::<SssProfileData>()) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.profiles_buffer = Some(Tracked::new(buffer));
        }

        // -- 2. Diff-sync profiles data ---------------------------------
        let registry = ctx.assets.sss_registry.read();
        if self.last_registry_version != registry.version {
            ctx.queue.write_buffer(
                self.profiles_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&registry.buffer_data),
            );
            self.last_registry_version = registry.version;
        }

        // -- 3. Lazy-create pipelines + layout --------------------------
        if self.horizontal_pipeline.is_none() {
            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("SSSS Bind Group Layout"),
                    entries: &[
                        // binding 0: t_color
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
                        // binding 1: t_normal
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
                        // binding 2: t_depth
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
                        // binding 3: u_profiles (storage)
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
                        // binding 4: s_sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        // binding 5: t_feature_id
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
                        // binding 6: t_specular
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
                label: Some("SSSS Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });

            // -- Horizontal shader (no defines) -------------------------
            let shader_defines = ShaderCompilationOptions::default();

            let (hor_shader, hor_hash) = ctx.shader_manager.get_or_compile_template(
                device,
                "passes/ssss",
                &shader_defines,
                "",
                "",
            );

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
                "SSSS Horizontal Pipeline",
            ));

            // -- Vertical shader (SSSS_VERTICAL_PASS = 1) -------------
            let mut vert_defines = ShaderCompilationOptions::default();
            vert_defines.add_define("SSSS_VERTICAL_PASS", "1");

            let (vert_shader, vert_hash) = ctx.shader_manager.get_or_compile_template(
                device,
                "passes/ssss",
                &vert_defines,
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
                "SSSS Vertical Pipeline",
            ));

            self.bind_group_layout = Some(Tracked::new(bind_group_layout));
        }
    }

    /// Build the ephemeral pass node and insert it into the graph.
    ///
    /// All inputs are explicitly wired — no blackboard lookups.
    /// The pass modifies `scene_color` in-place (read + write).
    pub fn add_to_graph(
        &self,
        graph: &mut RenderGraph,
        scene_color: TextureNodeId,
        scene_depth: TextureNodeId,
        scene_normals: TextureNodeId,
        feature_id: TextureNodeId,
        specular_tex: TextureNodeId,
    ) -> TextureNodeId {
        let ssss_out_color = graph.create_alias(scene_color, "Scene_Color_SSSS");

        let node = SsssPassNode {
            scene_color_in: scene_color,
            scene_color_out: ssss_out_color,
            temp_blur: TextureNodeId(0),
            depth_in: scene_depth,
            normal_in: scene_normals,
            feature_id,
            specular_tex,
            horizontal_pipeline: self.horizontal_pipeline.expect("SsssFeature not prepared"),
            vertical_pipeline: self.vertical_pipeline.expect("SsssFeature not prepared"),
            bind_group_layout: self.bind_group_layout.clone().unwrap(),
            profiles_buffer: self.profiles_buffer.clone().unwrap(),
            horizontal_bind_group: None,
            vertical_bind_group: None,
        };
        graph.add_pass(Box::new(node));
        ssss_out_color
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PassNode (ephemeral, created per frame)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct SsssPassNode {
    // ─── RDG Resource Slots (explicit wiring from add_to_graph) ───────
    scene_color_in: TextureNodeId,
    scene_color_out: TextureNodeId,
    temp_blur: TextureNodeId,
    depth_in: TextureNodeId,
    normal_in: TextureNodeId,
    feature_id: TextureNodeId,
    specular_tex: TextureNodeId,

    // --- Cloned from Feature (lightweight IDs / Tracked clones) ----
    horizontal_pipeline: RenderPipelineId,
    vertical_pipeline: RenderPipelineId,
    bind_group_layout: Tracked<wgpu::BindGroupLayout>,
    profiles_buffer: Tracked<wgpu::Buffer>,

    // --- Per-Frame BindGroups --------------------------------------
    horizontal_bind_group: Option<wgpu::BindGroup>,
    vertical_bind_group: Option<wgpu::BindGroup>,
}

impl PassNode for SsssPassNode {
    fn name(&self) -> &'static str {
        "Ssss_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        // Internal scratch texture for the horizontal blur.
        let (w, h) = builder.global_resolution();
        let hdr_format = builder.frame_config().hdr_format;
        let desc = TextureDesc::new_2d(
            w,
            h,
            hdr_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        self.temp_blur = builder.create_texture("SSSS_Temp", desc);

        // In-place read + write on scene color.
        builder.read_texture(self.scene_color_in);
        builder.declare_output(self.scene_color_out);

        // Upstream inputs.
        builder.read_texture(self.depth_in);
        builder.read_texture(self.normal_in);
        builder.read_texture(self.feature_id);
        builder.read_texture(self.specular_tex);
    }

    fn prepare(&mut self, ctx: &mut PrepareContext) {
        // -- Gather input views (RDG-managed) ---------------------------

        // Depth: need a DepthOnly sub-view for sampling
        let depth_sub_key = SubViewKey {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        };

        ctx.views
            .get_or_create_sub_view(self.depth_in, &depth_sub_key);

        let depth_only_view = ctx
            .views
            .get_sub_view(self.depth_in, &depth_sub_key)
            .expect("SSSS: depth-only view must exist");

        let color_in_view = ctx.views.get_texture_view(self.scene_color_in);

        let normal_view = ctx.views.get_texture_view(self.normal_in);
        let feature_view = ctx.views.get_texture_view(self.feature_id);
        let temp_blur_view = ctx.views.get_texture_view(self.temp_blur);
        let specular_view = ctx.views.get_texture_view(self.specular_tex);

        // -- Build bind groups (H: read color_in, V: read temp_blur) ----
        let layout = &self.bind_group_layout;
        let sampler = ctx.sampler_registry.get_common(CommonSampler::LinearClamp);
        let profiles_buffer = &self.profiles_buffer;

        // Horizontal: color_in -> temp_blur
        let horizontal_key = BindGroupKey::new(layout.id())
            .with_resource(color_in_view.id())
            .with_resource(normal_view.id())
            .with_resource(depth_only_view.id())
            .with_resource(profiles_buffer.id())
            .with_resource(sampler.id())
            .with_resource(feature_view.id())
            .with_resource(specular_view.id());

        if ctx.global_bind_group_cache.get(&horizontal_key).is_none() {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SSSS Horizontal Bind Group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(color_in_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(normal_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(depth_only_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: profiles_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(feature_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(specular_view),
                    },
                ],
            });
            ctx.global_bind_group_cache
                .insert(horizontal_key.clone(), bg);
        }
        self.horizontal_bind_group = ctx.global_bind_group_cache.get(&horizontal_key).cloned();

        // Vertical: temp_blur -> color_out
        let vertical_key = BindGroupKey::new(layout.id())
            .with_resource(temp_blur_view.id())
            .with_resource(normal_view.id())
            .with_resource(depth_only_view.id())
            .with_resource(profiles_buffer.id())
            .with_resource(sampler.id())
            .with_resource(feature_view.id())
            .with_resource(specular_view.id());

        if ctx.global_bind_group_cache.get(&vertical_key).is_none() {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SSSS Vertical Bind Group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(temp_blur_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(normal_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(depth_only_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: profiles_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(feature_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(specular_view),
                    },
                ],
            });
            ctx.global_bind_group_cache.insert(vertical_key.clone(), bg);
        }
        self.vertical_bind_group = ctx.global_bind_group_cache.get(&vertical_key).cloned();
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let depth_stencil_view = ctx.get_texture_view(self.depth_in);
        let temp_blur_view = ctx.get_texture_view(self.temp_blur);

        // Resolve pipeline handles -> actual GPU pipeline references (O(1))
        let hor_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.horizontal_pipeline);
        let vert_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.vertical_pipeline);

        // -- H Sub-Pass: color_in -> temp_blur --------------------------
        {
            // let rtt = ctx.get_color_attachment(self.temp_blur, Some(wgpu::Color::TRANSPARENT), None);
            // we use StoreOp::Store here. Todo: improve resource lifetime tracking.
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSSS Horizontal"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: temp_blur_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
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

            rpass.set_pipeline(hor_pipeline);
            rpass.set_stencil_reference(STENCIL_FEATURE_SSS);
            rpass.set_bind_group(0, self.horizontal_bind_group.as_ref().unwrap(), &[]);
            rpass.draw(0..3, 0..1);
        }

        // -- V Sub-Pass: temp_blur -> color_out -------------------------
        {
            let rtt = ctx.get_color_attachment(self.scene_color_out, None, None);

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSSS Vertical"),
                color_attachments: &[rtt],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_stencil_view,
                    depth_ops: None,
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            rpass.set_pipeline(vert_pipeline);
            rpass.set_stencil_reference(STENCIL_FEATURE_SSS);
            rpass.set_bind_group(0, self.vertical_bind_group.as_ref().unwrap(), &[]);
            rpass.draw(0..3, 0..1);
        }
    }
}
