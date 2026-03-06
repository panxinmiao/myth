//! RDG Screen-Space Sub-Surface Scattering (SSSSS) Post-Processing Pass
//!
//! Implements a separable Gaussian blur for SSS materials identified via the
//! **Thin G-Buffer** stencil channel.  Two sequential render passes are
//! executed (H then V) to reconstruct a 2-D scatter kernel.
//!
//! # Data Flow (SSA-Compliant)
//!
//! ```text
//!  Opaque / Transparent             RdgSssssPass
//!       |                 +-----------------------------------+
//! color_in  ------------>|  H Sub-Pass: Horizontal blur      |---> temp_blur
//! normal_in ----+------->|                                   |         |
//! depth_in  ----|        |  V Sub-Pass: Vertical blur        |<--------+
//! feature_id ---|        |  (writes NEW color_out, not       |
//! specular_tex -+        |   back to color_in)               |---> color_out
//!                        +-----------------------------------+
//! ```
//!
//! # SSA Principle
//!
//! The pass reads `color_in` but writes to a distinct `color_out` node.
//! The intermediate `temp_blur` is also a separate RDG transient resource.
//! No resource node is both read and written -- the DAG remains acyclic.
//!
//! # Integration
//!
//! Must come **after** `TransparentPass` and **before** `BloomPass` in the
//! `HighFidelity` render path.  Zero cost when disabled (`enabled == false`).
//!
//! # GPU Resources
//!
//! - **Profiles StorageBuffer**: 256 x `SssProfileData` (12 KB).
//!   Uploaded from `AssetServer.sss_registry`.
//! - **temp_blur TransientTexture**: `Rgba16Float`, same size as scene colour.

use crate::renderer::HDR_TEXTURE_FORMAT;
use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::resources::{CommonSampler, Tracked};
use crate::renderer::graph::rdg::allocator::SubViewKey;
use crate::renderer::graph::rdg::builder::PassBuilder;
use crate::renderer::graph::rdg::context::{RdgExecuteContext, RdgPrepareContext};
use crate::renderer::graph::rdg::node::PassNode;
use crate::renderer::graph::rdg::types::{RdgTextureDesc, TextureNodeId};
use crate::renderer::pipeline::{
    ColorTargetKey, DepthStencilKey, FullscreenPipelineKey, RenderPipelineId,
    ShaderCompilationOptions,
};
use crate::resources::screen_space::{SssProfileData, STENCIL_FEATURE_SSS};
use std::mem::size_of;

/// RDG Screen-Space Sub-Surface Scattering pass (SSA-compliant).
///
/// Performs a two-pass separable Gaussian blur on pixels marked as SSS in
/// the stencil buffer.  Reads from `color_in`, writes a distinct `color_out`.
pub struct RdgSssssPass {
    // --- RDG Resource Slots (set by Composer) ----------------------
    /// HDR scene colour
    pub scene_color: TextureNodeId,
    // /// HDR scene colour output (write only -- V sub-pass destination)
    // pub color_out: TextureNodeId,
    /// Intermediate blur target (write H, read V)
    pub temp_blur: TextureNodeId,
    /// Scene depth buffer (stencil for culling + depth-aware blur)
    pub depth_in: TextureNodeId,
    /// Scene normals (for normal-aware kernel weighting)
    pub normal_in: TextureNodeId,
    /// Feature ID texture (SSS material identification)
    pub feature_id: TextureNodeId,
    /// Specular MRT texture
    pub specular_tex: TextureNodeId,

    // --- Push Parameters (set by Composer) -------------------------
    /// Whether SSS is enabled this frame.
    pub enabled: bool,

    // --- Pipelines -------------------------------------------------
    horizontal_pipeline: Option<RenderPipelineId>,
    vertical_pipeline: Option<RenderPipelineId>,
    bind_group_layout: Option<Tracked<wgpu::BindGroupLayout>>,

    // --- Persistent GPU Resources ----------------------------------
    profiles_buffer: Option<Tracked<wgpu::Buffer>>,
    last_registry_version: u64,

    // --- Per-Frame BindGroups --------------------------------------
    horizontal_bind_group: Option<wgpu::BindGroup>,
    vertical_bind_group: Option<wgpu::BindGroup>,
}

impl RdgSssssPass {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scene_color: TextureNodeId(0),
            // color_out: TextureNodeId(0),
            temp_blur: TextureNodeId(0),
            depth_in: TextureNodeId(0),
            normal_in: TextureNodeId(0),
            feature_id: TextureNodeId(0),
            specular_tex: TextureNodeId(0),
            enabled: false,
            horizontal_pipeline: None,
            vertical_pipeline: None,
            bind_group_layout: None,
            profiles_buffer: None,
            last_registry_version: 0,
            horizontal_bind_group: None,
            vertical_bind_group: None,
        }
    }
}

impl PassNode for RdgSssssPass {
    fn name(&self) -> &'static str {
        "RDG SSSSS Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        if !self.enabled {
            return;
        }

        // Producer: create the temporary blur texture.
        let (w, h) = builder.global_resolution();
        let hdr_format = builder.frame_config().hdr_format;
        let desc = RdgTextureDesc::new_2d(
            w,
            h,
            hdr_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        self.temp_blur = builder.create_and_export("SSSSS_Temp", desc);

        // Consumer: wire upstream resources.
        self.scene_color = builder.write_blackboard("Scene_Color_HDR");
        builder.read_texture(self.scene_color);
        self.depth_in = builder.read_blackboard("Scene_Depth");
        self.normal_in = builder.read_blackboard("Scene_Normals");
        self.feature_id = builder.read_blackboard("Feature_ID");
        self.specular_tex = builder.read_blackboard("Specular_MRT");
    }

    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {
        if !self.enabled {
            return;
        }

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
                    label: Some("SSSSS Bind Group Layout"),
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
                label: Some("SSSSS Pipeline Layout"),
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
                "SSSSS Horizontal Pipeline",
            ));

            // -- Vertical shader (SSSSS_VERTICAL_PASS = 1) -------------
            let mut vert_defines = ShaderCompilationOptions::default();
            vert_defines.add_define("SSSSS_VERTICAL_PASS", "1");

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
                "SSSSS Vertical Pipeline",
            ));

            self.bind_group_layout = Some(Tracked::new(bind_group_layout));
        }

        // -- 4. Gather input views (RDG-managed) -----------------------
        // Extract IDs and raw pointers upfront to avoid holding the immutable
        // borrow on `ctx` across mutable cache operations (same pattern as SSAO).

        // let color_in_view_id = ctx.get_texture_view(self.scene_color).id();
        // let color_in_view_ptr =
        //     ctx.get_texture_view(self.scene_color) as *const Tracked<wgpu::TextureView>;

        // Depth: need a DepthOnly sub-view for sampling
        let depth_sub_key = SubViewKey {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        };
        
        ctx.views.get_or_create_sub_view(
            self.depth_in,
            depth_sub_key.clone(),
        );

        let depth_only_view = ctx.views.get_sub_view(self.depth_in, &depth_sub_key).expect(
            "RDG SSSSS: depth-only view must exist"
        );

        let color_in_view = ctx.views.get_texture_view(self.scene_color);

        let normal_view = ctx.views.get_texture_view(self.normal_in);
        let feature_view = ctx.views.get_texture_view(self.feature_id);
        let temp_blur_view = ctx.views.get_texture_view(self.temp_blur);
        let specular_view = ctx.views.get_texture_view(self.specular_tex);

        // -- 5. Build bind groups (H: read color_in, V: read temp_blur)
        let layout = self.bind_group_layout.as_ref().unwrap();
        let sampler = ctx.sampler_registry.get_common(CommonSampler::LinearClamp);
        let profiles_buffer = self.profiles_buffer.as_ref().unwrap();

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
                label: Some("SSSSS Horizontal Bind Group"),
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
        self.horizontal_bind_group = ctx
            .global_bind_group_cache
            .get(&horizontal_key)
            .cloned();

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
            // SAFETY: All views are alive for the entire frame scope.
                // let temp_blur_view = unsafe { &*temp_blur_view_ptr };
                // let normal_view = unsafe { &*normal_view_ptr };
                // let depth_view = unsafe { &*depth_view_ptr };
                // let feature_view = unsafe { &*feature_view_ptr };
                // let specular_view = unsafe { &*specular_view_ptr };

            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SSSSS Vertical Bind Group"),
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

    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        if !self.enabled {
            return;
        }

        let depth_stencil_view = ctx.get_texture_view(self.depth_in);
        let temp_blur_view = ctx.get_texture_view(self.temp_blur);
        let color_out_view = ctx.get_texture_view(self.scene_color);

        // Resolve pipeline handles -> actual GPU pipeline references (O(1))
        let hor_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.horizontal_pipeline.unwrap());
        let vert_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.vertical_pipeline.unwrap());

        // -- H Sub-Pass: color_in -> temp_blur --------------------------
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSSSS Horizontal"),
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
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSSSS Vertical"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_out_view,
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
            rpass.set_stencil_reference(STENCIL_FEATURE_SSS);
            rpass.set_bind_group(0, self.vertical_bind_group.as_ref().unwrap(), &[]);
            rpass.draw(0..3, 0..1);
        }
    }
}