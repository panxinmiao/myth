//! Screen-Space Sub-Surface Scattering (SSSSS) Feature & Transient Pass Node
//!
//! Implements a separable Gaussian blur for SSS materials identified via the
//! **Thin G-Buffer** stencil channel. Two sequential render passes are
//! executed (H then V) to reconstruct a 2-D scatter kernel.
//!
//! - [`SssssFeature`] (persistent): pipelines, layout, profiles storage buffer.
//! - [`RdgSssssPassNode`] (transient): per-frame bind groups, sub-pass execution.
//!
//! # Data Flow
//!
//! ```text
//!  Opaque / Transparent                 SSSSS
//!       |                    +-----------------------------------+
//! scene_color  ------------>|  H Sub-Pass: Horizontal blur      |---> temp_blur
//! normal_in  ----+--------->|                                   |         |
//! depth_in   ----|          |  V Sub-Pass: Vertical blur        |<--------+
//! feature_id ----|          |  (writes back to scene_color)     |
//! specular_tex --+          +-----------------------------------+---> scene_color
//! ```
//!
//! # Integration
//!
//! Must come **after** `OpaquePass` and **before** `BloomPass` in the
//! `HighFidelity` render path. Zero cost when disabled.

use std::mem::size_of;

use crate::renderer::HDR_TEXTURE_FORMAT;
use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::resources::{CommonSampler, Tracked};
use crate::renderer::graph::rdg::allocator::SubViewKey;
use crate::renderer::graph::rdg::builder::PassBuilder;
use crate::renderer::graph::rdg::context::{RdgExecuteContext, RdgPrepareContext};
use crate::renderer::graph::rdg::feature::ExtractContext;
use crate::renderer::graph::rdg::graph::RenderGraph;
use crate::renderer::graph::rdg::node::PassNode;
use crate::renderer::graph::rdg::types::{RdgTextureDesc, TextureNodeId};
use crate::renderer::pipeline::{
    ColorTargetKey, DepthStencilKey, FullscreenPipelineKey, RenderPipelineId,
    ShaderCompilationOptions,
};
use crate::resources::screen_space::{STENCIL_FEATURE_SSS, SssProfileData};

// =============================================================================
// Persistent Feature
// =============================================================================

/// Persistent SSSSS Feature — owns pipelines, bind group layout,
/// and the profiles storage buffer.
pub struct SssssFeature {
    horizontal_pipeline: Option<RenderPipelineId>,
    vertical_pipeline: Option<RenderPipelineId>,
    bind_group_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    profiles_buffer: Option<Tracked<wgpu::Buffer>>,
    last_registry_version: u64,
}

impl SssssFeature {
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

    /// Create persistent resources: profiles buffer, pipelines, layout.
    pub fn extract_and_prepare(&mut self, ctx: &mut ExtractContext) {
        self.ensure_profiles_buffer(ctx.device);
        self.sync_profiles(ctx);
        self.ensure_pipelines(ctx);
    }

    /// Build and inject a transient SSSSS node into the render graph.
    ///
    /// The pass reads `scene_color` (H sub-pass input), writes back to
    /// `scene_color` (V sub-pass output), and uses an internal `temp_blur`
    /// texture for the intermediate horizontal result.
    pub fn add_to_graph(
        &self,
        rdg: &mut RenderGraph,
        scene_color: TextureNodeId,
        scene_depth: TextureNodeId,
        scene_normals: TextureNodeId,
        feature_id: TextureNodeId,
        specular_tex: TextureNodeId,
    ) {
        let config = rdg.frame_config();
        let (w, h) = (config.width, config.height);
        let hdr_format = config.hdr_format;

        let temp_desc = RdgTextureDesc::new_2d(
            w,
            h,
            hdr_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let temp_blur = rdg.register_resource("SSSSS_Temp", temp_desc, false);

        let node = Box::new(RdgSssssPassNode {
            scene_color,
            temp_blur,
            depth_in: scene_depth,
            normal_in: scene_normals,
            feature_id,
            specular_tex,
            horizontal_pipeline: self.horizontal_pipeline.unwrap(),
            vertical_pipeline: self.vertical_pipeline.unwrap(),
            bind_group_layout: self.bind_group_layout.clone().unwrap(),
            profiles_buffer: self.profiles_buffer.clone().unwrap(),
            horizontal_bind_group: None,
            vertical_bind_group: None,
        });
        rdg.add_pass_owned(node);
    }

    // =========================================================================
    // Lazy Initialization
    // =========================================================================

    fn ensure_profiles_buffer(&mut self, device: &wgpu::Device) {
        if self.profiles_buffer.is_some() {
            return;
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SSS Profiles Buffer"),
            size: (256 * size_of::<SssProfileData>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.profiles_buffer = Some(Tracked::new(buffer));
    }

    fn sync_profiles(&mut self, ctx: &mut ExtractContext) {
        let registry = ctx.assets.sss_registry.read();
        if self.last_registry_version != registry.version {
            ctx.queue.write_buffer(
                self.profiles_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&registry.buffer_data),
            );
            self.last_registry_version = registry.version;
        }
    }

    fn ensure_pipelines(&mut self, ctx: &mut ExtractContext) {
        if self.horizontal_pipeline.is_some() {
            return;
        }

        let device = ctx.device;

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

        // Shared stencil / depth / color config
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

        // Horizontal pipeline
        let shader_defines = ShaderCompilationOptions::default();
        let (hor_shader, hor_hash) = ctx.shader_manager.get_or_compile_template(
            device,
            "passes/ssss",
            &shader_defines,
            "",
            "",
        );
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

        // Vertical pipeline (SSSSS_VERTICAL_PASS = 1)
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
}

// =============================================================================
// Transient Pass Node
// =============================================================================

struct RdgSssssPassNode {
    // RDG resource slots
    scene_color: TextureNodeId,
    temp_blur: TextureNodeId,
    depth_in: TextureNodeId,
    normal_in: TextureNodeId,
    feature_id: TextureNodeId,
    specular_tex: TextureNodeId,

    // Cloned from Feature
    horizontal_pipeline: RenderPipelineId,
    vertical_pipeline: RenderPipelineId,
    bind_group_layout: Tracked<wgpu::BindGroupLayout>,
    profiles_buffer: Tracked<wgpu::Buffer>,

    // Built during prepare()
    horizontal_bind_group: Option<wgpu::BindGroup>,
    vertical_bind_group: Option<wgpu::BindGroup>,
}

impl PassNode for RdgSssssPassNode {
    fn name(&self) -> &'static str {
        "RDG_SSSSS_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        builder.write_texture(self.temp_blur);
        builder.write_texture(self.scene_color);
        builder.read_texture(self.scene_color);
        builder.read_texture(self.depth_in);
        builder.read_texture(self.normal_in);
        builder.read_texture(self.feature_id);
        builder.read_texture(self.specular_tex);
    }

    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {
        // Depth-only sub-view for sampling
        let depth_sub_key = SubViewKey {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        };
        ctx.views
            .get_or_create_sub_view(self.depth_in, depth_sub_key.clone());

        let depth_only_view = ctx
            .views
            .get_sub_view(self.depth_in, &depth_sub_key)
            .expect("SSSSS: depth-only view must exist");

        let color_in_view = ctx.views.get_texture_view(self.scene_color);
        let normal_view = ctx.views.get_texture_view(self.normal_in);
        let feature_view = ctx.views.get_texture_view(self.feature_id);
        let temp_blur_view = ctx.views.get_texture_view(self.temp_blur);
        let specular_view = ctx.views.get_texture_view(self.specular_tex);

        let layout = &self.bind_group_layout;
        let sampler = ctx.sampler_registry.get_common(CommonSampler::LinearClamp);
        let profiles_buffer = &self.profiles_buffer;

        // Horizontal: reads scene_color → writes temp_blur
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
                label: Some("SSSSS Horizontal BG"),
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

        // Vertical: reads temp_blur → writes scene_color
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
                label: Some("SSSSS Vertical BG"),
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
        let depth_stencil_view = ctx.get_texture_view(self.depth_in);
        let temp_blur_view = ctx.get_texture_view(self.temp_blur);
        let color_out_view = ctx.get_texture_view(self.scene_color);

        let hor_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.horizontal_pipeline);
        let vert_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.vertical_pipeline);

        // H Sub-Pass: scene_color → temp_blur
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

        // V Sub-Pass: temp_blur → scene_color
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
