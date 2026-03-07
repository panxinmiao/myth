//! Skybox Feature & Transient Pass Node
//!
//! Renders the scene background (gradient, cubemap, equirectangular, or
//! planar texture) as a fullscreen triangle at the far depth plane.
//!
//! - [`SkyboxFeature`] (persistent): layouts, pipeline cache, bind groups
//!   (reference only persistent asset textures / uniform buffers).
//! - [`RdgSkyboxPassNode`] (transient): holds scene_color / scene_depth
//!   node IDs and the resolved pipeline + bind group for execution.
//!
//! # Special Case
//!
//! The skybox bind group references **only** persistent resources (uniform
//! buffer + asset texture + sampler), so it is created in
//! [`extract_and_prepare`] rather than the transient prepare phase. The
//! [`PreparedSkyboxDraw`] is stored for BasicForward inline rendering.

use rustc_hash::FxHashMap;

use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::resources::SamplerKey;
use crate::renderer::core::resources::Tracked;
use crate::renderer::graph::frame::PreparedSkyboxDraw;
use crate::renderer::graph::rdg::builder::PassBuilder;
use crate::renderer::graph::rdg::context::RdgExecuteContext;
use crate::renderer::graph::rdg::feature::{ExtractContext, SkyboxConfig};
use crate::renderer::graph::rdg::graph::RenderGraph;
use crate::renderer::graph::rdg::node::PassNode;
use crate::renderer::graph::rdg::types::TextureNodeId;
use crate::renderer::pipeline::{
    ColorTargetKey, DepthStencilKey, FullscreenPipelineKey, MultisampleKey, RenderPipelineId,
    ShaderCompilationOptions,
};
use crate::resources::shader_defines::ShaderDefines;
use crate::resources::texture::TextureSource;
use crate::resources::uniforms::WgslStruct;
use crate::scene::background::{BackgroundMapping, BackgroundMode, SkyboxParamsUniforms};

const SKYBOX_SAMPLER_KEY: SamplerKey = SamplerKey {
    address_mode_u: wgpu::AddressMode::Repeat,
    address_mode_v: wgpu::AddressMode::ClampToEdge,
    address_mode_w: wgpu::AddressMode::ClampToEdge,
    mag_filter: wgpu::FilterMode::Linear,
    min_filter: wgpu::FilterMode::Linear,
    mipmap_filter: wgpu::MipmapFilterMode::Linear,
    lod_min_clamp: 0.0,
    lod_max_clamp: 32.0,
    compare: None,
    anisotropy_clamp: 1,
    border_color: None,
};

// ─── Pipeline Variant Key ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SkyboxPipelineKey {
    variant: SkyboxVariant,
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SkyboxVariant {
    Gradient,
    Cube,
    Equirectangular,
    Planar,
}

impl SkyboxVariant {
    fn from_background(mode: &BackgroundMode) -> Option<Self> {
        match mode {
            BackgroundMode::Color(_) => None,
            BackgroundMode::Gradient { .. } => Some(Self::Gradient),
            BackgroundMode::Texture { mapping, .. } => match mapping {
                BackgroundMapping::Cube => Some(Self::Cube),
                BackgroundMapping::Equirectangular => Some(Self::Equirectangular),
                BackgroundMapping::Planar => Some(Self::Planar),
            },
        }
    }

    fn shader_define_key(self) -> &'static str {
        match self {
            Self::Gradient => "SKYBOX_GRADIENT",
            Self::Cube => "SKYBOX_CUBE",
            Self::Equirectangular => "SKYBOX_EQUIRECT",
            Self::Planar => "SKYBOX_PLANAR",
        }
    }

    fn needs_texture(self) -> bool {
        !matches!(self, Self::Gradient)
    }
}

// ─── Layout Helpers ───────────────────────────────────────────────────────────

fn create_uniform_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("RDG Skybox Layout (NoTex)"),
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
    })
}

fn create_texture_layout(
    device: &wgpu::Device,
    view_dimension: wgpu::TextureViewDimension,
    label: &str,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

// =============================================================================
// Persistent Feature
// =============================================================================

/// Persistent skybox Feature — owns layouts, pipeline cache, and creates
/// bind groups that reference only persistent resources (asset textures,
/// uniform buffers).
pub struct SkyboxFeature {
    layout_gradient: Option<Tracked<wgpu::BindGroupLayout>>,
    layout_cube: Option<Tracked<wgpu::BindGroupLayout>>,
    layout_2d: Option<Tracked<wgpu::BindGroupLayout>>,
    local_cache: FxHashMap<SkyboxPipelineKey, RenderPipelineId>,

    // Resolved at extract_and_prepare() time:
    current_bind_group: Option<wgpu::BindGroup>,
    current_pipeline: Option<RenderPipelineId>,
}

impl SkyboxFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            layout_gradient: None,
            layout_cube: None,
            layout_2d: None,
            local_cache: FxHashMap::default(),
            current_bind_group: None,
            current_pipeline: None,
        }
    }

    /// Prepare persistent GPU resources, compile pipeline, build bind group.
    ///
    /// The skybox bind group references only persistent resources (uniform
    /// buffer + asset texture + sampler), so it is safe to build here.
    pub fn extract_and_prepare(&mut self, ctx: &mut ExtractContext, config: &SkyboxConfig) {
        self.ensure_layouts(ctx.device);

        let Some(variant) = SkyboxVariant::from_background(&config.background_mode) else {
            self.current_bind_group = None;
            self.current_pipeline = None;
            return;
        };

        let params_gpu_id = config.bg_uniforms_gpu_id;
        let params_cpu_id = config.bg_uniforms_cpu_id;

        if let BackgroundMode::Texture {
            source: TextureSource::Asset(handle),
            ..
        } = &config.background_mode
        {
            ctx.resource_manager.prepare_texture(ctx.assets, *handle);
        }

        ctx.sampler_registry
            .get_custom(ctx.device, SKYBOX_SAMPLER_KEY);

        let texture_view = if let BackgroundMode::Texture {
            source, mapping, ..
        } = &config.background_mode
        {
            Self::resolve_texture_view(ctx.resource_manager, source, *mapping)
        } else {
            None
        };

        let layout = self.layout_for_variant(variant);
        let layout_id = layout.id();
        let sampler = ctx.sampler_registry.get_custom_ref(&SKYBOX_SAMPLER_KEY);

        let bind_group = if variant.needs_texture() {
            let Some(tex_view) = texture_view else {
                self.current_bind_group = None;
                self.current_pipeline = None;
                return;
            };

            let tex_view_key = std::ptr::from_ref::<wgpu::TextureView>(tex_view) as u64;
            let key = BindGroupKey::new(layout_id)
                .with_resource(params_gpu_id)
                .with_resource(tex_view_key)
                .with_resource(sampler.id());

            if let Some(cached) = ctx.global_bind_group_cache.get(&key) {
                cached.clone()
            } else {
                let params_gpu = ctx
                    .resource_manager
                    .gpu_buffers
                    .get(&params_cpu_id)
                    .expect("Skybox params GPU buffer must exist");

                let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RDG Skybox BG (Texture)"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: params_gpu.buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(tex_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                    ],
                });
                ctx.global_bind_group_cache.insert(key, bg.clone());
                bg
            }
        } else {
            let key = BindGroupKey::new(layout_id).with_resource(params_gpu_id);

            if let Some(cached) = ctx.global_bind_group_cache.get(&key) {
                cached.clone()
            } else {
                let params_gpu = ctx
                    .resource_manager
                    .gpu_buffers
                    .get(&params_cpu_id)
                    .expect("Skybox params GPU buffer must exist");

                let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RDG Skybox BG (Gradient)"),
                    layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_gpu.buffer.as_entire_binding(),
                    }],
                });
                ctx.global_bind_group_cache.insert(key, bg.clone());
                bg
            }
        };

        self.current_bind_group = Some(bind_group.clone());

        let pipeline_key = SkyboxPipelineKey {
            variant,
            color_format: config.color_format,
            depth_format: config.depth_format,
        };
        self.current_pipeline = Some(self.get_or_create_pipeline(ctx, pipeline_key));

        // Store for BasicForward inline rendering.
        if let (Some(pipeline_id), Some(bg)) = (self.current_pipeline, &self.current_bind_group) {
            ctx.render_lists.prepared_skybox = Some(PreparedSkyboxDraw {
                pipeline_id,
                bind_group: bg.clone(),
            });
        }
    }

    /// Build and inject a transient skybox node into the render graph.
    pub fn add_to_graph(
        &self,
        rdg: &mut RenderGraph,
        scene_color: TextureNodeId,
        scene_depth: TextureNodeId,
    ) {
        let node = Box::new(RdgSkyboxPassNode {
            scene_color,
            scene_depth,
            bind_group: self.current_bind_group.clone(),
            pipeline_id: self.current_pipeline,
        });
        rdg.add_pass_owned(node);
    }

    // =========================================================================
    // Internals
    // =========================================================================

    fn ensure_layouts(&mut self, device: &wgpu::Device) {
        if self.layout_gradient.is_some() {
            return;
        }
        self.layout_gradient = Some(Tracked::new(create_uniform_layout(device)));
        self.layout_cube = Some(Tracked::new(create_texture_layout(
            device,
            wgpu::TextureViewDimension::Cube,
            "RDG Skybox Layout (Cube)",
        )));
        self.layout_2d = Some(Tracked::new(create_texture_layout(
            device,
            wgpu::TextureViewDimension::D2,
            "RDG Skybox Layout (2D)",
        )));
    }

    fn layout_for_variant(&self, variant: SkyboxVariant) -> &Tracked<wgpu::BindGroupLayout> {
        match variant {
            SkyboxVariant::Gradient => self.layout_gradient.as_ref().unwrap(),
            SkyboxVariant::Cube => self.layout_cube.as_ref().unwrap(),
            SkyboxVariant::Equirectangular | SkyboxVariant::Planar => {
                self.layout_2d.as_ref().unwrap()
            }
        }
    }

    fn get_or_create_pipeline(
        &mut self,
        ctx: &mut ExtractContext,
        key: SkyboxPipelineKey,
    ) -> RenderPipelineId {
        if let Some(&pipeline_id) = self.local_cache.get(&key) {
            return pipeline_id;
        }

        let gpu_world = ctx
            .resource_manager
            .get_global_state(ctx.render_state.id, ctx.extracted_scene.scene_id)
            .expect("Global state must exist");

        let mut defines = ShaderDefines::new();
        defines.set(key.variant.shader_define_key(), "1");

        let mut options = ShaderCompilationOptions { defines };
        options.add_define(
            "struct_definitions",
            SkyboxParamsUniforms::wgsl_struct_def("SkyboxParams").as_str(),
        );

        let (shader_module, shader_hash) = ctx.shader_manager.get_or_compile_template(
            ctx.device,
            "passes/skybox",
            &options,
            "",
            &gpu_world.binding_wgsl,
        );

        let layout = self.layout_for_variant(key.variant);
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RDG Skybox Pipeline Layout"),
                bind_group_layouts: &[&gpu_world.layout, layout],
                immediate_size: 0,
            });

        let fullscreen_key = FullscreenPipelineKey {
            shader_hash,
            color_targets: smallvec::smallvec![ColorTargetKey::from(wgpu::ColorTargetState {
                format: key.color_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            depth_stencil: Some(DepthStencilKey::from(wgpu::DepthStencilState {
                format: key.depth_format,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::GreaterEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            })),
            multisample: MultisampleKey::from(wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            }),
        };

        let pipeline_id = ctx.pipeline_cache.get_or_create_fullscreen(
            ctx.device,
            shader_module,
            &pipeline_layout,
            &fullscreen_key,
            "RDG Skybox Pipeline",
        );

        self.local_cache.insert(key, pipeline_id);
        pipeline_id
    }

    fn resolve_texture_view<'a>(
        resource_manager: &'a crate::renderer::core::ResourceManager,
        source: &TextureSource,
        mapping: BackgroundMapping,
    ) -> Option<&'a wgpu::TextureView> {
        match source {
            TextureSource::Asset(handle) => {
                if let Some(binding) = resource_manager.texture_bindings.get(*handle)
                    && let Some(img) = resource_manager.gpu_images.get(&binding.cpu_image_id)
                {
                    match mapping {
                        BackgroundMapping::Cube => {
                            if img.default_view_dimension == wgpu::TextureViewDimension::Cube {
                                return Some(&img.default_view);
                            }
                        }
                        BackgroundMapping::Equirectangular | BackgroundMapping::Planar => {
                            if img.default_view_dimension == wgpu::TextureViewDimension::D2 {
                                return Some(&img.default_view);
                            }
                        }
                    }
                }
                None
            }
            TextureSource::Attachment(id, _) => resource_manager.internal_resources.get(id),
        }
    }
}

// =============================================================================
// Transient Pass Node
// =============================================================================

struct RdgSkyboxPassNode {
    scene_color: TextureNodeId,
    scene_depth: TextureNodeId,
    bind_group: Option<wgpu::BindGroup>,
    pipeline_id: Option<RenderPipelineId>,
}

impl PassNode for RdgSkyboxPassNode {
    fn name(&self) -> &'static str {
        "RDG_Skybox_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        builder.write_texture(self.scene_color);
        builder.read_texture(self.scene_depth);
    }

    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let (Some(pipeline_id), Some(bind_group)) =
            (self.pipeline_id, &self.bind_group)
        else {
            return;
        };

        let render_lists = &ctx.render_lists;
        let Some(gpu_global_bind_group) = &render_lists.gpu_global_bind_group else {
            return;
        };

        let color_view = ctx.get_texture_view(self.scene_color);
        let depth_view = ctx.get_texture_view(self.scene_depth);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RDG Skybox Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        let pipeline = ctx.pipeline_cache.get_render_pipeline(pipeline_id);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, gpu_global_bind_group, &[]);
        pass.set_bind_group(1, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
