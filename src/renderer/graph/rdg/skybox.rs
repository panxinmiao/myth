//! RDG Skybox / Background Render Pass
//!
//! Renders the scene background (gradient, cubemap, equirectangular, or planar
//! texture) as a fullscreen triangle at the far depth plane. Uses Reverse-Z
//! depth testing (`GreaterEqual`) so opaque geometry masks the background.
//!
//! # RDG Slots
//!
//! - `scene_color`: HDR color buffer (read + write, LoadOp::Load)
//! - `scene_depth`: Depth buffer (read, LoadOp::Load)
//!
//! # Push Parameters
//!
//! Pipeline variant and textures are resolved from `scene.background` during prepare.

use rustc_hash::FxHashMap;

use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::resources::Tracked;
use crate::renderer::graph::frame::PreparedSkyboxDraw;
use crate::renderer::graph::rdg::builder::PassBuilder;
use crate::renderer::graph::rdg::context::{RdgExecuteContext, RdgPrepareContext};
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

// ─── RDG Skybox Pass ──────────────────────────────────────────────────────────

/// RDG Skybox / Background render pass.
///
/// Renders the scene background using the global bind group (group 0) for
/// camera data and its own bind group (group 1) for skybox-specific params.
pub struct RdgSkyboxPass {
    // ─── RDG Resource Slots ────────────────────────────────────────
    pub scene_color: TextureNodeId,
    pub scene_depth: TextureNodeId,

    // ─── Bind Group Layouts (Group 1) ──────────────────────────────
    layout_gradient: Option<Tracked<wgpu::BindGroupLayout>>,
    layout_cube: Option<Tracked<wgpu::BindGroupLayout>>,
    layout_2d: Option<Tracked<wgpu::BindGroupLayout>>,

    // ─── Sampler ───────────────────────────────────────────────────
    sampler: Option<Tracked<wgpu::Sampler>>,

    // ─── Pipeline Cache ────────────────────────────────────────────
    local_cache: FxHashMap<SkyboxPipelineKey, RenderPipelineId>,

    // ─── Runtime State ─────────────────────────────────────────────
    current_bind_group: Option<wgpu::BindGroup>,
    current_pipeline: Option<RenderPipelineId>,
}

impl RdgSkyboxPass {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scene_color: TextureNodeId(0),
            scene_depth: TextureNodeId(0),
            layout_gradient: None,
            layout_cube: None,
            layout_2d: None,
            sampler: None,
            local_cache: FxHashMap::default(),
            current_bind_group: None,
            current_pipeline: None,
        }
    }

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

        self.sampler = Some(Tracked::new(device.create_sampler(
            &wgpu::SamplerDescriptor {
                label: Some("RDG Skybox Sampler"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            },
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
        ctx: &mut RdgPrepareContext,
        key: SkyboxPipelineKey,
    ) -> RenderPipelineId {
        if let Some(&pipeline_id) = self.local_cache.get(&key) {
            return pipeline_id;
        }

        let gpu_world = ctx
            .resource_manager
            .get_global_state(ctx.render_state.id, ctx.scene.id())
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
                count: 1, // HighFidelity always sample_count = 1
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
        ctx: &'a RdgPrepareContext,
        source: &TextureSource,
        mapping: BackgroundMapping,
    ) -> Option<&'a wgpu::TextureView> {
        match source {
            TextureSource::Asset(handle) => {
                if let Some(binding) = ctx.resource_manager.texture_bindings.get(*handle)
                    && let Some(img) = ctx.resource_manager.gpu_images.get(&binding.cpu_image_id)
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
            TextureSource::Attachment(id, _) => ctx.resource_manager.internal_resources.get(id),
        }
    }
}

impl PassNode for RdgSkyboxPass {
    fn name(&self) -> &'static str {
        "RDG_Skybox_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        builder.read_texture(self.scene_color);
        builder.write_texture(self.scene_color);
        builder.read_texture(self.scene_depth);
    }

    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {
        self.ensure_layouts(ctx.device);

        let background = &ctx.scene.background;

        let Some(variant) = SkyboxVariant::from_background(&background.mode) else {
            self.current_bind_group = None;
            self.current_pipeline = None;
            return;
        };

        // Ensure GPU buffer for skybox params
        let params_gpu_id = ctx.resource_manager.ensure_buffer_id(&background.uniforms);
        let params_cpu_id = background.uniforms.id();

        if let BackgroundMode::Texture {
            source: TextureSource::Asset(handle),
            ..
        } = &background.mode
        {
            ctx.resource_manager
                .prepare_texture(&ctx.scene.assets, *handle);
        }

        // Resolve texture view
        let texture_view = if let BackgroundMode::Texture {
            source, mapping, ..
        } = &background.mode
        {
            Self::resolve_texture_view(ctx, source, *mapping)
        } else {
            None
        };

        // Build bind group (group 1)
        let layout = self.layout_for_variant(variant);
        let layout_id = layout.id();
        let sampler = self.sampler.as_ref().unwrap();

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

                let bg = ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
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

                let bg = ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
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

        // Pipeline
        let color_format = ctx.graph.resources[self.scene_color.0 as usize].desc.format;
        let pipeline_key = SkyboxPipelineKey {
            variant,
            color_format,
            depth_format: ctx.wgpu_ctx.depth_format,
        };
        self.current_pipeline = Some(self.get_or_create_pipeline(ctx, pipeline_key));

        // Store prepared draw state for LDR path inline rendering
        if let (Some(pipeline_id), Some(bg)) = (self.current_pipeline, &self.current_bind_group) {
            ctx.render_lists.prepared_skybox = Some(PreparedSkyboxDraw {
                pipeline_id,
                bind_group: bg.clone(),
            });
        }
    }

    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let (Some(pipeline_id), Some(bind_group)) =
            (self.current_pipeline, &self.current_bind_group)
        else {
            return;
        };

        let render_lists = &ctx.render_lists;
        let Some(gpu_global_bind_group) = &render_lists.gpu_global_bind_group else {
            return;
        };

        let color_view = ctx.get_texture_view(self.scene_color);
        let depth_view = ctx.get_texture_view(self.scene_depth);

        let pass_desc = wgpu::RenderPassDescriptor {
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
        };

        let mut pass = encoder.begin_render_pass(&pass_desc);

        let pipeline = ctx.pipeline_cache.get_render_pipeline(pipeline_id);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, gpu_global_bind_group, &[]);
        pass.set_bind_group(1, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
