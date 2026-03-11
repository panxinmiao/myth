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
//! # Push Parameters (set by Composer)
//!
//! All scene-level configuration is pushed into the public fields by the
//! Composer before the RDG prepare loop. The pass itself never accesses
//! `Scene` directly.
//!
//! - `background_mode`: Background rendering mode (gradient, texture, etc.)
//! - `bg_uniforms_cpu_id`: CPU buffer ID for `CpuBuffer<SkyboxParamsUniforms>`
//! - `bg_uniforms_gpu_id`: GPU buffer ID (from `ensure_buffer_id`)
//! - `scene_id`: Scene unique ID for global state lookup

use rustc_hash::FxHashMap;

use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::gpu::SamplerKey;
use crate::renderer::core::gpu::Tracked;
use crate::renderer::graph::core::{
    ExecuteContext, ExtractContext, PassBuilder, PassNode, RenderGraph, TextureNodeId,
};
use crate::renderer::pipeline::{
    ColorTargetKey, DepthStencilKey, FullscreenPipelineKey, MultisampleKey, RenderPipelineId,
    ShaderCompilationOptions,
};
use crate::resources::buffer::CpuBuffer;
use crate::resources::shader_defines::ShaderDefines;
use crate::resources::texture::TextureSource;
use crate::resources::uniforms::WgslStruct;
use crate::scene::background::{BackgroundMapping, BackgroundMode, SkyboxParamsUniforms};

/// Sampler key for the skybox environment map: trilinear filtering with
/// horizontal repeat (seamless panorama wrap) and vertical/depth clamp.
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
    msaa_samples: u32,
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
        label: Some("Skybox Layout (NoTex)"),
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

/// Skybox / Background rendering feature.
///
/// Owns persistent GPU state (layouts, pipeline cache, bind groups) and
/// produces an ephemeral [`SkyboxPassNode`] each frame via [`Self::add_to_graph`].
pub struct SkyboxFeature {
    // ─── RDG Resource Slots ────────────────────────────────────────
    pub scene_color: TextureNodeId,
    pub scene_depth: TextureNodeId,

    // ─── Bind Group Layouts (Group 1) ──────────────────────────────
    layout_gradient: Option<Tracked<wgpu::BindGroupLayout>>,
    layout_cube: Option<Tracked<wgpu::BindGroupLayout>>,
    layout_2d: Option<Tracked<wgpu::BindGroupLayout>>,

    // ─── Pipeline Cache ────────────────────────────────────────────
    local_cache: FxHashMap<SkyboxPipelineKey, RenderPipelineId>,

    // ─── Runtime State ─────────────────────────────────────────────
    pub(crate) current_bind_group: Option<wgpu::BindGroup>,
    pub(crate) current_pipeline: Option<RenderPipelineId>,
}

impl Default for SkyboxFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl SkyboxFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scene_color: TextureNodeId(0),
            scene_depth: TextureNodeId(0),
            layout_gradient: None,
            layout_cube: None,
            layout_2d: None,
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
            "Skybox Layout (Cube)",
        )));
        self.layout_2d = Some(Tracked::new(create_texture_layout(
            device,
            wgpu::TextureViewDimension::D2,
            "Skybox Layout (2D)",
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
        global_state_key: (u32, u32),
    ) -> RenderPipelineId {
        if let Some(&pipeline_id) = self.local_cache.get(&key) {
            return pipeline_id;
        }

        let gpu_world = ctx
            .resource_manager
            .get_global_state(global_state_key.0, global_state_key.1)
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
                label: Some("Skybox Pipeline Layout"),
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
                count: key.msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            }),
        };

        let pipeline_id = ctx.pipeline_cache.get_or_create_fullscreen(
            ctx.device,
            shader_module,
            &pipeline_layout,
            &fullscreen_key,
            "Skybox Pipeline",
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

    /// Extract scene data and prepare GPU resources for skybox rendering.
    ///
    /// Called **before** the render graph is built. Caches bind groups and
    /// pipelines so the ephemeral [`SkyboxPassNode`] only carries lightweight IDs.
    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        background_mode: &BackgroundMode,
        bg_uniforms: &CpuBuffer<SkyboxParamsUniforms>,
        global_state_key: (u32, u32),
        color_format: wgpu::TextureFormat,
    ) {
        self.ensure_layouts(ctx.device);

        let Some(variant) = SkyboxVariant::from_background(background_mode) else {
            self.current_bind_group = None;
            self.current_pipeline = None;
            return;
        };

        // GPU buffer was already ensured by the Composer; use pushed IDs.
        let bg_uniforms_id = ctx.resource_manager.ensure_buffer_id(bg_uniforms);

        if let BackgroundMode::Texture {
            source: TextureSource::Asset(handle),
            ..
        } = background_mode
        {
            ctx.resource_manager.prepare_texture(ctx.assets, *handle);
        }

        // Ensure the custom sampler is created (first frame only; subsequent
        // frames are a no-op HashMap lookup). The mutable borrow is released
        // before we resolve the texture view below.
        ctx.sampler_registry
            .get_custom(ctx.device, SKYBOX_SAMPLER_KEY);

        // Resolve texture view
        let texture_view = if let BackgroundMode::Texture {
            source, mapping, ..
        } = background_mode
        {
            Self::resolve_texture_view(ctx.resource_manager, source, *mapping)
        } else {
            None
        };

        // Build bind group (group 1)
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
                .with_resource(bg_uniforms_id)
                .with_resource(tex_view_key)
                .with_resource(sampler.id());

            if let Some(cached) = ctx.global_bind_group_cache.get(&key) {
                cached.clone()
            } else {
                let params_gpu = ctx
                    .resource_manager
                    .gpu_buffers
                    .get(&bg_uniforms.id())
                    .expect("Skybox params GPU buffer must exist");

                let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Skybox BG (Texture)"),
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
            let key = BindGroupKey::new(layout_id).with_resource(bg_uniforms_id);

            if let Some(cached) = ctx.global_bind_group_cache.get(&key) {
                cached.clone()
            } else {
                let params_gpu = ctx
                    .resource_manager
                    .gpu_buffers
                    .get(&bg_uniforms.id())
                    .expect("Skybox params GPU buffer must exist");

                let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Skybox BG (Gradient)"),
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

        // Pipeline — uses pushed format fields instead of reading from the graph.
        let pipeline_key = SkyboxPipelineKey {
            variant,
            color_format,
            depth_format: ctx.wgpu_ctx.depth_format,
            msaa_samples: ctx.wgpu_ctx.msaa_samples,
        };
        self.current_pipeline =
            Some(self.get_or_create_pipeline(ctx, pipeline_key, global_state_key));
    }

    /// Create an ephemeral [`SkyboxPassNode`] and add it to the render graph.
    /// Build the ephemeral pass node and insert it into the graph.
    ///
    /// Creates an SSA alias of `scene_color` so that the dependency
    /// Opaque → Skybox is locked by graph edges, not by registration
    /// order.  Returns the new colour version for downstream threading.
    pub fn add_to_graph(
        &self,
        graph: &mut RenderGraph,
        scene_color: TextureNodeId,
        scene_depth: TextureNodeId,
    ) -> TextureNodeId {
        let color_output = graph.create_alias(scene_color, "Scene_Color_Skybox");
        let node = SkyboxPassNode {
            in_color: scene_color,
            out_color: color_output,
            scene_depth,
            pipeline_id: self.current_pipeline,
            bind_group: self.current_bind_group.clone(),
        };
        graph.add_pass(Box::new(node));
        color_output
    }
}

// ─── Skybox Pass Node ─────────────────────────────────────────────────────────

/// Ephemeral per-frame skybox render pass node.
pub struct SkyboxPassNode {
    /// Previous colour version (read dependency).
    in_color: TextureNodeId,
    /// New colour version — SSA alias of `in_color` (write dependency).
    out_color: TextureNodeId,
    scene_depth: TextureNodeId,
    pipeline_id: Option<RenderPipelineId>,
    bind_group: Option<wgpu::BindGroup>,
}

impl PassNode for SkyboxPassNode {
    fn name(&self) -> &'static str {
        "Skybox_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        builder.read_texture(self.in_color);
        builder.declare_output(self.out_color);
        builder.read_texture(self.scene_depth);
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let (Some(pipeline_id), Some(bind_group)) = (self.pipeline_id, &self.bind_group) else {
            return;
        };

        let gpu_global_bind_group = ctx.baked_lists.global_bind_group;

        let color_att = ctx.get_color_attachment(self.out_color, None, None);
        let depth_att = ctx.get_depth_stencil_attachment(self.scene_depth, 0.0);

        let pass_desc = wgpu::RenderPassDescriptor {
            label: Some("Skybox Pass"),
            color_attachments: &[color_att],
            depth_stencil_attachment: depth_att,
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
