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

use crate::core::gpu::{ResourceState, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    ExecuteContext, ExtractContext, PassNode, RawBufferBinding, RenderTargetOps, TextureNodeId,
};
use crate::graph::passes::atmosphere::ProceduralSkyboxResources;
use crate::pipeline::{
    ColorTargetKey, DepthStencilKey, FullscreenPipelineKey, MultisampleKey, RenderPipelineId,
    ShaderCompilationOptions, ShaderSource,
};
use myth_resources::buffer::CpuBuffer;
use myth_resources::shader_defines::ShaderDefines;
use myth_resources::texture::TextureSampler;
use myth_resources::texture::TextureSource;
use myth_resources::uniforms::WgslStruct;
use myth_scene::background::{BackgroundMapping, BackgroundMode, SkyboxParamsUniforms};

/// Sampler key for the skybox environment map: trilinear filtering with
/// horizontal repeat (seamless panorama wrap) and vertical/depth clamp.
const SKYBOX_SAMPLER_KEY: TextureSampler = TextureSampler {
    address_mode_u: wgpu::AddressMode::Repeat,
    address_mode_v: wgpu::AddressMode::ClampToEdge,
    address_mode_w: wgpu::AddressMode::ClampToEdge,
    mag_filter: wgpu::FilterMode::Linear,
    min_filter: wgpu::FilterMode::Linear,
    mipmap_filter: wgpu::MipmapFilterMode::Linear,
    lod_min_clamp: 0.0,
    lod_max_clamp: 32.0,
    compare: None,
    anisotropy_clamp: Some(1),
    border_color: None,
};

// ─── Pipeline Variant Key ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SkyboxPipelineKey {
    variant: SkyboxVariant,
    procedural_starbox: ProceduralStarboxKind,
    procedural_moon_texture: bool,
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    msaa_samples: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ProceduralStarboxKind {
    None,
    Equirectangular,
    Cube,
}

impl ProceduralStarboxKind {
    fn from_view_dimension(view_dimension: wgpu::TextureViewDimension) -> Option<Self> {
        match view_dimension {
            wgpu::TextureViewDimension::D2 => Some(Self::Equirectangular),
            wgpu::TextureViewDimension::Cube => Some(Self::Cube),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SkyboxVariant {
    Gradient,
    Cube,
    Equirectangular,
    Planar,
    Procedural,
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
            BackgroundMode::Procedural(_) => Some(Self::Procedural),
        }
    }

    fn shader_define_key(self) -> &'static str {
        match self {
            Self::Gradient => "SKYBOX_GRADIENT",
            Self::Cube => "SKYBOX_CUBE",
            Self::Equirectangular => "SKYBOX_EQUIRECT",
            Self::Planar => "SKYBOX_PLANAR",
            Self::Procedural => "SKYBOX_PROCEDURAL",
        }
    }

    fn needs_texture(self) -> bool {
        !matches!(self, Self::Gradient)
    }

    fn apply_shader_defines(
        self,
        defines: &mut ShaderDefines,
        starbox_kind: ProceduralStarboxKind,
        moon_texture: bool,
    ) {
        defines.set(self.shader_define_key(), "1");
        if self == Self::Procedural {
            match starbox_kind {
                ProceduralStarboxKind::None => {}
                ProceduralStarboxKind::Equirectangular => {
                    defines.set("CELESTIAL_STARBOX_EQUIRECT", "1");
                }
                ProceduralStarboxKind::Cube => {
                    defines.set("CELESTIAL_STARBOX_CUBE", "1");
                }
            }
            if moon_texture {
                defines.set("USE_MOON_TEXTURE", "1");
            }
        }
    }
}

#[derive(Clone, Copy)]
struct ResolvedTextureView<'a> {
    view: &'a wgpu::TextureView,
    view_dimension: wgpu::TextureViewDimension,
    resource_key: u64,
}

#[derive(Clone, Copy)]
struct ProceduralStarboxView<'a> {
    view: &'a wgpu::TextureView,
    kind: ProceduralStarboxKind,
    resource_key: u64,
}

#[derive(Clone, Copy)]
struct ProceduralMoonView<'a> {
    view: &'a wgpu::TextureView,
    resource_key: u64,
    enabled: bool,
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

fn create_procedural_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Skybox Layout (Procedural)"),
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
                    view_dimension: wgpu::TextureViewDimension::D2,
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
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

fn create_procedural_texture_layout(
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
                    view_dimension: wgpu::TextureViewDimension::D2,
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
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension,
                    multisampled: false,
                },
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
    layout_procedural: Option<Tracked<wgpu::BindGroupLayout>>,
    layout_procedural_eq: Option<Tracked<wgpu::BindGroupLayout>>,
    layout_procedural_cube: Option<Tracked<wgpu::BindGroupLayout>>,

    procedural_sampler: Option<Tracked<wgpu::Sampler>>,

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
            scene_color: TextureNodeId::from_index(0),
            scene_depth: TextureNodeId::from_index(0),
            layout_gradient: None,
            layout_cube: None,
            layout_2d: None,
            layout_procedural: None,
            layout_procedural_eq: None,
            layout_procedural_cube: None,
            procedural_sampler: None,
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
        self.layout_procedural = Some(Tracked::new(create_procedural_layout(device)));
        self.layout_procedural_eq = Some(Tracked::new(create_procedural_texture_layout(
            device,
            wgpu::TextureViewDimension::D2,
            "Skybox Layout (Procedural Eq)",
        )));
        self.layout_procedural_cube = Some(Tracked::new(create_procedural_texture_layout(
            device,
            wgpu::TextureViewDimension::Cube,
            "Skybox Layout (Procedural Cube)",
        )));
        self.procedural_sampler = Some(Tracked::new(device.create_sampler(
            &wgpu::SamplerDescriptor {
                label: Some("Skybox Procedural Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
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
            },
        )));
    }

    fn layout_for_variant(
        &self,
        variant: SkyboxVariant,
        procedural_starbox: ProceduralStarboxKind,
    ) -> &Tracked<wgpu::BindGroupLayout> {
        match variant {
            SkyboxVariant::Gradient => self.layout_gradient.as_ref().unwrap(),
            SkyboxVariant::Cube => self.layout_cube.as_ref().unwrap(),
            SkyboxVariant::Equirectangular | SkyboxVariant::Planar => {
                self.layout_2d.as_ref().unwrap()
            }
            SkyboxVariant::Procedural => match procedural_starbox {
                ProceduralStarboxKind::None => self.layout_procedural.as_ref().unwrap(),
                ProceduralStarboxKind::Equirectangular => {
                    self.layout_procedural_eq.as_ref().unwrap()
                }
                ProceduralStarboxKind::Cube => self.layout_procedural_cube.as_ref().unwrap(),
            },
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
        key.variant.apply_shader_defines(
            &mut defines,
            key.procedural_starbox,
            key.procedural_moon_texture,
        );

        let mut options = ShaderCompilationOptions {
            defines,
            ..Default::default()
        };
        options.add_define(
            "struct_definitions",
            SkyboxParamsUniforms::wgsl_struct_def("SkyboxParams").as_str(),
        );
        options.inject_code("binding_code", &gpu_world.binding_wgsl);

        let (shader_module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/utility/skybox"),
            &options,
        );

        let layout = self.layout_for_variant(key.variant, key.procedural_starbox);
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Skybox Pipeline Layout"),
                bind_group_layouts: &[Some(&gpu_world.layout), Some(layout)],
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
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
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

    fn resolve_texture_view_with_dimension<'a>(
        resource_manager: &'a crate::core::ResourceManager,
        source: &TextureSource,
    ) -> Option<ResolvedTextureView<'a>> {
        match source {
            TextureSource::Asset(handle) => {
                let binding = resource_manager.texture_bindings.get(*handle)?;
                let img = resource_manager.gpu_images.get(binding.image_handle)?;
                Some(ResolvedTextureView {
                    view: &img.default_view,
                    view_dimension: img.default_view_dimension,
                    resource_key: binding.view_id,
                })
            }
            TextureSource::Attachment(id, dim) => {
                resource_manager
                    .internal_resources
                    .get(id)
                    .map(|view| ResolvedTextureView {
                        view,
                        view_dimension: *dim,
                        resource_key: *id,
                    })
            }
        }
    }

    fn resolve_texture_view<'a>(
        resource_manager: &'a crate::core::ResourceManager,
        source: &TextureSource,
        mapping: BackgroundMapping,
    ) -> Option<ResolvedTextureView<'a>> {
        let resolved = Self::resolve_texture_view_with_dimension(resource_manager, source)?;
        match mapping {
            BackgroundMapping::Cube
                if resolved.view_dimension == wgpu::TextureViewDimension::Cube =>
            {
                Some(resolved)
            }
            BackgroundMapping::Equirectangular | BackgroundMapping::Planar
                if resolved.view_dimension == wgpu::TextureViewDimension::D2 =>
            {
                Some(resolved)
            }
            _ => None,
        }
    }

    fn resolve_procedural_starbox_view<'a>(
        resource_manager: &'a crate::core::ResourceManager,
        source: &TextureSource,
    ) -> Option<ProceduralStarboxView<'a>> {
        let resolved = Self::resolve_texture_view_with_dimension(resource_manager, source)?;
        let kind = ProceduralStarboxKind::from_view_dimension(resolved.view_dimension)?;
        Some(ProceduralStarboxView {
            view: resolved.view,
            kind,
            resource_key: resolved.resource_key,
        })
    }

    fn resolve_procedural_moon_view(
        resource_manager: &crate::core::ResourceManager,
        source: Option<TextureSource>,
    ) -> ProceduralMoonView<'_> {
        source
            .and_then(|source| {
                let resolved =
                    Self::resolve_texture_view_with_dimension(resource_manager, &source)?;
                if resolved.view_dimension != wgpu::TextureViewDimension::D2 {
                    return None;
                }

                Some(ProceduralMoonView {
                    view: resolved.view,
                    resource_key: resolved.resource_key,
                    enabled: true,
                })
            })
            .unwrap_or(ProceduralMoonView {
                view: &resource_manager.system_textures.white_2d,
                resource_key: resource_manager.system_textures.white_2d.id(),
                enabled: false,
            })
    }

    /// Extract scene data and prepare GPU resources for skybox rendering.
    ///
    /// Called **before** the render graph is built. Caches bind groups and
    /// pipelines so the ephemeral [`SkyboxPassNode`] only carries lightweight IDs.
    pub(crate) fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        background_mode: &BackgroundMode,
        bg_uniforms: &CpuBuffer<SkyboxParamsUniforms>,
        global_state_key: (u32, u32),
        color_format: wgpu::TextureFormat,
        procedural_resources: Option<ProceduralSkyboxResources<'_>>,
    ) {
        self.ensure_layouts(ctx.device);

        let Some(variant) = SkyboxVariant::from_background(background_mode) else {
            self.current_bind_group = None;
            self.current_pipeline = None;
            return;
        };

        if let BackgroundMode::Texture {
            source: TextureSource::Asset(handle),
            ..
        } = background_mode
        {
            let state = ctx.resource_manager.prepare_texture(ctx.assets, *handle);
            if matches!(state, ResourceState::Pending)
                && self.current_bind_group.is_some()
                && self.current_pipeline.is_some()
            {
                // Texture is still loading, but we have a valid bind group and pipeline from the previous frame.
                // Keep rendering the skybox with the old texture until the new one is ready, instead of stalling the GPU with an empty bind group.
                return;
            }
        }

        if let BackgroundMode::Procedural(params) = background_mode
            && let Some(TextureSource::Asset(handle)) = params.starbox_texture
        {
            let state = ctx.resource_manager.prepare_texture(ctx.assets, handle);
            if matches!(state, ResourceState::Pending)
                && self.current_bind_group.is_some()
                && self.current_pipeline.is_some()
            {
                return;
            }
        }

        if let BackgroundMode::Procedural(params) = background_mode
            && let Some(TextureSource::Asset(handle)) = params.moon_albedo_texture
        {
            let state = ctx.resource_manager.prepare_texture(ctx.assets, handle);
            if matches!(state, ResourceState::Pending)
                && self.current_bind_group.is_some()
                && self.current_pipeline.is_some()
            {
                return;
            }
        }

        // Ensure the custom sampler is created (first frame only; subsequent
        // frames are a no-op HashMap lookup). The mutable borrow is released
        // before we resolve the texture view below.
        let sampler_id = ctx
            .resource_manager
            .sampler_registry
            .get_custom(ctx.device, &SKYBOX_SAMPLER_KEY)
            .0;

        // Resolve texture view
        let texture_view = match background_mode {
            BackgroundMode::Texture {
                source, mapping, ..
            } => Self::resolve_texture_view(ctx.resource_manager, source, *mapping)
                .map(|resolved| (resolved.view.clone(), resolved.resource_key)),
            _ => None,
        };
        let procedural_starbox = match background_mode {
            BackgroundMode::Procedural(params) => params.starbox_texture.and_then(|source| {
                Self::resolve_procedural_starbox_view(ctx.resource_manager, &source)
                    .map(|starbox| (starbox.view.clone(), starbox.kind, starbox.resource_key))
            }),
            _ => None,
        };
        let procedural_moon = match background_mode {
            BackgroundMode::Procedural(params) => {
                let moon =
                    Self::resolve_procedural_moon_view(ctx.resource_manager, params.moon_albedo_texture);
                Some((moon.view.clone(), moon.resource_key, moon.enabled))
            }
            _ => None,
        };
        let procedural_starbox_kind = procedural_starbox
            .as_ref()
            .map_or(ProceduralStarboxKind::None, |(_, kind, _)| *kind);
        let procedural_moon_enabled = procedural_moon
            .as_ref()
            .is_some_and(|(_, _, enabled)| *enabled);
        let procedural_starbox_binding = procedural_starbox.clone();
        let procedural_moon_binding = procedural_moon.clone();

        // Build bind group (group 1)
        let layout = self.layout_for_variant(variant, procedural_starbox_kind);
        let sampler = ctx
            .resource_manager
            .sampler_registry
            .get_sampler_by_index(sampler_id)
            .expect("Skybox sampler index must be valid")
            .clone();

        let bind_group = if variant == SkyboxVariant::Procedural {
            let Some(resources) = procedural_resources else {
                self.current_bind_group = None;
                self.current_pipeline = None;
                return;
            };
            let (moon_view, moon_resource_key, _) =
                procedural_moon_binding.expect("procedural moon view must exist");
            let sampler = self
                .procedural_sampler
                .as_ref()
                .expect("procedural skybox sampler must exist");

            let label = if procedural_starbox.is_some() {
                Some("Skybox BG (Procedural+Starbox)")
            } else {
                Some("Skybox BG (Procedural)")
            };

            let builder = ctx
                .build_bind_group(layout, label)
                .bind_tracked_buffer(0, resources.bake_params_buffer)
                .bind_tracked_texture_view(1, resources.sky_view_view)
                .bind_tracked_sampler(2, sampler)
                .bind_tracked_texture_view(3, resources.transmittance_view)
                .bind_texture_view_with_id(4, &moon_view, moon_resource_key);

            match procedural_starbox_binding {
                Some((starbox_view, _, starbox_resource_key)) => builder
                    .bind_texture_view_with_id(5, &starbox_view, starbox_resource_key)
                    .build(),
                None => builder.build(),
            }
            .clone()
        } else if variant.needs_texture() {
            let Some((tex_view, tex_view_resource_key)) = texture_view else {
                self.current_bind_group = None;
                self.current_pipeline = None;
                return;
            };

            let (params_buffer, params_buffer_id) = {
                let params_gpu = bg_uniforms
                    .gpu_handle()
                    .and_then(|h| ctx.resource_manager.gpu_buffers.get(h))
                    .expect("Skybox params GPU buffer must exist");
                (params_gpu.buffer.clone(), params_gpu.id)
            };

            ctx.build_bind_group(layout, Some("Skybox BG (Texture)"))
                .bind_raw_buffer(0, RawBufferBinding::new(&params_buffer, params_buffer_id, None))
                .bind_texture_view_with_id(1, &tex_view, tex_view_resource_key)
                .bind_sampler_with_id(2, &sampler, sampler_id as u64)
                .build()
                .clone()
        } else {
            let (params_buffer, params_buffer_id) = {
                let params_gpu = bg_uniforms
                    .gpu_handle()
                    .and_then(|h| ctx.resource_manager.gpu_buffers.get(h))
                    .expect("Skybox params GPU buffer must exist");
                (params_gpu.buffer.clone(), params_gpu.id)
            };

            ctx.build_bind_group(layout, Some("Skybox BG (Gradient)"))
                .bind_raw_buffer(0, RawBufferBinding::new(&params_buffer, params_buffer_id, None))
                .build()
                .clone()
        };

        self.current_bind_group = Some(bind_group.clone());

        // Pipeline — uses pushed format fields instead of reading from the graph.
        let pipeline_key = SkyboxPipelineKey {
            variant,
            procedural_starbox: if variant == SkyboxVariant::Procedural {
                procedural_starbox_kind
            } else {
                ProceduralStarboxKind::None
            },
            procedural_moon_texture: if variant == SkyboxVariant::Procedural {
                procedural_moon_enabled
            } else {
                false
            },
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
    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        scene_color: TextureNodeId,
        scene_depth: TextureNodeId,
        dependency_textures: [Option<TextureNodeId>; 2],
    ) -> TextureNodeId {
        let pipeline = self
            .current_pipeline
            .map(|id| ctx.pipeline_cache.get_render_pipeline(id));
        let bind_group = self.current_bind_group.as_ref();
        ctx.graph.add_pass("Skybox_Pass", |builder| {
            let out_color = builder.mutate_texture(scene_color, "Scene_Color_Skybox");
            builder.read_texture(scene_depth);
            for dependency in dependency_textures.into_iter().flatten() {
                builder.read_texture(dependency);
            }
            let node = SkyboxPassNode {
                out_color,
                scene_depth,
                pipeline,
                bind_group,
            };
            (node, out_color)
        })
    }
}

// ─── Skybox Pass Node ─────────────────────────────────────────────────────────

/// Ephemeral per-frame skybox render pass node.
pub struct SkyboxPassNode<'a> {
    out_color: TextureNodeId,
    scene_depth: TextureNodeId,
    pipeline: Option<&'a wgpu::RenderPipeline>,
    bind_group: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for SkyboxPassNode<'a> {
    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let (Some(pipeline), Some(bind_group)) = (self.pipeline, self.bind_group) else {
            return;
        };

        let gpu_global_bind_group = ctx.baked_lists.global_bind_group;

        let color_att = ctx.get_color_attachment(self.out_color, RenderTargetOps::Load, None);
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

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, gpu_global_bind_group, &[]);
        pass.set_bind_group(1, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
