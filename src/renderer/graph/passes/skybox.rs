//! Skybox / Background Render Pass
//!
//! Renders the scene background (gradient, cubemap, equirectangular, or planar texture)
//! as a fullscreen triangle at the far depth plane. Uses Reverse-Z depth testing
//! (`GreaterEqual`) so that opaque geometry masks the background via Early-Z.
//!
//! # Render Graph Position
//!
//! ```text
//! OpaquePass → SkyboxPass → TransparentPass
//! ```
//!
//! This pass reads depth written by the opaque pass and outputs color.
//! All `LoadOp`s are `Load` since we build upon existing framebuffer content.
//!
//! # Pipeline Variants
//!
//! Each [`BackgroundMapping`] produces a distinct pipeline variant via `ShaderDefines`,
//! avoiding dynamic branching in the fragment shader:
//!
//! | Variant | ShaderDefine | Texture |
//! |---------|-------------|---------|
//! | Gradient | `SKYBOX_GRADIENT` | None |
//! | Cube | `SKYBOX_CUBE` | `texture_cube<f32>` |
//! | Equirectangular | `SKYBOX_EQUIRECT` | `texture_2d<f32>` |
//! | Planar | `SKYBOX_PLANAR` | `texture_2d<f32>` |

use rustc_hash::FxHashMap;

use crate::render::RenderNode;
use crate::renderer::core::{binding::BindGroupKey, resources::Tracked};
use crate::renderer::graph::context::{ExecuteContext, GraphResource, PrepareContext};
use crate::renderer::graph::frame::PreparedSkyboxDraw;
use crate::renderer::pipeline::{
    ColorTargetKey, DepthStencilKey, FullscreenPipelineKey, MultisampleKey, RenderPipelineId,
    ShaderCompilationOptions,
};
use crate::resources::shader_defines::ShaderDefines;
use crate::resources::texture::TextureSource;
use crate::resources::uniforms::WgslStruct;
use crate::scene::background::{BackgroundMapping, BackgroundMode, SkyboxParamsUniforms};

// ============================================================================
// GPU Uniform Structs
// ============================================================================

// Camera data (view_projection_inverse, camera_position) is now obtained from
// the global bind group's RenderStateUniforms, eliminating the need for a
// separate SkyboxCameraUniforms.
//
// SkyboxParamsUniforms is defined in scene::background alongside BackgroundSettings,
// which owns the CpuBuffer<SkyboxParamsUniforms>. The render pass only reads from it.

// ============================================================================
// Pipeline variant key
// ============================================================================

/// Identifies a skybox pipeline variant.
///
/// The pipeline differs by:
/// - Background mapping type (determines shader variant + bind group layout)
/// - Output color format (HDR vs LDR)
/// - Depth format
/// - MSAA sample count
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SkyboxPipelineKey {
    variant: SkyboxVariant,
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    sample_count: u32,
}

/// Which shader variant to compile.
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

    /// Whether this variant requires a texture binding.
    fn needs_texture(self) -> bool {
        !matches!(self, Self::Gradient)
    }

    /// The texture view dimension for this variant's bind group layout.
    #[allow(dead_code)]
    fn texture_view_dimension(self) -> wgpu::TextureViewDimension {
        match self {
            Self::Cube => wgpu::TextureViewDimension::Cube,
            _ => wgpu::TextureViewDimension::D2,
        }
    }
}

// ============================================================================
// Layout helpers
// ============================================================================

/// Create uniform-only bind group layout (for gradient variant — Group 1).
///
/// Camera data is obtained from the global bind group (Group 0).
fn create_uniform_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Skybox Layout (NoTex)"),
        entries: &[
            // Binding 0: Skybox params
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
        ],
    })
}

/// Create uniform + texture + sampler bind group layout (Group 1).
///
/// Camera data is obtained from the global bind group (Group 0).
fn create_texture_layout(
    device: &wgpu::Device,
    view_dimension: wgpu::TextureViewDimension,
    label: &str,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            // Binding 0: Skybox params
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
            // Binding 1: Texture
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
            // Binding 2: Sampler
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

// ============================================================================
// SkyboxPass
// ============================================================================

/// Skybox / Background render pass.
///
/// Self-contained pass that uses the global bind group (Group 0) for camera
/// data (`view_projection_inverse`, `camera_position`) and its own bind group
/// (Group 1) for skybox-specific parameters and textures.
///
/// Three pre-created layouts cover all variants:
/// - `layout_gradient`: Binding 0 (params uniform only)
/// - `layout_cube`: Bindings 0–2 (params uniform + cube texture + sampler)
/// - `layout_2d`: Bindings 0–2 (params uniform + 2D texture + sampler)
///
/// # Lifecycle
///
/// 1. `prepare()`: Syncs skybox params, resolves textures,
///    creates/caches pipeline and bind group.
/// 2. `run()`: Emits a single fullscreen draw call with `LoadOp::Load`.
pub struct SkyboxPass {
    // --- Bind Group Layouts (one per texture dimension, Group 1) ---
    layout_gradient: Tracked<wgpu::BindGroupLayout>,
    layout_cube: Tracked<wgpu::BindGroupLayout>,
    layout_2d: Tracked<wgpu::BindGroupLayout>,

    // --- Sampler ---
    sampler: Tracked<wgpu::Sampler>,

    // --- Pipeline Cache ---
    local_cache: FxHashMap<SkyboxPipelineKey, RenderPipelineId>,

    // --- Runtime State (set during prepare, consumed during run) ---
    current_bind_group: Option<wgpu::BindGroup>,
    current_pipeline: Option<RenderPipelineId>,
}

impl SkyboxPass {
    /// Creates a new skybox pass.
    ///
    /// Pre-creates all three bind group layouts and a shared sampler.
    /// No pipelines are compiled until the first frame that requires them.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let layout_gradient = create_uniform_layout(device);
        let layout_cube = create_texture_layout(
            device,
            wgpu::TextureViewDimension::Cube,
            "Skybox Layout (Cube)",
        );
        let layout_2d =
            create_texture_layout(device, wgpu::TextureViewDimension::D2, "Skybox Layout (2D)");

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Skybox Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat, // use Repeat for seamless cubemap/equirectangular sampling
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        Self {
            layout_gradient: Tracked::new(layout_gradient),
            layout_cube: Tracked::new(layout_cube),
            layout_2d: Tracked::new(layout_2d),
            sampler: Tracked::new(sampler),
            local_cache: FxHashMap::default(),
            current_bind_group: None,
            current_pipeline: None,
        }
    }

    /// Returns the bind group layout for the given variant.
    fn layout_for_variant(&self, variant: SkyboxVariant) -> &Tracked<wgpu::BindGroupLayout> {
        match variant {
            SkyboxVariant::Gradient => &self.layout_gradient,
            SkyboxVariant::Cube => &self.layout_cube,
            SkyboxVariant::Equirectangular | SkyboxVariant::Planar => &self.layout_2d,
        }
    }

    /// Gets or creates a pipeline for the given key.
    fn get_or_create_pipeline(
        &mut self,
        ctx: &mut PrepareContext,
        key: SkyboxPipelineKey,
    ) -> RenderPipelineId {
        if let Some(&pipeline_id) = self.local_cache.get(&key) {
            return pipeline_id;
        }

        log::debug!(
            "Compiling Skybox pipeline: variant={:?}, format={:?}, samples={}",
            key.variant,
            key.color_format,
            key.sample_count
        );

        let gpu_world = ctx
            .resource_manager
            .get_global_state(ctx.render_state.id, ctx.scene.id)
            .expect("Global state must exist");

        // 1. Shader defines
        let mut defines = ShaderDefines::new();
        defines.set(key.variant.shader_define_key(), "1");

        // 2. Generate shader with auto-generated struct definitions and global binding code
        let mut options = ShaderCompilationOptions { defines };

        options.add_define(
            "struct_definitions",
            SkyboxParamsUniforms::wgsl_struct_def("SkyboxParams").as_str(),
        );

        let device = &ctx.wgpu_ctx.device;

        let (shader_module, shader_hash) = ctx.shader_manager.get_or_compile_template(
            device,
            "passes/skybox",
            &options,
            "",
            &gpu_world.binding_wgsl,
        );

        // 3. Pipeline layout: Group 0 = global, Group 1 = skybox-specific
        let layout = self.layout_for_variant(key.variant);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Skybox Pipeline Layout"),
            bind_group_layouts: &[
                &gpu_world.layout, // Group 0: Global bind group (frame-level resources)
                layout,            // Group 1: Skybox-specific bind group
            ],
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
                count: key.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            }),
        };

        let pipeline_id = ctx.pipeline_cache.get_or_create_fullscreen(
            device,
            shader_module,
            &pipeline_layout,
            &fullscreen_key,
            "Skybox Pipeline",
        );

        self.local_cache.insert(key, pipeline_id);
        pipeline_id
    }

    /// Resolves the texture view for a skybox background texture source.
    ///
    /// Returns `None` if the texture is not yet uploaded or unavailable.
    fn resolve_texture_view<'a>(
        ctx: &'a PrepareContext,
        source: &TextureSource,
        mapping: BackgroundMapping,
    ) -> Option<&'a wgpu::TextureView> {
        match source {
            // 1. Handle Asset (regular texture resource)
            TextureSource::Asset(handle) => {
                if let Some(binding) = ctx.resource_manager.texture_bindings.get(*handle)
                    && let Some(img) = ctx.resource_manager.gpu_images.get(&binding.cpu_image_id)
                {
                    match mapping {
                        BackgroundMapping::Cube => {
                            if img.default_view_dimension == wgpu::TextureViewDimension::Cube {
                                return Some(&img.default_view);
                            }
                            log::warn!(
                                "Skybox mapping is Cube but texture is {:?}",
                                img.default_view_dimension
                            );
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
            // 2. Handle Attachment (dynamically generated RenderTarget, etc.)
            TextureSource::Attachment(id, _) => ctx.resource_manager.internal_resources.get(id),
        }
    }
}

impl RenderNode for SkyboxPass {
    fn name(&self) -> &'static str {
        "Skybox Pass"
    }
    #[allow(clippy::similar_names)]
    fn prepare(&mut self, ctx: &mut PrepareContext) {
        let background = &ctx.scene.background;

        // 1. Determine variant from the BackgroundMode
        let Some(variant) = SkyboxVariant::from_background(&background.mode) else {
            // Color mode — no skybox pass needed
            self.current_bind_group = None;
            self.current_pipeline = None;
            return;
        };

        // 2. Ensure GPU buffer is up-to-date.
        //    Uniform values are written by user API (BackgroundSettings setters)
        //    which automatically bumps the CpuBuffer version. We only sync to GPU
        //    when the version has actually changed — no per-frame writes here.
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

        // 3. Resolve texture view (if textured variant)
        let texture_view: Option<&wgpu::TextureView> =
            if let BackgroundMode::Texture {
                source, mapping, ..
            } = &background.mode
            {
                Self::resolve_texture_view(ctx, source, *mapping)
            } else {
                None
            };

        // 4. Build or retrieve cached bind group (Group 1: skybox-specific)
        let layout = self.layout_for_variant(variant);
        let layout_id = layout.id();

        let bind_group = if variant.needs_texture() {
            // --- Textured variant ---
            let Some(tex_view) = texture_view else {
                log::warn!("SkyboxPass: texture not yet available, skipping frame");
                self.current_bind_group = None;
                self.current_pipeline = None;
                return;
            };

            let tex_view_key = std::ptr::from_ref::<wgpu::TextureView>(tex_view) as u64;
            let key = BindGroupKey::new(layout_id)
                .with_resource(params_gpu_id)
                .with_resource(tex_view_key)
                .with_resource(self.sampler.id());

            if let Some(cached) = ctx.global_bind_group_cache.get(&key) {
                cached.clone()
            } else {
                let params_gpu = ctx
                    .resource_manager
                    .gpu_buffers
                    .get(&params_cpu_id)
                    .expect("Skybox params GPU buffer must exist after ensure");

                let bg = ctx
                    .wgpu_ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Skybox BindGroup (Texture)"),
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
                                resource: wgpu::BindingResource::Sampler(&self.sampler),
                            },
                        ],
                    });
                ctx.global_bind_group_cache.insert(key, bg.clone());
                bg
            }
        } else {
            // --- Gradient variant (no texture) ---
            let key = BindGroupKey::new(layout_id).with_resource(params_gpu_id);

            if let Some(cached) = ctx.global_bind_group_cache.get(&key) {
                cached.clone()
            } else {
                let params_gpu = ctx
                    .resource_manager
                    .gpu_buffers
                    .get(&params_cpu_id)
                    .expect("Skybox params GPU buffer must exist after ensure");

                let bg = ctx
                    .wgpu_ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Skybox BindGroup (Gradient)"),
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

        self.current_bind_group = Some(bind_group);

        // 5. Ensure pipeline exists for the current variant + output format
        let pipeline_key = SkyboxPipelineKey {
            variant,
            color_format: ctx.get_scene_render_target_format(),
            depth_format: ctx.wgpu_ctx.depth_format,
            sample_count: ctx.wgpu_ctx.msaa_samples,
        };

        self.current_pipeline = Some(self.get_or_create_pipeline(ctx, pipeline_key));

        // 6. Store prepared draw state for potential inline use (LDR path)
        //    SimpleForwardPass reads this to draw skybox within its own render pass.
        if let (Some(pipeline_id), Some(bind_group)) =
            (self.current_pipeline, &self.current_bind_group)
        {
            ctx.render_lists.prepared_skybox = Some(PreparedSkyboxDraw {
                pipeline_id,
                bind_group: bind_group.clone(),
            });
        }
    }

    fn run(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        // In LDR mode, skybox is drawn inline by SimpleForwardPass
        // (between opaque and transparent draws within a single render pass).
        if !ctx.wgpu_ctx.render_path.supports_post_processing() {
            return;
        }

        // Skip if no pipeline/bind group (Color mode or missing resources)
        let (Some(pipeline_id), Some(bind_group)) =
            (self.current_pipeline, &self.current_bind_group)
        else {
            return;
        };

        let render_lists = ctx.render_lists;
        let Some(gpu_global_bind_group) = &render_lists.gpu_global_bind_group else {
            return;
        };

        // --- Determine render targets ---
        let target_view = ctx.get_scene_render_target_view();
        let depth_view = ctx.get_resource_view(GraphResource::DepthStencil);

        // MSAA: render into the multisample attachment; do NOT resolve here.
        // In the HDR pipeline, TransparentPass handles the final MSAA resolve.
        let attachment_view =
            if let Some(msaa_view) = ctx.try_get_resource_view(GraphResource::SceneMsaa) {
                msaa_view as &wgpu::TextureView
            } else {
                target_view
            };

        // --- Create RenderPass (all Load — we inherit from OpaquePass) ---
        let pass_desc = wgpu::RenderPassDescriptor {
            label: Some("Skybox Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: attachment_view,
                resolve_target: None, // No resolve — TransparentPass will do it
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // Inherit opaque results
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load, // Read opaque depth
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

        // Draw fullscreen triangle (3 vertices, 1 instance, no vertex buffer)
        pass.draw(0..3, 0..1);
    }
}
