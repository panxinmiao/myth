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

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use rustc_hash::FxHashMap;

use crate::render::{RenderContext, RenderNode};
use crate::renderer::core::{binding::BindGroupKey, resources::Tracked};
use crate::renderer::graph::frame::PreparedSkyboxDraw;
use crate::renderer::pipeline::{ShaderCompilationOptions, shader_gen::ShaderGenerator};
use crate::resources::buffer::{CpuBuffer, GpuData};
use crate::resources::shader_defines::ShaderDefines;
use crate::resources::texture::TextureSource;
use crate::scene::background::{BackgroundMapping, BackgroundMode};

// ============================================================================
// GPU Uniform Structs
// ============================================================================

/// Camera data for skybox ray reconstruction.
///
/// Kept separate from the global `RenderStateUniforms` so the skybox pass
/// is self-contained and does not depend on the global bind group layout.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct SkyboxCameraUniforms {
    /// Inverse of the view-projection matrix (for clip → world reconstruction)
    view_projection_inverse: Mat4,
    /// Camera world position
    camera_position: Vec3,
    _pad0: f32,
}

impl Default for SkyboxCameraUniforms {
    fn default() -> Self {
        Self {
            view_projection_inverse: Mat4::IDENTITY,
            camera_position: Vec3::ZERO,
            _pad0: 0.0,
        }
    }
}

impl GpuData for SkyboxCameraUniforms {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
    fn byte_size(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

/// Skybox parameters uniform.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct SkyboxParamsUniforms {
    color_top: Vec4,
    color_bottom: Vec4,
    rotation: f32,
    intensity: f32,
    _pad0: f32,
    _pad1: f32,
}

impl Default for SkyboxParamsUniforms {
    fn default() -> Self {
        Self {
            color_top: Vec4::ZERO,
            color_bottom: Vec4::ZERO,
            rotation: 0.0,
            intensity: 1.0,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

impl GpuData for SkyboxParamsUniforms {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
    fn byte_size(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

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
            Self::Equirectangular | Self::Planar => wgpu::TextureViewDimension::D2,
            Self::Gradient => wgpu::TextureViewDimension::D2, // unused
        }
    }
}

// ============================================================================
// Layout helpers
// ============================================================================

/// Create uniform-only bind group layout (for gradient variant).
fn create_uniform_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Skybox Layout (NoTex)"),
        entries: &[
            // Binding 0: Camera uniforms
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 1: Skybox params
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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

/// Create uniform + texture + sampler bind group layout.
fn create_texture_layout(
    device: &wgpu::Device,
    view_dimension: wgpu::TextureViewDimension,
    label: &str,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            // Binding 0: Camera uniforms
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 1: Skybox params
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 2: Texture
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension,
                    multisampled: false,
                },
                count: None,
            },
            // Binding 3: Sampler
            wgpu::BindGroupLayoutEntry {
                binding: 3,
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
/// Self-contained pass (like [`ToneMapPass`]) with its own bind group layouts,
/// pipeline cache, and uniform buffers. Does not depend on the global bind
/// group (Group 0) used by mesh rendering passes.
///
/// Three pre-created layouts cover all variants:
/// - `layout_gradient`: Bindings 0–1 (uniforms only)
/// - `layout_cube`: Bindings 0–3 (uniforms + cube texture + sampler)
/// - `layout_2d`: Bindings 0–3 (uniforms + 2D texture + sampler)
///
/// # Lifecycle
///
/// 1. `prepare()`: Syncs camera & skybox uniforms, resolves textures,
///    creates/caches pipeline and bind group.
/// 2. `run()`: Emits a single fullscreen draw call with `LoadOp::Load`.
pub struct SkyboxPass {
    // --- Bind Group Layouts (one per texture dimension) ---
    layout_gradient: Tracked<wgpu::BindGroupLayout>,
    layout_cube: Tracked<wgpu::BindGroupLayout>,
    layout_2d: Tracked<wgpu::BindGroupLayout>,

    // --- Sampler ---
    sampler: Tracked<wgpu::Sampler>,

    // --- Uniform Buffers ---
    camera_uniforms: CpuBuffer<SkyboxCameraUniforms>,
    params_uniforms: CpuBuffer<SkyboxParamsUniforms>,

    // --- Pipeline Cache ---
    pipeline_cache: FxHashMap<SkyboxPipelineKey, wgpu::RenderPipeline>,

    // --- Runtime State (set during prepare, consumed during run) ---
    current_bind_group: Option<wgpu::BindGroup>,
    current_pipeline: Option<wgpu::RenderPipeline>,
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
        let layout_2d = create_texture_layout(
            device,
            wgpu::TextureViewDimension::D2,
            "Skybox Layout (2D)",
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Skybox Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let camera_uniforms = CpuBuffer::new(
            SkyboxCameraUniforms::default(),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some("Skybox Camera Uniforms"),
        );
        let params_uniforms = CpuBuffer::new(
            SkyboxParamsUniforms::default(),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some("Skybox Params Uniforms"),
        );

        Self {
            layout_gradient: Tracked::new(layout_gradient),
            layout_cube: Tracked::new(layout_cube),
            layout_2d: Tracked::new(layout_2d),
            sampler: Tracked::new(sampler),
            camera_uniforms,
            params_uniforms,
            pipeline_cache: FxHashMap::default(),
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
        device: &wgpu::Device,
        key: SkyboxPipelineKey,
    ) -> wgpu::RenderPipeline {
        if let Some(pipeline) = self.pipeline_cache.get(&key) {
            return pipeline.clone();
        }

        log::debug!(
            "Compiling Skybox pipeline: variant={:?}, format={:?}, samples={}",
            key.variant,
            key.color_format,
            key.sample_count
        );

        // 1. Shader defines
        let mut defines = ShaderDefines::new();
        defines.set(key.variant.shader_define_key(), "1");

        // 2. Generate shader
        let options = ShaderCompilationOptions { defines };
        let shader_code = ShaderGenerator::generate_shader("", "", "passes/skybox", &options);

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("Skybox Shader ({:?})", key.variant)),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_code)),
        });

        // 3. Pipeline layout (uses the variant-appropriate bind group layout)
        let layout = self.layout_for_variant(key.variant);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Skybox Pipeline Layout"),
            bind_group_layouts: &[layout],
            immediate_size: 0,
        });

        // 4. Render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("Skybox Pipeline ({:?})", key.variant)),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                buffers: &[], // Fullscreen triangle — no vertex buffers
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: key.color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Fullscreen triangle — no culling
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: key.depth_format,
                // Skybox sits at Z=0.0 (Reverse-Z far plane).
                // depth_write: false — skybox never writes depth (infinitely far).
                depth_write_enabled: false,
                // GreaterEqual: pass if fragment_z >= buffer_z.
                //   - Where opaque exists (buffer > 0): 0.0 >= buffer → fail → culled ✓
                //   - Where empty (buffer = 0.0): 0.0 >= 0.0 → pass → draws ✓
                depth_compare: wgpu::CompareFunction::GreaterEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: key.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        self.pipeline_cache.insert(key, pipeline.clone());
        pipeline
    }

    /// Resolves the texture view for a skybox background texture source.
    ///
    /// Returns `None` if the texture is not yet uploaded or unavailable.
    fn resolve_texture_view<'a>(
        ctx: &'a RenderContext,
        source: &TextureSource,
        mapping: BackgroundMapping,
    ) -> Option<&'a wgpu::TextureView> {
        match mapping {
            BackgroundMapping::Cube => {
                // 1. Try the processed cubemap from the environment map cache
                if let Some(gpu_env) = ctx.resource_manager.environment_map_cache.get(source) {
                    if !gpu_env.needs_compute {
                        if let Some(view) =
                            ctx.resource_manager.internal_resources.get(&gpu_env.cube_view_id)
                        {
                            return Some(view);
                        }
                    }
                }
                // 2. Fallback: raw cubemap asset already uploaded as a Cube view
                if let TextureSource::Asset(handle) = source {
                    if let Some(binding) = ctx.resource_manager.texture_bindings.get(*handle) {
                        if let Some(img) =
                            ctx.resource_manager.gpu_images.get(&binding.cpu_image_id)
                        {
                            if img.default_view_dimension == wgpu::TextureViewDimension::Cube {
                                return Some(&img.default_view);
                            }
                        }
                    }
                }
                None
            }
            BackgroundMapping::Equirectangular | BackgroundMapping::Planar => {
                // 2D texture asset
                if let TextureSource::Asset(handle) = source {
                    if let Some(binding) = ctx.resource_manager.texture_bindings.get(*handle) {
                        if let Some(img) =
                            ctx.resource_manager.gpu_images.get(&binding.cpu_image_id)
                        {
                            return Some(&img.default_view);
                        }
                    }
                }
                None
            }
        }
    }
}

impl RenderNode for SkyboxPass {
    fn name(&self) -> &str {
        "Skybox Pass"
    }

    fn prepare(&mut self, ctx: &mut RenderContext) {
        let background = &ctx.scene.background;

        // 1. Determine variant
        let Some(variant) = SkyboxVariant::from_background(background) else {
            // Color mode — no skybox pass needed
            self.current_bind_group = None;
            self.current_pipeline = None;
            return;
        };

        // 2. Update camera uniforms from current render state
        {
            let rs = ctx.render_state.uniforms().read();
            let mut u = self.camera_uniforms.write();
            u.view_projection_inverse = rs.view_projection_inverse;
            u.camera_position = rs.camera_position;
        }

        // 3. Update skybox params from scene background
        {
            let mut p = self.params_uniforms.write();
            match background {
                BackgroundMode::Gradient { top, bottom } => {
                    p.color_top = *top;
                    p.color_bottom = *bottom;
                    p.rotation = 0.0;
                    p.intensity = 1.0;
                }
                BackgroundMode::Texture {
                    rotation,
                    intensity,
                    ..
                } => {
                    p.color_top = Vec4::ZERO;
                    p.color_bottom = Vec4::ZERO;
                    p.rotation = *rotation;
                    p.intensity = *intensity;
                }
                BackgroundMode::Color(_) => unreachable!(),
            }
        }

        // 4. Ensure GPU buffers are up-to-date
        let camera_gpu_id = ctx.resource_manager.ensure_buffer_id(&self.camera_uniforms);
        let params_gpu_id = ctx.resource_manager.ensure_buffer_id(&self.params_uniforms);
        let camera_cpu_id = self.camera_uniforms.id();
        let params_cpu_id = self.params_uniforms.id();

        // 5. Resolve texture view (if textured variant)
        let texture_view: Option<&wgpu::TextureView> =
            if let BackgroundMode::Texture {
                source, mapping, ..
            } = background
            {
                Self::resolve_texture_view(ctx, source, *mapping)
            } else {
                None
            };

        // 6. Build or retrieve cached bind group
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

            let tex_view_key = tex_view as *const wgpu::TextureView as u64;
            let key = BindGroupKey::new(layout_id)
                .with_resource(camera_gpu_id)
                .with_resource(params_gpu_id)
                .with_resource(tex_view_key)
                .with_resource(self.sampler.id());

            if let Some(cached) = ctx.global_bind_group_cache.get(&key) {
                cached.clone()
            } else {
                let camera_gpu = ctx
                    .resource_manager
                    .gpu_buffers
                    .get(&camera_cpu_id)
                    .expect("Skybox camera GPU buffer must exist after ensure");
                let params_gpu = ctx
                    .resource_manager
                    .gpu_buffers
                    .get(&params_cpu_id)
                    .expect("Skybox params GPU buffer must exist after ensure");

                let bg =
                    ctx.wgpu_ctx
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("Skybox BindGroup (Texture)"),
                            layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: camera_gpu.buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: params_gpu.buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::TextureView(tex_view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                                },
                            ],
                        });
                ctx.global_bind_group_cache.insert(key, bg.clone());
                bg
            }
        } else {
            // --- Gradient variant (no texture) ---
            let key = BindGroupKey::new(layout_id)
                .with_resource(camera_gpu_id)
                .with_resource(params_gpu_id);

            if let Some(cached) = ctx.global_bind_group_cache.get(&key) {
                cached.clone()
            } else {
                let camera_gpu = ctx
                    .resource_manager
                    .gpu_buffers
                    .get(&camera_cpu_id)
                    .expect("Skybox camera GPU buffer must exist after ensure");
                let params_gpu = ctx
                    .resource_manager
                    .gpu_buffers
                    .get(&params_cpu_id)
                    .expect("Skybox params GPU buffer must exist after ensure");

                let bg =
                    ctx.wgpu_ctx
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("Skybox BindGroup (Gradient)"),
                            layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: camera_gpu.buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: params_gpu.buffer.as_entire_binding(),
                                },
                            ],
                        });
                ctx.global_bind_group_cache.insert(key, bg.clone());
                bg
            }
        };

        self.current_bind_group = Some(bind_group);

        // 7. Ensure pipeline exists for the current variant + output format
        let pipeline_key = SkyboxPipelineKey {
            variant,
            color_format: ctx.get_scene_render_target_format(),
            depth_format: ctx.wgpu_ctx.depth_format,
            sample_count: ctx.wgpu_ctx.msaa_samples,
        };

        self.current_pipeline =
            Some(self.get_or_create_pipeline(&ctx.wgpu_ctx.device, pipeline_key));

        // 8. Store prepared draw state for potential inline use (LDR path)
        //    SimpleForwardPass reads this to draw skybox within its own render pass.
        if let (Some(pipeline), Some(bind_group)) =
            (&self.current_pipeline, &self.current_bind_group)
        {
            ctx.render_lists.prepared_skybox = Some(PreparedSkyboxDraw {
                pipeline: pipeline.clone(),
                bind_group: bind_group.clone(),
            });
        }
    }

    fn run(&self, ctx: &mut RenderContext, encoder: &mut wgpu::CommandEncoder) {
        // In LDR mode, skybox is drawn inline by SimpleForwardPass
        // (between opaque and transparent draws within a single render pass).
        if !ctx.wgpu_ctx.enable_hdr {
            return;
        }

        // Skip if no pipeline/bind group (Color mode or missing resources)
        let (Some(pipeline), Some(bind_group)) =
            (&self.current_pipeline, &self.current_bind_group)
        else {
            return;
        };

        // --- Determine render targets ---
        let target_view = ctx.get_scene_render_target_view();
        let depth_view = &ctx.frame_resources.depth_view;

        // MSAA: render into the multisample attachment; do NOT resolve here.
        // In the HDR pipeline, TransparentPass handles the final MSAA resolve.
        let attachment_view = if ctx.wgpu_ctx.msaa_samples > 1 {
            ctx.frame_resources
                .scene_msaa_view
                .as_ref()
                .expect("MSAA view must exist when msaa_samples > 1")
                as &wgpu::TextureView
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

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);

        // Draw fullscreen triangle (3 vertices, 1 instance, no vertex buffer)
        pass.draw(0..3, 0..1);
    }
}
