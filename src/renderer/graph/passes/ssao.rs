//! Screen Space Ambient Occlusion (SSAO) Post-Processing Pass
//!
//! Implements a production-grade SSAO with two sub-passes:
//!
//! 1. **Raw SSAO**: Hemisphere occlusion sampling using depth + normal buffers,
//!    a configurable sample kernel, and a 4×4 tiled rotation noise texture.
//! 2. **Cross-Bilateral Blur**: A depth-aware + normal-aware spatial filter
//!    that smooths the raw noise while preserving geometric edges.
//!
//! # Data Flow
//!
//! ```text
//! DepthNormalPrepass        SsaoPass
//!        │             ┌──────────────────────────┐
//!  SceneDepth  ───┬───►│  Sub-Pass 1: Raw SSAO    │──► R8Unorm (noisy)
//!  SceneNormal ───┤    │                           │         │
//!                 └───►│  Sub-Pass 2: Bilateral    │◄────────┘
//!                      │           Blur            │──► R8Unorm (smooth)
//!                      └──────────────────────────┘
//!                                   │
//!           stored as transient texture → used by FrameResources
//!                  (screen_bind_group, binding 2)
//! ```
//!
//! # Performance
//!
//! - Pipelines are created once and cached
//! - The 4×4 noise texture is uploaded once and reused
//! - BindGroups use `GlobalBindGroupCache` for cross-frame reuse
//! - Transient textures are recycled through `TransientTexturePool`
//! - The bilateral blur is a single-pass 5×5 filter (no separable passes
//!   needed at this kernel size)

use crate::render::RenderNode;
use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::resources::Tracked;
use crate::renderer::graph::context::{ExecuteContext, PrepareContext};
use crate::renderer::graph::transient_pool::{TransientTextureDesc, TransientTextureId};
use crate::renderer::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions,
};
use crate::resources::ssao::{SsaoUniforms, generate_ssao_noise};
use crate::resources::uniforms::WgslStruct;

/// The SSAO output texture format: single-channel unsigned normalized.
const SSAO_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// Screen Space Ambient Occlusion pass.
///
/// Owns two internal sub-pipelines (raw generation + bilateral blur),
/// a persistent 4×4 noise texture, and per-frame transient textures for
/// the intermediate and final AO results.
///
/// The final blurred AO texture ID is exposed via [`ssao_output_id`](Self::ssao_output_id)
/// so that `FrameResources` can bind it into the `screen_bind_group`
/// at group 3 binding 2.
pub struct SsaoPass {
    // === Pipelines (cached as IDs in the global PipelineCache) ===
    raw_pipeline: Option<RenderPipelineId>,
    blur_pipeline: Option<RenderPipelineId>,

    // === Bind Group Layouts ===
    raw_layout: Tracked<wgpu::BindGroupLayout>,
    raw_uniforms_layout: Tracked<wgpu::BindGroupLayout>,
    blur_layout: Tracked<wgpu::BindGroupLayout>,

    // === Shared Resources ===
    /// Linear, clamp-to-edge sampler for depth/normal/AO textures.
    linear_sampler: Tracked<wgpu::Sampler>,
    /// Nearest-neighbor, repeat sampler for the 4×4 noise texture.
    noise_sampler: Tracked<wgpu::Sampler>,
    /// Persistent 4×4 tiled rotation noise texture.
    noise_texture_view: Option<Tracked<wgpu::TextureView>>,

    // === Transient Textures (allocated per-frame) ===
    /// Handle to the raw (noisy) AO texture.
    raw_texture_id: Option<TransientTextureId>,
    /// Handle to the final blurred AO texture.
    blur_texture_id: Option<TransientTextureId>,

    // === Cached BindGroups ===
    raw_bind_group: Option<wgpu::BindGroup>,
    raw_uniforms_bind_group: Option<wgpu::BindGroup>,
    blur_bind_group: Option<wgpu::BindGroup>,

    // === Runtime State ===
    enabled: bool,
}

impl SsaoPass {
    /// Creates a new SSAO pass, allocating GPU layouts and samplers.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        // --- Bind Group Layout: Raw SSAO (Group 1) ---
        // depth, normal, noise textures + samplers
        let raw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSAO Raw Layout"),
            entries: &[
                // Binding 0: Depth texture
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
                // Binding 1: Normal texture
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
                // Binding 2: Noise texture (4×4, tiled)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 3: Linear sampler (depth/normal)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Binding 4: Noise sampler (nearest, repeat)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        // --- Bind Group Layout: Raw SSAO Uniforms (Group 2) ---
        let raw_uniforms_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SSAO Raw Uniforms Layout"),
                entries: &[
                    // Binding 0: SsaoUniforms
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
            });

        // --- Bind Group Layout: Bilateral Blur (Group 0) ---
        // raw AO texture + depth texture + normal texture + sampler
        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSAO Blur Layout"),
            entries: &[
                // Binding 0: Raw SSAO texture
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
                // Binding 1: Depth texture
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
                // Binding 2: Normal texture
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
                // Binding 3: Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // --- Samplers ---
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SSAO Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SSAO Noise Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            raw_pipeline: None,
            blur_pipeline: None,

            raw_layout: Tracked::new(raw_layout),
            raw_uniforms_layout: Tracked::new(raw_uniforms_layout),
            blur_layout: Tracked::new(blur_layout),

            linear_sampler: Tracked::new(linear_sampler),
            noise_sampler: Tracked::new(noise_sampler),
            noise_texture_view: None,

            raw_texture_id: None,
            blur_texture_id: None,

            raw_bind_group: None,
            raw_uniforms_bind_group: None,
            blur_bind_group: None,

            enabled: false,
        }
    }

    /// Returns the transient texture ID of the final blurred SSAO output.
    ///
    /// Returns `None` if SSAO is disabled or hasn't been prepared yet.
    #[inline]
    #[must_use]
    pub fn ssao_output_id(&self) -> Option<TransientTextureId> {
        if self.enabled {
            self.blur_texture_id
        } else {
            None
        }
    }

    // =========================================================================
    // Noise Texture (uploaded once)
    // =========================================================================

    fn ensure_noise_texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.noise_texture_view.is_some() {
            return;
        }

        let noise_data = generate_ssao_noise();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSAO Noise 4x4"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload the 4×4 noise data
        let flat: Vec<u8> = noise_data.iter().flat_map(|p| p.iter().copied()).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &flat,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * 4), // 4 pixels × 4 bytes/pixel (RGBA8)
                rows_per_image: Some(4),
            },
            wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.noise_texture_view = Some(Tracked::new(view));
    }

    // =========================================================================
    // Pipeline Creation
    // =========================================================================

    fn ensure_pipelines(&mut self, ctx: &mut PrepareContext) {
        if self.raw_pipeline.is_some() {
            return;
        }

        let device = &ctx.wgpu_ctx.device;

        let gpu_world = ctx
            .resource_manager
            .get_global_state(ctx.render_state.id, ctx.scene.id)
            .expect("Global state must exist");

        let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
            format: SSAO_TEXTURE_FORMAT,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        });

        // --- Raw SSAO Pipeline ---
        {
            let mut options = ShaderCompilationOptions::default();
            options.add_define(
                "struct_definitions",
                SsaoUniforms::wgsl_struct_def("SsaoUniforms").as_str(),
            );

            let (module, shader_hash) = ctx.shader_manager.get_or_compile_template(
                device,
                "passes/ssao_raw",
                &options,
                "",
                &gpu_world.binding_wgsl,
            );

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SSAO Raw Pipeline Layout"),
                bind_group_layouts: &[
                    &gpu_world.layout,
                    &self.raw_layout,
                    &self.raw_uniforms_layout,
                ],
                immediate_size: 0,
            });

            let key = FullscreenPipelineKey::fullscreen(
                shader_hash,
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

        // --- Bilateral Blur Pipeline ---
        {
            let (module, shader_hash) = ctx.shader_manager.get_or_compile_template(
                device,
                "passes/ssao_blur",
                &ShaderCompilationOptions::default(),
                "",
                "",
            );

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SSAO Blur Pipeline Layout"),
                bind_group_layouts: &[&self.blur_layout],
                immediate_size: 0,
            });

            let key = FullscreenPipelineKey::fullscreen(
                shader_hash,
                smallvec::smallvec![color_target],
                None,
            );

            self.blur_pipeline = Some(ctx.pipeline_cache.get_or_create_fullscreen(
                device,
                module,
                &pipeline_layout,
                &key,
                "SSAO Blur Pipeline",
            ));
        }
    }

    // =========================================================================
    // Transient Texture Allocation
    // =========================================================================

    fn allocate_textures(&mut self, ctx: &mut PrepareContext) {
        let size = ctx.wgpu_ctx.size();

        // Raw SSAO output (noisy)
        self.raw_texture_id = Some(ctx.transient_pool.allocate(
            &ctx.wgpu_ctx.device,
            &TransientTextureDesc {
                width: size.0 / 2,
                height: size.1 / 2,
                format: SSAO_TEXTURE_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                mip_level_count: 1,
                label: "SSAO Raw",
            },
        ));

        // Blurred SSAO output (smooth)
        self.blur_texture_id = Some(ctx.transient_pool.allocate(
            &ctx.wgpu_ctx.device,
            &TransientTextureDesc {
                width: size.0,
                height: size.1,
                format: SSAO_TEXTURE_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                mip_level_count: 1,
                label: "SSAO Blurred",
            },
        ));
    }

    // =========================================================================
    // BindGroup Construction
    // =========================================================================

    fn build_bind_groups(&mut self, ctx: &mut PrepareContext) {
        let device = &ctx.wgpu_ctx.device;

        // Access frame_resources fields directly to avoid borrowing the whole ctx
        let depth_view = &ctx.frame_resources.depth_only_view;

        let normal_view = ctx.transient_pool.get_view(
            ctx.blackboard
                .scene_normal_texture_id
                .expect("SceneNormal must exist for SSAO"),
        );

        let noise_view = self
            .noise_texture_view
            .as_ref()
            .expect("Noise texture must exist");

        // --- Raw SSAO BindGroup (Group 1) ---
        {
            let key = BindGroupKey::new(self.raw_layout.id())
                .with_resource(depth_view.id())
                .with_resource(normal_view.id())
                .with_resource(noise_view.id())
                .with_resource(self.linear_sampler.id())
                .with_resource(self.noise_sampler.id());

            let bind_group = if let Some(cached) = ctx.global_bind_group_cache.get(&key) {
                cached.clone()
            } else {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SSAO Raw BindGroup"),
                    layout: &self.raw_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(normal_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(noise_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(&self.noise_sampler),
                        },
                    ],
                });
                ctx.global_bind_group_cache.insert(key, bg.clone());
                bg
            };
            self.raw_bind_group = Some(bind_group);
        }

        // --- Raw SSAO Uniforms BindGroup (Group 2) ---
        {
            let uniforms = &ctx.scene.ssao.uniforms;
            let gpu_buffer_id = ctx.resource_manager.ensure_buffer_id(uniforms);
            let cpu_buffer_id = uniforms.id();

            let key = BindGroupKey::new(self.raw_uniforms_layout.id()).with_resource(gpu_buffer_id);

            let bind_group = if let Some(cached) = ctx.global_bind_group_cache.get(&key) {
                cached.clone()
            } else {
                let gpu_buffer = ctx
                    .resource_manager
                    .gpu_buffers
                    .get(&cpu_buffer_id)
                    .expect("SSAO GpuBuffer must exist after ensure_buffer_id");

                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SSAO Uniforms BindGroup"),
                    layout: &self.raw_uniforms_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: gpu_buffer.buffer.as_entire_binding(),
                    }],
                });
                ctx.global_bind_group_cache.insert(key, bg.clone());
                bg
            };
            self.raw_uniforms_bind_group = Some(bind_group);
        }

        // --- Blur BindGroup (Group 0) ---
        {
            let raw_view = ctx
                .transient_pool
                .get_view(self.raw_texture_id.expect("Raw texture must be allocated"));

            let key = BindGroupKey::new(self.blur_layout.id())
                .with_resource(raw_view.id())
                .with_resource(depth_view.id())
                .with_resource(normal_view.id())
                .with_resource(self.linear_sampler.id());

            let bind_group = if let Some(cached) = ctx.global_bind_group_cache.get(&key) {
                cached.clone()
            } else {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SSAO Blur BindGroup"),
                    layout: &self.blur_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(raw_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(normal_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                        },
                    ],
                });
                ctx.global_bind_group_cache.insert(key, bg.clone());
                bg
            };
            self.blur_bind_group = Some(bind_group);
        }
    }
}

impl RenderNode for SsaoPass {
    fn name(&self) -> &'static str {
        "SSAO Pass"
    }

    fn prepare(&mut self, ctx: &mut PrepareContext) {
        // =====================================================================
        // 1. Check if SSAO should run
        // =====================================================================
        let settings = &ctx.scene.ssao;
        self.enabled = settings.enabled;

        if !self.enabled {
            return;
        }

        // =====================================================================
        // 2. Update noise scale from current screen size
        // =====================================================================
        let size = ctx.wgpu_ctx.size();
        ctx.scene.ssao.update_noise_scale(size.0, size.1);

        // =====================================================================
        // 3. Ensure persistent resources
        // =====================================================================
        self.ensure_noise_texture(&ctx.wgpu_ctx.device, &ctx.wgpu_ctx.queue);
        self.ensure_pipelines(ctx);

        // =====================================================================
        // 4. Allocate per-frame transient textures
        // =====================================================================
        self.allocate_textures(ctx);

        // =====================================================================
        // 5. Publish SSAO output ID for downstream draw passes (group 3 binding)
        // =====================================================================
        ctx.blackboard.ssao_texture_id = self.blur_texture_id;

        // =====================================================================
        // 6. Build bind groups
        // =====================================================================
        self.build_bind_groups(ctx);
    }

    fn run(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        if !self.enabled {
            return;
        }

        let Some(raw_pipeline_id) = self.raw_pipeline else {
            return;
        };
        let Some(blur_pipeline_id) = self.blur_pipeline else {
            return;
        };
        let Some(raw_bg) = &self.raw_bind_group else {
            return;
        };
        let Some(raw_uniforms_bg) = &self.raw_uniforms_bind_group else {
            return;
        };
        let Some(blur_bg) = &self.blur_bind_group else {
            return;
        };

        let render_lists = ctx.render_lists;
        let Some(gpu_global_bind_group) = &render_lists.gpu_global_bind_group else {
            return;
        };

        let raw_pipeline = ctx.pipeline_cache.get_render_pipeline(raw_pipeline_id);
        let blur_pipeline = ctx.pipeline_cache.get_render_pipeline(blur_pipeline_id);

        // =====================================================================
        // Sub-Pass 1: Raw SSAO
        // =====================================================================
        {
            let raw_view = ctx
                .transient_pool
                .get_view(self.raw_texture_id.expect("Raw texture allocated"));

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSAO Raw Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: raw_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::DontCare(wgpu::LoadOpDontCare::default()),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });

            pass.set_pipeline(raw_pipeline);
            pass.set_bind_group(0, gpu_global_bind_group, &[]);
            pass.set_bind_group(1, raw_bg, &[]);
            pass.set_bind_group(2, raw_uniforms_bg, &[]);
            pass.draw(0..3, 0..1); // fullscreen triangle
        }

        // =====================================================================
        // Sub-Pass 2: Cross-Bilateral Blur
        // =====================================================================
        {
            let blur_view = ctx
                .transient_pool
                .get_view(self.blur_texture_id.expect("Blur texture allocated"));

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSAO Blur Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: blur_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::DontCare(wgpu::LoadOpDontCare::default()),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });

            pass.set_pipeline(blur_pipeline);
            pass.set_bind_group(0, blur_bg, &[]);
            pass.draw(0..3, 0..1); // fullscreen triangle
        }
    }
}
