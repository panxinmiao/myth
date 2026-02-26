//! Screen-Space Sub-Surface Scattering (SSSSS) Post-Processing Pass
//!
//! Implements a separable Gaussian blur for SSS materials identified via the
//! **Thin G-Buffer** Normal.a channel.  Two sequential render passes are
//! executed (H then V) to reconstruct a 2-D scatter.
//!
//! # Data Flow
//!
//! ```text
//!  DepthNormalPrepass          SssssPass
//!       │              ┌────────────────────────────┐
//! SceneColor ─────────►│  H Sub-Pass: Horizontal blur │──► "ssss_ping" (transient)
//! SceneNormal ────┬───►│                              │         │
//! SceneDepth  ────┤    │  V Sub-Pass: Vertical blur   │◄────────┘
//!                 └───►│   (overwrites SceneColor)    │──► SceneColor (in-place)
//!                      └────────────────────────────┘
//! ```
//!
//! # Integration
//!
//! Must come **after** `TransparentPass` and **before** `BloomPass` in the
//! `HighFidelity` render path.  Zero cost when disabled.
//!
//! # GPU Resources
//!
//! - **Profiles StorageBuffer**: 256 × `ScreenSpaceMaterialData` (12 KB).
//!   Uploaded from `ExtractedScene.current_screen_space_profiles`.
//! - **H/V Uniforms UniformBuffer**: 2 × `SsssUniforms` (64 bytes total).
//! - **Ping TransientTexture**: `Rgba16Float`, same size as scene colour.

use std::borrow::Cow;

use bytemuck::cast_slice;

use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::resources::Tracked;
use crate::renderer::graph::context::{ExecuteContext, PrepareContext};
use crate::renderer::graph::transient_pool::{TransientTextureDesc, TransientTextureId};
use crate::renderer::graph::RenderNode;
use crate::renderer::pipeline::shader_gen::ShaderGenerator;
use crate::renderer::pipeline::ShaderCompilationOptions;
use crate::renderer::HDR_TEXTURE_FORMAT;
use crate::resources::screen_space::{ScreenSpaceMaterialData, SsssUniforms};

// ── Constants ────────────────────────────────────────────────────────────────

/// SSS ping texture format — same as the HDR scene colour.
const SSSS_COLOR_FORMAT: wgpu::TextureFormat = HDR_TEXTURE_FORMAT;

/// GPU storage buffer capacity: 256 profiles × 48 bytes = 12 KB.
const PROFILES_BUFFER_CAPACITY: u64 =
    256 * std::mem::size_of::<ScreenSpaceMaterialData>() as u64;

// ── SssssPass ────────────────────────────────────────────────────────────────

/// Screen-Space Sub-Surface Scattering pass.
///
/// Owns one pipeline (shared for H and V passes), persistent GPU buffers
/// (uniforms, SSS profiles storage buffer), a linear sampler, and per-frame
/// transient resources.
pub struct SssssPass {
    // === Runtime State ===
    enabled: bool,

    // === Pipeline (created once, cached) ===
    pipeline: Option<wgpu::RenderPipeline>,

    // === Bind Group Layouts ===
    /// Group 0: `SsssUniforms` uniform buffer.
    uniforms_layout: Tracked<wgpu::BindGroupLayout>,
    /// Group 1: colour texture + normal + depth + profiles storage + sampler.
    textures_layout: Tracked<wgpu::BindGroupLayout>,

    // === Shared Resources (persistent) ===
    linear_sampler: Tracked<wgpu::Sampler>,

    /// 256-entry GPU storage buffer for SSS profiles.
    profiles_buffer: Option<wgpu::Buffer>,
    /// Uniform buffer for the horizontal pass (direction = [1, 0]).
    h_uniforms_buffer: Option<wgpu::Buffer>,
    /// Uniform buffer for the vertical pass (direction = [0, 1]).
    v_uniforms_buffer: Option<wgpu::Buffer>,

    // === Per-Frame State ===
    /// Intermediate "ping" texture: H pass writes here, V pass reads here.
    ping_texture_id: Option<TransientTextureId>,

    // Bind groups for H sub-pass
    h_uniforms_bg: Option<wgpu::BindGroup>,
    h_textures_bg: Option<wgpu::BindGroup>,

    // Bind groups for V sub-pass
    v_uniforms_bg: Option<wgpu::BindGroup>,
    v_textures_bg: Option<wgpu::BindGroup>,
}

impl SssssPass {
    /// Creates a new SSSSS pass, allocating persistent GPU resources.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        // ── Group 0: Uniforms Layout ─────────────────────────────────────────
        let uniforms_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SSSSS Uniforms Layout"),
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
            });

        // ── Group 1: Textures + Storage Buffer Layout ────────────────────────
        let textures_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SSSSS Textures Layout"),
                entries: &[
                    // Binding 0: Colour texture (HDR input — scene colour or ping)
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
                    // Binding 1: Normal texture (Rgba8Unorm, alpha = SSS ID)
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
                    // Binding 2: Depth texture
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
                    // Binding 3: SSS Profiles StorageBuffer (array<ScreenSpaceMaterialData>)
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
                    // Binding 4: Linear sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // ── Linear Sampler ───────────────────────────────────────────────────
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SSSSS Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            enabled: false,
            pipeline: None,

            uniforms_layout: Tracked::new(uniforms_layout),
            textures_layout: Tracked::new(textures_layout),
            linear_sampler: Tracked::new(linear_sampler),

            profiles_buffer: None,
            h_uniforms_buffer: None,
            v_uniforms_buffer: None,

            ping_texture_id: None,
            h_uniforms_bg: None,
            h_textures_bg: None,
            v_uniforms_bg: None,
            v_textures_bg: None,
        }
    }

    // ── Pipeline ─────────────────────────────────────────────────────────────

    fn ensure_pipeline(&mut self, ctx: &PrepareContext) {
        if self.pipeline.is_some() {
            return;
        }

        let device = &ctx.wgpu_ctx.device;

        let shader_code = ShaderGenerator::generate_shader(
            "",
            "",
            "passes/ssss",
            &ShaderCompilationOptions::default(),
        );

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SSSSS Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_code)),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSSSS Pipeline Layout"),
            bind_group_layouts: &[&self.uniforms_layout, &self.textures_layout],
            immediate_size: 0,
        });

        self.pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SSSSS Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SSSS_COLOR_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        }));
    }

    // ── GPU Buffers ───────────────────────────────────────────────────────────

    /// Lazily creates the profiles storage buffer with capacity for 256 entries.
    fn ensure_profiles_buffer(&mut self, device: &wgpu::Device) {
        if self.profiles_buffer.is_some() {
            return;
        }
        self.profiles_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SSSSS Profiles StorageBuffer"),
            size: PROFILES_BUFFER_CAPACITY,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }

    fn ensure_uniforms_buffers(&mut self, device: &wgpu::Device) {
        let size = std::mem::size_of::<SsssUniforms>() as u64;
        let usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;

        if self.h_uniforms_buffer.is_none() {
            self.h_uniforms_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SSSSS H Uniforms Buffer"),
                size,
                usage,
                mapped_at_creation: false,
            }));
        }
        if self.v_uniforms_buffer.is_none() {
            self.v_uniforms_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SSSSS V Uniforms Buffer"),
                size,
                usage,
                mapped_at_creation: false,
            }));
        }
    }

    // ── Transient Allocation ─────────────────────────────────────────────────

    fn allocate_ping_texture(&mut self, ctx: &mut PrepareContext) {
        let (w, h) = ctx.wgpu_ctx.size();
        self.ping_texture_id = Some(ctx.transient_pool.allocate(
            &ctx.wgpu_ctx.device,
            &TransientTextureDesc {
                width: w,
                height: h,
                format: SSSS_COLOR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                mip_level_count: 1,
                label: "SSSSS Ping",
            },
        ));
    }

    // ── BindGroup Construction ────────────────────────────────────────────────

    fn build_bind_groups(&mut self, ctx: &mut PrepareContext) {
        let device = &ctx.wgpu_ctx.device;

        let normal_view = ctx
            .frame_resources
            .scene_normal_view
            .as_ref()
            .expect("SceneNormal must exist for SSSSS pass");
        let depth_view = &ctx.frame_resources.depth_view;
        let scene_color_view = &ctx.frame_resources.scene_color_view[0];
        let ping_view = ctx
            .transient_pool
            .get_view(self.ping_texture_id.expect("Ping texture must be allocated"));

        let profiles_buf = self.profiles_buffer.as_ref().unwrap();
        let h_buf = self.h_uniforms_buffer.as_ref().unwrap();
        let v_buf = self.v_uniforms_buffer.as_ref().unwrap();

        // H Uniforms BindGroup (Group 0 for H pass)
        {
            let key = BindGroupKey::new(self.uniforms_layout.id())
                .with_resource(h_buf as *const _ as u64);
            let bg = ctx.global_bind_group_cache.get(&key).cloned().unwrap_or_else(|| {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SSSSS H Uniforms BG"),
                    layout: &self.uniforms_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: h_buf.as_entire_binding(),
                    }],
                });
                ctx.global_bind_group_cache.insert(key, bg.clone());
                bg
            });
            self.h_uniforms_bg = Some(bg);
        }

        // H Textures BindGroup (Group 1 for H pass — reads scene colour)
        {
            let key = BindGroupKey::new(self.textures_layout.id())
                .with_resource(scene_color_view.id())
                .with_resource(normal_view.id())
                .with_resource(depth_view.id())
                .with_resource(profiles_buf as *const _ as u64)
                .with_resource(self.linear_sampler.id());

            let bg = ctx.global_bind_group_cache.get(&key).cloned().unwrap_or_else(|| {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SSSSS H Textures BG"),
                    layout: &self.textures_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(scene_color_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(normal_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: profiles_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                        },
                    ],
                });
                ctx.global_bind_group_cache.insert(key, bg.clone());
                bg
            });
            self.h_textures_bg = Some(bg);
        }

        // V Uniforms BindGroup (Group 0 for V pass)
        {
            let key = BindGroupKey::new(self.uniforms_layout.id())
                .with_resource(v_buf as *const _ as u64);
            let bg = ctx.global_bind_group_cache.get(&key).cloned().unwrap_or_else(|| {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SSSSS V Uniforms BG"),
                    layout: &self.uniforms_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: v_buf.as_entire_binding(),
                    }],
                });
                ctx.global_bind_group_cache.insert(key, bg.clone());
                bg
            });
            self.v_uniforms_bg = Some(bg);
        }

        // V Textures BindGroup (Group 1 for V pass — reads ping texture)
        {
            let key = BindGroupKey::new(self.textures_layout.id())
                .with_resource(ping_view.id())
                .with_resource(normal_view.id())
                .with_resource(depth_view.id())
                .with_resource(profiles_buf as *const _ as u64)
                .with_resource(self.linear_sampler.id());

            let bg = ctx.global_bind_group_cache.get(&key).cloned().unwrap_or_else(|| {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("SSSSS V Textures BG"),
                    layout: &self.textures_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(ping_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(normal_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: profiles_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                        },
                    ],
                });
                ctx.global_bind_group_cache.insert(key, bg.clone());
                bg
            });
            self.v_textures_bg = Some(bg);
        }
    }
}

impl Default for SssssPass {
    fn default() -> Self {
        panic!("SssssPass requires a wgpu Device; use SssssPass::new(&device)")
    }
}

impl RenderNode for SssssPass {
    fn name(&self) -> &'static str {
        "SSSSS Pass"
    }

    fn prepare(&mut self, ctx: &mut PrepareContext) {
        // Zero-overhead guard: only activate when SSS materials are present.
        self.enabled = ctx.extracted_scene.has_screen_space_features;
        if !self.enabled {
            return;
        }

        // Guard: SSSSS requires a normal buffer (HighFidelity path with Z-prepass).
        if ctx.frame_resources.scene_normal_view.is_none() {
            log::warn!(
                "[SssssPass] SceneNormal buffer unavailable \
                 (requires HighFidelity path + depth-normal prepass). Skipping."
            );
            self.enabled = false;
            return;
        }

        let device = &ctx.wgpu_ctx.device;
        let (w, h) = ctx.wgpu_ctx.size();
        let aspect = w as f32 / h as f32;
        let texel_size = [1.0 / w as f32, 1.0 / h as f32];

        // 1. Ensure persistent GPU resources
        self.ensure_pipeline(ctx);
        self.ensure_profiles_buffer(device);
        self.ensure_uniforms_buffers(device);

        // 2. Upload SSS profiles to GPU (only when profiles changed)
        if ctx.extracted_scene.screen_space_profiles_changed {
            let profiles = &ctx.extracted_scene.current_screen_space_profiles;
            let data = cast_slice::<ScreenSpaceMaterialData, u8>(profiles.as_slice());
            // Upload only as many bytes as are populated (rest keep previous zeros)
            ctx.wgpu_ctx.queue.write_buffer(
                self.profiles_buffer.as_ref().unwrap(),
                0,
                data,
            );
        }

        // 3. Upload H/V uniforms
        let h_uniforms = SsssUniforms {
            texel_size,
            direction: [1.0, 0.0],
            aspect_ratio: aspect,
            _padding: [0.0; 3],
        };
        let v_uniforms = SsssUniforms {
            texel_size,
            direction: [0.0, 1.0],
            aspect_ratio: aspect,
            _padding: [0.0; 3],
        };
        ctx.wgpu_ctx.queue.write_buffer(
            self.h_uniforms_buffer.as_ref().unwrap(),
            0,
            cast_slice::<SsssUniforms, u8>(&[h_uniforms]),
        );
        ctx.wgpu_ctx.queue.write_buffer(
            self.v_uniforms_buffer.as_ref().unwrap(),
            0,
            cast_slice::<SsssUniforms, u8>(&[v_uniforms]),
        );

        // 4. Allocate ping transient texture
        self.allocate_ping_texture(ctx);

        // 5. Publish ping texture ID for potential downstream compositing
        ctx.blackboard.sss_texture_id = self.ping_texture_id;

        // 6. Build per-frame bind groups
        self.build_bind_groups(ctx);
    }

    fn run(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        if !self.enabled {
            return;
        }

        let Some(pipeline) = &self.pipeline else {
            return;
        };
        let Some(h_uniforms_bg) = &self.h_uniforms_bg else {
            return;
        };
        let Some(h_textures_bg) = &self.h_textures_bg else {
            return;
        };
        let Some(v_uniforms_bg) = &self.v_uniforms_bg else {
            return;
        };
        let Some(v_textures_bg) = &self.v_textures_bg else {
            return;
        };
        let ping_id = self.ping_texture_id.expect("Ping texture allocated");

        // ── H Sub-Pass: Horizontal blur  scene_colour → ping ─────────────────
        {
            let ping_view = ctx.transient_pool.get_view(ping_id);

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSSSS H Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ping_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::DontCare(wgpu::LoadOpDontCare::default()),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });

            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, h_uniforms_bg, &[]);
            pass.set_bind_group(1, h_textures_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ── V Sub-Pass: Vertical blur  ping → scene_colour ───────────────────
        {
            let scene_color_view: &wgpu::TextureView = ctx.get_scene_render_target_view();

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSSSS V Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: scene_color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Load existing colour so non-SSS pixels are preserved unchanged.
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });

            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, v_uniforms_bg, &[]);
            pass.set_bind_group(1, v_textures_bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
