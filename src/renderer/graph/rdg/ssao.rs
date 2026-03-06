//! RDG Screen Space Ambient Occlusion (SSAO) Pass
//!
//! Implements production-grade SSAO within the new RDG framework.
//! This pass receives depth and normal textures as external resources
//! from the old system and produces a blurred AO texture.
//!
//! # RDG Slots
//!
//! - `depth_tex`: Scene depth buffer (external, from old system prepass)
//! - `normal_tex`: Scene normal buffer (external, from old system prepass)
//! - `output_tex`: Blurred AO texture (transient, R8Unorm)
//!
//! # Internal Sub-Passes
//!
//! 1. **Raw SSAO**: Hemisphere sampling with kernel, produces noisy R8Unorm
//! 2. **Cross-Bilateral Blur**: Depth/normal-aware spatial filter
//!
//! # Push Model
//!
//! All parameters (enabled flag, uniform buffer ID, noise scale) are pushed
//! by the Composer. The pass never accesses Scene directly.

use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::resources::Tracked;
use crate::renderer::graph::rdg::allocator::SubViewKey;
use crate::renderer::graph::rdg::builder::PassBuilder;
use crate::renderer::graph::rdg::context::{RdgExecuteContext, RdgPrepareContext};
use crate::renderer::graph::rdg::node::PassNode;
use crate::renderer::graph::rdg::types::TextureNodeId;
use crate::renderer::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions,
};
use crate::resources::ssao::{SsaoUniforms, generate_ssao_noise};
use crate::resources::uniforms::WgslStruct;

/// The SSAO output texture format: single-channel unsigned normalized.
const SSAO_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// RDG Screen Space Ambient Occlusion pass.
///
/// Produces a single-channel AO texture from depth and normal inputs.
/// The output can be consumed by downstream passes (e.g. composited
/// during tone mapping or bound as a screen-space resource).
pub struct RdgSsaoPass {
    // ─── RDG Resource Slots (set by Composer) ──────────────────────
    pub depth_tex: TextureNodeId,
    pub normal_tex: TextureNodeId,
    pub output_tex: TextureNodeId,

    // ─── Push Parameters (set by Composer from Scene) ──────────────
    /// CPU buffer ID for `CpuBuffer<SsaoUniforms>`.
    pub uniforms_cpu_id: u64,
    /// Global state key (render_state_id, scene_id).
    pub global_state_key: (u32, u32),

    // ─── Pipelines ─────────────────────────────────────────────────
    raw_pipeline: Option<RenderPipelineId>,
    blur_pipeline: Option<RenderPipelineId>,

    // ─── Bind Group Layouts ────────────────────────────────────────
    raw_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    raw_uniforms_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    blur_layout: Option<Tracked<wgpu::BindGroupLayout>>,

    // ─── Samplers ──────────────────────────────────────────────────
    linear_sampler: Option<Tracked<wgpu::Sampler>>,
    noise_sampler: Option<Tracked<wgpu::Sampler>>,
    point_sampler: Option<Tracked<wgpu::Sampler>>,

    // ─── Persistent Resources ──────────────────────────────────────
    noise_texture_view: Option<Tracked<wgpu::TextureView>>,

    // ─── Internal Transient Texture (raw AO, owned by pass) ───────
    raw_texture: Option<wgpu::Texture>,
    raw_texture_view: Option<Tracked<wgpu::TextureView>>,
    raw_texture_size: (u32, u32),

    // ─── Cached BindGroups ─────────────────────────────────────────
    raw_bind_group_key: Option<BindGroupKey>,
    raw_uniforms_bind_group_key: Option<BindGroupKey>,
    blur_bind_group_key: Option<BindGroupKey>,
}

impl RdgSsaoPass {
    #[must_use]
    pub fn new() -> Self {
        Self {
            depth_tex: TextureNodeId(0),
            normal_tex: TextureNodeId(0),
            output_tex: TextureNodeId(0),

            uniforms_cpu_id: 0,
            global_state_key: (0, 0),

            raw_pipeline: None,
            blur_pipeline: None,

            raw_layout: None,
            raw_uniforms_layout: None,
            blur_layout: None,

            linear_sampler: None,
            noise_sampler: None,
            point_sampler: None,
            noise_texture_view: None,

            raw_texture: None,
            raw_texture_view: None,
            raw_texture_size: (0, 0),

            raw_bind_group_key: None,
            raw_uniforms_bind_group_key: None,
            blur_bind_group_key: None,
        }
    }

    // =========================================================================
    // Lazy Initialization
    // =========================================================================

    fn ensure_layouts_and_samplers(&mut self, device: &wgpu::Device) {
        if self.raw_layout.is_some() {
            return;
        }

        // ─── Raw SSAO Layout (Group 1): depth, normal, noise + samplers ───
        let raw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RDG SSAO Raw Layout"),
            entries: &[
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        // ─── Uniforms Layout (Group 2) ─────────────────────────────
        let raw_uniforms_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RDG SSAO Uniforms Layout"),
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

        // ─── Blur Layout (Group 0): raw AO + depth + normal + samplers ────
        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RDG SSAO Blur Layout"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        self.raw_layout = Some(Tracked::new(raw_layout));
        self.raw_uniforms_layout = Some(Tracked::new(raw_uniforms_layout));
        self.blur_layout = Some(Tracked::new(blur_layout));

        // ─── Samplers ──────────────────────────────────────────────
        self.linear_sampler = Some(Tracked::new(device.create_sampler(
            &wgpu::SamplerDescriptor {
                label: Some("RDG SSAO Linear Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            },
        )));

        self.noise_sampler = Some(Tracked::new(device.create_sampler(
            &wgpu::SamplerDescriptor {
                label: Some("RDG SSAO Noise Sampler"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            },
        )));

        self.point_sampler = Some(Tracked::new(device.create_sampler(
            &wgpu::SamplerDescriptor {
                label: Some("RDG SSAO Point Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            },
        )));
    }

    fn ensure_noise_texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.noise_texture_view.is_some() {
            return;
        }

        let noise_data = generate_ssao_noise();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RDG SSAO Noise 4x4"),
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
                bytes_per_row: Some(4 * 4),
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

    fn ensure_pipelines(&mut self, ctx: &mut RdgPrepareContext) {
        if self.raw_pipeline.is_some() {
            return;
        }

        let device = ctx.device;
        let raw_layout = self.raw_layout.as_ref().unwrap();
        let uniforms_layout = self.raw_uniforms_layout.as_ref().unwrap();
        let blur_layout = self.blur_layout.as_ref().unwrap();

        let gpu_world = ctx
            .resource_manager
            .get_global_state(self.global_state_key.0, self.global_state_key.1)
            .expect("RDG SSAO: GpuGlobalState must exist");

        let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
            format: SSAO_TEXTURE_FORMAT,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        });

        // ─── Raw SSAO Pipeline ─────────────────────────────────────
        {
            let mut options = ShaderCompilationOptions::default();
            options.add_define(
                "struct_definitions",
                SsaoUniforms::wgsl_struct_def("SsaoUniforms").as_str(),
            );

            let (module, hash) = ctx.shader_manager.get_or_compile_template(
                device,
                "passes/ssao_raw",
                &options,
                "",
                &gpu_world.binding_wgsl,
            );

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RDG SSAO Raw Pipeline Layout"),
                bind_group_layouts: &[&gpu_world.layout, raw_layout, uniforms_layout],
                immediate_size: 0,
            });

            let key = FullscreenPipelineKey::fullscreen(
                hash,
                smallvec::smallvec![color_target.clone()],
                None,
            );

            self.raw_pipeline = Some(ctx.pipeline_cache.get_or_create_fullscreen(
                device,
                module,
                &pipeline_layout,
                &key,
                "RDG SSAO Raw Pipeline",
            ));
        }

        // ─── Blur Pipeline ─────────────────────────────────────────
        {
            let (module, hash) = ctx.shader_manager.get_or_compile_template(
                device,
                "passes/ssao_blur",
                &ShaderCompilationOptions::default(),
                "",
                "",
            );

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RDG SSAO Blur Pipeline Layout"),
                bind_group_layouts: &[blur_layout],
                immediate_size: 0,
            });

            let key =
                FullscreenPipelineKey::fullscreen(hash, smallvec::smallvec![color_target], None);

            self.blur_pipeline = Some(ctx.pipeline_cache.get_or_create_fullscreen(
                device,
                module,
                &pipeline_layout,
                &key,
                "RDG SSAO Blur Pipeline",
            ));
        }
    }

    // =========================================================================
    // Internal Raw AO Texture
    // =========================================================================

    fn ensure_raw_texture(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        // SSAO raw pass runs at half resolution
        let raw_w = (width / 2).max(1);
        let raw_h = (height / 2).max(1);

        if self.raw_texture_size == (raw_w, raw_h) && self.raw_texture.is_some() {
            return;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RDG SSAO Raw Texture"),
            size: wgpu::Extent3d {
                width: raw_w,
                height: raw_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SSAO_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.raw_texture = Some(texture);
        self.raw_texture_view = Some(Tracked::new(view));
        self.raw_texture_size = (raw_w, raw_h);

        // Invalidate bind groups
        self.raw_bind_group_key = None;
        self.blur_bind_group_key = None;
    }

    // =========================================================================
    // BindGroup Construction
    // =========================================================================

    fn build_bind_groups(&mut self, ctx: &mut RdgPrepareContext) {
        let device = ctx.device;

        // Extract physical texture view IDs and raw pointers up front
        // to avoid holding the immutable borrow on `ctx` across mutable cache operations.
        // let depth_view_id = ctx.get_texture_view(self.depth_tex).id();
        let normal_view_id = ctx.get_texture_view(self.normal_tex).id();
        // SAFETY: We use raw pointers to break the borrow conflict. The views
        // remain valid for the entire scope since ctx/pool outlive them.
        let depth_only_view = ctx.get_or_create_sub_view(
            self.depth_tex,
            SubViewKey {
                aspect: wgpu::TextureAspect::DepthOnly,
                ..Default::default()
            },
        );
        let depth_view_id = depth_only_view.id();
        let depth_view_ptr = depth_only_view as *const Tracked<wgpu::TextureView>;

        // let depth_view_ptr =
        //     ctx.get_texture_view(self.depth_tex) as *const Tracked<wgpu::TextureView>;
        let normal_view_ptr =
            ctx.get_texture_view(self.normal_tex) as *const Tracked<wgpu::TextureView>;

        let noise_view = self.noise_texture_view.as_ref().unwrap();
        let raw_view = self.raw_texture_view.as_ref().unwrap();

        let linear_sampler = self.linear_sampler.as_ref().unwrap();
        let noise_sampler = self.noise_sampler.as_ref().unwrap();
        let point_sampler = self.point_sampler.as_ref().unwrap();

        let raw_layout = self.raw_layout.as_ref().unwrap();
        let uniforms_layout = self.raw_uniforms_layout.as_ref().unwrap();
        let blur_layout = self.blur_layout.as_ref().unwrap();

        // ─── Raw SSAO BindGroup (Group 1) ──────────────────────────
        {
            let key = BindGroupKey::new(raw_layout.id())
                .with_resource(depth_view_id)
                .with_resource(normal_view_id)
                .with_resource(noise_view.id())
                .with_resource(linear_sampler.id())
                .with_resource(noise_sampler.id())
                .with_resource(point_sampler.id());

            if self.raw_bind_group_key.as_ref() != Some(&key) {
                if ctx.global_bind_group_cache.get(&key).is_none() {
                    // SAFETY: depth/normal views are alive for entire frame
                    let depth_view = unsafe { &*depth_view_ptr };
                    let normal_view = unsafe { &*normal_view_ptr };
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("RDG SSAO Raw BG"),
                        layout: raw_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&**depth_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&**normal_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&**noise_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::Sampler(&**linear_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::Sampler(&**noise_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::Sampler(&**point_sampler),
                            },
                        ],
                    });
                    ctx.global_bind_group_cache.insert(key.clone(), bg);
                }
                self.raw_bind_group_key = Some(key);
            }
        }

        // ─── Raw SSAO Uniforms BindGroup (Group 2) ─────────────────
        {
            let gpu_buffer = ctx
                .resource_manager
                .gpu_buffers
                .get(&self.uniforms_cpu_id)
                .expect("RDG SSAO: uniforms GPU buffer must exist");

            let key = BindGroupKey::new(uniforms_layout.id()).with_resource(gpu_buffer.id);

            if self.raw_uniforms_bind_group_key.as_ref() != Some(&key) {
                if ctx.global_bind_group_cache.get(&key).is_none() {
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("RDG SSAO Uniforms BG"),
                        layout: uniforms_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: gpu_buffer.buffer.as_entire_binding(),
                        }],
                    });
                    ctx.global_bind_group_cache.insert(key.clone(), bg);
                }
                self.raw_uniforms_bind_group_key = Some(key);
            }
        }

        // ─── Blur BindGroup (Group 0) ──────────────────────────────
        {
            let key = BindGroupKey::new(blur_layout.id())
                .with_resource(raw_view.id())
                .with_resource(depth_view_id)
                .with_resource(normal_view_id)
                .with_resource(linear_sampler.id())
                .with_resource(point_sampler.id());

            if self.blur_bind_group_key.as_ref() != Some(&key) {
                if ctx.global_bind_group_cache.get(&key).is_none() {
                    // SAFETY: depth/normal views alive for entire frame
                    let depth_view = unsafe { &*depth_view_ptr };
                    let normal_view = unsafe { &*normal_view_ptr };
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("RDG SSAO Blur BG"),
                        layout: blur_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&**raw_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&**depth_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&**normal_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::Sampler(&**linear_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::Sampler(&**point_sampler),
                            },
                        ],
                    });
                    ctx.global_bind_group_cache.insert(key.clone(), bg);
                }
                self.blur_bind_group_key = Some(key);
            }
        }
    }
}

impl PassNode for RdgSsaoPass {
    fn name(&self) -> &'static str {
        "RDG_SSAO_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        // Self-wire well-known resources from the registry.
        self.depth_tex = builder.find_resource("Scene_Depth")
            .expect("Scene_Depth must be registered before RdgSsaoPass");
        self.normal_tex = builder.find_resource("Scene_Normals")
            .expect("Scene_Normals must be registered before RdgSsaoPass");
        self.output_tex = builder.find_resource("SSAO_Output")
            .expect("SSAO_Output must be registered before RdgSsaoPass");

        builder.read_texture(self.depth_tex);
        builder.read_texture(self.normal_tex);
        builder.write_texture(self.output_tex);
    }

    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {
        // 1. Lazy initialization
        self.ensure_layouts_and_samplers(ctx.device);
        self.ensure_noise_texture(ctx.device, ctx.queue);
        self.ensure_pipelines(ctx);

        // 2. Internal raw AO texture
        let output_desc = &ctx.graph.resources[self.output_tex.0 as usize].desc;
        self.ensure_raw_texture(ctx.device, output_desc.size.width, output_desc.size.height);

        // 3. Bind groups
        self.build_bind_groups(ctx);
    }

    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let Some(raw_pipeline_id) = self.raw_pipeline else {
            return;
        };
        let Some(blur_pipeline_id) = self.blur_pipeline else {
            return;
        };
        let Some(raw_bg_key) = &self.raw_bind_group_key else {
            return;
        };
        let Some(uniforms_bg_key) = &self.raw_uniforms_bind_group_key else {
            return;
        };
        let Some(blur_bg_key) = &self.blur_bind_group_key else {
            return;
        };
        let Some(global_bg) = ctx.global_bind_group else {
            log::warn!("RDG SSAO: global_bind_group missing, skipping");
            return;
        };

        let raw_bg = ctx
            .global_bind_group_cache
            .get(raw_bg_key)
            .expect("SSAO raw BG should exist");
        let uniforms_bg = ctx
            .global_bind_group_cache
            .get(uniforms_bg_key)
            .expect("SSAO uniforms BG should exist");
        let blur_bg = ctx
            .global_bind_group_cache
            .get(blur_bg_key)
            .expect("SSAO blur BG should exist");

        let raw_pipeline = ctx.pipeline_cache.get_render_pipeline(raw_pipeline_id);
        let blur_pipeline = ctx.pipeline_cache.get_render_pipeline(blur_pipeline_id);

        let raw_view = self
            .raw_texture_view
            .as_ref()
            .expect("Raw texture must be allocated");

        // =====================================================================
        // Sub-Pass 1: Raw SSAO
        // =====================================================================
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("RDG SSAO Raw Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: raw_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::DontCare(wgpu::LoadOpDontCare::default()),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_pipeline(raw_pipeline);
            pass.set_bind_group(0, global_bg, &[]);
            pass.set_bind_group(1, raw_bg, &[]);
            pass.set_bind_group(2, uniforms_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // =====================================================================
        // Sub-Pass 2: Cross-Bilateral Blur
        // =====================================================================
        {
            let output_view = ctx.get_texture_view(self.output_tex);

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("RDG SSAO Blur Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::DontCare(wgpu::LoadOpDontCare::default()),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_pipeline(blur_pipeline);
            pass.set_bind_group(0, blur_bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
