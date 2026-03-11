//! RDG Compute Passes (BRDF LUT & IBL)
//!
//! # BRDF LUT
//!
//! Precomputes the Cook-Torrance BRDF integration lookup table into a 2D
//! storage texture. This is a one-shot task that runs once when the engine
//! initialises the environment. The compute dispatch is recorded into the
//! shared RDG encoder during the execute phase.
//!
//! # IBL (Image-Based Lighting)
//!
//! Converts an equirectangular / cube-map source into a mipmapped PMREM
//! cube-map used for specular environment reflections. Due to its multi-phase
//! nature (equirect → cube, mipmap generation, PMREM prefiltering) the work
//! is performed during the prepare phase with a dedicated command encoder.

use crate::renderer::core::gpu::SamplerKey;
use crate::renderer::core::gpu::{BRDF_LUT_SIZE, CubeSourceType};
use crate::renderer::graph::core::builder::PassBuilder;
use crate::renderer::graph::core::context::{ExecuteContext, ExtractContext};
use crate::renderer::graph::core::graph::RenderGraph;
use crate::renderer::graph::core::node::PassNode;
use crate::renderer::pipeline::{
    ColorTargetKey, ComputePipelineId, ComputePipelineKey, FullscreenPipelineKey, RenderPipelineId,
};
use crate::resources::texture::TextureSource;
use wgpu::TextureViewDimension;

// ============================================================================
// BRDF LUT Compute Pass
// ============================================================================

/// BRDF LUT compute feature.
///
/// Dispatches a single compute shader that fills a 128×128 `Rgba16Float`
/// storage texture with pre-integrated BRDF data. The texture is managed
/// by [`ResourceManager`] and referenced through the global bind group.
///
/// Produces an ephemeral [`BrdfLutPassNode`] each frame via [`Self::add_to_graph`].
pub struct BrdfLutFeature {
    pipeline_id: Option<ComputePipelineId>,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    active: bool,
}

impl BrdfLutFeature {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BRDF LUT BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });

        Self {
            pipeline_id: None,
            bind_group_layout,
            bind_group: None,
            active: false,
        }
    }

    /// Lazily create the compute pipeline via the global [`PipelineCache`].
    fn ensure_pipeline(&mut self, ctx: &mut ExtractContext) {
        if self.pipeline_id.is_some() {
            return;
        }

        let source = include_str!("../../pipeline/shaders/program/brdf_lut.wgsl");
        let (module, shader_hash) =
            ctx.shader_manager
                .get_or_compile_raw(ctx.device, "BRDF LUT Shader", source);

        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("BRDF LUT Pipeline Layout"),
                bind_group_layouts: &[&self.bind_group_layout],
                immediate_size: 0,
            });

        let key = ComputePipelineKey { shader_hash };
        self.pipeline_id = Some(ctx.pipeline_cache.get_or_create_compute(
            ctx.device,
            module,
            &layout,
            &key,
            "BRDF LUT Pipeline",
        ));
    }
}

impl BrdfLutFeature {
    /// Extract and prepare BRDF LUT compute resources.
    pub fn extract_and_prepare(&mut self, ctx: &mut ExtractContext) {
        if !ctx.resource_manager.needs_brdf_compute {
            self.active = false;
            return;
        }

        self.ensure_pipeline(ctx);

        let texture = ctx
            .resource_manager
            .brdf_lut_texture
            .as_ref()
            .expect("LUT texture must be created by ensure_brdf_lut");

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BRDF LUT BG"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        }));

        self.active = true;
        ctx.resource_manager.needs_brdf_compute = false;
    }

    /// Create an ephemeral [`BrdfLutPassNode`] and add it to the render graph.
    pub fn add_to_graph(&self, graph: &mut RenderGraph) {
        let node = BrdfLutPassNode {
            pipeline_id: self.pipeline_id,
            bind_group: self.bind_group.clone(),
            active: self.active,
        };
        graph.add_pass(Box::new(node));
    }
}

// ─── BRDF LUT Pass Node ──────────────────────────────────────────────────────

/// Ephemeral per-frame BRDF LUT compute pass node.
struct BrdfLutPassNode {
    pipeline_id: Option<ComputePipelineId>,
    bind_group: Option<wgpu::BindGroup>,
    active: bool,
}

impl PassNode for BrdfLutPassNode {
    fn name(&self) -> &'static str {
        "BRDF_LUT"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        builder.mark_side_effect();
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        if !self.active {
            return;
        }

        let Some(bg) = &self.bind_group else {
            return;
        };

        let pipeline = ctx
            .pipeline_cache
            .get_compute_pipeline(self.pipeline_id.expect("Pipeline must exist"));

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("BRDF LUT"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, bg, &[]);
        cpass.dispatch_workgroups(BRDF_LUT_SIZE / 8, BRDF_LUT_SIZE / 8, 1);
    }
}

// ============================================================================
// IBL Compute Pass
// ============================================================================

/// Trilinear filtering, all-axes clamped. Used for the face blit sub-pass.
const IBL_BLIT_SAMPLER_KEY: SamplerKey = SamplerKey::LINEAR_CLAMP;

/// Linear filtering with horizontal repeat (seamless equirectangular wrap)
/// and vertical/depth clamp. Mipmap filtering is not needed for the single-mip
/// equirect source.
const IBL_EQUIRECT_SAMPLER_KEY: SamplerKey = SamplerKey {
    address_mode_u: wgpu::AddressMode::Repeat,
    address_mode_v: wgpu::AddressMode::ClampToEdge,
    address_mode_w: wgpu::AddressMode::ClampToEdge,
    mag_filter: wgpu::FilterMode::Linear,
    min_filter: wgpu::FilterMode::Linear,
    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
    lod_min_clamp: 0.0,
    lod_max_clamp: 32.0,
    compare: None,
    anisotropy_clamp: 1,
    border_color: None,
};

/// IBL (Image-Based Lighting) compute feature.
///
/// Performs equirectangular → cube conversion, mipmap generation, and PMREM
/// pre-filtering for specular environment reflections. Due to the multi-phase
/// nature of the work (render passes for blit + mipmap, compute for PMREM),
/// all GPU commands are submitted during [`Self::extract_and_prepare`] with a
/// dedicated command encoder.
///
/// Produces an ephemeral [`IblPassNode`] each frame via [`Self::add_to_graph`].
pub struct IblComputeFeature {
    // PMREM prefilter
    pmrem_pipeline_id: Option<ComputePipelineId>,
    pmrem_layout_source: wgpu::BindGroupLayout,
    pmrem_layout_dest: wgpu::BindGroupLayout,

    // Equirectangular → Cube
    equirect_pipeline_id: Option<ComputePipelineId>,
    equirect_layout: wgpu::BindGroupLayout,

    // Face blit (CubeNoMipmaps → owned cube)
    blit_pipeline_id: Option<RenderPipelineId>,
    blit_layout: wgpu::BindGroupLayout,
}

impl IblComputeFeature {
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn new(device: &wgpu::Device) -> Self {
        // ====== PMREM prefilter layouts ======
        let pmrem_layout_source =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("IBL Source BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pmrem_layout_dest = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("IBL Dest BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                },
                count: None,
            }],
        });

        // ====== Equirectangular → Cube layouts ======
        let equirect_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Equirect BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ],
        });

        // ====== Blit layout + sampler ======
        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("IBL Blit BGL"),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        Self {
            pmrem_pipeline_id: None,
            pmrem_layout_source,
            pmrem_layout_dest,
            equirect_pipeline_id: None,
            equirect_layout,
            blit_pipeline_id: None,
            blit_layout,
        }
    }

    /// Lazily create all three pipelines via the global [`PipelineCache`].
    fn ensure_pipelines(&mut self, ctx: &mut ExtractContext) {
        if self.pmrem_pipeline_id.is_some() {
            return;
        }

        let device = ctx.device;

        // --- PMREM compute pipeline ---
        {
            let source = include_str!("../../pipeline/shaders/program/ibl.wgsl");
            let (module, hash) =
                ctx.shader_manager
                    .get_or_compile_raw(device, "IBL Prefilter Shader", source);

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("IBL Pipeline Layout"),
                bind_group_layouts: &[&self.pmrem_layout_source, &self.pmrem_layout_dest],
                immediate_size: 0,
            });

            let key = ComputePipelineKey { shader_hash: hash };
            self.pmrem_pipeline_id = Some(ctx.pipeline_cache.get_or_create_compute(
                device,
                module,
                &layout,
                &key,
                "IBL Compute Pipeline",
            ));
        }

        // --- Equirectangular → Cube compute pipeline ---
        {
            let source = include_str!("../../pipeline/shaders/program/equirect_to_cube.wgsl");
            let (module, hash) =
                ctx.shader_manager
                    .get_or_compile_raw(device, "Equirect to Cube Shader", source);

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Equirect Pipeline Layout"),
                bind_group_layouts: &[&self.equirect_layout],
                immediate_size: 0,
            });

            let key = ComputePipelineKey { shader_hash: hash };
            self.equirect_pipeline_id = Some(ctx.pipeline_cache.get_or_create_compute(
                device,
                module,
                &layout,
                &key,
                "Equirect to Cube Pipeline",
            ));
        }

        // --- Blit render pipeline ---
        {
            let source = include_str!("../../pipeline/shaders/program/blit.wgsl");
            let (module, hash) =
                ctx.shader_manager
                    .get_or_compile_raw(device, "IBL Blit Shader", source);

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("IBL Blit Pipeline Layout"),
                bind_group_layouts: &[&self.blit_layout],
                immediate_size: 0,
            });

            let key = FullscreenPipelineKey::fullscreen(
                hash,
                smallvec::smallvec![ColorTargetKey::from(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                None,
            );

            self.blit_pipeline_id = Some(ctx.pipeline_cache.get_or_create_fullscreen(
                device,
                module,
                &layout,
                &key,
                "IBL Blit Pipeline",
            ));
        }
    }

    /// Blit individual cube faces from `src_texture` to `dst_texture` mip 0.
    fn blit_cube_faces(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src_texture: &wgpu::Texture,
        dst_texture: &wgpu::Texture,
        blit_pipeline: &wgpu::RenderPipeline,
        blit_sampler: &wgpu::Sampler,
    ) {
        let layer_count = src_texture
            .depth_or_array_layers()
            .min(dst_texture.depth_or_array_layers());

        for layer in 0..layer_count {
            let src_view = src_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Blit Src Face"),
                dimension: Some(TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                base_mip_level: 0,
                mip_level_count: Some(1),
                usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
                ..Default::default()
            });

            let dst_view = dst_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Blit Dst Face"),
                dimension: Some(TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                base_mip_level: 0,
                mip_level_count: Some(1),
                usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
                ..Default::default()
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Blit BG"),
                layout: &self.blit_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(blit_sampler),
                    },
                ],
            });

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blit Cube Face"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst_view,
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
            rpass.set_pipeline(blit_pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }
    }
}

impl IblComputeFeature {
    /// Extract and prepare IBL compute resources.
    ///
    /// All IBL compute work is performed here with a dedicated command
    /// encoder. The multi-phase nature of the task (equirect → cube,
    /// mipmap generation via render passes, PMREM compute dispatches)
    /// requires its own submission boundary.
    #[allow(clippy::too_many_lines)]
    pub fn extract_and_prepare(&mut self, ctx: &mut ExtractContext) {
        let Some(source) = ctx.resource_manager.pending_ibl_source.take() else {
            return;
        };

        self.ensure_pipelines(ctx);

        // Ensure custom samplers exist in the registry. The mutable borrows
        // are released immediately; later code uses get_custom_ref() (shared).
        ctx.sampler_registry
            .get_custom(ctx.device, IBL_BLIT_SAMPLER_KEY);
        ctx.sampler_registry
            .get_custom(ctx.device, IBL_EQUIRECT_SAMPLER_KEY);

        let pmrem_pipeline = ctx
            .pipeline_cache
            .get_compute_pipeline(self.pmrem_pipeline_id.expect("PMREM pipeline must exist"));
        let equirect_pipeline = ctx.pipeline_cache.get_compute_pipeline(
            self.equirect_pipeline_id
                .expect("Equirect pipeline must exist"),
        );
        let blit_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.blit_pipeline_id.expect("Blit pipeline must exist"));

        let mut gpu_env = match ctx.resource_manager.environment_map_cache.remove(&source) {
            Some(env) if env.needs_compute => env,
            Some(env) => {
                ctx.resource_manager
                    .environment_map_cache
                    .insert(source, env);
                return;
            }
            None => return,
        };

        let source_type = gpu_env.source_type;

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("IBL Compute Encoder"),
            });

        // ── Phase 1: Prepare mipmapped cube source for PMREM ───────────
        match source_type {
            CubeSourceType::Equirectangular => {
                let cube_texture = gpu_env
                    .cube_texture
                    .as_ref()
                    .expect("owned cube_texture must exist for Equirectangular source");

                let cube_size = cube_texture.width();

                {
                    let source_view = match &source {
                        TextureSource::Asset(handle) => {
                            let tex_asset = ctx.assets.textures.get(*handle).unwrap();
                            let img_id = tex_asset.image.id();
                            &ctx.resource_manager.get_image(img_id).unwrap().default_view
                        }
                        TextureSource::Attachment(id, _) => {
                            ctx.resource_manager.internal_resources.get(id).unwrap()
                        }
                    };

                    let dest_view = cube_texture.create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(TextureViewDimension::D2Array),
                        mip_level_count: Some(1),
                        base_mip_level: 0,
                        ..Default::default()
                    });

                    let equirect_sampler = ctx
                        .sampler_registry
                        .get_custom_ref(&IBL_EQUIRECT_SAMPLER_KEY);

                    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Equirect BindGroup"),
                        layout: &self.equirect_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(source_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(equirect_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&dest_view),
                            },
                        ],
                    });

                    {
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Equirect to Cube"),
                            timestamp_writes: None,
                        });
                        cpass.set_pipeline(equirect_pipeline);
                        cpass.set_bind_group(0, &bind_group, &[]);
                        let group_count = cube_size.div_ceil(8);
                        cpass.dispatch_workgroups(group_count, group_count, 6);
                    }
                }

                ctx.resource_manager.mipmap_generator.generate(
                    ctx.device,
                    &mut encoder,
                    cube_texture,
                );
            }

            CubeSourceType::CubeNoMipmaps => {
                let cube_texture = gpu_env
                    .cube_texture
                    .as_ref()
                    .expect("owned cube_texture must exist for CubeNoMipmaps source");

                {
                    let source_texture = match &source {
                        TextureSource::Asset(handle) => {
                            let tex_asset = ctx.assets.textures.get(*handle).unwrap();
                            let img_id = tex_asset.image.id();
                            &ctx.resource_manager.get_image(img_id).unwrap().texture
                        }
                        TextureSource::Attachment(_, _) => {
                            gpu_env.needs_compute = false;
                            ctx.resource_manager
                                .environment_map_cache
                                .insert(source, gpu_env);
                            return;
                        }
                    };

                    let blit_sampler = ctx.sampler_registry.get_custom_ref(&IBL_BLIT_SAMPLER_KEY);

                    self.blit_cube_faces(
                        ctx.device,
                        &mut encoder,
                        source_texture,
                        cube_texture,
                        blit_pipeline,
                        blit_sampler,
                    );
                }

                ctx.resource_manager.mipmap_generator.generate(
                    ctx.device,
                    &mut encoder,
                    cube_texture,
                );
            }

            CubeSourceType::CubeWithMipmaps => {}
        }

        // ── Phase 2: PMREM prefiltering ────────────────────────────────
        let pmrem_source_view = match source_type {
            CubeSourceType::Equirectangular | CubeSourceType::CubeNoMipmaps => gpu_env
                .cube_texture
                .as_ref()
                .unwrap()
                .create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(TextureViewDimension::Cube),
                    ..Default::default()
                }),
            CubeSourceType::CubeWithMipmaps => match &source {
                TextureSource::Asset(handle) => {
                    let tex_asset = ctx.assets.textures.get(*handle).unwrap();
                    let img_id = tex_asset.image.id();
                    ctx.resource_manager
                        .get_image(img_id)
                        .unwrap()
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::Cube),
                            ..Default::default()
                        })
                }
                TextureSource::Attachment(_, _) => {
                    gpu_env.needs_compute = false;
                    ctx.resource_manager
                        .environment_map_cache
                        .insert(source, gpu_env);
                    return;
                }
            },
        };

        let mip_levels = gpu_env.pmrem_texture.mip_level_count();
        let pmrem_size = gpu_env.pmrem_texture.width();
        let format = wgpu::TextureFormat::Rgba16Float;

        for mip in 0..mip_levels {
            let mip_size = (pmrem_size >> mip).max(1);
            let roughness = mip as f32 / (mip_levels - 1) as f32;

            let params = [roughness, mip_size as f32, 0.0, 0.0];
            let param_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("IBL Params Buffer"),
                size: std::mem::size_of::<[f32; 4]>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            {
                let mut view = param_buffer.slice(..).get_mapped_range_mut();
                view.copy_from_slice(bytemuck::cast_slice(&params));
            }
            param_buffer.unmap();

            let bg_src = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pmrem_layout_source,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&pmrem_source_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            &ctx.resource_manager.dummy_sampler.sampler,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: param_buffer.as_entire_binding(),
                    },
                ],
            });

            let dest_view = gpu_env
                .pmrem_texture
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("PMREM Mip {mip}")),
                    format: Some(format),
                    dimension: Some(TextureViewDimension::D2Array),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(6),
                    usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                });

            let bg_dst = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pmrem_layout_dest,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&dest_view),
                }],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("IBL Gen"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(pmrem_pipeline);
                cpass.set_bind_group(0, &bg_src, &[]);
                cpass.set_bind_group(1, &bg_dst, &[]);
                let group_count = mip_size.div_ceil(8);
                cpass.dispatch_workgroups(group_count, group_count, 6);
            }
        }

        ctx.queue.submit(std::iter::once(encoder.finish()));

        gpu_env.needs_compute = false;
        ctx.resource_manager
            .environment_map_cache
            .insert(source, gpu_env);

        log::info!("IBL PMREM generated. Source type: {source_type:?}");
    }

    /// Create an ephemeral [`IblPassNode`] and add it to the render graph.
    pub fn add_to_graph(&self, graph: &mut RenderGraph, source: TextureSource) {
        let node = IblPassNode { _source: source };
        graph.add_pass(Box::new(node));
    }
}

// ─── IBL Pass Node ────────────────────────────────────────────────────────────

/// Ephemeral per-frame IBL pass node.
///
/// All IBL work is completed during [`IblComputeFeature::extract_and_prepare`];
/// this node is a no-op placeholder so the graph stays consistent.
struct IblPassNode {
    // todo: move logic from extract_and_prepare here and make this a real pass node that executes the compute work.
    // This would allow better integration with the graph (e.g. explicit dependencies) and remove the need for a separate command encoder and submission in extract_and_prepare.
    // for now, ibl compute is a one-off special case, so it's not worth the refactor yet.
    _source: TextureSource,
}

impl PassNode for IblPassNode {
    fn name(&self) -> &'static str {
        "IBL_Compute"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        builder.mark_side_effect();
    }

    fn execute(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {
        // All IBL compute work is completed during extract_and_prepare.
    }
}
