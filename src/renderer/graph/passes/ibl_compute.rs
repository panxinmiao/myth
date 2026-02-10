use crate::renderer::core::resources::environment::CubeSourceType;
use crate::renderer::graph::{RenderContext, RenderNode};
use crate::resources::texture::TextureSource;
use std::borrow::Cow;
use wgpu::{PipelineCompilationOptions, TextureViewDimension};

pub struct IBLComputePass {
    // PMREM prefilter
    pmrem_pipeline: wgpu::ComputePipeline,
    pmrem_layout_source: wgpu::BindGroupLayout,
    pmrem_layout_dest: wgpu::BindGroupLayout,

    // Equirectangular → Cube
    equirect_pipeline: wgpu::ComputePipeline,
    equirect_layout: wgpu::BindGroupLayout,

    // Face blit (for CubeNoMipmaps: copy source cube → owned cube)
    blit_pipeline: wgpu::RenderPipeline,
    blit_layout: wgpu::BindGroupLayout,
    blit_sampler: wgpu::Sampler,
}

impl IBLComputePass {
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn new(device: &wgpu::Device) -> Self {
        // ====== PMREM prefilter pipeline ======
        let pmrem_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("IBL Prefilter Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../../pipeline/shaders/program/ibl.wgsl"
            ))),
        });

        let pmrem_layout_source =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("IBL Source Layout"),
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
            label: Some("IBL Dest Layout"),
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

        let pmrem_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("IBL Pipeline Layout"),
            bind_group_layouts: &[&pmrem_layout_source, &pmrem_layout_dest],
            immediate_size: 0,
        });

        let pmrem_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("IBL Compute Pipeline"),
            layout: Some(&pmrem_layout),
            module: &pmrem_shader,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        // ====== Equirectangular → Cube pipeline ======
        let equirect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Equirect to Cube Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../../pipeline/shaders/program/equirect_to_cube.wgsl"
            ))),
        });

        let equirect_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Equirect Layout"),
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

        let equirect_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Equirect Pipeline Layout"),
                bind_group_layouts: &[&equirect_layout],
                immediate_size: 0,
            });

        let equirect_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Equirect to Cube Pipeline"),
            layout: Some(&equirect_pipeline_layout),
            module: &equirect_shader,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        // ====== Blit pipeline (for CubeNoMipmaps face copy) ======
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("IBL Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../../pipeline/shaders/program/blit.wgsl"
            ))),
        });

        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("IBL Blit Layout"),
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

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("IBL Blit Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("IBL Blit Pipeline Layout"),
                    bind_group_layouts: &[&blit_layout],
                    immediate_size: 0,
                }),
            ),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("IBL Blit Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        Self {
            pmrem_pipeline,
            pmrem_layout_source,
            pmrem_layout_dest,
            equirect_pipeline,
            equirect_layout,
            blit_pipeline,
            blit_layout,
            blit_sampler,
        }
    }

    /// Blit individual cube faces from `src_texture` to `dst_texture` mip 0
    /// using the blit render pipeline.
    fn blit_cube_faces(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src_texture: &wgpu::Texture,
        dst_texture: &wgpu::Texture,
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
                        resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                    },
                ],
            });

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blit Cube Face"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&self.blit_pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }
    }
}

impl RenderNode for IBLComputePass {
    fn name(&self) -> &'static str {
        "IBL Compute Pass"
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &mut RenderContext, encoder: &mut wgpu::CommandEncoder) {
        let Some(source) = ctx.resource_manager.pending_ibl_source.take() else {
            return;
        };

        // Temporarily take GpuEnvironment to split borrows:
        // we need simultaneous access to textures (in gpu_env) and
        // mipmap_generator / gpu_images (in resource_manager).
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

        // ================================================================
        // Phase 1: Prepare mipmapped cube source for PMREM
        // ================================================================
        match source_type {
            CubeSourceType::Equirectangular => {
                // Step 1a: Equirectangular 2D → Cube mip 0
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

                    let clamp_sampler =
                        ctx.wgpu_ctx
                            .device
                            .create_sampler(&wgpu::SamplerDescriptor {
                                label: Some("IBL Clamp Sampler"),
                                address_mode_u: wgpu::AddressMode::Repeat,
                                address_mode_v: wgpu::AddressMode::ClampToEdge,
                                address_mode_w: wgpu::AddressMode::ClampToEdge,
                                mag_filter: wgpu::FilterMode::Linear,
                                min_filter: wgpu::FilterMode::Linear,
                                ..Default::default()
                            });

                    let bind_group =
                        ctx.wgpu_ctx
                            .device
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("Equirect BindGroup"),
                                layout: &self.equirect_layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(source_view),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::Sampler(&clamp_sampler),
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
                        cpass.set_pipeline(&self.equirect_pipeline);
                        cpass.set_bind_group(0, &bind_group, &[]);

                        let group_count = cube_size.div_ceil(8);
                        cpass.dispatch_workgroups(group_count, group_count, 6);
                    }
                }

                // Step 1b: Generate mipmaps for owned cube
                ctx.resource_manager.mipmap_generator.generate(
                    &ctx.wgpu_ctx.device,
                    encoder,
                    cube_texture,
                );
            }

            CubeSourceType::CubeNoMipmaps => {
                // Step 1a: Blit source cube faces → owned cube mip 0
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
                            // Attachment cube without mipmaps not supported
                            gpu_env.needs_compute = false;
                            ctx.resource_manager
                                .environment_map_cache
                                .insert(source, gpu_env);
                            return;
                        }
                    };

                    self.blit_cube_faces(
                        &ctx.wgpu_ctx.device,
                        encoder,
                        source_texture,
                        cube_texture,
                    );
                }

                // Step 1b: Generate mipmaps for owned cube
                ctx.resource_manager.mipmap_generator.generate(
                    &ctx.wgpu_ctx.device,
                    encoder,
                    cube_texture,
                );
            }

            CubeSourceType::CubeWithMipmaps => {
                // No cube pre-processing needed; PMREM reads directly from
                // the asset's mipmapped cube texture.
            }
        }

        // ================================================================
        // Phase 2: PMREM prefiltering (common to all source types)
        // ================================================================
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
            let param_buffer = ctx.wgpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
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

            let bg_src = ctx
                .wgpu_ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
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

            let bg_dst = ctx
                .wgpu_ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
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
                cpass.set_pipeline(&self.pmrem_pipeline);
                cpass.set_bind_group(0, &bg_src, &[]);
                cpass.set_bind_group(1, &bg_dst, &[]);

                let group_count = mip_size.div_ceil(8);
                cpass.dispatch_workgroups(group_count, group_count, 6);
            }
        }

        // Mark done and return to cache
        gpu_env.needs_compute = false;
        ctx.resource_manager
            .environment_map_cache
            .insert(source, gpu_env);

        log::info!("IBL PMREM generated. Source type: {source_type:?}");
    }
}
