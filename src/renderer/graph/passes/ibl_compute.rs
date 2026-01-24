use std::borrow::Cow;
use std::cell::RefCell;
use wgpu::TextureViewDimension;
use crate::renderer::graph::{RenderNode, RenderContext};
use crate::resources::texture::TextureSource;

const EQUIRECT_CUBE_SIZE: u32 = 1024;
const PMREM_SIZE: u32 = 512;

pub struct IBLComputePass {
    pmrem_pipeline: wgpu::ComputePipeline,
    pmrem_layout_source: wgpu::BindGroupLayout,
    pmrem_layout_dest: wgpu::BindGroupLayout,
    
    equirect_pipeline: wgpu::ComputePipeline,
    equirect_layout: wgpu::BindGroupLayout,
    
    last_processed_source: RefCell<Option<(TextureSource, u64)>>,
}

impl IBLComputePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let pmrem_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("IBL Prefilter Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../../pipeline/shaders/program/ibl.wgsl"))),
        });

        let pmrem_layout_source = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
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
            compilation_options: Default::default(),
            cache: None,
        });

        let equirect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Equirect to Cube Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../../pipeline/shaders/program/equirect_to_cube.wgsl"))),
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

        let equirect_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Equirect Pipeline Layout"),
            bind_group_layouts: &[&equirect_layout],
            immediate_size: 0,
        });

        let equirect_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Equirect to Cube Pipeline"),
            layout: Some(&equirect_pipeline_layout),
            module: &equirect_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pmrem_pipeline,
            pmrem_layout_source,
            pmrem_layout_dest,
            equirect_pipeline,
            equirect_layout,
            last_processed_source: RefCell::new(None),
        }
    }
}

impl RenderNode for IBLComputePass {
    fn name(&self) -> &str { "IBL Compute Pass" }

    fn run(&self, ctx: &mut RenderContext, encoder: &mut wgpu::CommandEncoder) {
        let env = &ctx.scene.environment;

        let current_source = match &env.source_env_map {
            Some(s) => s.clone(),
            None => return,
        };

        let mut current_version = 0;
        
        if let TextureSource::Asset(handle) = &current_source {
            if let Some(tex) = ctx.assets.get_texture(*handle) {
                current_version = tex.version();
            }
        }

        let needs_update = if let Some((last_src, last_ver)) = &*self.last_processed_source.borrow() {
            *last_src != current_source || *last_ver != current_version
        } else {
            true
        };

        if !needs_update && env.pmrem_map.is_some() {
            return;
        }

        let (source_view_dim, cpu_image_id) = match &current_source {
            TextureSource::Asset(handle) => {
                ctx.resource_manager.prepare_texture(ctx.assets, *handle);

                let texture_asset = match ctx.assets.get_texture(*handle) {
                    Some(asset) => asset,
                    None => {
                        log::error!("IBL Source Asset missing: {:?}", handle);
                        return;
                    }
                };
                let view_dim = texture_asset.view_dimension;
                let img_id = texture_asset.image.id();

                if ctx.resource_manager.get_image(img_id).is_none() {
                    log::error!("IBL Source GpuImage missing for handle: {:?}", handle);
                    return;
                }
                (view_dim, Some(img_id))
            },
            TextureSource::Attachment(id) => {
                if ctx.resource_manager.internal_resources.get(id).is_none() {
                    log::error!("Missing internal attachment: {}", id);
                    return;
                }
                (TextureViewDimension::Cube, None)
            }
        };

        let is_2d_source = source_view_dim == TextureViewDimension::D2;
        
        let converted_cube_texture = if is_2d_source {
            let source_view = match &current_source {
                TextureSource::Asset(_) => {
                    &ctx.resource_manager.get_image(cpu_image_id.unwrap()).unwrap().default_view
                },
                TextureSource::Attachment(id) => {
                    ctx.resource_manager.internal_resources.get(id).unwrap()
                }
            };
            
            let cube_texture = ctx.wgpu_ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Env Equirect2Cube"),
                size: wgpu::Extent3d { 
                    width: EQUIRECT_CUBE_SIZE, 
                    height: EQUIRECT_CUBE_SIZE, 
                    depth_or_array_layers: 6 
                },
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                mip_level_count: 1,
                sample_count: 1,
                view_formats: &[],
            });

            let dest_view = cube_texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });

            let bind_group = ctx.wgpu_ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Equirect BindGroup"),
                layout: &self.equirect_layout,
                entries: &[
                    wgpu::BindGroupEntry { 
                        binding: 0, 
                        resource: wgpu::BindingResource::TextureView(source_view) 
                    },
                    wgpu::BindGroupEntry { 
                        binding: 1, 
                        resource: wgpu::BindingResource::Sampler(&ctx.resource_manager.dummy_sampler.sampler) 
                    },
                    wgpu::BindGroupEntry { 
                        binding: 2, 
                        resource: wgpu::BindingResource::TextureView(&dest_view) 
                    },
                ],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { 
                    label: Some("Equirect to Cube"), 
                    timestamp_writes: None 
                });
                cpass.set_pipeline(&self.equirect_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                
                let group_count = (EQUIRECT_CUBE_SIZE + 7) / 8;
                cpass.dispatch_workgroups(group_count, group_count, 6);
            }

            Some(cube_texture)
        } else {
            None
        };

        let pmrem_source_view = if let Some(ref cube_tex) = converted_cube_texture {
            let cube_view = cube_tex.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::Cube),
                ..Default::default()
            });
            
            let converted_view_id = ctx.resource_manager.register_internal_texture(cube_view);
            ctx.scene.environment.processed_env_map = Some(TextureSource::Attachment(converted_view_id));
            
            cube_tex.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::Cube),
                ..Default::default()
            })
        } else {
            ctx.scene.environment.processed_env_map = Some(current_source.clone());
            
            match &current_source {
                TextureSource::Asset(_) => {
                    ctx.resource_manager.get_image(cpu_image_id.unwrap()).unwrap()
                        .texture.create_view(&wgpu::TextureViewDescriptor {
                            dimension: Some(wgpu::TextureViewDimension::Cube),
                            ..Default::default()
                        })
                },
                TextureSource::Attachment(_) => {
                    return;
                }
            }
        };

        let mip_levels = (PMREM_SIZE as f32).log2().floor() as u32 + 1;
        let format = wgpu::TextureFormat::Rgba16Float;

        let pmrem_texture = ctx.wgpu_ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PMREM Cubemap"),
            size: wgpu::Extent3d { 
                width: PMREM_SIZE, 
                height: PMREM_SIZE, 
                depth_or_array_layers: 6 
            },
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        for mip in 0..mip_levels {
            let mip_size = (PMREM_SIZE >> mip).max(1);
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

            let bg_src = ctx.wgpu_ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pmrem_layout_source,
                entries: &[
                    wgpu::BindGroupEntry { 
                        binding: 0, 
                        resource: wgpu::BindingResource::TextureView(&pmrem_source_view) 
                    },
                    wgpu::BindGroupEntry { 
                        binding: 1, 
                        resource: wgpu::BindingResource::Sampler(&ctx.resource_manager.dummy_sampler.sampler) 
                    },
                    wgpu::BindGroupEntry { 
                        binding: 2, 
                        resource: param_buffer.as_entire_binding() 
                    },
                ],
            });

            let dest_view = pmrem_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("PMREM Mip {}", mip)),
                format: Some(format),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(6),
                usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
            });

            let bg_dst = ctx.wgpu_ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pmrem_layout_dest,
                entries: &[
                    wgpu::BindGroupEntry { 
                        binding: 0, 
                        resource: wgpu::BindingResource::TextureView(&dest_view) 
                    },
                ],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { 
                    label: Some("IBL Gen"), 
                    timestamp_writes: None 
                });
                cpass.set_pipeline(&self.pmrem_pipeline);
                cpass.set_bind_group(0, &bg_src, &[]);
                cpass.set_bind_group(1, &bg_dst, &[]);
                
                let group_count = (mip_size + 7) / 8;
                cpass.dispatch_workgroups(group_count, group_count, 6);
            }
        }

        let pmrem_cube_view = pmrem_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("PMREM Cube View"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        let id = ctx.resource_manager.register_internal_texture_by_name("PMREM_Map", pmrem_cube_view);
        
        ctx.scene.environment.pmrem_map = Some(TextureSource::Attachment(id));
        ctx.scene.environment.env_map_max_mip_level = (mip_levels - 1) as f32;
        *self.last_processed_source.borrow_mut() = Some((current_source, current_version));

        log::info!("IBL PMREM generated. Source type: {}", if is_2d_source { "2D HDR" } else { "CubeMap" });
    }
}