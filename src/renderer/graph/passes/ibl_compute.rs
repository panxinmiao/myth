use std::borrow::Cow;
use std::cell::RefCell;
use crate::renderer::graph::{RenderNode, RenderContext};
use crate::resources::texture::TextureSource;

pub struct IBLComputePass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout_source: wgpu::BindGroupLayout,
    bind_group_layout_dest: wgpu::BindGroupLayout,
    
    // 追踪上一次处理的源环境图，用于检测变化
    last_processed_source: RefCell<Option<(TextureSource, u64)>>,
}

impl IBLComputePass {
    pub fn new(device: &wgpu::Device) -> Self {
        // 1. Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("IBL Prefilter Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../../pipeline/shaders/program/ibl.wgsl"))),
        });

        // 2. Layouts
        // Source: Cube Texture + Sampler + Uniforms
        let bind_group_layout_source = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        // Dest: Storage Texture 2D Array (One Mip Level)
        let bind_group_layout_dest = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("IBL Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout_source, &bind_group_layout_dest],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("IBL Compute Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout_source,
            bind_group_layout_dest,
            last_processed_source: RefCell::new(None),
        }
    }

}

impl RenderNode for IBLComputePass {
    fn name(&self) -> &str { "IBL Compute Pass" }

    fn run(&self, ctx: &mut RenderContext, encoder: &mut wgpu::CommandEncoder) {
        let env = &mut ctx.scene.environment;

        // 1. 获取当前的 Source 配置
        let current_source = match &env.source_env_map {
            Some(s) => s.clone(),
            None => return, // 没有源，无法生成
        };

        // 2. [关键修复] 获取内容版本号
        let mut current_version = 0;
        
        // 尝试获取版本号。如果 Asset 还在加载中，这里的 version 可能是 0 或初始值。
        // 当 Asset 加载完成，AssetServer 会调用 texture.needs_update()，version 会增加。
        if let TextureSource::Asset(handle) = &current_source {
            if let Some(tex) = ctx.assets.get_texture(*handle) {
                current_version = tex.version();
            }
        }

        // 2. 检查是否需要更新 (Source 变了)
        // 注意：我们通过比较 TextureSource Enum 的相等性来判断
        let needs_update = if let Some((last_src, last_ver)) = &*self.last_processed_source.borrow() {
            *last_src != current_source || *last_ver != current_version
        } else {
            true // 第一次运行
        };

        if !needs_update && env.pmrem_map.is_some() {
            return;
        }


        // 3. 获取源纹理的 View
        // 这一步需要 RenderContext 不可变借用，所以我们在上面 match 之后做
        let source_view = match &current_source {
            TextureSource::Asset(handle) => {
                // 显式准备资源
                ctx.resource_manager.prepare_texture(ctx.assets, *handle);
                println!("Using asset texture for IBL source: {:?}", handle);

                // 获取 Asset 以拿到 CPU ID
                // 显式处理 Option::None
                let texture_asset = match ctx.assets.get_texture(*handle) {
                    Some(asset) => asset,
                    None => {
                        log::error!("IBL Source Asset missing: {:?}", handle);
                        return;
                    }
                };
                let cpu_image_id = texture_asset.image.id();

                // 使用 CPU ID 查询 GpuImage
                match ctx.resource_manager.get_image(cpu_image_id) {
                    Some(img) => &img.default_view,
                    None => {
                        log::error!("IBL Source GpuImage missing for handle: {:?}", handle);
                        return;
                    }
                }
            },
            TextureSource::Attachment(id) => {
                if let Some(v) = ctx.resource_manager.internal_resources.get(id) {
                    v
                } else {
                    log::error!("Missing internal attachment: {}", id);
                    return;
                }
            }
        };

        // 4. 准备 PMREM 纹理
        let size = 512;
        let mip_levels = (size as f32).log2().floor() as u32 + 1;
        let format = wgpu::TextureFormat::Rgba16Float;

        let pmrem_texture = ctx.wgpu_ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PMREM Cubemap"),
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 6 },
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // 5. 逐级生成
        for mip in 0..mip_levels {
            let mip_size = (size >> mip).max(1);
            let roughness = mip as f32 / (mip_levels - 1) as f32;

            let params = [roughness, mip_size as f32, 0.0, 0.0];
            let param_buffer = ctx.wgpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("IBL Params Buffer"),
                size: std::mem::size_of::<[f32; 4]>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });

            // 写入数据
            {
                let mut view = param_buffer.slice(..).get_mapped_range_mut();
                view.copy_from_slice(bytemuck::cast_slice(&params));
            }
            param_buffer.unmap();

            // Uniforms
            // let param_buffer = ctx.resource_manager.create_temp_uniform_buffer(&params);
            // ctx.wgpu_ctx.queue.write_buffer(&param_buffer, 0, bytemuck::cast_slice(&params));

            // BindGroups
            let bg_src = ctx.wgpu_ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout_source,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(source_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&ctx.resource_manager.dummy_sampler.sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: param_buffer.as_entire_binding() },
                ],
            });

            // Target View (D2Array for Storage)
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
                layout: &self.bind_group_layout_dest,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&dest_view) },
                ],
            });

            // Dispatch
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("IBL Gen"), timestamp_writes: None });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bg_src, &[]);
            cpass.set_bind_group(1, &bg_dst, &[]);
            
            let group_count = (mip_size + 7) / 8;
            cpass.dispatch_workgroups(group_count, group_count, 6);
        }

        // 6. 注册最终结果 (作为 Cube View)
        // 注意：我们注册的是 Cube 视图，以便在 Shader 中作为 texture_cube 采样
        let pmrem_cube_view = pmrem_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("PMREM Cube View"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        // 调用 register_internal_texture_by_name，获取 ID
        let id = ctx.resource_manager.register_internal_texture_by_name("PMREM_Map", pmrem_cube_view);
        
        // 7. 更新状态
        ctx.scene.environment.pmrem_map = Some(TextureSource::Attachment(id));
        ctx.scene.environment.env_map_max_mip_level = (mip_levels - 1) as f32;
        *self.last_processed_source.borrow_mut() = Some((current_source, current_version));

        log::info!("IBL PMREM generated and registered. Source updated.");
        println!("IBL PMREM generated and registered. Source updated.");
    }
}