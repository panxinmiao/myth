use std::borrow::Cow;
use wgpu::util::DeviceExt;
use crate::renderer::graph::{RenderNode, RenderContext};

pub struct IBLComputePass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout_source: wgpu::BindGroupLayout,
    bind_group_layout_dest: wgpu::BindGroupLayout,
    // 标记是否需要运行（例如加载了新环境图后设为 true）
    pub dirty: std::cell::Cell<bool>,
}

impl IBLComputePass {
    pub fn new(device: &wgpu::Device) -> Self {
        // 1. 加载 Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("IBL Prefilter Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../../pipeline/shaders/program/ibl.wgsl"))),
        });

        // 2. 创建 BindGroup Layouts (参考 Python 代码中的 @group(0) 和 @group(1))
        let bind_group_layout_source = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("IBL Source Layout"),
            entries: &[
                // binding 0: srcTex (texture_cube)
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
                // binding 1: sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 2: params (Uniform)
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

        let bind_group_layout_dest = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("IBL Dest Layout"),
            entries: &[
                // binding 0: destTex (storage_2d_array)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float, // 注意检查设备支持，或者改用 Rgba32Float
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ],
        });

        // 3. 创建 Pipeline
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
            dirty: std::cell::Cell::new(false),
        }
    }
}

impl RenderNode for IBLComputePass {
    fn name(&self) -> &str { "IBL Compute Pass" }

    fn run(&self, ctx: &mut RenderContext, encoder: &mut wgpu::CommandEncoder) {
        if !self.dirty.get() { return; }

        // 检查是否需要生成：
        // 1. 有 source_env_map
        // 2. 没有 pmrem_map (说明是刚加载或被重置了)
        let env = &mut ctx.scene.environment;
        if env.source_env_map.is_none() { return; } 
        if env.pmrem_map.is_some() { return; }

        let source_handle = env.source_env_map.unwrap();

        // 获取 GPU 资源
        let gpu_source = match ctx.resource_manager.get_texture(source_handle) {
            Some(t) => t,
            None => return, // 还没上传，下一帧再说
        };

        // 关键：我们需要目标纹理 (PMREM Map)
        // 建议在 Scene.environment 中增加一个 pmrem_texture handle
        // 或者在这里临时创建（如果是预计算）
        let target_handle = ctx.scene.environment.bindings().pmrem_texture.expect("PMREM texture not init");
        let gpu_target = ctx.resource_manager.get_texture(target_handle).expect("Target not uploaded");

        // === 循环 Mip Levels ===
        let mip_levels = 6; // 或从 texture info 获取
        let size = 256; // 目标尺寸

        for mip in 0..mip_levels {
            let mip_size = (size >> mip).max(1);
            let roughness = mip as f32 / (mip_levels - 1) as f32;

            // 1. 更新 Params Buffer (Uniform)
            // 注意：因为要在循环中使用，需要为每个 mip 创建一个 buffer 或者使用 offset
            // 这里为了简单演示，假设 ResourceManager 有 helper 方法分配临时 Uniform
            let params = [roughness, mip_size as f32]; 
            let params_buffer = ctx.resource_manager.create_temp_uniform_buffer(&params);

            // 2. 创建 Source BindGroup
            let bind_group_source = ctx.wgpu_ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout_source,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&gpu_env_map.view) }, // Cubemap View
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&ctx.resource_manager.dummy_sampler) }, // Linear Sampler
                    wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
                ],
            });

            // 3. 创建 Dest BindGroup (针对当前 Mip 的 View)
            // 关键：创建 D2Array View 指向特定 Mip
            let dest_view = gpu_target.texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("PMREM Mip View"),
                format: Some(wgpu::TextureFormat::Rgba16Float),
                dimension: Some(wgpu::TextureViewDimension::D2Array), // 必须是 2D Array 以便写入 Cube 的 6 层
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(6),
                usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
            });

            let bind_group_dest = ctx.wgpu_ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_group_layout_dest,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&dest_view) },
                ],
            });

            // 4. Dispatch
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("IBL Gen"), timestamp_writes: None });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group_source, &[]);
            cpass.set_bind_group(1, &bind_group_dest, &[]);
            
            // 修复 Python 代码中的 BUG：向上取整
            let group_count = (mip_size + 7) / 8; 
            cpass.dispatch_workgroups(group_count, group_count, 6); // z=6 对应 Cube 的 6 个面
        }
        
        self.dirty.set(false);
    }
}