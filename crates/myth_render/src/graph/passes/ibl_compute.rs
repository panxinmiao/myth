//! PMREM generation pass.
//!
//! Reads the persistent scene-owned base cubemap and writes the persistent
//! scene-owned PMREM cubemap. The pass owns only command dispatch during
//! execute; all static bind groups and parameter buffers are prepared during
//! the feature extract stage.

use rustc_hash::FxHashMap;

use crate::core::gpu::Tracked;
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::TextureNodeId;
use crate::graph::core::context::{ExecuteContext, ExtractContext};
use crate::graph::core::node::PassNode;
use crate::pipeline::{
    ComputePipelineId, ComputePipelineKey, ShaderCompilationOptions, ShaderSource,
};

const SCENE_CACHE_TTL: u64 = 120;
const STATIC_PMREM_SAMPLE_COUNT: u32 = 4096;
const DYNAMIC_PMREM_SAMPLE_COUNT: u32 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum IblPipelineVariant {
    Static,
    Dynamic,
}

struct IblSceneState {
    base_cube_view_id: u64,
    pmrem_view_ids: Vec<u64>,
    pmrem_size: u32,
    pipeline_variant: IblPipelineVariant,
    _params_buffers: Vec<Tracked<wgpu::Buffer>>,
    source_bind_groups: Vec<wgpu::BindGroup>,
    dest_bind_groups: Vec<wgpu::BindGroup>,
    last_used_frame: u64,
}

pub struct IblComputeFeature {
    pipeline_ids: FxHashMap<IblPipelineVariant, ComputePipelineId>,
    source_layout: wgpu::BindGroupLayout,
    dest_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    scene_states: FxHashMap<u32, IblSceneState>,
}

impl IblComputeFeature {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let source_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let dest_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("IBL PMREM Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        Self {
            pipeline_ids: FxHashMap::default(),
            source_layout,
            dest_layout,
            sampler,
            scene_states: FxHashMap::default(),
        }
    }

    pub fn extract_and_prepare(&mut self, ctx: &mut ExtractContext, scene_id: u32) {
        self.prune_scene_states(ctx.resource_manager.frame_index());

        let Some(pipeline_variant) = ctx
            .resource_manager
            .gpu_environment(scene_id)
            .map(|gpu_env| Self::pipeline_variant(gpu_env.source_type))
        else {
            self.scene_states.remove(&scene_id);
            return;
        };

        self.ensure_pipeline(ctx, pipeline_variant);

        let Some(gpu_env) = ctx.resource_manager.gpu_environment(scene_id) else {
            self.scene_states.remove(&scene_id);
            return;
        };

        let frame_index = ctx.resource_manager.frame_index();
        let pmrem_view_ids: Vec<u64> = gpu_env
            .pmrem_storage_views
            .iter()
            .map(Tracked::id)
            .collect();
        let needs_rebuild = self.scene_states.get(&scene_id).is_none_or(|state| {
            state.base_cube_view_id != gpu_env.base_cube_view.id()
                || state.pmrem_size != gpu_env.pmrem_texture.width()
                || state.pmrem_view_ids != pmrem_view_ids
                || state.pipeline_variant != pipeline_variant
        });

        if needs_rebuild {
            let mip_levels = gpu_env.pmrem_texture.mip_level_count();
            let pmrem_size = gpu_env.pmrem_texture.width();
            let roughness_denominator = (mip_levels.saturating_sub(1)).max(1) as f32;

            let mut params_buffers = Vec::with_capacity(mip_levels as usize);
            let mut source_bind_groups = Vec::with_capacity(mip_levels as usize);
            let mut dest_bind_groups = Vec::with_capacity(mip_levels as usize);

            for mip in 0..mip_levels {
                let mip_size = (pmrem_size >> mip).max(1);
                let params = [mip as f32 / roughness_denominator, mip_size as f32, 0.0, 0.0];

                let buffer = Tracked::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("IBL Params"),
                    size: std::mem::size_of::<[f32; 4]>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                ctx.queue
                    .write_buffer(&buffer, 0, bytemuck::cast_slice(&params));

                let source_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("IBL Source BG"),
                    layout: &self.source_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&gpu_env.base_cube_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: buffer.as_entire_binding(),
                        },
                    ],
                });

                let dest_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("IBL Dest BG"),
                    layout: &self.dest_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(gpu_env.pmrem_mip_view(mip)),
                    }],
                });

                params_buffers.push(buffer);
                source_bind_groups.push(source_bg);
                dest_bind_groups.push(dest_bg);
            }

            self.scene_states.insert(
                scene_id,
                IblSceneState {
                    base_cube_view_id: gpu_env.base_cube_view.id(),
                    pmrem_view_ids,
                    pmrem_size,
                    pipeline_variant,
                    _params_buffers: params_buffers,
                    source_bind_groups,
                    dest_bind_groups,
                    last_used_frame: frame_index,
                },
            );
        } else if let Some(state) = self.scene_states.get_mut(&scene_id) {
            state.last_used_frame = frame_index;
        }
    }

    fn pipeline_variant(source_type: crate::core::gpu::CubeSourceType) -> IblPipelineVariant {
        match source_type {
            crate::core::gpu::CubeSourceType::Procedural => IblPipelineVariant::Dynamic,
            crate::core::gpu::CubeSourceType::Equirectangular
            | crate::core::gpu::CubeSourceType::Cubemap => IblPipelineVariant::Static,
        }
    }

    fn sample_count(variant: IblPipelineVariant) -> u32 {
        match variant {
            IblPipelineVariant::Static => STATIC_PMREM_SAMPLE_COUNT,
            IblPipelineVariant::Dynamic => DYNAMIC_PMREM_SAMPLE_COUNT,
        }
    }

    fn ensure_pipeline(&mut self, ctx: &mut ExtractContext, variant: IblPipelineVariant) {
        if self.pipeline_ids.contains_key(&variant) {
            return;
        }

        let (module, shader_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/utility/ibl"),
            &ShaderCompilationOptions::default(),
        );

        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("IBL Compute PL"),
                bind_group_layouts: &[Some(&self.source_layout), Some(&self.dest_layout)],
                immediate_size: 0,
            });

        let sample_count = Self::sample_count(variant);
        let constants = [("SAMPLE_COUNT", f64::from(sample_count))];
        let compilation_options = wgpu::PipelineCompilationOptions {
            constants: &constants,
            ..Default::default()
        };

        let pipeline_id = ctx.pipeline_cache.get_or_create_compute(
            ctx.device,
            module,
            &layout,
            &ComputePipelineKey::new(shader_hash).with_compilation_options(&compilation_options),
            &compilation_options,
            match variant {
                IblPipelineVariant::Static => "IBL Compute Pipeline (Static)",
                IblPipelineVariant::Dynamic => "IBL Compute Pipeline (Dynamic)",
            },
        );

        self.pipeline_ids.insert(variant, pipeline_id);
    }

    fn prune_scene_states(&mut self, current_frame: u64) {
        self.scene_states.retain(|_, state| {
            current_frame.saturating_sub(state.last_used_frame) <= SCENE_CACHE_TTL
        });
    }

    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        scene_id: u32,
        base_cube: TextureNodeId,
        pmrem: TextureNodeId,
    ) {
        let state = self
            .scene_states
            .get(&scene_id)
            .expect("scene IBL state must be prepared before graph build");
        let pipeline = self
            .pipeline_ids
            .get(&state.pipeline_variant)
            .map(|&id| ctx.pipeline_cache.get_compute_pipeline(id));

        ctx.graph.add_pass("IBL_Compute", |builder| {
            builder.read_texture(base_cube);
            builder.write_texture(pmrem);
            let node = IblComputePassNode {
                pmrem,
                pipeline,
                source_bind_groups: &state.source_bind_groups,
                dest_bind_groups: &state.dest_bind_groups,
            };
            (node, ())
        });
    }
}

struct IblComputePassNode<'a> {
    pmrem: TextureNodeId,
    pipeline: Option<&'a wgpu::ComputePipeline>,
    source_bind_groups: &'a [wgpu::BindGroup],
    dest_bind_groups: &'a [wgpu::BindGroup],
}

impl PassNode<'_> for IblComputePassNode<'_> {
    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let pipeline = self.pipeline.expect("IBL pipeline must exist");
        let mip_levels = ctx.get_texture(self.pmrem).mip_level_count();

        for mip in 0..mip_levels as usize {
            let mip_size = (ctx.get_texture(self.pmrem).width() >> mip as u32).max(1);
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("IBL Compute"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &self.source_bind_groups[mip], &[]);
            cpass.set_bind_group(1, &self.dest_bind_groups[mip], &[]);
            let group_count = mip_size.div_ceil(8);
            cpass.dispatch_workgroups(group_count, group_count, 6);
        }
    }
}