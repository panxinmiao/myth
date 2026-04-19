//! PMREM generation pass.
//!
//! Reads the persistent scene-owned base cubemap and writes the persistent
//! scene-owned PMREM cubemap. Parameter buffers are prepared during the
//! feature extract stage, while mip-specific bind groups are rebuilt in the
//! RDG prepare phase so they participate in the shared transient binding path.

use rustc_hash::FxHashMap;

use crate::core::gpu::Tracked;
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{BufferDesc, BufferNodeId, PrepareContext, TextureNodeId};
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
    params_buffers: Vec<Tracked<wgpu::Buffer>>,
    pmrem_views: Vec<Tracked<wgpu::TextureView>>,
    last_used_frame: u64,
}

pub struct IblComputeFeature {
    pipeline_ids: FxHashMap<IblPipelineVariant, ComputePipelineId>,
    source_layout: Tracked<wgpu::BindGroupLayout>,
    dest_layout: Tracked<wgpu::BindGroupLayout>,
    sampler: wgpu::Sampler,
    scene_states: FxHashMap<u32, IblSceneState>,
}

impl IblComputeFeature {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let source_layout = Tracked::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        }));

        let dest_layout = Tracked::new(device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        }));

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
            let mut pmrem_views = Vec::with_capacity(mip_levels as usize);

            for mip in 0..mip_levels {
                let mip_size = (pmrem_size >> mip).max(1);
                let params = [
                    mip as f32 / roughness_denominator,
                    mip_size as f32,
                    0.0,
                    0.0,
                ];

                let buffer = Tracked::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("IBL Params"),
                    size: std::mem::size_of::<[f32; 4]>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                ctx.queue
                    .write_buffer(&buffer, 0, bytemuck::cast_slice(&params));

                params_buffers.push(buffer);
                pmrem_views.push(gpu_env.pmrem_mip_view(mip).clone());
            }

            self.scene_states.insert(
                scene_id,
                IblSceneState {
                    base_cube_view_id: gpu_env.base_cube_view.id(),
                    pmrem_view_ids,
                    pmrem_size,
                    pipeline_variant,
                    params_buffers,
                    pmrem_views,
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
        let source_layout = &self.source_layout;
        let dest_layout = &self.dest_layout;
        let sampler = &self.sampler;
        let params_desc = BufferDesc::new(
            std::mem::size_of::<[f32; 4]>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        ctx.graph.add_pass("IBL_Compute", |builder| {
            builder.read_texture(base_cube);
            builder.write_texture(pmrem);

            let mut mip_states = Vec::with_capacity(state.params_buffers.len());
            for (params_buffer, pmrem_mip_view) in
                state.params_buffers.iter().zip(state.pmrem_views.iter())
            {
                let params_buffer =
                    builder.read_external_buffer("IBL_Params", params_desc, params_buffer);
                mip_states.push(IblMipState {
                    params_buffer,
                    pmrem_mip_view,
                    source_bg: None,
                    dest_bg: None,
                });
            }
            let mip_states = builder.graph.alloc_slice_mut(&mip_states);

            let node = IblComputePassNode {
                base_cube,
                pmrem,
                pipeline,
                source_layout,
                dest_layout,
                sampler,
                mips: mip_states,
            };
            (node, ())
        });
    }
}

#[derive(Clone, Copy)]
struct IblMipState<'a> {
    params_buffer: BufferNodeId,
    pmrem_mip_view: &'a Tracked<wgpu::TextureView>,
    source_bg: Option<&'a wgpu::BindGroup>,
    dest_bg: Option<&'a wgpu::BindGroup>,
}

struct IblComputePassNode<'a> {
    base_cube: TextureNodeId,
    pmrem: TextureNodeId,
    pipeline: Option<&'a wgpu::ComputePipeline>,
    source_layout: &'a Tracked<wgpu::BindGroupLayout>,
    dest_layout: &'a Tracked<wgpu::BindGroupLayout>,
    sampler: &'a wgpu::Sampler,
    mips: &'a mut [IblMipState<'a>],
}

impl<'a> PassNode<'a> for IblComputePassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        for mip in self.mips.iter_mut() {
            mip.source_bg = Some(
                ctx.build_bind_group(self.source_layout, Some("IBL Source BG"))
                    .bind_texture(0, self.base_cube)
                    .bind_sampler(1, self.sampler)
                    .bind_buffer(2, mip.params_buffer)
                    .build(),
            );
            mip.dest_bg = Some(
                ctx.build_bind_group(self.dest_layout, Some("IBL Dest BG"))
                    .bind_tracked_texture_view(0, mip.pmrem_mip_view)
                    .build(),
            );
        }
    }

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
            cpass.set_bind_group(0, self.mips[mip].source_bg.expect("IBL source BG missing"), &[]);
            cpass.set_bind_group(1, self.mips[mip].dest_bg.expect("IBL dest BG missing"), &[]);
            let group_count = mip_size.div_ceil(8);
            cpass.dispatch_workgroups(group_count, group_count, 6);
        }
    }
}
