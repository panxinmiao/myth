//! Environment source conversion pass.
//!
//! Converts a scene environment source into the persistent scene-owned base
//! cubemap. All bind groups in this pass are static because they only reference
//! persistent source/destination views, so they are prepared during the
//! feature extract stage rather than in the node execute stage.

use rustc_hash::FxHashMap;

use crate::core::ResourceManager;
use crate::core::gpu::CubeSourceType;
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::TextureNodeId;
use crate::graph::core::context::{ExecuteContext, ExtractContext};
use crate::graph::core::node::PassNode;
use crate::pipeline::{
    ComputePipelineId, ComputePipelineKey, ShaderCompilationOptions, ShaderSource,
};
use myth_resources::texture::{TextureSampler, TextureSource};
use myth_scene::Scene;

const EQUIRECT_SAMPLER_KEY: TextureSampler = TextureSampler {
    address_mode_u: wgpu::AddressMode::Repeat,
    address_mode_v: wgpu::AddressMode::ClampToEdge,
    address_mode_w: wgpu::AddressMode::ClampToEdge,
    mag_filter: wgpu::FilterMode::Linear,
    min_filter: wgpu::FilterMode::Linear,
    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
    lod_min_clamp: 0.0,
    lod_max_clamp: 32.0,
    compare: None,
    anisotropy_clamp: Some(1),
    border_color: None,
};

const CUBEMAP_SAMPLER_KEY: TextureSampler = TextureSampler::LINEAR_CLAMP;
const SCENE_CACHE_TTL: u64 = 120;

pub struct ResolvedSourceView<'a> {
    pub view: &'a wgpu::TextureView,
    pub view_id: u64,
}

struct SceneSourceConvertState {
    source_type: CubeSourceType,
    source_view_id: u64,
    dest_view_id: u64,
    bind_group: wgpu::BindGroup,
    last_used_frame: u64,
}

pub struct EquirectToCubeFeature {
    equirect_pipeline_id: Option<ComputePipelineId>,
    cubemap_pipeline_id: Option<ComputePipelineId>,
    equirect_layout: wgpu::BindGroupLayout,
    cubemap_layout: wgpu::BindGroupLayout,
    equirect_sampler: wgpu::Sampler,
    cubemap_sampler: wgpu::Sampler,
    scene_states: FxHashMap<u32, SceneSourceConvertState>,
}

impl EquirectToCubeFeature {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let equirect_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Environment EquirectToCube BGL"),
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

        let cubemap_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Environment CubeToCube BGL"),
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
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ],
        });

        let equirect_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Environment Equirect Sampler"),
            address_mode_u: EQUIRECT_SAMPLER_KEY.address_mode_u,
            address_mode_v: EQUIRECT_SAMPLER_KEY.address_mode_v,
            address_mode_w: EQUIRECT_SAMPLER_KEY.address_mode_w,
            mag_filter: EQUIRECT_SAMPLER_KEY.mag_filter,
            min_filter: EQUIRECT_SAMPLER_KEY.min_filter,
            mipmap_filter: EQUIRECT_SAMPLER_KEY.mipmap_filter,
            lod_min_clamp: EQUIRECT_SAMPLER_KEY.lod_min_clamp,
            lod_max_clamp: EQUIRECT_SAMPLER_KEY.lod_max_clamp,
            compare: EQUIRECT_SAMPLER_KEY.compare,
            anisotropy_clamp: EQUIRECT_SAMPLER_KEY.anisotropy_clamp.unwrap_or(1),
            border_color: EQUIRECT_SAMPLER_KEY.border_color,
        });

        let cubemap_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Environment Cubemap Sampler"),
            address_mode_u: CUBEMAP_SAMPLER_KEY.address_mode_u,
            address_mode_v: CUBEMAP_SAMPLER_KEY.address_mode_v,
            address_mode_w: CUBEMAP_SAMPLER_KEY.address_mode_w,
            mag_filter: CUBEMAP_SAMPLER_KEY.mag_filter,
            min_filter: CUBEMAP_SAMPLER_KEY.min_filter,
            mipmap_filter: CUBEMAP_SAMPLER_KEY.mipmap_filter,
            lod_min_clamp: CUBEMAP_SAMPLER_KEY.lod_min_clamp,
            lod_max_clamp: CUBEMAP_SAMPLER_KEY.lod_max_clamp,
            compare: CUBEMAP_SAMPLER_KEY.compare,
            anisotropy_clamp: CUBEMAP_SAMPLER_KEY.anisotropy_clamp.unwrap_or(1),
            border_color: CUBEMAP_SAMPLER_KEY.border_color,
        });

        Self {
            equirect_pipeline_id: None,
            cubemap_pipeline_id: None,
            equirect_layout,
            cubemap_layout,
            equirect_sampler,
            cubemap_sampler,
            scene_states: FxHashMap::default(),
        }
    }

    pub fn extract_and_prepare(&mut self, ctx: &mut ExtractContext, scene: &Scene) {
        self.ensure_pipelines(ctx);
        self.prune_scene_states(ctx.resource_manager.frame_index());

        let scene_id = scene.id();
        let Some(gpu_env) = ctx.resource_manager.gpu_environment(scene_id) else {
            self.scene_states.remove(&scene_id);
            return;
        };

        let Some(source) = scene.environment.source_env_map() else {
            self.scene_states.remove(&scene_id);
            return;
        };

        let Some(resolved_source) = Self::resolve_source_view(ctx.resource_manager, source) else {
            return;
        };

        let source_type = gpu_env.source_type;
        if self.layout_and_sampler(source_type).is_none() {
            self.scene_states.remove(&scene_id);
            return;
        }

        let dest_view = &gpu_env.base_cube_storage_view;
        let frame_index = ctx.resource_manager.frame_index();
        let needs_rebuild = self.scene_states.get(&scene_id).is_none_or(|state| {
            state.source_type != source_type
                || state.source_view_id != resolved_source.view_id
                || state.dest_view_id != dest_view.id()
        });

        if needs_rebuild {
            let (layout, sampler, _) = self
                .layout_and_sampler(source_type)
                .expect("procedural atmosphere should not reach source conversion");
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Environment Source Convert BG"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(resolved_source.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(dest_view),
                    },
                ],
            });
            self.scene_states.insert(
                scene_id,
                SceneSourceConvertState {
                    source_type,
                    source_view_id: resolved_source.view_id,
                    dest_view_id: dest_view.id(),
                    bind_group,
                    last_used_frame: frame_index,
                },
            );
        } else if let Some(state) = self.scene_states.get_mut(&scene_id) {
            state.last_used_frame = frame_index;
        }
    }

    pub fn resolve_source_view<'a>(
        resource_manager: &'a ResourceManager,
        source: &TextureSource,
    ) -> Option<ResolvedSourceView<'a>> {
        match source {
            TextureSource::Asset(handle) => resource_manager
                .texture_bindings
                .get(*handle)
                .and_then(|binding| {
                    resource_manager
                        .gpu_images
                        .get(binding.image_handle)
                        .map(|img| ResolvedSourceView {
                            view: &img.default_view,
                            view_id: binding.view_id,
                        })
                }),
            TextureSource::Attachment(id, _) => resource_manager
                .internal_resources
                .get(id)
                .map(|view| ResolvedSourceView { view, view_id: *id }),
        }
    }

    fn ensure_pipelines(&mut self, ctx: &mut ExtractContext) {
        if self.equirect_pipeline_id.is_some() && self.cubemap_pipeline_id.is_some() {
            return;
        }

        let options = ShaderCompilationOptions::default();
        let compilation_options = wgpu::PipelineCompilationOptions::default();

        let (equirect_module, equirect_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/utility/equirect_to_cube"),
            &options,
        );
        let equirect_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Environment EquirectToCube PL"),
                bind_group_layouts: &[Some(&self.equirect_layout)],
                immediate_size: 0,
            });
        self.equirect_pipeline_id = Some(ctx.pipeline_cache.get_or_create_compute(
            ctx.device,
            equirect_module,
            &equirect_layout,
            &ComputePipelineKey::new(equirect_hash).with_compilation_options(&compilation_options),
            &compilation_options,
            "Environment EquirectToCube Pipeline",
        ));

        let (cubemap_module, cubemap_hash) = ctx.shader_manager.get_or_compile(
            ctx.device,
            ShaderSource::File("entry/utility/cube_to_cube"),
            &options,
        );
        let cubemap_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Environment CubeToCube PL"),
                bind_group_layouts: &[Some(&self.cubemap_layout)],
                immediate_size: 0,
            });
        self.cubemap_pipeline_id = Some(ctx.pipeline_cache.get_or_create_compute(
            ctx.device,
            cubemap_module,
            &cubemap_layout,
            &ComputePipelineKey::new(cubemap_hash).with_compilation_options(&compilation_options),
            &compilation_options,
            "Environment CubeToCube Pipeline",
        ));
    }

    fn layout_and_sampler(
        &self,
        source_type: CubeSourceType,
    ) -> Option<(&wgpu::BindGroupLayout, &wgpu::Sampler, CubeSourceType)> {
        match source_type {
            CubeSourceType::Equirectangular => {
                Some((&self.equirect_layout, &self.equirect_sampler, source_type))
            }
            CubeSourceType::Cubemap => {
                Some((&self.cubemap_layout, &self.cubemap_sampler, source_type))
            }
            CubeSourceType::Procedural => None,
        }
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
        source_type: CubeSourceType,
        base_cube: TextureNodeId,
    ) {
        let pipeline = match source_type {
            CubeSourceType::Equirectangular => self
                .equirect_pipeline_id
                .map(|id| ctx.pipeline_cache.get_compute_pipeline(id)),
            CubeSourceType::Cubemap => self
                .cubemap_pipeline_id
                .map(|id| ctx.pipeline_cache.get_compute_pipeline(id)),
            CubeSourceType::Procedural => None,
        };

        let state = self
            .scene_states
            .get(&scene_id)
            .expect("scene source conversion state must be prepared before graph build");

        ctx.graph.add_pass("Environment_Source_Convert", |builder| {
            builder.write_texture(base_cube);
            let node = EquirectToCubePassNode {
                base_cube,
                pipeline,
                bind_group: &state.bind_group,
            };
            (node, ())
        });
    }
}

struct EquirectToCubePassNode<'a> {
    base_cube: TextureNodeId,
    pipeline: Option<&'a wgpu::ComputePipeline>,
    bind_group: &'a wgpu::BindGroup,
}

impl PassNode<'_> for EquirectToCubePassNode<'_> {
    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let pipeline = self
            .pipeline
            .expect("environment source pipeline must exist");
        let group_count = ctx.get_texture(self.base_cube).width().div_ceil(8);

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Environment Source Convert"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, self.bind_group, &[]);
            cpass.dispatch_workgroups(group_count, group_count, 6);
        }

        ctx.mipmap_generator
            .generate(ctx.device, encoder, ctx.get_texture(self.base_cube));
    }
}
