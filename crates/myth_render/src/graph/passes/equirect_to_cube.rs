//! Environment source conversion pass.
//!
//! Converts scene environment sources into the persistent scene-owned base cube
//! texture. The pass is intentionally narrow in responsibility:
//!
//! - `Equirectangular` source: 2D panorama -> cubemap
//! - `Cubemap` source: cubemap -> persistent scene cubemap
//!
//! PMREM generation is handled by a separate node so RenderGraph can model the
//! dependency as `Source Convert -> PMREM -> Scene Shading`.

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

pub struct EquirectToCubeFeature {
    equirect_pipeline_id: Option<ComputePipelineId>,
    cubemap_pipeline_id: Option<ComputePipelineId>,
    equirect_layout: wgpu::BindGroupLayout,
    cubemap_layout: wgpu::BindGroupLayout,
    equirect_sampler: wgpu::Sampler,
    cubemap_sampler: wgpu::Sampler,
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
        }
    }

    pub fn extract_and_prepare(&mut self, ctx: &mut ExtractContext) {
        self.ensure_pipelines(ctx);
    }

    pub fn resolve_source_view<'a>(
        resource_manager: &'a ResourceManager,
        source: &TextureSource,
    ) -> Option<&'a wgpu::TextureView> {
        match source {
            TextureSource::Asset(handle) => resource_manager
                .texture_bindings
                .get(*handle)
                .and_then(|binding| resource_manager.gpu_images.get(binding.image_handle))
                .map(|img| &img.default_view),
            TextureSource::Attachment(id, _) => resource_manager.internal_resources.get(id),
        }
    }

    fn ensure_pipelines(&mut self, ctx: &mut ExtractContext) {
        if self.equirect_pipeline_id.is_some() && self.cubemap_pipeline_id.is_some() {
            return;
        }

        let options = ShaderCompilationOptions::default();

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
            &ComputePipelineKey {
                shader_hash: equirect_hash,
            },
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
            &ComputePipelineKey {
                shader_hash: cubemap_hash,
            },
            "Environment CubeToCube Pipeline",
        ));
    }

    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        source_type: CubeSourceType,
        source_view: &'a wgpu::TextureView,
        base_cube: TextureNodeId,
        base_cube_texture: &'a wgpu::Texture,
    ) {
        debug_assert!(matches!(source_type, CubeSourceType::Equirectangular | CubeSourceType::Cubemap));

        let equirect_pipeline = self
            .equirect_pipeline_id
            .map(|id| ctx.pipeline_cache.get_compute_pipeline(id));
        let cubemap_pipeline = self
            .cubemap_pipeline_id
            .map(|id| ctx.pipeline_cache.get_compute_pipeline(id));

        ctx.graph.add_pass("Environment_Source_Convert", |builder| {
            builder.write_texture(base_cube);
            let node = EquirectToCubePassNode {
                source_type,
                source_view,
                base_cube_texture,
                equirect_pipeline,
                cubemap_pipeline,
                equirect_layout: &self.equirect_layout,
                cubemap_layout: &self.cubemap_layout,
                equirect_sampler: &self.equirect_sampler,
                cubemap_sampler: &self.cubemap_sampler,
            };
            (node, ())
        });
    }
}

struct EquirectToCubePassNode<'a> {
    source_type: CubeSourceType,
    source_view: &'a wgpu::TextureView,
    base_cube_texture: &'a wgpu::Texture,
    equirect_pipeline: Option<&'a wgpu::ComputePipeline>,
    cubemap_pipeline: Option<&'a wgpu::ComputePipeline>,
    equirect_layout: &'a wgpu::BindGroupLayout,
    cubemap_layout: &'a wgpu::BindGroupLayout,
    equirect_sampler: &'a wgpu::Sampler,
    cubemap_sampler: &'a wgpu::Sampler,
}

impl PassNode<'_> for EquirectToCubePassNode<'_> {
    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let dest_view = self
            .base_cube_texture
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("Environment Base Cube Mip0"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(6),
                usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                ..Default::default()
            });

        let (pipeline, layout, sampler) = match self.source_type {
            CubeSourceType::Equirectangular => (
                self.equirect_pipeline.expect("equirect pipeline must exist"),
                self.equirect_layout,
                self.equirect_sampler,
            ),
            CubeSourceType::Cubemap => (
                self.cubemap_pipeline.expect("cubemap pipeline must exist"),
                self.cubemap_layout,
                self.cubemap_sampler,
            ),
            CubeSourceType::Procedural => return,
        };

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Environment Source Convert BG"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(self.source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&dest_view),
                },
            ],
        });

        let group_count = self.base_cube_texture.width().div_ceil(8);

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Environment Source Convert"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(group_count, group_count, 6);
        }

        ctx.mipmap_generator
            .generate(ctx.device, encoder, self.base_cube_texture);
    }
}