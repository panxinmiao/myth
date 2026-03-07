//! Tone Mapping Feature & Transient Pass Node
//!
//! Converts HDR scene color to LDR for display, split into Feature +
//! transient PassNode architecture:
//!
//! - [`ToneMapFeature`] (persistent): layouts, pipeline cache (by mode/format/lut).
//! - [`RdgToneMapPassNode`] (transient): per-frame bind group assembly and execution.
//!
//! # Algorithms
//!
//! Linear, Neutral, Reinhard, Cineon, ACES, AgX — selectable per-frame.
//! Optional vignette, color grading (3D LUT), film grain, chromatic aberration.

use rustc_hash::FxHashMap;

use crate::ShaderDefines;
use crate::assets::TextureHandle;
use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::resources::{CommonSampler, Tracked};
use crate::renderer::graph::rdg::builder::PassBuilder;
use crate::renderer::graph::rdg::context::{RdgExecuteContext, RdgPrepareContext};
use crate::renderer::graph::rdg::feature::ExtractContext;
use crate::renderer::graph::rdg::graph::RenderGraph;
use crate::renderer::graph::rdg::node::PassNode;
use crate::renderer::graph::rdg::types::TextureNodeId;
use crate::renderer::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions,
};
use crate::resources::texture::TextureSource;
use crate::resources::tone_mapping::{ToneMappingMode, ToneMappingUniforms};
use crate::resources::uniforms::WgslStruct;

/// Pipeline cache key: (mode, output_format, has_lut).
type PipelineCacheKey = (ToneMappingMode, wgpu::TextureFormat, bool);

// =============================================================================
// Configuration
// =============================================================================

/// Parameters passed to [`ToneMapFeature::add_to_graph`] each frame.
pub struct ToneMapParams {
    pub mode: ToneMappingMode,
    pub has_lut: bool,
    pub uniforms_cpu_id: u64,
    pub lut_handle: Option<TextureHandle>,
    pub global_state_key: (u32, u32),
    pub output_format: wgpu::TextureFormat,
}

// =============================================================================
// Persistent Feature
// =============================================================================

/// Persistent tone mapping Feature — owns layouts and a local pipeline cache
/// keyed by `(mode, format, has_lut)`.
pub struct ToneMapFeature {
    layout_base: Option<Tracked<wgpu::BindGroupLayout>>,
    layout_with_lut: Option<Tracked<wgpu::BindGroupLayout>>,
    local_cache: FxHashMap<PipelineCacheKey, RenderPipelineId>,
}

impl ToneMapFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            layout_base: None,
            layout_with_lut: None,
            local_cache: FxHashMap::default(),
        }
    }

    /// Prepare persistent GPU resources and compile the pipeline for the
    /// current frame's configuration.
    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        params: &ToneMapParams,
    ) -> RenderPipelineId {
        self.ensure_layouts(ctx.device);

        if let Some(lut_handle) = params.lut_handle {
            ctx.resource_manager.prepare_texture(ctx.assets, lut_handle);
        }

        self.get_or_create_pipeline(ctx, params)
    }

    /// Build and inject a transient tone-mapping node into the render graph.
    pub fn add_to_graph(
        &self,
        rdg: &mut RenderGraph,
        input_tex: TextureNodeId,
        output_tex: TextureNodeId,
        pipeline_id: RenderPipelineId,
        params: &ToneMapParams,
    ) {
        let layout = if params.has_lut {
            self.layout_with_lut.clone().unwrap()
        } else {
            self.layout_base.clone().unwrap()
        };

        let node = Box::new(RdgToneMapPassNode {
            input_tex,
            output_tex,
            has_lut: params.has_lut,
            uniforms_cpu_id: params.uniforms_cpu_id,
            lut_handle: params.lut_handle,
            pipeline_id,
            layout,
            current_bind_group_key: None,
        });
        rdg.add_pass_owned(node);
    }

    // =========================================================================
    // Lazy Initialization
    // =========================================================================

    fn ensure_layouts(&mut self, device: &wgpu::Device) {
        if self.layout_base.is_some() {
            return;
        }

        let base_entries = [
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];

        self.layout_base = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("RDG ToneMap Layout (base)"),
                entries: &base_entries,
            },
        )));

        let mut lut_entries = base_entries.to_vec();
        lut_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        });
        lut_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });

        self.layout_with_lut = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("RDG ToneMap Layout (LUT)"),
                entries: &lut_entries,
            },
        )));
    }

    fn get_or_create_pipeline(
        &mut self,
        ctx: &mut ExtractContext,
        params: &ToneMapParams,
    ) -> RenderPipelineId {
        let cache_key = (params.mode, params.output_format, params.has_lut);

        if let Some(&id) = self.local_cache.get(&cache_key) {
            return id;
        }

        log::debug!(
            "RDG ToneMap: compiling pipeline for {:?}, fmt={:?}, lut={}",
            params.mode,
            params.output_format,
            params.has_lut,
        );

        let device = ctx.device;
        let mut defines = ShaderDefines::new();
        params.mode.apply_to_defines(&mut defines);
        if params.has_lut {
            defines.set("USE_LUT", "1");
        }

        let gpu_world = ctx
            .resource_manager
            .get_global_state(params.global_state_key.0, params.global_state_key.1)
            .expect("RDG ToneMap: GpuGlobalState must exist");

        let mut options = ShaderCompilationOptions { defines };
        options.add_define(
            "struct_definitions",
            ToneMappingUniforms::wgsl_struct_def("Uniforms").as_str(),
        );

        let (shader_module, shader_hash) = ctx.shader_manager.get_or_compile_template(
            device,
            "passes/tone_mapping",
            &options,
            "",
            &gpu_world.binding_wgsl,
        );

        let layout = if params.has_lut {
            self.layout_with_lut.as_ref().unwrap()
        } else {
            self.layout_base.as_ref().unwrap()
        };

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RDG ToneMap Pipeline Layout"),
            bind_group_layouts: &[&gpu_world.layout, layout],
            immediate_size: 0,
        });

        let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
            format: params.output_format,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        });

        let key =
            FullscreenPipelineKey::fullscreen(shader_hash, smallvec::smallvec![color_target], None);

        let id = ctx.pipeline_cache.get_or_create_fullscreen(
            device,
            shader_module,
            &pipeline_layout,
            &key,
            &format!(
                "RDG ToneMap Pipeline {:?} lut={}",
                params.mode, params.has_lut
            ),
        );

        self.local_cache.insert(cache_key, id);
        id
    }
}

// =============================================================================
// Transient Pass Node
// =============================================================================

/// Per-frame tone mapping pass node.
struct RdgToneMapPassNode {
    input_tex: TextureNodeId,
    output_tex: TextureNodeId,

    has_lut: bool,
    uniforms_cpu_id: u64,
    lut_handle: Option<TextureHandle>,
    pipeline_id: RenderPipelineId,
    layout: Tracked<wgpu::BindGroupLayout>,

    current_bind_group_key: Option<BindGroupKey>,
}

impl PassNode for RdgToneMapPassNode {
    fn name(&self) -> &'static str {
        "RDG_ToneMap_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        builder.read_texture(self.input_tex);
        builder.write_texture(self.output_tex);
    }

    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {
        let input_view = ctx.views.get_texture_view(self.input_tex);
        let sampler = ctx.sampler_registry.get_common(CommonSampler::LinearClamp);

        let gpu_buffer = ctx
            .resource_manager
            .gpu_buffers
            .get(&self.uniforms_cpu_id)
            .expect("RDG ToneMap: GPU buffer for ToneMappingUniforms must exist");

        let mut key = BindGroupKey::new(self.layout.id())
            .with_resource(input_view.id())
            .with_resource(sampler.id())
            .with_resource(gpu_buffer.id);

        let lut_view_id = if self.has_lut {
            self.lut_handle.and_then(|h| {
                ctx.resource_manager
                    .get_texture_binding(h)
                    .map(|b| b.view_id)
            })
        } else {
            None
        };

        if let Some(lut_id) = lut_view_id {
            key = key.with_resource(lut_id).with_resource(sampler.id());
        }

        if self.current_bind_group_key.as_ref() != Some(&key) {
            if ctx.global_bind_group_cache.get(&key).is_none() {
                let mut entries = vec![
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: gpu_buffer.buffer.as_entire_binding(),
                    },
                ];

                if let Some(lut_handle) = self.lut_handle {
                    if self.has_lut {
                        let lut_view = ctx
                            .resource_manager
                            .get_texture_view(&TextureSource::Asset(lut_handle));
                        entries.push(wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(lut_view),
                        });
                        entries.push(wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        });
                    }
                }

                let new_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RDG ToneMap BindGroup"),
                    layout: &*self.layout,
                    entries: &entries,
                });
                ctx.global_bind_group_cache.insert(key.clone(), new_bg);
            }
            self.current_bind_group_key = Some(key);
        }
    }

    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let Some(global_bind_group) = ctx.global_bind_group else {
            log::warn!("RDG ToneMap: global_bind_group missing, skipping");
            return;
        };

        let output_view = ctx.get_texture_view(self.output_tex);
        let pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.pipeline_id);
        let bind_group_key = self
            .current_bind_group_key
            .as_ref()
            .expect("BindGroupKey should have been set in prepare!");
        let bind_group = ctx
            .global_bind_group_cache
            .get(bind_group_key)
            .expect("BindGroup should have been prepared!");

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RDG ToneMap Pass"),
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

        rpass.set_pipeline(pipeline);
        rpass.set_bind_group(0, global_bind_group, &[]);
        rpass.set_bind_group(1, bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }
}
