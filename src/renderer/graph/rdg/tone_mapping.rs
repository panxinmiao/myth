//! RDG Tone Mapping Post-Processing Pass
//!
//! Converts HDR scene color to LDR for display. Operates in Push mode:
//! all scene-level parameters (mode, uniforms, LUT) are set externally
//! by the Composer before the RDG prepare loop.
//!
//! # RDG Slots
//!
//! - `input_tex`: HDR scene color (after Bloom, if enabled)
//! - `output_tex`: LDR output (fed to FXAA or directly to surface)
//!
//! # Features
//!
//! - Multiple tone mapping algorithms (Linear, Neutral, Reinhard, Cineon, ACES, AgX)
//! - Vignette, color grading (3D LUT), film grain, chromatic aberration
//! - Version-tracked uniform buffer via `CpuBuffer<ToneMappingUniforms>`
//! - L1 pipeline cache with (mode, format, has_lut) key

use rustc_hash::FxHashMap;

use crate::ShaderDefines;
use crate::assets::TextureHandle;
use crate::renderer::core::binding::BindGroupKey;
use crate::renderer::core::resources::Tracked;
use crate::renderer::graph::rdg::builder::PassBuilder;
use crate::renderer::graph::rdg::context::{RdgExecuteContext, RdgPrepareContext};
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

/// RDG Tone Mapping pass.
///
/// All scene-level configuration is pushed into the public fields by the
/// Composer/Renderer before the RDG prepare loop. The pass itself never
/// accesses `Scene` directly.
///
/// # GPU Resource Lifecycle
///
/// - `BindGroupLayout`s and samplers are created lazily on first use.
/// - Pipelines are cached by `(mode, format, has_lut)`.
/// - `BindGroup`s are cached via `GlobalBindGroupCache` with content-addressed keys.
pub struct RdgToneMapPass {
    // ─── RDG Resource Slots (set by Composer) ──────────────────────
    pub input_tex: TextureNodeId,
    pub output_tex: TextureNodeId,

    // ─── Push Parameters (set by Composer from Scene) ──────────────
    /// Current tone mapping algorithm.
    pub mode: ToneMappingMode,
    /// Whether a 3D LUT texture is active.
    pub has_lut: bool,
    /// CPU-side buffer ID for `ToneMappingUniforms`.
    /// The Composer must call `resource_manager.ensure_buffer_id()` before
    /// the RDG prepare loop so the GPU buffer is ready.
    pub uniforms_cpu_id: u64,
    /// Optional LUT texture handle. If `Some`, the Composer must have called
    /// `resource_manager.prepare_texture()` beforehand.
    pub lut_handle: Option<TextureHandle>,
    /// Global state key (render_state_id, scene_id) for looking up
    /// `GpuGlobalState` in `ResourceManager`.
    pub global_state_key: (u32, u32),

    // ─── Internal Stateful Cache ───────────────────────────────────

    /// Base bind group layout (without LUT bindings).
    layout_base: Option<Tracked<wgpu::BindGroupLayout>>,
    /// Extended bind group layout (with LUT bindings).
    layout_with_lut: Option<Tracked<wgpu::BindGroupLayout>>,
    /// Linear sampler for input texture.
    sampler: Option<Tracked<wgpu::Sampler>>,
    /// Linear sampler dedicated to 3D LUT texture.
    lut_sampler: Option<Tracked<wgpu::Sampler>>,

    /// Cached pipeline IDs by (mode, output_format, has_lut).
    local_cache: FxHashMap<PipelineCacheKey, RenderPipelineId>,
    /// Pipeline ID for the current frame.
    current_pipeline: Option<RenderPipelineId>,
    /// Content-addressed key of the current BindGroup.
    current_bind_group_key: Option<BindGroupKey>,
}

impl RdgToneMapPass {
    /// Creates a new tone mapping pass.
    ///
    /// All GPU resources are lazily initialized on first `prepare()` call.
    #[must_use]
    pub fn new() -> Self {
        Self {
            input_tex: TextureNodeId(0),
            output_tex: TextureNodeId(0),

            mode: ToneMappingMode::default(),
            has_lut: false,
            uniforms_cpu_id: 0,
            lut_handle: None,
            global_state_key: (0, 0),

            layout_base: None,
            layout_with_lut: None,
            sampler: None,
            lut_sampler: None,

            local_cache: FxHashMap::default(),
            current_pipeline: None,
            current_bind_group_key: None,
        }
    }

    // ─── Lazy Initialization ───────────────────────────────────────

    fn ensure_layouts(&mut self, device: &wgpu::Device) {
        if self.layout_base.is_some() {
            return;
        }

        // Base entries: input texture + sampler + uniforms
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

        self.layout_base = Some(Tracked::new(
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RDG ToneMap Layout (base)"),
                entries: &base_entries,
            }),
        ));

        // Extended layout: base + LUT 3D texture + LUT sampler
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

        self.layout_with_lut = Some(Tracked::new(
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RDG ToneMap Layout (LUT)"),
                entries: &lut_entries,
            }),
        ));
    }

    fn ensure_samplers(&mut self, device: &wgpu::Device) {
        if self.sampler.is_some() {
            return;
        }

        self.sampler = Some(Tracked::new(device.create_sampler(
            &wgpu::SamplerDescriptor {
                label: Some("RDG ToneMap Sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            },
        )));

        self.lut_sampler = Some(Tracked::new(device.create_sampler(
            &wgpu::SamplerDescriptor {
                label: Some("RDG ToneMap LUT Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            },
        )));
    }

    // ─── Helpers ───────────────────────────────────────────────────

    #[inline]
    fn current_layout(&self) -> &Tracked<wgpu::BindGroupLayout> {
        if self.has_lut {
            self.layout_with_lut.as_ref().unwrap()
        } else {
            self.layout_base.as_ref().unwrap()
        }
    }

    /// Gets or creates a pipeline for the current (mode, format, has_lut) triple.
    fn get_or_create_pipeline(&mut self, ctx: &mut RdgPrepareContext) -> RenderPipelineId {
        let output_format = ctx.graph.resources[self.output_tex.0 as usize].desc.format;
        let cache_key = (self.mode, output_format, self.has_lut);

        if let Some(&id) = self.local_cache.get(&cache_key) {
            return id;
        }

        log::debug!(
            "RDG ToneMap: compiling pipeline for {:?}, fmt={:?}, lut={}",
            self.mode,
            output_format,
            self.has_lut,
        );

        let device = ctx.device;

        // Shader defines
        let mut defines = ShaderDefines::new();
        self.mode.apply_to_defines(&mut defines);
        if self.has_lut {
            defines.set("USE_LUT", "1");
        }

        let gpu_world = ctx
            .resource_manager
            .get_global_state(self.global_state_key.0, self.global_state_key.1)
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

        let layout = self.current_layout();
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RDG ToneMap Pipeline Layout"),
            bind_group_layouts: &[&gpu_world.layout, layout],
            immediate_size: 0,
        });

        let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
            format: output_format,
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
                self.mode, self.has_lut
            ),
        );

        self.local_cache.insert(cache_key, id);
        id
    }
}

impl PassNode for RdgToneMapPass {
    fn name(&self) -> &'static str {
        "RDG_ToneMap_Pass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        builder.read_texture(self.input_tex);
        builder.write_texture(self.output_tex);
    }

    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {
        // ─── 1. Lazy initialization ────────────────────────────────
        self.ensure_layouts(ctx.device);
        self.ensure_samplers(ctx.device);

        // ─── 2. Pipeline (re)creation ──────────────────────────────
        //
        // Check if mode/format/lut changed since last frame.
        let output_format = ctx.graph.resources[self.output_tex.0 as usize].desc.format;
        let cache_key = (self.mode, output_format, self.has_lut);

        if self.current_pipeline.is_none()
            || !self.local_cache.contains_key(&cache_key)
        {
            self.current_pipeline = Some(self.get_or_create_pipeline(ctx));
        } else {
            self.current_pipeline = self.local_cache.get(&cache_key).copied();
        }

        // ─── 3. BindGroup construction ─────────────────────────────
        let input_view = ctx.get_physical_texture(self.input_tex);
        let sampler = self.sampler.as_ref().unwrap();
        let layout = self.current_layout();

        // Look up GPU buffer for uniforms
        let gpu_buffer = ctx
            .resource_manager
            .gpu_buffers
            .get(&self.uniforms_cpu_id)
            .expect("RDG ToneMap: GPU buffer for ToneMappingUniforms must exist");

        let mut key = BindGroupKey::new(layout.id())
            .with_resource(input_view.id())
            .with_resource(sampler.id())
            .with_resource(gpu_buffer.id);

        // LUT resources
        let lut_view_id = if self.has_lut {
            if let Some(lut_handle) = self.lut_handle {
                let binding = ctx.resource_manager.get_texture_binding(lut_handle);
                binding.map(|b| b.view_id)
            } else {
                None
            }
        } else {
            None
        };

        if let Some(lut_id) = lut_view_id {
            let lut_sampler = self.lut_sampler.as_ref().unwrap();
            key = key
                .with_resource(lut_id)
                .with_resource(lut_sampler.id());
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
                        let lut_sampler = self.lut_sampler.as_ref().unwrap();

                        entries.push(wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(lut_view),
                        });
                        entries.push(wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(lut_sampler),
                        });
                    }
                }

                let new_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RDG ToneMap BindGroup"),
                    layout: &**layout,
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
            .get_render_pipeline(self.current_pipeline.expect("Pipeline not initialized!"));

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
