//! Debug View Feature + Ephemeral PassNode
//!
//! Provides runtime visualisation of intermediate RDG textures (depth,
//! normals, velocity, SSAO, bloom, etc.) without invasive pipeline changes.
//!
//! - **`DebugViewFeature`** (long-lived): owns the pipeline, bind-group
//!   layouts, and a small uniform buffer for the `view_mode` selector.
//!   `extract_and_prepare()` compiles the fullscreen pipeline once and
//!   re-creates it only when the surface format changes.
//!
//! - **`DebugViewPassNode`** (ephemeral per-frame): carries the source
//!   texture ID, target surface ID, and borrowed references to the
//!   Feature's persistent resources.  Created by
//!   `DebugViewFeature::add_to_graph()` each frame.
//!
//! # Bind Group Model
//!
//! - **Group 0 (static)**: sampler + uniform buffer — Feature-owned,
//!   rebuilt only when the uniform buffer identity changes.
//! - **Group 1 (transient)**: source texture — PassNode-owned, rebuilt
//!   each frame during the RDG prepare phase.
//!
//! # Safety
//!
//! The entire module is gated behind `#[cfg(feature = "debug_view")]`.
//! When the feature is disabled, no code is compiled and the engine has
//! zero overhead from this system.

#![cfg(feature = "debug_view")]

use bytemuck::{Pod, Zeroable};
use wgpu::CommandEncoder;

use crate::core::gpu::{CommonSampler, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    ExecuteContext, ExtractContext, PassNode, PrepareContext, RenderTargetOps, TextureNodeId,
};
use crate::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions, ShaderSource,
};
use myth_resources::buffer::CpuBuffer;

// ─── GPU Uniform Layout ─────────────────────────────────────────────────────

/// Maps to the `DebugUniforms` struct in `debug_view.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DebugViewUniforms {
    pub view_mode: u32,
    pub custom_scale: f32,
    pub z_near: f32,
    pub z_far: f32,
}

impl Default for DebugViewUniforms {
    fn default() -> Self {
        Self {
            view_mode: 0,
            custom_scale: 100.0,
            z_near: 0.1,
            z_far: 100.0,
        }
    }
}

impl myth_resources::buffer::GpuData for DebugViewUniforms {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
    fn byte_size(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Feature (long-lived, stored in RendererState)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Persistent resources for the debug-view overlay pass.
pub struct DebugViewFeature {
    /// L1 cache key: surface format the pipeline was compiled for.
    l1_cache_format: Option<wgpu::TextureFormat>,
    l1_cache_is_depth: Option<bool>,
    pipeline_id: Option<RenderPipelineId>,

    /// Group 0 (static): sampler + uniform buffer.
    static_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    /// Group 1 (transient): source texture.
    transient_layout_color: Option<Tracked<wgpu::BindGroupLayout>>,
    transient_layout_depth: Option<Tracked<wgpu::BindGroupLayout>>,

    /// Feature-owned static bind group (Group 0).
    static_bg: Option<wgpu::BindGroup>,
    /// Staleness tracking for uniform buffer identity.
    last_uniforms_buffer_id: u64,

    uniforms: CpuBuffer<DebugViewUniforms>,
}

impl Default for DebugViewFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl DebugViewFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            l1_cache_format: None,
            l1_cache_is_depth: None,
            pipeline_id: None,
            static_layout: None,
            transient_layout_color: None,
            transient_layout_depth: None,
            static_bg: None,
            last_uniforms_buffer_id: 0,
            uniforms: CpuBuffer::new(
                DebugViewUniforms::default(),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("DebugView Uniforms"),
            ),
        }
    }

    /// Lazily create the two bind group layouts.
    fn ensure_layouts(&mut self, device: &wgpu::Device) {
        if self.static_layout.is_some() {
            return;
        }

        // Group 0 (static): sampler + uniforms
        self.static_layout = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("DebugView Static Layout (G0)"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            },
        )));

        // Group 1 (transient): source texture
        self.transient_layout_color = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("DebugView Transient Layout Color (G1)"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            },
        )));

        self.transient_layout_depth = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("DebugView Transient Layout Depth (G1)"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            },
        )));
    }

    /// Pre-RDG preparation: ensure layouts, compile pipeline, upload
    /// uniforms, and build the static bind group.
    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        output_format: wgpu::TextureFormat,
        params: DebugViewUniforms,
        is_depth: bool,
    ) {
        self.ensure_layouts(ctx.device);

        // Update the CPU-side uniform and flush to GPU.
        {
            let mut guard = self.uniforms.write();
            *guard = params;
        }
        ctx.resource_manager.ensure_buffer(&self.uniforms);

        // ── Pipeline (re)creation on format change ─────────────────
        if self.l1_cache_format != Some(output_format) || self.l1_cache_is_depth != Some(is_depth) {
            let mut options = ShaderCompilationOptions::default();

            if is_depth {
                options.add_define("IS_DEPTH", "1");
            }

            let (shader_module, shader_hash) = ctx.shader_manager.get_or_compile(
                ctx.device,
                ShaderSource::File("entry/post_process/debug_view"),
                &options,
            );

            let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
                format: output_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            });

            let key = FullscreenPipelineKey::fullscreen(
                shader_hash,
                smallvec::smallvec![color_target],
                None,
            );

            let transient_layout = if is_depth {
                self.transient_layout_depth.as_deref()
            } else {
                self.transient_layout_color.as_deref()
            };

            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("DebugView Pipeline Layout"),
                        bind_group_layouts: &[self.static_layout.as_deref(), transient_layout],
                        immediate_size: 0,
                    });

            self.pipeline_id = Some(ctx.pipeline_cache.get_or_create_fullscreen(
                ctx.device,
                shader_module,
                &pipeline_layout,
                &key,
                "DebugView Pipeline",
            ));
            self.l1_cache_format = Some(output_format);
            self.l1_cache_is_depth = Some(is_depth);
        }

        // ── Static bind group (Group 0) — rebuild on buffer identity change
        if let Some(handle) = self.uniforms.gpu_handle()
            && let Some(gpu_buf) = ctx.resource_manager.gpu_buffers.get(handle)
            && (self.static_bg.is_none() || self.last_uniforms_buffer_id != gpu_buf.id)
        {
            let sampler = ctx
                .resource_manager
                .sampler_registry
                .get_common(CommonSampler::NearestClamp);
            let layout = self.static_layout.as_ref().unwrap();

            self.static_bg = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DebugView Static BG (G0)"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: gpu_buf.buffer.as_entire_binding(),
                    },
                ],
            }));
            self.last_uniforms_buffer_id = gpu_buf.id;
        }
    }

    /// Inject the debug-view pass into the render graph.
    ///
    /// Reads `source_tex`, writes to `target_surface` via SSA relay, and
    /// returns the updated surface handle.
    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        source_tex: TextureNodeId,
        target_surface: TextureNodeId,
        is_depth: bool,
    ) -> TextureNodeId {
        let pipeline_id = self.pipeline_id.expect("DebugViewFeature not prepared");
        let pipeline = ctx.pipeline_cache.get_render_pipeline(pipeline_id);
        let static_bg = self
            .static_bg
            .as_ref()
            .expect("DebugViewFeature: static BG not built");
        let transient_layout = if is_depth {
            self.transient_layout_depth.as_ref().unwrap()
        } else {
            self.transient_layout_color.as_ref().unwrap()
        };

        ctx.graph.add_pass("DebugView_Pass", |builder| {
            builder.read_texture(source_tex);
            let output = builder.mutate_texture(target_surface, "Surface_DebugView");

            let node = DebugViewPassNode {
                source_tex,
                output_tex: output,
                pipeline,
                static_bg,
                transient_layout,
                transient_bg: None,
            };
            (node, output)
        })
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PassNode (ephemeral, created per frame)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct DebugViewPassNode<'a> {
    source_tex: TextureNodeId,
    output_tex: TextureNodeId,
    pipeline: &'a wgpu::RenderPipeline,
    /// Feature-owned static bind group (Group 0): sampler + uniforms.
    static_bg: &'a wgpu::BindGroup,
    /// Layout for transient bind group (Group 1).
    transient_layout: &'a Tracked<wgpu::BindGroupLayout>,
    /// Transient bind group built in `prepare()`.
    transient_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for DebugViewPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.transient_bg = Some(
            crate::myth_bind_group!(ctx, self.transient_layout, Some("DebugView Transient BG (G1)"), [
                0 => self.source_tex,
            ]),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut CommandEncoder) {
        let transient_bg = self
            .transient_bg
            .expect("DebugView transient BG not prepared!");

        let rtt = ctx.get_color_attachment(self.output_tex, RenderTargetOps::DontCare, None);

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("DebugView Pass"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(self.pipeline);
        rpass.set_bind_group(0, self.static_bg, &[]);
        rpass.set_bind_group(1, transient_bg, &[]);
        rpass.draw(0..3, 0..1);
    }
}
