//! Unified Pipeline Cache
//!
//! Central owner of **all** `wgpu::RenderPipeline` and `wgpu::ComputePipeline`
//! instances. Pipelines are stored in contiguous `Vec`s and addressed through
//! lightweight [`RenderPipelineId`] / [`ComputePipelineId`] handles.
//!
//! # Two-Level Caching (L1 / L2)
//!
//! Material-driven geometry pipelines benefit from the existing **L1 fast cache**
//! (`FastPipelineKey` → `RenderPipelineId`) that avoids rebuilding the full
//! descriptor key every frame. L1 keys are cheap `Copy` structs based on
//! resource handles and version counters.
//!
//! All pipeline families share the **L2 canonical cache** keyed by the full
//! state descriptor (`GraphicsPipelineKey`, `FullscreenPipelineKey`,
//! `ComputePipelineKey`, `SimpleGeometryPipelineKey`). A full-state hash is computed
//! only on L1 miss (or for families without L1).
//!
//! # Shader Modules
//!
//! Shader module caching has been extracted into [`ShaderManager`] to decouple
//! concerns. `PipelineCache` depends on `ShaderManager` for module lookup but
//! does not own the module cache.
//!
//! [`ShaderManager`]: super::shader_manager::ShaderManager

#[cfg(feature = "debug_view")]
use myth_scene::DebugViewMode;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::core::BindGroupContext;
use crate::core::gpu::{GpuGlobalState, GpuMaterial, Tracked};
use crate::graph::extracted::SceneFeatures;
use crate::pipeline::pipeline_id::{ComputePipelineId, RenderPipelineId};
use crate::pipeline::pipeline_key::{
    ComputePipelineKey, FullscreenPipelineKey, GraphicsPipelineKey, PipelineFlags,
    SimpleGeometryPipelineKey, fx_hash_key,
};
use crate::pipeline::shader_gen::ShaderCompilationOptions;
use crate::pipeline::shader_manager::{ShaderManager, ShaderSource};
use crate::pipeline::vertex::GeneratedVertexLayout;
use myth_assets::{GeometryHandle, MaterialHandle};

// ─── L1 Fast Keys ────────────────────────────────────────────────────────────

/// L1 fast key for material-driven geometry pipelines.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub struct FastPipelineKey {
    pub material_handle: MaterialHandle,
    pub material_version: u64,
    pub geometry_handle: GeometryHandle,
    pub geometry_version: u64,
    pub instance_variants: u32,
    pub global_state_id: u32,
    pub scene_variants: SceneFeatures,
    pub taa_enabled: bool,
    pub pipeline_settings_version: u64,
    #[cfg(feature = "debug_view")]
    pub debug_view_mode: DebugViewMode,
}

/// L1 fast key for shadow depth-only pipelines.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub struct FastShadowPipelineKey {
    pub material_handle: MaterialHandle,
    pub material_version: u64,
    pub geometry_handle: GeometryHandle,
    pub geometry_version: u64,
    pub instance_variants: u32,
    pub pipeline_settings_version: u64,
}

// ─── Pipeline Cache ──────────────────────────────────────────────────────────

/// Central pipeline storage and deduplication cache.
///
/// Alongside every compiled pipeline, the cache can optionally store the
/// pipeline's [`Tracked<wgpu::BindGroupLayout>`] array.  This allows
/// downstream code (e.g. RDG pass nodes) to look up layouts by pipeline
/// ID without holding onto layout objects themselves, enabling stateless
/// pass-node designs.
pub struct PipelineCache {
    // ---- Storage (contiguous, indexed by Id) ----
    render_pipelines: Vec<wgpu::RenderPipeline>,
    compute_pipelines: Vec<wgpu::ComputePipeline>,

    /// Per-render-pipeline tracked bind-group layouts (parallel to `render_pipelines`).
    /// An entry is `None` if the caller did not register layouts for that pipeline.
    render_layouts: Vec<Option<SmallVec<[Tracked<wgpu::BindGroupLayout>; 4]>>>,
    /// Per-compute-pipeline tracked bind-group layouts.
    compute_layouts: Vec<Option<SmallVec<[Tracked<wgpu::BindGroupLayout>; 4]>>>,

    // ---- L2 canonical lookups (full-state hash → Id) ----
    graphics_lookup: FxHashMap<u64, RenderPipelineId>,
    fullscreen_lookup: FxHashMap<u64, RenderPipelineId>,
    simple_geometry_lookup: FxHashMap<u64, RenderPipelineId>,
    compute_lookup: FxHashMap<u64, ComputePipelineId>,

    // ---- L1 fast lookups (handle+version → Id) ----
    fast_cache: FxHashMap<FastPipelineKey, RenderPipelineId>,
    fast_shadow_cache: FxHashMap<FastShadowPipelineKey, RenderPipelineId>,
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineCache {
    #[must_use]
    pub fn new() -> Self {
        Self {
            render_pipelines: Vec::with_capacity(64),
            compute_pipelines: Vec::with_capacity(8),
            render_layouts: Vec::with_capacity(64),
            compute_layouts: Vec::with_capacity(8),
            graphics_lookup: FxHashMap::default(),
            fullscreen_lookup: FxHashMap::default(),
            simple_geometry_lookup: FxHashMap::default(),
            compute_lookup: FxHashMap::default(),
            fast_cache: FxHashMap::default(),
            fast_shadow_cache: FxHashMap::default(),
        }
    }

    // ── Pipeline Retrieval (execute-phase, O(1)) ─────────────────────────────

    /// Retrieve a render pipeline by handle. **Panics** if the id is invalid.
    #[inline]
    #[must_use]
    pub fn get_render_pipeline(&self, id: RenderPipelineId) -> &wgpu::RenderPipeline {
        &self.render_pipelines[id.index()]
    }

    /// Retrieve a compute pipeline by handle. **Panics** if the id is invalid.
    #[inline]
    #[must_use]
    pub fn get_compute_pipeline(&self, id: ComputePipelineId) -> &wgpu::ComputePipeline {
        &self.compute_pipelines[id.index()]
    }

    // ── Layout Registration & Retrieval ──────────────────────────────────────

    /// Associate tracked bind-group layouts with an existing render pipeline.
    ///
    /// Call this immediately after `get_or_create_*` to enable downstream code
    /// to retrieve layouts via [`get_tracked_layout`](Self::get_tracked_layout)
    /// using only the [`RenderPipelineId`].
    pub fn register_render_layouts(
        &mut self,
        id: RenderPipelineId,
        layouts: SmallVec<[Tracked<wgpu::BindGroupLayout>; 4]>,
    ) {
        let idx = id.index();
        if idx >= self.render_layouts.len() {
            self.render_layouts.resize_with(idx + 1, || None);
        }
        self.render_layouts[idx] = Some(layouts);
    }

    /// Associate tracked bind-group layouts with an existing compute pipeline.
    pub fn register_compute_layouts(
        &mut self,
        id: ComputePipelineId,
        layouts: SmallVec<[Tracked<wgpu::BindGroupLayout>; 4]>,
    ) {
        let idx = id.index();
        if idx >= self.compute_layouts.len() {
            self.compute_layouts.resize_with(idx + 1, || None);
        }
        self.compute_layouts[idx] = Some(layouts);
    }

    /// Retrieve a tracked bind-group layout for a render pipeline.
    ///
    /// **Panics** if no layouts were registered for `id`, or if `group_index`
    /// is out of range.
    #[inline]
    #[must_use]
    pub fn get_tracked_layout(
        &self,
        id: RenderPipelineId,
        group_index: usize,
    ) -> &Tracked<wgpu::BindGroupLayout> {
        self.render_layouts[id.index()]
            .as_ref()
            .expect("No layouts registered for this pipeline")
            .get(group_index)
            .expect("group_index out of range")
    }

    /// Retrieve a tracked bind-group layout for a compute pipeline.
    #[inline]
    #[must_use]
    pub fn get_tracked_compute_layout(
        &self,
        id: ComputePipelineId,
        group_index: usize,
    ) -> &Tracked<wgpu::BindGroupLayout> {
        self.compute_layouts[id.index()]
            .as_ref()
            .expect("No layouts registered for this compute pipeline")
            .get(group_index)
            .expect("group_index out of range")
    }

    // ── Cache Invalidation ───────────────────────────────────────────────────

    /// Clears **all** cached pipelines.
    ///
    /// The `ShaderManager` module cache is *not* affected — shader source code
    /// is independent of these settings.
    #[deprecated(note = "Prefer targeted invalidation via version increments")]
    pub fn clear(&mut self) {
        self.render_pipelines.clear();
        self.compute_pipelines.clear();
        self.graphics_lookup.clear();
        self.fullscreen_lookup.clear();
        self.simple_geometry_lookup.clear();
        self.compute_lookup.clear();
        self.fast_cache.clear();
        self.fast_shadow_cache.clear();
    }

    // ── L1 Fast Cache (material geometry pipelines) ──────────────────────────

    #[must_use]
    pub fn get_pipeline_fast(&self, fast_key: FastPipelineKey) -> Option<RenderPipelineId> {
        self.fast_cache.get(&fast_key).copied()
    }

    pub fn insert_pipeline_fast(&mut self, fast_key: FastPipelineKey, id: RenderPipelineId) {
        self.fast_cache.insert(fast_key, id);
    }

    #[must_use]
    pub fn get_shadow_pipeline_fast(
        &self,
        fast_key: FastShadowPipelineKey,
    ) -> Option<RenderPipelineId> {
        self.fast_shadow_cache.get(&fast_key).copied()
    }

    pub fn insert_shadow_pipeline_fast(
        &mut self,
        fast_key: FastShadowPipelineKey,
        id: RenderPipelineId,
    ) {
        self.fast_shadow_cache.insert(fast_key, id);
    }

    // ── L2 Canonical: Material Geometry Pipeline ─────────────────────────────

    /// Look up or create a material-driven geometry pipeline.
    ///
    /// This is the main entry point for `SceneCullPass`.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn get_or_create_graphics(
        &mut self,
        device: &wgpu::Device,
        shader_manager: &mut ShaderManager,
        template_name: &str,
        canonical_key: &GraphicsPipelineKey,
        options: &ShaderCompilationOptions,
        vertex_layout: &GeneratedVertexLayout,
        gpu_material: &GpuMaterial,
        object_bind_group: &BindGroupContext,
        gpu_world: &GpuGlobalState,
        screen_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> RenderPipelineId {
        let hash = fx_hash_key(&canonical_key);
        if let Some(&id) = self.graphics_lookup.get(&hash) {
            return id;
        }

        // Compile shader via ShaderManager
        let binding_code = format!(
            "{}\n{}\n{}",
            &gpu_world.binding_wgsl, &gpu_material.binding_wgsl, &object_bind_group.binding_wgsl
        );

        let mut opts = options.clone();
        opts.inject_code("vertex_input_code", &vertex_layout.vertex_input_code);
        opts.inject_code("binding_code", binding_code);

        let (shader_module, _code_hash) = shader_manager.get_or_compile(
            device,
            ShaderSource::File(template_name),
            &opts,
        );

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                Some(&gpu_world.layout),
                Some(&gpu_material.layout),
                Some(&object_bind_group.layout),
                Some(screen_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let vertex_buffers_layout: Vec<_> =
            vertex_layout.buffers.iter().map(|l| l.as_wgpu()).collect();

        let blend_state: Option<wgpu::BlendState> =
            canonical_key
                .blend_state
                .as_ref()
                .map(|bk| wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: bk.color.src_factor,
                        dst_factor: bk.color.dst_factor,
                        operation: bk.color.operation,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: bk.alpha.src_factor,
                        dst_factor: bk.alpha.dst_factor,
                        operation: bk.alpha.operation,
                    },
                });

        let mut color_targets = vec![Some(wgpu::ColorTargetState {
            format: canonical_key.color_format,
            blend: blend_state,
            write_mask: wgpu::ColorWrites::ALL,
        })];

        // Specular split requires a second render target for the specular output, which is appended after the main color target.
        if canonical_key.flags.contains(PipelineFlags::SPECULAR_SPLIT) {
            color_targets.push(Some(wgpu::ColorTargetState {
                format: canonical_key.color_format,
                blend: blend_state,
                write_mask: wgpu::ColorWrites::ALL,
            }));
        }

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Scene Render Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: shader_module,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers_layout,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader_module,
                entry_point: Some("fs_main"),
                targets: &color_targets,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: canonical_key.topology,
                front_face: canonical_key.front_face,
                cull_mode: canonical_key.cull_mode,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: canonical_key.depth_format,
                depth_write_enabled: Some(canonical_key.flags.contains(PipelineFlags::DEPTH_WRITE)),
                depth_compare: Some(canonical_key.depth_compare),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: canonical_key.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: canonical_key
                    .flags
                    .contains(PipelineFlags::ALPHA_TO_COVERAGE),
            },
            multiview_mask: None,
            cache: None,
        });

        let id = self.push_render_pipeline(pipeline);
        self.graphics_lookup.insert(hash, id);
        id
    }

    // ── L2 Canonical: Fullscreen / Post-Process Pipeline ─────────────────────

    /// Look up or create a fullscreen / post-processing render pipeline.
    ///
    /// The caller must supply the pre-created `pipeline_layout` because
    /// bind-group layouts vary widely across post-processing passes.
    ///
    /// Primitive state is hardcoded to standard fullscreen-triangle values:
    /// `TriangleList`, no culling, CCW, no vertex buffers.
    pub fn get_or_create_fullscreen(
        &mut self,
        device: &wgpu::Device,
        shader_module: &wgpu::ShaderModule,
        pipeline_layout: &wgpu::PipelineLayout,
        canonical_key: &FullscreenPipelineKey,
        label: &str,
    ) -> RenderPipelineId {
        let hash = fx_hash_key(canonical_key);
        if let Some(&id) = self.fullscreen_lookup.get(&hash) {
            return id;
        }

        // Rebuild wgpu types from key mirrors
        let color_targets: Vec<Option<wgpu::ColorTargetState>> = canonical_key
            .color_targets
            .iter()
            .map(|ct| {
                Some(wgpu::ColorTargetState {
                    format: ct.format,
                    blend: ct.blend.map(|bk| wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: bk.color.src_factor,
                            dst_factor: bk.color.dst_factor,
                            operation: bk.color.operation,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: bk.alpha.src_factor,
                            dst_factor: bk.alpha.dst_factor,
                            operation: bk.alpha.operation,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::from_bits_truncate(ct.write_mask),
                })
            })
            .collect();

        let depth_stencil = canonical_key
            .depth_stencil
            .map(|dk| wgpu::DepthStencilState {
                format: dk.format,
                depth_write_enabled: dk.depth_write_enabled,
                depth_compare: dk.depth_compare,
                stencil: wgpu::StencilState {
                    front: wgpu::StencilFaceState {
                        compare: dk.stencil.front.compare,
                        fail_op: dk.stencil.front.fail_op,
                        depth_fail_op: dk.stencil.front.depth_fail_op,
                        pass_op: dk.stencil.front.pass_op,
                    },
                    back: wgpu::StencilFaceState {
                        compare: dk.stencil.back.compare,
                        fail_op: dk.stencil.back.fail_op,
                        depth_fail_op: dk.stencil.back.depth_fail_op,
                        pass_op: dk.stencil.back.pass_op,
                    },
                    read_mask: dk.stencil.read_mask,
                    write_mask: dk.stencil.write_mask,
                },
                bias: wgpu::DepthBiasState {
                    constant: dk.bias.constant,
                    slope_scale: f32::from_bits(dk.bias.slope_scale_bits),
                    clamp: f32::from_bits(dk.bias.clamp_bits),
                },
            });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader_module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader_module,
                entry_point: Some("fs_main"),
                targets: &color_targets,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil,
            multisample: wgpu::MultisampleState {
                count: canonical_key.multisample.count,
                mask: canonical_key.multisample.mask,
                alpha_to_coverage_enabled: canonical_key.multisample.alpha_to_coverage_enabled,
            },
            multiview_mask: None,
            cache: None,
        });

        let id = self.push_render_pipeline(pipeline);
        self.fullscreen_lookup.insert(hash, id);
        id
    }

    // ── L2 Canonical: Simple Geometry Pipeline (Prepass, Shadow) ──────────

    /// Look up or create a simplified geometry pipeline (prepass or shadow).
    ///
    /// These passes render actual meshes with vertex input but skip complex
    /// material/lighting state. The caller supplies pre-compiled shader module,
    /// pipeline layout, and vertex buffer layouts.
    pub fn get_or_create_simple_geometry(
        &mut self,
        device: &wgpu::Device,
        shader_module: &wgpu::ShaderModule,
        pipeline_layout: &wgpu::PipelineLayout,
        canonical_key: &SimpleGeometryPipelineKey,
        label: &str,
        vertex_buffers: &[wgpu::VertexBufferLayout<'_>],
    ) -> RenderPipelineId {
        let hash = fx_hash_key(canonical_key);
        if let Some(&id) = self.simple_geometry_lookup.get(&hash) {
            return id;
        }

        // Rebuild wgpu types from key mirrors
        let color_targets: Vec<Option<wgpu::ColorTargetState>> = canonical_key
            .color_targets
            .iter()
            .map(|ct| {
                Some(wgpu::ColorTargetState {
                    format: ct.format,
                    blend: ct.blend.map(|bk| wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: bk.color.src_factor,
                            dst_factor: bk.color.dst_factor,
                            operation: bk.color.operation,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: bk.alpha.src_factor,
                            dst_factor: bk.alpha.dst_factor,
                            operation: bk.alpha.operation,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::from_bits_truncate(ct.write_mask),
                })
            })
            .collect();

        let dk = &canonical_key.depth_stencil;
        let depth_stencil = Some(wgpu::DepthStencilState {
            format: dk.format,
            depth_write_enabled: dk.depth_write_enabled,
            depth_compare: dk.depth_compare,
            stencil: wgpu::StencilState {
                front: wgpu::StencilFaceState {
                    compare: dk.stencil.front.compare,
                    fail_op: dk.stencil.front.fail_op,
                    depth_fail_op: dk.stencil.front.depth_fail_op,
                    pass_op: dk.stencil.front.pass_op,
                },
                back: wgpu::StencilFaceState {
                    compare: dk.stencil.back.compare,
                    fail_op: dk.stencil.back.fail_op,
                    depth_fail_op: dk.stencil.back.depth_fail_op,
                    pass_op: dk.stencil.back.pass_op,
                },
                read_mask: dk.stencil.read_mask,
                write_mask: dk.stencil.write_mask,
            },
            bias: wgpu::DepthBiasState {
                constant: dk.bias.constant,
                slope_scale: f32::from_bits(dk.bias.slope_scale_bits),
                clamp: f32::from_bits(dk.bias.clamp_bits),
            },
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader_module,
                entry_point: Some("vs_main"),
                buffers: vertex_buffers,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader_module,
                entry_point: Some("fs_main"),
                targets: &color_targets,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: canonical_key.topology,
                front_face: canonical_key.front_face,
                cull_mode: canonical_key.cull_mode,
                ..Default::default()
            },
            depth_stencil,
            multisample: wgpu::MultisampleState {
                count: canonical_key.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        let id = self.push_render_pipeline(pipeline);
        self.simple_geometry_lookup.insert(hash, id);
        id
    }

    // ── L2 Canonical: Compute Pipeline ───────────────────────────────────────

    /// Look up or create a compute pipeline.
    pub fn get_or_create_compute(
        &mut self,
        device: &wgpu::Device,
        shader_module: &wgpu::ShaderModule,
        pipeline_layout: &wgpu::PipelineLayout,
        canonical_key: &ComputePipelineKey,
        label: &str,
    ) -> ComputePipelineId {
        let hash = fx_hash_key(canonical_key);
        if let Some(&id) = self.compute_lookup.get(&hash) {
            return id;
        }

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(pipeline_layout),
            module: shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let id = self.push_compute_pipeline(pipeline);
        self.compute_lookup.insert(hash, id);
        id
    }

    // ── Stats ────────────────────────────────────────────────────────────────

    /// Number of cached render pipelines.
    #[must_use]
    pub fn render_pipeline_count(&self) -> usize {
        self.render_pipelines.len()
    }

    /// Number of cached compute pipelines.
    #[must_use]
    pub fn compute_pipeline_count(&self) -> usize {
        self.compute_pipelines.len()
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    fn push_render_pipeline(&mut self, pipeline: wgpu::RenderPipeline) -> RenderPipelineId {
        let id = RenderPipelineId(self.render_pipelines.len() as u32);
        self.render_pipelines.push(pipeline);
        self.render_layouts.push(None);
        id
    }

    fn push_compute_pipeline(&mut self, pipeline: wgpu::ComputePipeline) -> ComputePipelineId {
        let id = ComputePipelineId(self.compute_pipelines.len() as u32);
        self.compute_pipelines.push(pipeline);
        self.compute_layouts.push(None);
        id
    }
}
