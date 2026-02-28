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
//! `ComputePipelineKey`, `ShadowPipelineKey`). A full-state hash is computed
//! only on L1 miss (or for families without L1).
//!
//! # Shader Modules
//!
//! Shader module caching has been extracted into [`ShaderManager`] to decouple
//! concerns. `PipelineCache` depends on `ShaderManager` for module lookup but
//! does not own the module cache.
//!
//! [`ShaderManager`]: super::shader_manager::ShaderManager

use rustc_hash::FxHashMap;

use crate::assets::{GeometryHandle, MaterialHandle};
use crate::renderer::core::BindGroupContext;
use crate::renderer::core::resources::{GpuGlobalState, GpuMaterial};
use crate::renderer::graph::context::FrameResources;
use crate::renderer::graph::extracted::SceneFeatures;
use crate::renderer::pipeline::pipeline_id::{ComputePipelineId, RenderPipelineId};
use crate::renderer::pipeline::pipeline_key::{
    ComputePipelineKey, FullscreenPipelineKey, GraphicsPipelineKey, ShadowPipelineKey, fx_hash_key,
};
use crate::renderer::pipeline::shader_gen::ShaderCompilationOptions;
use crate::renderer::pipeline::shader_manager::ShaderManager;
use crate::renderer::pipeline::vertex::GeneratedVertexLayout;

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
    pub pipeline_settings_version: u64,
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
pub struct PipelineCache {
    // ---- Storage (contiguous, indexed by Id) ----
    render_pipelines: Vec<wgpu::RenderPipeline>,
    compute_pipelines: Vec<wgpu::ComputePipeline>,

    // ---- L2 canonical lookups (full-state hash → Id) ----
    graphics_lookup: FxHashMap<u64, RenderPipelineId>,
    fullscreen_lookup: FxHashMap<u64, RenderPipelineId>,
    shadow_lookup: FxHashMap<u64, RenderPipelineId>,
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
            graphics_lookup: FxHashMap::default(),
            fullscreen_lookup: FxHashMap::default(),
            shadow_lookup: FxHashMap::default(),
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

    // ── Cache Invalidation ───────────────────────────────────────────────────

    /// Clears **all** cached pipelines (called when MSAA / HDR settings change).
    ///
    /// The `ShaderManager` module cache is *not* affected — shader source code
    /// is independent of these settings.
    pub fn clear(&mut self) {
        self.render_pipelines.clear();
        self.compute_pipelines.clear();
        self.graphics_lookup.clear();
        self.fullscreen_lookup.clear();
        self.shadow_lookup.clear();
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
        frame_resources: &FrameResources,
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

        let (shader_module, _code_hash) = shader_manager.get_or_compile_template(
            device,
            template_name,
            options,
            &vertex_layout.vertex_input_code,
            &binding_code,
        );

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &gpu_world.layout,
                &gpu_material.layout,
                &object_bind_group.layout,
                &frame_resources.screen_bind_group_layout,
            ],
            immediate_size: 0,
        });

        let vertex_buffers_layout: Vec<_> =
            vertex_layout.buffers.iter().map(|l| l.as_wgpu()).collect();

        let blend_state: Option<wgpu::BlendState> =
            canonical_key.blend_state.as_ref().map(|bk| wgpu::BlendState {
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

        let color_targets = if canonical_key.is_specular_split {
            vec![
                Some(wgpu::ColorTargetState {
                    format: canonical_key.color_format,
                    blend: blend_state,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: canonical_key.color_format,
                    blend: blend_state,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ]
        } else {
            vec![Some(wgpu::ColorTargetState {
                format: canonical_key.color_format,
                blend: blend_state,
                write_mask: wgpu::ColorWrites::ALL,
            })]
        };

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
                depth_write_enabled: canonical_key.depth_write,
                depth_compare: canonical_key.depth_compare,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: canonical_key.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: canonical_key.alpha_to_coverage,
            },
            multiview_mask: None,
            cache: None,
        });

        let id = self.push_render_pipeline(pipeline);
        self.graphics_lookup.insert(hash, id);
        id
    }

    // ── L2 Canonical: Shadow Pipeline ────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    pub fn get_or_create_shadow(
        &mut self,
        device: &wgpu::Device,
        shader_manager: &mut ShaderManager,
        canonical_key: &ShadowPipelineKey,
        options: &ShaderCompilationOptions,
        vertex_layout: &GeneratedVertexLayout,
        shadow_global_layout: &wgpu::BindGroupLayout,
        shadow_binding_wgsl: &str,
        gpu_material: &GpuMaterial,
        object_bind_group: &BindGroupContext,
    ) -> RenderPipelineId {
        let hash = fx_hash_key(canonical_key);
        if let Some(&id) = self.shadow_lookup.get(&hash) {
            return id;
        }

        let binding_code = format!(
            "{}\n{}\n{}",
            shadow_binding_wgsl, &gpu_material.binding_wgsl, &object_bind_group.binding_wgsl
        );

        let (shader_module, _code_hash) = shader_manager.get_or_compile_template(
            device,
            "passes/depth",
            options,
            &vertex_layout.vertex_input_code,
            &binding_code,
        );

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Pipeline Layout"),
            bind_group_layouts: &[
                shadow_global_layout,
                &gpu_material.layout,
                &object_bind_group.layout,
            ],
            immediate_size: 0,
        });

        let vertex_buffers_layout: Vec<_> =
            vertex_layout.buffers.iter().map(|l| l.as_wgpu()).collect();

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Pipeline"),
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
                targets: &[],
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
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let id = self.push_render_pipeline(pipeline);
        self.shadow_lookup.insert(hash, id);
        id
    }

    // ── L2 Canonical: Fullscreen / Post-Process Pipeline ─────────────────────

    /// Look up or create a fullscreen / post-processing render pipeline.
    ///
    /// The caller must supply the pre-created `pipeline_layout` because
    /// bind-group layouts vary widely across post-processing passes.
    pub fn get_or_create_fullscreen(
        &mut self,
        device: &wgpu::Device,
        shader_module: &wgpu::ShaderModule,
        pipeline_layout: &wgpu::PipelineLayout,
        canonical_key: &FullscreenPipelineKey,
        label: &str,
        vertex_buffers: &[wgpu::VertexBufferLayout<'_>],
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

        let depth_stencil = canonical_key.depth_stencil.map(|dk| wgpu::DepthStencilState {
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
        id
    }

    fn push_compute_pipeline(&mut self, pipeline: wgpu::ComputePipeline) -> ComputePipelineId {
        let id = ComputePipelineId(self.compute_pipelines.len() as u32);
        self.compute_pipelines.push(pipeline);
        id
    }
}
