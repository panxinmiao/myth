//! Depth-Normal Pre Pass
//!
//! Writes depth and view-space normals to the scene depth & normal buffers
//! before the main opaque pass. This enables:
//!
//! 1. **Z-Prepass optimisation**: the subsequent `OpaquePass` can run with
//!    `depth_write = false` and `depth_compare = Equal`, guaranteeing zero
//!    overdraw for expensive PBR shading.
//! 2. **Mini G-Buffer**: the normal texture is consumed by SSAO (and
//!    potentially other screen-space effects) without a full deferred pass.
//!
//! # Data Flow
//! ```text
//! RenderLists.opaque → DepthNormalPrepass → SceneDepth + SceneNormal
//! ```
//!
//! # Pipeline Cache
//!
//! The pass maintains its own `FxHashMap<u16, wgpu::RenderPipeline>` keyed
//! by the main-pass `pipeline_id`.  Two objects that share a main pipeline
//! also share a prepass pipeline (same vertex layout, topology, cull mode,
//! and material macros).

use rustc_hash::FxHashMap;

use crate::renderer::core::resources::Tracked;
use crate::renderer::graph::context::{ExecuteContext, GraphResource, PrepareContext};
use crate::renderer::graph::{RenderNode, TrackedRenderPass};
use crate::renderer::pipeline::shader_gen::{ShaderCompilationOptions, ShaderGenerator};
use crate::resources::material::{AlphaMode, Side};

/// Normal texture format — Rgba8Unorm ([-1,1] → [0,1] mapping).
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Returns `true` when the given depth format also contains a stencil aspect.
/// Used to gate the optional stencil-write optimisation for the SSSSS pass.
#[inline]
fn depth_format_has_stencil(fmt: wgpu::TextureFormat) -> bool {
    matches!(
        fmt,
        wgpu::TextureFormat::Depth24PlusStencil8 | wgpu::TextureFormat::Depth32FloatStencil8
    )
}

pub struct DepthNormalPrepass {
    pub(crate) needs_normal: bool, // Whether to output normals (for SSAO) or just do a Z-prepass

    /// Whether to encode `screen_space_id` into Normal.a and write the geometry
    /// stencil mask for the SSSSS pass.
    ///
    /// Set to `true` by `prepare()` when `extracted_scene.has_screen_space_features` is set.
    /// Invalidates the pipeline cache dimension so that new pipelines are compiled
    /// with the `USE_SCREEN_SPACE_FEATURES` define.
    pub(crate) use_screen_space: bool,

    depth_view: Option<Tracked<wgpu::TextureView>>,
    normal_view: Option<Tracked<wgpu::TextureView>>,

    /// Pipeline cache keyed by `(pipeline_id, needs_normal, use_screen_space)`.
    ///
    /// The third dimension ensures pipelines are re-compiled when SSS features
    /// are toggled — the shader variant differs between the two states.
    pipeline_cache: FxHashMap<(u16, bool, bool), wgpu::RenderPipeline>,
}

impl DepthNormalPrepass {
    #[must_use]
    pub fn new() -> Self {
        Self {
            needs_normal: false,
            use_screen_space: false,
            depth_view: None,
            normal_view: None,
            pipeline_cache: FxHashMap::default(),
        }
    }

    /// Pre-build prepass pipelines for every unique `pipeline_id` in the
    /// opaque command list.  Called during `prepare()` so that `run()` can
    /// remain `&self`.
    fn prepare_pipelines(&mut self, ctx: &mut PrepareContext) {
        let render_state_id = ctx.render_state.id;
        let scene_id = ctx.extracted_scene.scene_id;

        let Some(gpu_world) = ctx
            .resource_manager
            .get_global_state(render_state_id, scene_id)
        else {
            return;
        };

        let depth_format = ctx.wgpu_ctx.depth_format;
        let has_stencil = depth_format_has_stencil(depth_format);
        let gpu_world_binding_wgsl = gpu_world.binding_wgsl.clone();
        let gpu_world_layout_clone = gpu_world.layout.clone();

        let geo_guard = ctx.assets.geometries.read_lock();
        let mat_guard = ctx.assets.materials.read_lock();

        for cmd in &ctx.render_lists.opaque {
            // Skip if already cached (all three key dimensions must match)
            if self
                .pipeline_cache
                .contains_key(&(cmd.pipeline_id, self.needs_normal, self.use_screen_space))
            {
                continue;
            }

            let Some(geometry) = geo_guard.map.get(cmd.geometry_handle) else {
                continue;
            };
            let Some(material) = mat_guard.map.get(cmd.material_handle) else {
                continue;
            };

            // Transparent / blend materials should not be in the opaque list,
            // but guard just in case.
            if material.alpha_mode() == AlphaMode::Blend {
                continue;
            }

            let Some(gpu_geometry) = ctx.resource_manager.get_geometry(cmd.geometry_handle) else {
                continue;
            };
            let Some(gpu_material) = ctx.resource_manager.get_material(cmd.material_handle) else {
                continue;
            };

            // Find the corresponding extracted item to get per-instance info
            // (item_shader_defines, item_variant_flags).
            // This lookup runs only once per unique pipeline_id.
            let extracted_item = ctx.extracted_scene.render_items.iter().find(|item| {
                item.geometry == cmd.geometry_handle && item.material == cmd.material_handle
            });

            let (item_variant_flags, item_shader_defines) = match extracted_item {
                Some(item) => (item.item_variant_flags, Some(&item.item_shader_defines)),
                None => (0, None),
            };

            // ── Shader defines ─────────────────────────────────────────
            let geo_defines = geometry.shader_defines();
            let mat_defines = material.shader_defines();

            let empty_defines = crate::resources::shader_defines::ShaderDefines::new();
            let item_def = item_shader_defines.unwrap_or(&empty_defines);

            let mut options = ShaderCompilationOptions::from_merged(
                &mat_defines,
                geo_defines,
                &empty_defines,
                item_def,
            );

            // Inject prepass-specific macros
            options.add_define("IS_PREPASS", "1");

            if self.needs_normal {
                options.add_define("OUTPUT_NORMAL", "1");
            }

            if self.use_screen_space {
                // Activates the `screen_space_id → Normal.a` encoding branch in depth.wgsl.
                options.add_define("USE_SCREEN_SPACE_FEATURES", "1");
            }

            // ── Shader generation ──────────────────────────────────────
            let binding_code = format!(
                "{}\n{}\n{}",
                &gpu_world_binding_wgsl,
                &gpu_material.binding_wgsl,
                &cmd.object_bind_group.binding_wgsl
            );

            let shader_source = ShaderGenerator::generate_shader(
                &gpu_geometry.layout_info.vertex_input_code,
                &binding_code,
                "passes/depth",
                &options,
            );

            let shader_module =
                ctx.wgpu_ctx
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Prepass Shader"),
                        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
                    });

            // ── Pipeline layout ────────────────────────────────────────
            let layout =
                ctx.wgpu_ctx
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Prepass Pipeline Layout"),
                        bind_group_layouts: &[
                            &gpu_world_layout_clone,       // Set 0: Global
                            &gpu_material.layout,          // Set 1: Material
                            &cmd.object_bind_group.layout, // Set 2: Object
                        ],
                        immediate_size: 0,
                    });

            let vertex_buffers_layout: Vec<_> = gpu_geometry
                .layout_info
                .buffers
                .iter()
                .map(|l| l.as_wgpu())
                .collect();

            let cull_mode = match material.side() {
                Side::Front => Some(wgpu::Face::Back),
                Side::Back => Some(wgpu::Face::Front),
                Side::Double => None,
            };

            let front_face = if item_variant_flags & 0x1 != 0 {
                wgpu::FrontFace::Cw
            } else {
                wgpu::FrontFace::Ccw
            };

            // ── Pipeline ───────────────────────────────────────────────
            let pipeline =
                ctx.wgpu_ctx
                    .device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("Prepass Pipeline"),
                        layout: Some(&layout),
                        vertex: wgpu::VertexState {
                            module: &shader_module,
                            entry_point: Some("vs_main"),
                            buffers: &vertex_buffers_layout,
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &shader_module,
                            entry_point: Some("fs_main"),
                            targets: if self.needs_normal {
                                &[Some(wgpu::ColorTargetState {
                                    format: NORMAL_FORMAT,
                                    blend: None,
                                    write_mask: wgpu::ColorWrites::ALL,
                                })]
                            } else {
                                &[]
                            },
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                        }),
                        primitive: wgpu::PrimitiveState {
                            topology: geometry.topology,
                            cull_mode,
                            front_face,
                            ..Default::default()
                        },
                        depth_stencil: Some(wgpu::DepthStencilState {
                            format: depth_format,
                            depth_write_enabled: true,
                            // Reverse-Z: Greater
                            depth_compare: wgpu::CompareFunction::Greater,
                            stencil: if self.use_screen_space && has_stencil {
                                // Write a 1 into the stencil for every drawn geometry fragment.
                                // The SSSSS pass reads stencil != 0 to skip background pixels,
                                // then uses Normal.a to identify SSS vs non-SSS geometry.
                                wgpu::StencilState {
                                    front: wgpu::StencilFaceState {
                                        compare: wgpu::CompareFunction::Always,
                                        depth_fail_op: wgpu::StencilOperation::Keep,
                                        fail_op: wgpu::StencilOperation::Keep,
                                        pass_op: wgpu::StencilOperation::Replace,
                                    },
                                    back: wgpu::StencilFaceState {
                                        compare: wgpu::CompareFunction::Always,
                                        depth_fail_op: wgpu::StencilOperation::Keep,
                                        fail_op: wgpu::StencilOperation::Keep,
                                        pass_op: wgpu::StencilOperation::Replace,
                                    },
                                    read_mask: 0xFF,
                                    write_mask: 0xFF,
                                }
                            } else {
                                wgpu::StencilState::default()
                            },
                            bias: wgpu::DepthBiasState::default(),
                        }),
                        multisample: wgpu::MultisampleState {
                            count: 1, // Prepass is always 1× (no MSAA)
                            ..Default::default()
                        },
                        multiview_mask: None,
                        cache: None,
                    });

            self.pipeline_cache
                .insert((cmd.pipeline_id, self.needs_normal, self.use_screen_space), pipeline);
        }
    }
}

impl Default for DepthNormalPrepass {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderNode for DepthNormalPrepass {
    fn name(&self) -> &'static str {
        "Depth Normal Prepass"
    }

    fn prepare(&mut self, ctx: &mut PrepareContext) {
        if !ctx.wgpu_ctx.render_path.requires_z_prepass() {
            return;
        }

        self.depth_view = Some(ctx.get_resource_view(GraphResource::SceneDepth).clone());
        self.normal_view = Some(ctx.get_resource_view(GraphResource::SceneNormal).clone());

        // Mirror the screen-space feature state from the extracted scene.
        // A change here invalidates the pipeline cache key, triggering re-compilation.
        self.use_screen_space = ctx.extracted_scene.has_screen_space_features;

        // Pre-build pipelines for all unique pipeline IDs in the opaque list
        self.prepare_pipelines(ctx);
    }

    fn run(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        if !ctx.wgpu_ctx.render_path.requires_z_prepass() {
            return;
        }

        let render_lists = &ctx.render_lists;
        let Some(gpu_global_bind_group) = &render_lists.gpu_global_bind_group else {
            return;
        };

        let depth_view = self
            .depth_view
            .as_ref()
            .expect("Prepass: missing depth view");
        let normal_view = self
            .normal_view
            .as_ref()
            .expect("Prepass: missing normal view");

        let color_attachments = if self.needs_normal {
            vec![Some(wgpu::RenderPassColorAttachment {
                view: normal_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Background: no valid normal (alpha = 0)
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.5,
                        g: 0.5,
                        b: 1.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })]
        } else {
            vec![]
        };

        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Depth Normal Prepass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    // Reverse-Z: 0.0 = far plane
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: if self.use_screen_space
                    && depth_format_has_stencil(ctx.wgpu_ctx.depth_format)
                {
                    // Clear stencil to 0 (background).  Drawn fragments will write 1.
                    Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: wgpu::StoreOp::Store,
                    })
                } else {
                    None
                },
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        let mut tracked_pass = TrackedRenderPass::new(pass);

        // When writing stencil, set a uniform reference of 1 for all geometry.
        if self.use_screen_space && depth_format_has_stencil(ctx.wgpu_ctx.depth_format) {
            tracked_pass.raw_pass().set_stencil_reference(1);
        }

        tracked_pass.set_bind_group(
            0,
            render_lists.gpu_global_bind_group_id,
            gpu_global_bind_group,
            &[],
        );

        for cmd in &render_lists.opaque {
            let Some(pipeline) = self
                .pipeline_cache
                .get(&(cmd.pipeline_id, self.needs_normal, self.use_screen_space))
            else {
                continue;
            };

            tracked_pass.set_pipeline(cmd.pipeline_id, pipeline);

            // Set 1: Material (for alpha-tested materials to read the base map)
            if let Some(gpu_material) = ctx.resource_manager.get_material(cmd.material_handle) {
                tracked_pass.set_bind_group(
                    1,
                    gpu_material.bind_group_id,
                    &gpu_material.bind_group,
                    &[],
                );
            }

            // Set 2: Object (model matrix, skinning, morph targets)
            tracked_pass.set_bind_group(
                2,
                cmd.object_bind_group.bind_group_id,
                &cmd.object_bind_group.bind_group,
                &[cmd.dynamic_offset],
            );

            // Draw
            if let Some(gpu_geometry) = ctx.resource_manager.get_geometry(cmd.geometry_handle) {
                for (slot, buffer) in gpu_geometry.vertex_buffers.iter().enumerate() {
                    tracked_pass.set_vertex_buffer(
                        slot as u32,
                        gpu_geometry.vertex_buffer_ids[slot],
                        buffer.slice(..),
                    );
                }

                if let Some((index_buffer, index_format, count, id)) = &gpu_geometry.index_buffer {
                    tracked_pass.set_index_buffer(*id, index_buffer.slice(..), *index_format);
                    tracked_pass.draw_indexed(0..*count, 0, gpu_geometry.instance_range.clone());
                } else {
                    tracked_pass.draw(
                        gpu_geometry.draw_range.clone(),
                        gpu_geometry.instance_range.clone(),
                    );
                }
            }
        }
    }
}
