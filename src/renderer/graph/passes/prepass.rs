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
use crate::renderer::graph::{RenderNode, TrackedRenderPass, TransientTextureDesc};
use crate::renderer::pipeline::{
    ColorTargetKey, DepthStencilKey, SimpleGeometryPipelineKey, RenderPipelineId,
    ShaderCompilationOptions,
};
use crate::resources::material::{AlphaMode, Side};
use crate::resources::screen_space::STENCIL_WRITE_MASK;

/// Normal texture format — Rgba8Unorm ([-1,1] → [0,1] mapping).
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

const FEATURE_ID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Uint;

pub struct DepthNormalPrepass {
    pub(crate) needs_normal: bool, // Whether to output normals (for SSAO)
    pub(crate) needs_feature_id: bool, // Whether to output feature ID (SSS, SSR)

    depth_view: Option<Tracked<wgpu::TextureView>>,

    /// Pipeline cache keyed by the main-pass `pipeline_id`, `needs_normal` and `needs_feature_id`.
    ///
    /// Two objects sharing a main pipeline will share a prepass pipeline
    /// (identical vertex layout, topology, cull mode and material macros).
    local_cache: FxHashMap<(RenderPipelineId, bool, bool), RenderPipelineId>,
}

impl DepthNormalPrepass {
    #[must_use]
    pub fn new() -> Self {
        Self {
            needs_normal: false,
            needs_feature_id: false,
            depth_view: None,
            local_cache: FxHashMap::default(),
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
        let gpu_world_binding_wgsl = gpu_world.binding_wgsl.clone();
        let gpu_world_layout_clone = gpu_world.layout.clone();

        let geo_guard = ctx.assets.geometries.read_lock();
        let mat_guard = ctx.assets.materials.read_lock();

        for cmd in &ctx.render_lists.opaque {
            // Skip if already cached
            if self.local_cache.contains_key(&(
                cmd.pipeline_id,
                self.needs_normal,
                self.needs_feature_id,
            )) {
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
                &ctx.extracted_scene.scene_defines,
                item_def,
            );

            // Inject prepass-specific macros
            options.add_define("IS_PREPASS", "1");

            if self.needs_normal {
                options.add_define("OUTPUT_NORMAL", "1");
            }

            // ── Shader generation ──────────────────────────────────────
            let binding_code = format!(
                "{}\n{}\n{}",
                &gpu_world_binding_wgsl,
                &gpu_material.binding_wgsl,
                &cmd.object_bind_group.binding_wgsl
            );

            let (shader_module, shader_hash) = ctx.shader_manager.get_or_compile_template(
                &ctx.wgpu_ctx.device,
                "passes/depth",
                &options,
                &gpu_geometry.layout_info.vertex_input_code,
                &binding_code,
            );

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

            let stencil_state = if self.needs_feature_id {
                Some(wgpu::StencilFaceState {
                    compare: wgpu::CompareFunction::Always,
                    fail_op: wgpu::StencilOperation::Keep,
                    depth_fail_op: wgpu::StencilOperation::Keep,
                    pass_op: wgpu::StencilOperation::Replace,
                })
            } else {
                None
            };

            // ── Build color targets ────────────────────────────────────
            let color_targets: smallvec::SmallVec<[ColorTargetKey; 2]> =
                if self.needs_feature_id {
                    smallvec::smallvec![
                        ColorTargetKey::from(wgpu::ColorTargetState {
                            format: NORMAL_FORMAT,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        ColorTargetKey::from(wgpu::ColorTargetState {
                            format: FEATURE_ID_FORMAT,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ]
                } else if self.needs_normal {
                    smallvec::smallvec![ColorTargetKey::from(wgpu::ColorTargetState {
                        format: NORMAL_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })]
                } else {
                    smallvec::smallvec![]
                };

            // ── Pipeline key ───────────────────────────────────────────
            let prepass_key = SimpleGeometryPipelineKey {
                shader_hash,
                vertex_layout_id: gpu_geometry.layout_id,
                color_targets,
                depth_stencil: DepthStencilKey::from(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Greater,
                    stencil: wgpu::StencilState {
                        front: stencil_state.unwrap_or_default(),
                        back: stencil_state.unwrap_or_default(),
                        read_mask: 0xFF,
                        write_mask: STENCIL_WRITE_MASK,
                    },
                    bias: wgpu::DepthBiasState::default(),
                }),
                topology: geometry.topology,
                cull_mode,
                front_face,
            };

            let pipeline_id = ctx.pipeline_cache.get_or_create_simple_geometry(
                &ctx.wgpu_ctx.device,
                shader_module,
                &layout,
                &prepass_key,
                "Prepass Pipeline",
                &vertex_buffers_layout,
            );

            self.local_cache.insert(
                (cmd.pipeline_id, self.needs_normal, self.needs_feature_id),
                pipeline_id,
            );
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

        self.depth_view = Some(ctx.get_resource_view(GraphResource::DepthStencil).clone());

        if self.needs_normal {
            let size = ctx.wgpu_ctx.size();
            let normal_tex = ctx.transient_pool.allocate(
                &ctx.wgpu_ctx.device,
                &TransientTextureDesc {
                    width: size.0,
                    height: size.1,
                    format: NORMAL_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    mip_level_count: 1,
                    label: "Transient Scene Normal",
                },
            );
            ctx.blackboard.scene_normal_texture_id = Some(normal_tex);
        }

        if self.needs_feature_id {
            let size = ctx.wgpu_ctx.size();
            let feature_id_tex = ctx.transient_pool.allocate(
                &ctx.wgpu_ctx.device,
                &TransientTextureDesc {
                    width: size.0,
                    height: size.1,
                    format: FEATURE_ID_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    mip_level_count: 1,
                    label: "Transient Feature ID",
                },
            );
            ctx.blackboard.feature_id_texture_id = Some(feature_id_tex);
        }

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

        let mut color_attachments = if self.needs_normal {
            let normal_view = ctx.transient_pool.get_view(
                ctx.blackboard
                    .scene_normal_texture_id
                    .expect("Normal texture allocated"),
            );

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

        if self.needs_feature_id {
            let feature_view = ctx.transient_pool.get_view(
                ctx.blackboard
                    .feature_id_texture_id
                    .expect("Feature ID texture allocated"),
            );

            color_attachments.push(Some(wgpu::RenderPassColorAttachment {
                view: feature_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Clear to 0 (ID = 0 means no effect)
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            }));
        }

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
                stencil_ops: if self.needs_feature_id {
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

        tracked_pass.set_bind_group(
            0,
            render_lists.gpu_global_bind_group_id,
            gpu_global_bind_group,
            &[],
        );

        for cmd in &render_lists.opaque {
            let Some(&prepass_pipeline_id) = self.local_cache.get(&(
                cmd.pipeline_id,
                self.needs_normal,
                self.needs_feature_id,
            )) else {
                continue;
            };

            let pipeline = ctx.pipeline_cache.get_render_pipeline(prepass_pipeline_id);
            tracked_pass.set_pipeline(prepass_pipeline_id.0, pipeline);

            // Set 1: Material (for alpha-tested materials to read the base map)
            if let Some(gpu_material) = ctx.resource_manager.get_material(cmd.material_handle) {
                tracked_pass.set_bind_group(
                    1,
                    gpu_material.bind_group_id,
                    &gpu_material.bind_group,
                    &[],
                );
            }

            if self.needs_feature_id {
                tracked_pass
                    .raw_pass()
                    .set_stencil_reference(cmd.ss_feature_mask);
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
