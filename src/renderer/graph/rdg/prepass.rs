//! RDG Depth-Normal Pre Pass
//!
//! Writes depth and optional view-space normals before the main opaque pass.
//! Enables Z-Prepass optimisation (zero overdraw) and provides a mini G-Buffer
//! for SSAO and other screen-space effects.
//!
//! # RDG Slots
//!
//! - `scene_depth`: Scene depth buffer (created, Clear(0.0))
//! - `scene_normals`: Optional normal buffer (created, Rgba8Unorm)
//!
//! # Push Parameters
//!
//! - `needs_normal`: Whether to output view-space normals (for SSAO)
//! - `needs_feature_id`: Whether to output feature IDs (for SSS/SSR)

use rustc_hash::FxHashMap;

use crate::renderer::graph::rdg::builder::PassBuilder;
use crate::renderer::graph::rdg::context::{RdgExecuteContext, RdgPrepareContext};
use crate::renderer::graph::rdg::node::PassNode;
use crate::renderer::graph::rdg::types::TextureNodeId;
use crate::renderer::graph::TrackedRenderPass;
use crate::renderer::graph::transient_pool::TransientTextureDesc;
use crate::renderer::pipeline::{
    ColorTargetKey, DepthStencilKey, RenderPipelineId, ShaderCompilationOptions,
    SimpleGeometryPipelineKey,
};
use crate::resources::material::{AlphaMode, Side};
use crate::resources::screen_space::STENCIL_WRITE_MASK;

/// Normal texture format — Rgba8Unorm ([-1,1] → [0,1] mapping).
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Feature ID texture format — Rg8Uint.
const FEATURE_ID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Uint;

/// RDG Depth-Normal Pre Pass.
///
/// Writes scene depth (reverse-Z, Clear(0.0)) and optional normals to RDG
/// transient textures. Subsequent passes (Opaque, SSAO) read the depth via
/// `LoadOp::Load`.
pub struct RdgPrepass {
    // ─── RDG Resource Slots (set by Composer) ──────────────────────
    pub scene_depth: TextureNodeId,
    pub scene_normals: TextureNodeId,

    // ─── Push Parameters ───────────────────────────────────────────
    pub needs_normal: bool,
    pub needs_feature_id: bool,

    // ─── Internal Cache ────────────────────────────────────────────
    /// Pipeline cache: (main_pipeline_id, needs_normal, needs_feature_id) → prepass pipeline.
    local_cache: FxHashMap<(RenderPipelineId, bool, bool), RenderPipelineId>,
}

impl RdgPrepass {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scene_depth: TextureNodeId(0),
            scene_normals: TextureNodeId(0),
            needs_normal: false,
            needs_feature_id: false,
            local_cache: FxHashMap::default(),
        }
    }

    /// Build prepass pipelines for every unique `pipeline_id` in the opaque
    /// command list.
    fn prepare_pipelines(&mut self, ctx: &mut RdgPrepareContext) {
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

        for cmd in ctx.render_lists.opaque.iter() {
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

            if material.alpha_mode() == AlphaMode::Blend {
                continue;
            }

            let Some(gpu_geometry) = ctx.resource_manager.get_geometry(cmd.geometry_handle) else {
                continue;
            };
            let Some(gpu_material) = ctx.resource_manager.get_material(cmd.material_handle) else {
                continue;
            };

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
                ctx.device,
                "passes/depth",
                &options,
                &gpu_geometry.layout_info.vertex_input_code,
                &binding_code,
            );

            // ── Pipeline layout ────────────────────────────────────────
            let layout = ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("RDG Prepass Pipeline Layout"),
                    bind_group_layouts: &[
                        &gpu_world_layout_clone,
                        &gpu_material.layout,
                        &cmd.object_bind_group.layout,
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

            // ── Color targets ──────────────────────────────────────────
            let color_targets: smallvec::SmallVec<[ColorTargetKey; 2]> = if self.needs_feature_id {
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
                sample_count: 1, // HighFidelity path always uses sample_count = 1
            };

            let pipeline_id = ctx.pipeline_cache.get_or_create_simple_geometry(
                ctx.device,
                shader_module,
                &layout,
                &prepass_key,
                "RDG Prepass Pipeline",
                &vertex_buffers_layout,
            );

            self.local_cache.insert(
                (cmd.pipeline_id, self.needs_normal, self.needs_feature_id),
                pipeline_id,
            );
        }
    }
}

impl PassNode for RdgPrepass {
    fn name(&self) -> &'static str {
        "RDG_Prepass"
    }

    fn setup(&mut self, builder: &mut PassBuilder) {
        builder.write_texture(self.scene_depth);
        if self.needs_normal {
            builder.write_texture(self.scene_normals);
        }
    }

    fn prepare(&mut self, ctx: &mut RdgPrepareContext) {
        // Allocate feature ID texture via old transient pool if needed
        if self.needs_feature_id {
            let size = ctx.wgpu_ctx.size();
            let feature_id_tex = ctx.transient_pool.allocate(
                ctx.device,
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

        self.prepare_pipelines(ctx);
    }

    fn execute(&self, ctx: &RdgExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let render_lists = &ctx.render_lists;
        let Some(gpu_global_bind_group) = &render_lists.gpu_global_bind_group else {
            return;
        };

        let depth_view = ctx.get_texture_view(self.scene_depth);

        // Build color attachments based on normal/feature_id needs
        let mut color_attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = Vec::new();

        if self.needs_normal {
            let normal_view = ctx.get_texture_view(self.scene_normals);
            color_attachments.push(Some(wgpu::RenderPassColorAttachment {
                view: normal_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.5,
                        g: 0.5,
                        b: 1.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            }));
        }

        if self.needs_feature_id {
            if let Some(feature_id) = ctx.blackboard.feature_id_texture_id {
                let feature_view = ctx.transient_pool.get_view(feature_id);
                color_attachments.push(Some(wgpu::RenderPassColorAttachment {
                    view: feature_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }));
            }
        }

        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RDG Depth-Normal Prepass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0), // Reverse-Z: 0.0 = far
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
            let Some(&prepass_pipeline_id) =
                self.local_cache
                    .get(&(cmd.pipeline_id, self.needs_normal, self.needs_feature_id))
            else {
                continue;
            };

            let pipeline = ctx.pipeline_cache.get_render_pipeline(prepass_pipeline_id);
            tracked_pass.set_pipeline(prepass_pipeline_id.0, pipeline);

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

            tracked_pass.set_bind_group(
                2,
                cmd.object_bind_group.bind_group_id,
                &cmd.object_bind_group.bind_group,
                &[cmd.dynamic_offset],
            );

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
