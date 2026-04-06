//! Prepass Feature + Ephemeral PassNode
//!
//! - **`PrepassFeature`** (long-lived): owns the pipeline cache.
//!   `extract_and_prepare()` compiles prepass pipelines for the current
//!   opaque draw list.
//! - **`PrepassPassNode`** (ephemeral per-frame): carries cloned pipeline
//!   cache and lightweight resource IDs.  Created by
//!   `PrepassFeature::add_to_graph()`.
//!
//! # RDG Slots (explicit wiring)
//!
//! - `scene_depth`: Scene depth buffer (created & returned by `add_to_graph`)
//! - `scene_normals`: Optional normal buffer (created & returned by `add_to_graph`)
//! - `feature_id`: Optional feature-ID buffer (created & returned by `add_to_graph`)
//! - `velocity_buffer`: Optional velocity buffer for TAA (created & returned by `add_to_graph`)

use rustc_hash::FxHashMap;

use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    ExecuteContext, ExtractContext, PassNode, RenderTargetOps, TextureDesc, TextureNodeId,
};
use crate::graph::passes::draw::submit_draw_commands;
use crate::pipeline::{
    ColorTargetKey, DepthStencilKey, RenderPipelineId, ShaderCompilationOptions,
    ShaderSource, SimpleGeometryPipelineKey,
};
use myth_resources::material::Side;

/// Normal texture format — Rgba8Unorm ([-1,1] → [0,1] mapping).
pub(crate) const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Feature ID texture format — Rg8Uint.
pub(crate) const FEATURE_ID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Uint;

/// Velocity buffer format — Rg16Float (2-component half-precision motion vectors).
pub(crate) const VELOCITY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

/// Outputs produced by the Prepass, returned to the Composer for
/// explicit downstream wiring.
#[must_use = "SSA Graph: You must use the outputs of prepass to wire downstream passes!"]
pub struct PrepassOutputs {
    /// Single-sample scene depth (written by the prepass).
    pub scene_depth: TextureNodeId,
    /// View-space normals (if screen-space effects require them).
    pub scene_normals: Option<TextureNodeId>,
    /// Feature-ID colour attachment (if SSS/SSR require it).
    pub feature_id: Option<TextureNodeId>,
    /// Screen-space velocity buffer for TAA reprojection (`None` when
    /// TAA is not active).  Format: `Rg16Float`.
    pub velocity_buffer: Option<TextureNodeId>,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Feature (long-lived, stored in RenderFeatures)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Long-lived prepass feature.
///
/// Compiles depth/normal prepass pipelines during `extract_and_prepare()` and
/// stores them in `local_cache`.  The ephemeral [`PrepassPassNode`] receives
/// a clone of the cache via [`add_to_graph()`](Self::add_to_graph).
pub struct PrepassFeature {
    // ─── Push Parameters (set before extract_and_prepare) ──────────
    needs_normal: bool,
    needs_feature_id: bool,
    needs_velocity: bool,

    // ─── Internal Cache ────────────────────────────────────────────
    /// Pipeline cache: (main_pipeline_id, needs_normal, needs_feature_id, needs_velocity) → prepass pipeline.
    local_cache: FxHashMap<(RenderPipelineId, bool, bool, bool), RenderPipelineId>,
}

impl Default for PrepassFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl PrepassFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            needs_normal: false,
            needs_feature_id: false,
            needs_velocity: false,
            local_cache: FxHashMap::default(),
        }
    }

    /// Returns the prepass pipeline cache (for baking prepass draw commands).
    #[inline]
    #[must_use]
    pub fn local_cache(
        &self,
    ) -> &FxHashMap<(RenderPipelineId, bool, bool, bool), RenderPipelineId> {
        &self.local_cache
    }

    /// Whether the prepass outputs view-space normals.
    #[inline]
    #[must_use]
    pub fn needs_normal(&self) -> bool {
        self.needs_normal
    }

    /// Whether the prepass outputs a feature-ID colour target.
    #[inline]
    #[must_use]
    pub fn needs_feature_id(&self) -> bool {
        self.needs_feature_id
    }

    /// Whether the prepass outputs velocity vectors for TAA.
    #[inline]
    #[must_use]
    pub fn needs_velocity(&self) -> bool {
        self.needs_velocity
    }

    /// Pre-RDG resource preparation: compile prepass pipelines for every
    /// unique `pipeline_id` in the opaque command list.
    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        needs_normal: bool,
        needs_feature_id: bool,
        needs_velocity: bool,
    ) {
        self.needs_normal = needs_normal;
        self.needs_feature_id = needs_feature_id;
        self.needs_velocity = needs_velocity;
        self.prepare_pipelines(ctx);
    }

    /// Build prepass pipelines for every unique `pipeline_id` in the opaque
    /// command list.
    fn prepare_pipelines(&mut self, ctx: &mut ExtractContext) {
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
            if self.local_cache.contains_key(&(
                cmd.pipeline_id,
                self.needs_normal,
                self.needs_feature_id,
                self.needs_velocity,
            )) {
                continue;
            }

            let Some(geometry) = geo_guard.get_loaded(cmd.geometry_handle) else {
                continue;
            };
            let Some(material) = mat_guard.get_loaded(cmd.material_handle) else {
                continue;
            };

            if material.is_transparent() {
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

            let empty_defines = myth_resources::shader_defines::ShaderDefines::new();
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
            if self.needs_velocity {
                options.add_define("HAS_VELOCITY_TARGET", "1");
            }
            if self.needs_feature_id {
                options.add_define("OUTPUT_FEATURE_ID", "1");
            }

            // ── Shader generation ──────────────────────────────────────
            let binding_code = format!(
                "{}\n{}\n{}",
                &gpu_world_binding_wgsl,
                &gpu_material.binding_wgsl,
                &cmd.object_bind_group.binding_wgsl
            );

            options.inject_code("vertex_input_code", &gpu_geometry.layout_info.vertex_input_code);
            options.inject_code("binding_code", binding_code);

            let (shader_module, shader_hash) = ctx.shader_manager.get_or_compile(
                ctx.device,
                ShaderSource::File("passes/depth"),
                &options,
            );

            // ── Pipeline layout ────────────────────────────────────────
            let layout = ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Prepass Pipeline Layout"),
                    bind_group_layouts: &[
                        Some(&gpu_world_layout_clone),
                        Some(&gpu_material.layout),
                        Some(&cmd.object_bind_group.layout),
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

            // ── Color targets ──────────────────────────────────────────
            let mut color_targets: smallvec::SmallVec<[ColorTargetKey; 3]> = smallvec::smallvec![];

            if self.needs_normal {
                color_targets.push(ColorTargetKey::from(wgpu::ColorTargetState {
                    format: NORMAL_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }));
            }
            if self.needs_feature_id {
                color_targets.push(ColorTargetKey::from(wgpu::ColorTargetState {
                    format: FEATURE_ID_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }));
            }
            if self.needs_velocity {
                color_targets.push(ColorTargetKey::from(wgpu::ColorTargetState {
                    format: VELOCITY_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }));
            }

            let prepass_key = SimpleGeometryPipelineKey {
                shader_hash,
                vertex_layout_id: gpu_geometry.layout_id,
                color_targets,
                depth_stencil: DepthStencilKey::from(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Greater),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                topology: geometry.topology,
                cull_mode,
                front_face,
                sample_count: 1,
            };

            let pipeline_id = ctx.pipeline_cache.get_or_create_simple_geometry(
                ctx.device,
                shader_module,
                &layout,
                &prepass_key,
                "Prepass Pipeline",
                &vertex_buffers_layout,
            );

            self.local_cache.insert(
                (
                    cmd.pipeline_id,
                    self.needs_normal,
                    self.needs_feature_id,
                    self.needs_velocity,
                ),
                pipeline_id,
            );
        }
    }

    /// Build the ephemeral pass node, register resources, and insert it
    /// into the graph.
    ///
    /// All shared resources (`Scene_Depth`, `Scene_Normals`, `Feature_ID`)
    /// are created here so the Composer can wire them to downstream passes
    /// via explicit [`TextureNodeId`] connections.
    pub fn add_to_graph(
        &self,
        ctx: &mut GraphBuilderContext<'_, '_>,
        needs_normal: bool,
        needs_feature_id: bool,
        needs_velocity: bool,
    ) -> PrepassOutputs {
        let fc = ctx.frame_config;

        // Single-sample scene depth (always created).
        // COPY_SRC is required so TAA can archive scene depth to history.
        let depth_desc = TextureDesc::new(
            fc.width,
            fc.height,
            1,
            1,
            1,
            wgpu::TextureDimension::D2,
            fc.depth_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );
        let scene_depth = ctx
            .graph
            .register_resource("Scene_Depth", depth_desc, false);

        let scene_normals = if needs_normal {
            let desc = TextureDesc::new_2d(
                fc.width,
                fc.height,
                NORMAL_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );
            Some(ctx.graph.register_resource("Scene_Normals", desc, false))
        } else {
            None
        };

        let feature_id = if needs_feature_id {
            let desc = TextureDesc::new_2d(
                fc.width,
                fc.height,
                FEATURE_ID_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );
            Some(ctx.graph.register_resource("Feature_ID", desc, false))
        } else {
            None
        };

        let velocity_buffer = if needs_velocity {
            let desc = TextureDesc::new_2d(
                fc.width,
                fc.height,
                VELOCITY_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );
            Some(ctx.graph.register_resource("Velocity_Buffer", desc, false))
        } else {
            None
        };

        let node = PrepassPassNode {
            scene_depth,
            scene_normals: scene_normals.unwrap_or(TextureNodeId(0)),
            feature_id: feature_id.unwrap_or(TextureNodeId(0)),
            velocity_buffer: velocity_buffer.unwrap_or(TextureNodeId(0)),
            needs_normal,
            needs_feature_id,
            needs_velocity,
        };
        ctx.graph.add_pass("Pre_Pass", |builder| {
            builder.write_texture(scene_depth);
            if let Some(n) = scene_normals {
                builder.write_texture(n);
            }
            if let Some(f) = feature_id {
                builder.write_texture(f);
            }
            if let Some(v) = velocity_buffer {
                builder.write_texture(v);
            }
            (node, ())
        });

        PrepassOutputs {
            scene_depth,
            scene_normals,
            feature_id,
            velocity_buffer,
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PassNode (ephemeral per-frame)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Ephemeral per-frame prepass node.
///
/// Carries lightweight RDG resource IDs.  Pipeline remapping and draw
/// command baking are now handled by [`bake::bake_render_lists`]; this
/// node simply submits the pre-baked prepass commands.
pub struct PrepassPassNode {
    // ─── RDG Resource Slots (filled in setup) ──────────────────────
    scene_depth: TextureNodeId,
    scene_normals: TextureNodeId,
    feature_id: TextureNodeId,
    velocity_buffer: TextureNodeId,

    // ─── Push Parameters ───────────────────────────────────────────
    needs_normal: bool,
    needs_feature_id: bool,
    needs_velocity: bool,
}

impl PassNode<'_> for PrepassPassNode {
    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let gpu_global_bind_group = ctx.baked_lists.global_bind_group;

        let mut color_attachments: smallvec::SmallVec<
            [Option<wgpu::RenderPassColorAttachment>; 3],
        > = smallvec::SmallVec::with_capacity(3);

        let normal_clear = wgpu::Color {
            r: 0.5,
            g: 0.5,
            b: 1.0,
            a: 0.0,
        };
        if self.needs_normal
            && let Some(att) = ctx.get_color_attachment(
                self.scene_normals,
                RenderTargetOps::Clear(normal_clear),
                None,
            )
        {
            color_attachments.push(Some(att));
        }
        if self.needs_feature_id
            && let Some(att) = ctx.get_color_attachment(
                self.feature_id,
                RenderTargetOps::Clear(wgpu::Color::TRANSPARENT),
                None,
            )
        {
            color_attachments.push(Some(att));
        }
        if self.needs_velocity
            && let Some(att) = ctx.get_color_attachment(
                self.velocity_buffer,
                RenderTargetOps::Clear(wgpu::Color::TRANSPARENT),
                None,
            )
        {
            color_attachments.push(Some(att));
        }

        let dtt = ctx.get_depth_stencil_attachment(self.scene_depth, 0.0);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Depth-Normal-Velocity Prepass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: dtt,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_bind_group(0, gpu_global_bind_group, &[]);

        submit_draw_commands(&mut pass, &ctx.baked_lists.prepass);
    }
}
