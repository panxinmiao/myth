//! Render Command Baking
//!
//! Converts handle-based [`RenderCommand`] / [`ShadowRenderCommand`] lists
//! into physical [`DrawCommand`] lists by resolving every asset handle to its
//! underlying `wgpu` resource reference.
//!
//! This baking step bridges the **cull** phase (which works with lightweight
//! handles) and the **execute** phase (which should be "blind" — touching only
//! pre-resolved GPU state).
//!
//! # Call Site
//!
//! [`bake_render_lists`] is invoked once per frame inside
//! [`FrameComposer::render()`](super::composer::FrameComposer::render) after
//! the RDG prepare phase and before the execute loop.

use rustc_hash::FxHashMap;

use crate::renderer::core::ResourceManager;
use crate::renderer::pipeline::{PipelineCache, RenderPipelineId};

use super::frame::{
    BakedRenderLists, DrawCommand, RenderCommand, RenderLists, ShadowRenderCommand,
};

// ============================================================================
// Public API
// ============================================================================

/// Configuration for baking prepass draw commands.
///
/// Passed as `Some(...)` in the HighFidelity render path and `None` when
/// the prepass is disabled (BasicForward).
pub struct PrepassBakeConfig<'a> {
    /// Pipeline remapping table: `(main_pipeline, needs_normal, needs_fid, taa) → prepass_pipeline`.
    pub local_cache: &'a FxHashMap<(RenderPipelineId, bool, bool, bool), RenderPipelineId>,
    /// Whether the prepass outputs view-space normals.
    pub needs_normal: bool,
    /// Whether the prepass outputs a feature-ID stencil mask.
    pub needs_feature_id: bool,
    /// Whether the prepass outputs velocity vectors for TAA.
    pub taa_enabled: bool,
}

/// Bake all render lists into pre-resolved [`DrawCommand`] lists.
///
/// Every handle lookup (`get_geometry`, `get_material`,
/// `get_render_pipeline`) is performed exactly once here; the resulting
/// [`BakedRenderLists`] can be consumed by the execute phase without any
/// further indirection.
pub fn bake_render_lists<'a>(
    render_lists: &'a RenderLists,
    resource_manager: &'a ResourceManager,
    pipeline_cache: &'a PipelineCache,
    prepass_config: &Option<PrepassBakeConfig<'_>>,
) -> BakedRenderLists<'a> {
    let opaque = bake_main_commands(&render_lists.opaque, resource_manager, pipeline_cache);
    let transparent =
        bake_main_commands(&render_lists.transparent, resource_manager, pipeline_cache);

    let prepass = if let Some(cfg) = prepass_config {
        bake_prepass_commands(&render_lists.opaque, resource_manager, pipeline_cache, cfg)
    } else {
        Vec::new()
    };

    let shadow_queues = bake_shadow_queues(
        &render_lists.shadow_queues,
        resource_manager,
        pipeline_cache,
    );

    let global_bind_group = &render_lists
        .gpu_global_bind_group
        .as_ref()
        .expect("Global bind group not built!");

    BakedRenderLists {
        opaque,
        transparent,
        prepass,
        shadow_queues,
        global_bind_group,
    }
}

// ============================================================================
// Internals
// ============================================================================

/// Bake a slice of main-camera [`RenderCommand`]s into [`DrawCommand`]s.
///
/// The output preserves the input order (already sorted by
/// [`RenderLists::sort`]).
fn bake_main_commands<'a>(
    commands: &'a [RenderCommand],
    rm: &'a ResourceManager,
    pc: &'a PipelineCache,
) -> Vec<DrawCommand<'a>> {
    let mut out = Vec::with_capacity(commands.len());

    for cmd in commands {
        let pipeline = pc.get_render_pipeline(cmd.pipeline_id);

        let Some(gpu_mat) = rm.get_material(cmd.material_handle) else {
            continue;
        };
        let Some(gpu_geom) = rm.get_geometry(cmd.geometry_handle) else {
            continue;
        };

        let vertex_buffers: Vec<&wgpu::Buffer> = gpu_geom.vertex_buffers.iter().collect();

        let index_buffer = gpu_geom
            .index_buffer
            .as_ref()
            .map(|(buf, fmt, count, _id)| (buf as &wgpu::Buffer, *fmt, *count));

        out.push(DrawCommand {
            sort_key: cmd.sort_key.bits(),
            pipeline,
            vertex_buffers,
            index_buffer,
            bind_group_1: Some(&gpu_mat.bind_group),
            bind_group_2: (&cmd.object_bind_group.bind_group, cmd.dynamic_offset),
            bind_group_3: None, // Set at the pass level
            stencil_reference: None,
            vertex_range: gpu_geom.draw_range.clone(),
            instance_range: gpu_geom.instance_range.clone(),
        });
    }

    out
}

/// Bake prepass draw commands from the opaque list with pipeline remapping.
///
/// For each opaque command that has a corresponding prepass pipeline in
/// `cfg.local_cache`, a [`DrawCommand`] is emitted using the prepass
/// pipeline.  The `stencil_reference` field is populated when feature-ID
/// writing is enabled.
fn bake_prepass_commands<'a>(
    opaque_commands: &'a [RenderCommand],
    rm: &'a ResourceManager,
    pc: &'a PipelineCache,
    cfg: &PrepassBakeConfig<'_>,
) -> Vec<DrawCommand<'a>> {
    let mut out = Vec::with_capacity(opaque_commands.len());

    for cmd in opaque_commands {
        let Some(&prepass_pipeline_id) = cfg.local_cache.get(&(
            cmd.pipeline_id,
            cfg.needs_normal,
            cfg.needs_feature_id,
            cfg.taa_enabled,
        )) else {
            continue;
        };

        let pipeline = pc.get_render_pipeline(prepass_pipeline_id);

        let Some(gpu_mat) = rm.get_material(cmd.material_handle) else {
            continue;
        };
        let Some(gpu_geom) = rm.get_geometry(cmd.geometry_handle) else {
            continue;
        };

        let vertex_buffers: Vec<&wgpu::Buffer> = gpu_geom.vertex_buffers.iter().collect();

        let index_buffer = gpu_geom
            .index_buffer
            .as_ref()
            .map(|(buf, fmt, count, _id)| (buf as &wgpu::Buffer, *fmt, *count));

        out.push(DrawCommand {
            sort_key: cmd.sort_key.bits(),
            pipeline,
            vertex_buffers,
            index_buffer,
            bind_group_1: Some(&gpu_mat.bind_group),
            bind_group_2: (&cmd.object_bind_group.bind_group, cmd.dynamic_offset),
            bind_group_3: None,
            stencil_reference: None,
            vertex_range: gpu_geom.draw_range.clone(),
            instance_range: gpu_geom.instance_range.clone(),
        });
    }

    out
}

/// Bake all per-view shadow command queues and sort them by pipeline ID.
fn bake_shadow_queues<'a>(
    shadow_queues: &'a FxHashMap<(u64, u32), Vec<ShadowRenderCommand>>,
    rm: &'a ResourceManager,
    pc: &'a PipelineCache,
) -> FxHashMap<(u64, u32), Vec<DrawCommand<'a>>> {
    let mut out = FxHashMap::default();

    for (&key, commands) in shadow_queues {
        let mut baked = Vec::with_capacity(commands.len());

        for cmd in commands {
            let pipeline = pc.get_render_pipeline(cmd.pipeline_id);

            let Some(gpu_mat) = rm.get_material(cmd.material_handle) else {
                continue;
            };
            let Some(gpu_geom) = rm.get_geometry(cmd.geometry_handle) else {
                continue;
            };

            let vertex_buffers: Vec<&wgpu::Buffer> = gpu_geom.vertex_buffers.iter().collect();

            let index_buffer = gpu_geom
                .index_buffer
                .as_ref()
                .map(|(buf, fmt, count, _id)| (buf as &wgpu::Buffer, *fmt, *count));

            // Shadow sort key: pipeline ID in high bits, material bind group
            // in low bits — minimises both pipeline and bind-group switches.
            let sort_key =
                u64::from(cmd.pipeline_id.0) << 32 | (gpu_mat.bind_group_id & 0xFFFF_FFFF);

            baked.push(DrawCommand {
                sort_key,
                pipeline,
                vertex_buffers,
                index_buffer,
                bind_group_1: Some(&gpu_mat.bind_group),
                bind_group_2: (&cmd.object_bind_group.bind_group, cmd.dynamic_offset),
                bind_group_3: None,
                stencil_reference: None,
                vertex_range: gpu_geom.draw_range.clone(),
                instance_range: gpu_geom.instance_range.clone(),
            });
        }

        // Sort shadow commands to minimise state switches.
        baked.sort_unstable_by_key(|c| c.sort_key);

        out.insert(key, baked);
    }

    out
}
