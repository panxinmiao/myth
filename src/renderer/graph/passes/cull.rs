//! Scene Cull Pass
//!
//! Performs unified view-based culling, render command generation, and sorting.
//! This Pass only executes the `prepare` phase, no actual drawing.
//!
//! # "Everything is a View" Architecture
//!
//! All rendering viewpoints (main camera, shadow cascades, etc.) are expressed
//! as `RenderView` instances. The cull pass tests every renderable against
//! every view's frustum, producing per-view command queues.
//!
//! # Data Flow
//! ```text
//! ExtractedScene.render_items (all active renderables)
//!     → [Main Camera View]  → RenderLists.opaque / .transparent
//!     → [Shadow View 1..N]  → RenderLists.shadow_queues
//! ```

use log::{error, warn};
use slotmap::Key;

use crate::RenderPath;
use crate::renderer::core::view::{RenderView, ViewTarget};
use crate::renderer::graph::RenderNode;
use crate::renderer::graph::context::{ExecuteContext, PrepareContext};
use crate::renderer::graph::extracted::SceneFeatures;
use crate::renderer::graph::frame::{RenderCommand, RenderKey, ShadowRenderCommand};
use crate::renderer::graph::shadow_utils;
use crate::renderer::pipeline::shader_gen::ShaderCompilationOptions;
use crate::renderer::pipeline::{
    FastPipelineKey, FastShadowPipelineKey, PipelineKey, ShadowPipelineKey,
};
use crate::resources::material::{AlphaMode, Side};
use crate::resources::uniforms::{DynamicModelUniforms, Mat3Uniform};
use crate::scene::light::LightKind;

/// Scene Cull Pass
///
/// Unified culling and command generation for all `RenderView`s.
///
/// # Performance Considerations
/// - L1/L2 Pipeline cache avoids repeated shader compilation
/// - Batch dynamic Uniform upload reduces GPU calls
/// - `sort_unstable_by` avoids extra allocation
/// - Pre-computed world bounding spheres in `ExtractedRenderItem` avoid geometry lookups during culling
pub struct SceneCullPass;

const SHADOW_BINDING_WGSL: &str = "
struct Struct_shadow_light {
    view_projection: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> u_shadow_light: Struct_shadow_light;
";

impl SceneCullPass {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Prepare main camera render commands with frustum culling and sorting.
    ///
    /// # Flow
    /// 1. Clear `render_lists`
    /// 2. Frustum-cull `extracted_scene.render_items` against main camera
    /// 3. For visible items: lookup/create Pipeline, generate `RenderCommand`
    /// 4. Classify into opaque/transparent lists
    /// 5. Sort command lists
    #[allow(clippy::too_many_lines)]
    fn prepare_and_sort_commands(ctx: &mut PrepareContext) {
        // Pre-fetch config to avoid borrow conflicts
        let color_format = ctx.get_scene_render_target_format();
        let depth_format = ctx.wgpu_ctx.depth_format;
        let sample_count = ctx.wgpu_ctx.msaa_samples;
        let render_state_id = ctx.render_state.id;
        let scene_id = ctx.extracted_scene.scene_id;
        let pipeline_settings_version = ctx.wgpu_ctx.pipeline_settings_version;
        let camera_frustum = ctx.camera.frustum;
        let camera_pos = ctx.camera.position;

        let render_lists = &mut *ctx.render_lists;
        render_lists.clear();
        ctx.blackboard.clear();

        let Some(gpu_world) = ctx
            .resource_manager
            .get_global_state(render_state_id, scene_id)
        else {
            error!(
                "Render Environment missing for render_state_id {render_state_id}, scene_id {scene_id}"
            );
            return;
        };

        render_lists.gpu_global_bind_group_id = gpu_world.bind_group_id;
        render_lists.gpu_global_bind_group = Some(gpu_world.bind_group.clone());

        // Store main camera view for downstream use
        render_lists.active_views.push(RenderView::new_main_camera(
            ctx.camera.view_projection_matrix,
            camera_frustum,
            ctx.wgpu_ctx.size(),
        ));

        let mut use_transmission = false;
        {
            let geo_guard = ctx.assets.geometries.read_lock();
            let mat_guard = ctx.assets.materials.read_lock();

            for item_idx in 0..ctx.extracted_scene.render_items.len() {
                let item = &ctx.extracted_scene.render_items[item_idx];

                // ========== Main Camera Frustum Culling ==========
                let aabb = item.world_aabb;
                if aabb.is_finite() && !camera_frustum.intersects_aabb(&aabb) {
                    continue;
                }

                let Some(geometry) = geo_guard.map.get(item.geometry) else {
                    warn!("Geometry {:?} missing during render prepare", item.geometry);
                    continue;
                };
                let Some(material) = mat_guard.map.get(item.material) else {
                    warn!("Material {:?} missing during render prepare", item.material);
                    continue;
                };

                let object_bind_group = &item.object_bind_group;

                let Some(gpu_geometry) = ctx.resource_manager.get_geometry(item.geometry) else {
                    error!("CRITICAL: GpuGeometry missing for {:?}", item.geometry);
                    continue;
                };
                let Some(gpu_material) = ctx.resource_manager.get_material(item.material) else {
                    error!("CRITICAL: GpuMaterial missing for {:?}", item.material);
                    continue;
                };

                // Build fast cache key using version numbers
                let fast_key = FastPipelineKey {
                    material_handle: item.material,
                    material_version: gpu_material.version,
                    geometry_handle: item.geometry,
                    geometry_version: geometry.layout_version(),
                    instance_variants: item.item_variant_flags,
                    global_state_id: gpu_world.id,
                    scene_variants: ctx.extracted_scene.scene_variants,
                    pipeline_settings_version,
                };

                // ========== Hot Path Optimization: Check L1 Cache First ==========
                let (pipeline, pipeline_id) =
                    if let Some(p) = ctx.pipeline_cache.get_pipeline_fast(fast_key) {
                        // L1 cache hit: Directly use cached Pipeline
                        p.clone()
                    } else {
                        // L1 cache miss: Need full shader_defines computation
                        let geo_defines = geometry.shader_defines();
                        let mat_defines = material.shader_defines();

                        let final_a2c_enable = match material.alpha_mode() {
                            AlphaMode::Mask(_, a2c) => a2c,
                            AlphaMode::Blend | AlphaMode::Opaque => false,
                        };

                        let mut options = ShaderCompilationOptions::from_merged(
                            &mat_defines,
                            geo_defines,
                            &ctx.extracted_scene.scene_defines,
                            &item.item_shader_defines,
                        );

                        if final_a2c_enable {
                            options.add_define("ALPHA_TO_COVERAGE", "1");
                        }

                        if ctx.wgpu_ctx.render_path.supports_post_processing() {
                            options.add_define("HDR", "1");
                        }

                        let shader_hash = options.compute_hash();

                        // Determine if this is an opaque item (not transparent
                        // and not using transmission) so that Z-prepass depth
                        // compare overrides are only applied to opaque pipelines.
                        let is_opaque_item = material.alpha_mode() != AlphaMode::Blend
                            && !material.use_transmission();

                        let is_specular_split = match ctx.wgpu_ctx.render_path {
                            RenderPath::HighFidelity => {
                                is_opaque_item
                                    && ctx
                                        .extracted_scene
                                        .scene_variants
                                        .contains(SceneFeatures::USE_SSS)
                            }
                            RenderPath::BasicForward { .. } => false,
                        };

                        let canonical_key = PipelineKey {
                            shader_hash,
                            vertex_layout_id: gpu_geometry.layout_id,
                            bind_group_layout_ids: [
                                gpu_world.layout_id,
                                gpu_material.layout_id,
                                object_bind_group.layout_id,
                                ctx.frame_resources.screen_bind_group_layout.id(),
                            ],
                            topology: geometry.topology,
                            cull_mode: match material.side() {
                                Side::Front => Some(wgpu::Face::Back),
                                Side::Back => Some(wgpu::Face::Front),
                                Side::Double => None,
                            },
                            // When the Z-Normal prepass has written depth for
                            // opaque objects, the main pass can use Equal compare
                            // with depth writes disabled — every fragment that
                            // survives is guaranteed visible, eliminating
                            // overdraw.  Transparent objects still need the
                            // normal depth path.
                            depth_write: if is_opaque_item
                                && ctx.wgpu_ctx.render_path.requires_z_prepass()
                            {
                                false
                            } else {
                                material.depth_write()
                            },
                            depth_compare: if is_opaque_item
                                && ctx.wgpu_ctx.render_path.requires_z_prepass()
                                && material.depth_test()
                            {
                                wgpu::CompareFunction::Equal
                            } else if material.depth_test() {
                                wgpu::CompareFunction::Greater
                            } else {
                                wgpu::CompareFunction::Always
                            },
                            blend_state: if material.alpha_mode() == AlphaMode::Blend {
                                Some(wgpu::BlendState::ALPHA_BLENDING)
                            } else {
                                None
                            },
                            color_format,
                            depth_format,
                            sample_count,
                            alpha_to_coverage: final_a2c_enable,
                            front_face: if item.item_variant_flags & 0x1 != 0 {
                                wgpu::FrontFace::Cw
                            } else {
                                wgpu::FrontFace::Ccw
                            },

                            is_specular_split,
                        };

                        let (pipeline, pipeline_id) = ctx.pipeline_cache.get_pipeline(
                            &ctx.wgpu_ctx.device,
                            material.shader_name(),
                            canonical_key,
                            &options,
                            &gpu_geometry.layout_info,
                            gpu_material,
                            object_bind_group,
                            gpu_world,
                            ctx.frame_resources,
                        );

                        ctx.pipeline_cache
                            .insert_pipeline_fast(fast_key, (pipeline.clone(), pipeline_id));
                        (pipeline, pipeline_id)
                    };

                let mat_id = item.material.data().as_ffi() as u32;

                let has_transmission = material.use_transmission();
                if has_transmission {
                    use_transmission = true;
                }

                let is_transparent = material.alpha_mode() == AlphaMode::Blend || has_transmission;

                // Compute distance to camera for sorting (deferred from Extract phase)
                let item_pos = glam::Vec3A::from(item.world_matrix.w_axis.truncate());
                let distance_sq = camera_pos.distance_squared(item_pos);
                let sort_key = RenderKey::new(pipeline_id, mat_id, distance_sq, is_transparent);

                let ss_feature_mask = material.ss_feature_mask();

                let cmd = RenderCommand {
                    object_bind_group: object_bind_group.clone(),
                    geometry_handle: item.geometry,
                    material_handle: item.material,
                    pipeline_id,
                    pipeline,
                    model_matrix: item.world_matrix,
                    sort_key,
                    dynamic_offset: 0,
                    ss_feature_mask,
                };

                if is_transparent {
                    ctx.render_lists.insert_transparent(cmd);
                } else {
                    ctx.render_lists.insert_opaque(cmd);
                }
            }
        }

        ctx.render_lists.use_transmission = use_transmission;
        ctx.render_lists.sort();
    }

    /// Generate shadow [`RenderView`]s and per-view culled shadow commands.
    ///
    /// # Flow
    /// 1. Compute `scene_caster_extent` from `render_items` (items with `cast_shadows`).
    /// 2. For each shadow-casting light, build `RenderView`(s) via `shadow_utils`.
    /// 3. For each view, frustum-cull `render_items` (filtered by `cast_shadows`) → per-view `ShadowRenderCommand`.
    /// 4. Append views to `render_lists.active_views`, commands to `render_lists.shadow_queues`.
    #[allow(clippy::too_many_lines)]
    fn prepare_shadow_commands(ctx: &mut PrepareContext) {
        let depth_format = wgpu::TextureFormat::Depth32Float;
        let pipeline_settings_version = ctx.wgpu_ctx.pipeline_settings_version;
        let shadow_layout_entries = [wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<glam::Mat4>() as u64),
            },
            count: None,
        }];
        let (shadow_global_layout, _) = ctx
            .resource_manager
            .get_or_create_layout(&shadow_layout_entries);

        // ================================================================
        // 1. Compute scene caster extent (for CSM Z extension)
        //    Uses pre-computed world_bounding_sphere from Extract phase.
        // ================================================================
        let scene_caster_extent = {
            let camera_pos: glam::Vec3 = ctx.camera.position.to_array().into();
            let mut max_distance = 0.0f32;

            for item in &ctx.extracted_scene.render_items {
                if !item.cast_shadows {
                    continue;
                }

                let aabb = item.world_aabb;
                // Items with infinite radius are unbounded; use position only
                let effective_radius = if aabb.is_finite() {
                    aabb.size().length() * 0.5
                } else {
                    0.0
                };
                let center_ws = aabb.center();
                let distance = camera_pos.distance(center_ws) + effective_radius;
                max_distance = max_distance.max(distance);
            }

            max_distance.max(50.0)
        };

        // ================================================================
        // 2. Build shadow RenderViews for all shadow-casting lights
        // ================================================================
        let mut shadow_views: Vec<RenderView> = Vec::with_capacity(16);

        for (light_buffer_index, light) in ctx.extracted_scene.lights.iter().enumerate() {
            if !light.cast_shadows {
                continue;
            }

            let shadow_cfg = light.shadow.clone().unwrap_or_default();

            match &light.kind {
                LightKind::Directional(_) => {
                    let cam_far = if ctx.camera.far.is_finite() {
                        ctx.camera.far
                    } else {
                        shadow_cfg.max_shadow_distance
                    };
                    let shadow_far = shadow_cfg.max_shadow_distance.min(cam_far);
                    let caster_extension = scene_caster_extent.max(shadow_cfg.max_shadow_distance);

                    // Compute base layer: count previously emitted views
                    let base_layer = shadow_views.len() as u32;

                    let (views, _splits) = shadow_utils::build_directional_views(
                        light.id,
                        light.direction,
                        light_buffer_index,
                        ctx.camera,
                        &shadow_cfg,
                        shadow_far,
                        caster_extension,
                        base_layer,
                    );
                    shadow_views.extend(views);
                }
                LightKind::Spot(spot) => {
                    let base_layer = shadow_views.len() as u32;
                    shadow_views.push(shadow_utils::build_spot_view(
                        light.id,
                        light_buffer_index,
                        light.position,
                        light.direction,
                        spot,
                        &shadow_cfg,
                        base_layer,
                    ));
                }
                LightKind::Point(_) => {
                    // Future: build 6 cubemap face views
                }
            }
        }

        // ================================================================
        // 3. Per-view frustum culling + command generation
        //    Uses pre-computed world_bounding_sphere for fast culling.
        // ================================================================
        let geo_guard = ctx.assets.geometries.read_lock();
        let mat_guard = ctx.assets.materials.read_lock();

        for view in &shadow_views {
            let ViewTarget::ShadowLight {
                light_id,
                layer_index,
            } = view.target
            else {
                continue;
            };

            let queue = ctx
                .render_lists
                .shadow_queues
                .entry((light_id, layer_index))
                .or_default();

            for item in &ctx.extracted_scene.render_items {
                // Attribute filter: only shadow casters
                if !item.cast_shadows {
                    continue;
                }

                // Per-view frustum culling using pre-computed bounding boxes
                let aabb = item.world_aabb;
                if aabb.is_finite() && !view.frustum.intersects_aabb(&aabb) {
                    continue;
                }

                let Some(geometry) = geo_guard.map.get(item.geometry) else {
                    continue;
                };

                let Some(material) = mat_guard.map.get(item.material) else {
                    continue;
                };

                if material.alpha_mode() == AlphaMode::Blend {
                    continue;
                }

                let Some(gpu_geometry) = ctx.resource_manager.get_geometry(item.geometry) else {
                    continue;
                };
                let Some(gpu_material) = ctx.resource_manager.get_material(item.material) else {
                    continue;
                };

                let fast_key = FastShadowPipelineKey {
                    material_handle: item.material,
                    material_version: gpu_material.version,
                    geometry_handle: item.geometry,
                    geometry_version: geometry.layout_version(),
                    instance_variants: item.item_variant_flags,
                    pipeline_settings_version,
                };

                let pipeline =
                    if let Some(p) = ctx.pipeline_cache.get_shadow_pipeline_fast(fast_key) {
                        p.clone()
                    } else {
                        let geo_defines = geometry.shader_defines();
                        let mat_defines = material.shader_defines();

                        let mut options = ShaderCompilationOptions::from_merged(
                            &mat_defines,
                            geo_defines,
                            &crate::resources::shader_defines::ShaderDefines::new(),
                            &item.item_shader_defines,
                        );

                        options.add_define("SHADOW_PASS", "1");

                        let shader_hash = options.compute_hash();
                        let canonical_key = ShadowPipelineKey {
                            shader_hash,
                            topology: geometry.topology,
                            cull_mode: match material.side() {
                                Side::Front => Some(wgpu::Face::Back),
                                Side::Back => Some(wgpu::Face::Front),
                                Side::Double => None,
                            },
                            depth_format,
                            front_face: if item.item_variant_flags & 0x1 != 0 {
                                wgpu::FrontFace::Cw
                            } else {
                                wgpu::FrontFace::Ccw
                            },
                        };

                        let pipeline = ctx.pipeline_cache.get_shadow_pipeline(
                            &ctx.wgpu_ctx.device,
                            canonical_key,
                            &options,
                            &gpu_geometry.layout_info,
                            &shadow_global_layout,
                            SHADOW_BINDING_WGSL,
                            gpu_material,
                            &item.object_bind_group,
                        );

                        ctx.pipeline_cache
                            .insert_shadow_pipeline_fast(fast_key, pipeline.clone());
                        pipeline
                    };

                let world_matrix_inverse = item.world_matrix.inverse();
                let normal_matrix = Mat3Uniform::from_mat4(world_matrix_inverse.transpose());
                let dynamic_offset =
                    ctx.resource_manager
                        .allocate_model_uniform(DynamicModelUniforms {
                            world_matrix: item.world_matrix,
                            world_matrix_inverse,
                            normal_matrix,
                            ..Default::default()
                        });

                queue.push(ShadowRenderCommand {
                    object_bind_group: item.object_bind_group.clone(),
                    geometry_handle: item.geometry,
                    material_handle: item.material,
                    pipeline,
                    dynamic_offset,
                });
            }
        }

        drop(geo_guard);
        drop(mat_guard);

        // ================================================================
        // 4. Append shadow views to active_views (main camera view is already there)
        // ================================================================
        ctx.render_lists.active_views.extend(shadow_views);
    }

    /// 上传动态 Uniform 数据
    ///
    /// 为每个渲染命令计算并上传模型矩阵、逆矩阵、法线矩阵等。
    fn upload_dynamic_uniforms(ctx: &mut PrepareContext) {
        let render_lists = &mut *ctx.render_lists;

        if render_lists.is_empty() {
            return;
        }

        // 处理不透明物体
        for cmd in &mut render_lists.opaque {
            let world_matrix_inverse = cmd.model_matrix.inverse();
            let normal_matrix = Mat3Uniform::from_mat4(world_matrix_inverse.transpose());

            let offset = ctx
                .resource_manager
                .allocate_model_uniform(DynamicModelUniforms {
                    world_matrix: cmd.model_matrix,
                    world_matrix_inverse,
                    normal_matrix,
                    ..Default::default()
                });

            cmd.dynamic_offset = offset;
        }

        // 处理透明物体
        for cmd in &mut render_lists.transparent {
            let world_matrix_inverse = cmd.model_matrix.inverse();
            let normal_matrix = Mat3Uniform::from_mat4(world_matrix_inverse.transpose());

            let offset = ctx
                .resource_manager
                .allocate_model_uniform(DynamicModelUniforms {
                    world_matrix: cmd.model_matrix,
                    world_matrix_inverse,
                    normal_matrix,
                    ..Default::default()
                });

            cmd.dynamic_offset = offset;
        }

        ctx.resource_manager.upload_model_buffer();
    }
}

impl Default for SceneCullPass {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderNode for SceneCullPass {
    fn name(&self) -> &'static str {
        "Scene Cull Pass"
    }

    fn prepare(&mut self, ctx: &mut PrepareContext) {
        // 1. 准备并排序渲染命令
        Self::prepare_and_sort_commands(ctx);

        // 2. 生成阴影渲染命令
        Self::prepare_shadow_commands(ctx);

        // 3. 上传动态 Uniform
        Self::upload_dynamic_uniforms(ctx);
    }

    fn run(&self, _ctx: &ExecuteContext, _encoder: &mut wgpu::CommandEncoder) {
        // SceneCullPass 不执行实际绘制
        // 所有工作在 prepare 阶段完成
    }
}
