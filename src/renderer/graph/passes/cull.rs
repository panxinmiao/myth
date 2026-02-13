//! Scene Cull Pass
//!
//! 负责场景剔除、渲染命令生成与排序。
//! 此 Pass 仅执行 `prepare` 阶段，不进行实际绘制。
//!
//! # 职责
//! - 遍历 `ExtractedScene` 中的可见物体
//! - 生成 `RenderCommand` 并分类到 opaque/transparent 列表
//! - 上传动态 Uniform（模型矩阵、法线矩阵等）
//! - 对命令列表进行排序
//!
//! # 数据流
//! ```text
//! ExtractedScene → SceneCullPass → RenderLists
//! ```

use log::{error, warn};
use slotmap::Key;

use crate::renderer::graph::frame::{RenderCommand, RenderKey, ShadowRenderCommand};
use crate::renderer::graph::{RenderContext, RenderNode};
use crate::renderer::pipeline::shader_gen::ShaderCompilationOptions;
use crate::renderer::pipeline::{
    FastPipelineKey, FastShadowPipelineKey, PipelineKey, ShadowPipelineKey,
};
use crate::resources::material::{AlphaMode, Side};
use crate::resources::uniforms::{DynamicModelUniforms, Mat3Uniform};
use crate::scene::light::LightKind;

/// 场景剔除 Pass
///
/// 仅执行 prepare 阶段，将渲染命令写入 `RenderFrame.render_lists`。
///
/// # 性能考虑
/// - 利用 L1/L2 Pipeline 缓存避免重复编译
/// - 批量上传动态 Uniform 减少 GPU 调用
/// - 排序使用 `sort_unstable_by` 避免额外分配
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

    /// 准备渲染命令并排序
    ///
    /// # 流程
    /// 1. 清空 `render_lists`
    /// 2. 遍历 `extracted_scene.render_items`
    /// 3. 查找/创建 Pipeline
    /// 4. 生成 `RenderCommand` 并分类
    /// 5. 排序命令列表
    #[allow(clippy::too_many_lines)]
    fn prepare_and_sort_commands(ctx: &mut RenderContext) {
        // 预先获取需要的配置（避免后续借用冲突）
        let color_format = ctx.get_scene_render_target_format();
        let depth_format = ctx.wgpu_ctx.depth_format;
        let sample_count = ctx.wgpu_ctx.msaa_samples;
        let render_state_id = ctx.render_state.id;
        let scene_id = ctx.extracted_scene.scene_id;
        let pipeline_settings_version = ctx.wgpu_ctx.pipeline_settings_version;

        // 获取 render_lists 的可变引用
        let render_lists = &mut *ctx.render_lists;
        render_lists.clear();

        // 获取全局状态
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

        let mut use_transmission = false;
        {
            let geo_guard = ctx.assets.geometries.read_lock();
            let mat_guard = ctx.assets.materials.read_lock();

            for item_idx in 0..ctx.extracted_scene.render_items.len() {
                let item = &ctx.extracted_scene.render_items[item_idx];

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
                            &geo_defines,
                            &ctx.extracted_scene.scene_defines,
                            &item.item_shader_defines,
                        );

                        if final_a2c_enable {
                            options.add_define("ALPHA_TO_COVERAGE", "1");
                        }

                        if ctx.wgpu_ctx.enable_hdr {
                            options.add_define("HDR", "1");
                        }

                        let shader_hash = options.compute_hash();

                        let canonical_key = PipelineKey {
                            shader_hash,
                            topology: geometry.topology,
                            cull_mode: match material.side() {
                                Side::Front => Some(wgpu::Face::Back),
                                Side::Back => Some(wgpu::Face::Front),
                                Side::Double => None,
                            },
                            depth_write: material.depth_write(),
                            depth_compare: if material.depth_test() {
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
                let sort_key =
                    RenderKey::new(pipeline_id, mat_id, item.distance_sq, is_transparent);

                let cmd = RenderCommand {
                    object_bind_group: object_bind_group.clone(),
                    geometry_handle: item.geometry,
                    material_handle: item.material,
                    pipeline_id,
                    pipeline,
                    model_matrix: item.world_matrix,
                    sort_key,
                    dynamic_offset: 0,
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

    #[allow(clippy::too_many_lines)]
    fn prepare_shadow_commands(ctx: &mut RenderContext) {
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

        let geo_guard = ctx.assets.geometries.read_lock();
        let mat_guard = ctx.assets.materials.read_lock();

        for light in &ctx.extracted_scene.lights {
            if !light.cast_shadows {
                continue;
            }

            if !matches!(light.kind, LightKind::Directional(_) | LightKind::Spot(_)) {
                continue;
            }

            let queue = ctx.render_lists.shadow_queues.entry(light.id).or_default();

            // Use shadow_caster_items: these include ALL cast_shadows=true objects,
            // even those outside the main camera frustum.
            for item in &ctx.extracted_scene.shadow_caster_items {

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

                let pipeline = if let Some(p) = ctx.pipeline_cache.get_shadow_pipeline_fast(fast_key)
                {
                    p.clone()
                } else {
                    let geo_defines = geometry.shader_defines();
                    let mat_defines = material.shader_defines();

                    let mut options = ShaderCompilationOptions::from_merged(
                        &mat_defines,
                        &geo_defines,
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
                let dynamic_offset = ctx
                    .resource_manager
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
    }

    /// 上传动态 Uniform 数据
    ///
    /// 为每个渲染命令计算并上传模型矩阵、逆矩阵、法线矩阵等。
    fn upload_dynamic_uniforms(ctx: &mut RenderContext) {
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

    fn prepare(&mut self, ctx: &mut RenderContext) {
        // 1. 准备并排序渲染命令
        Self::prepare_and_sort_commands(ctx);

        // 2. 生成阴影渲染命令
        Self::prepare_shadow_commands(ctx);

        // 3. 上传动态 Uniform
        Self::upload_dynamic_uniforms(ctx);
    }

    fn run(&self, _ctx: &mut RenderContext, _encoder: &mut wgpu::CommandEncoder) {
        // SceneCullPass 不执行实际绘制
        // 所有工作在 prepare 阶段完成
    }
}
