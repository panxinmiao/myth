//! Forward Render Pass
//!
//! Implements the main drawing logic for the forward rendering pipeline.

use log::{warn, error};
use slotmap::Key;

use crate::renderer::graph::{RenderNode, RenderContext, TrackedRenderPass};
use crate::renderer::graph::frame::{RenderKey};
use crate::renderer::pipeline::{PipelineKey, FastPipelineKey};
use crate::renderer::pipeline::shader_gen::ShaderCompilationOptions;
use crate::resources::material::{AlphaMode, Side};
use crate::resources::uniforms::{DynamicModelUniforms, Mat3Uniform};

/// Forward Render Pass
/// 
/// Executes the standard forward rendering pipeline:
/// 1. Prepare and sort render commands
/// 2. Begin render pass
/// 3. Draw opaque objects (front-to-back sorting)
/// 4. Draw transparent objects (back-to-front sorting)
/// 
/// # Performance Considerations
/// - Command lists are pre-allocated and reuse memory, avoiding per-frame allocations
/// - `RefCell` runtime borrow check overhead is minimal (roughly equivalent to one atomic operation in single-threaded scenarios)
/// - Uses `TrackedRenderPass` to avoid redundant state changes
/// - Pipeline and BindGroup caching reduces GPU state changes
pub struct ForwardRenderPass {
    /// Clear color
    pub clear_color: wgpu::Color,

    pub(crate) output_to_screen: bool,

    commands_bundle : CommandsBundle,
}

impl ForwardRenderPass {
    pub fn new(clear_color: wgpu::Color, output_to_screen: bool) -> Self {
        Self {
            clear_color,
            commands_bundle: CommandsBundle::new(),
            output_to_screen,
        }
    }

    // color view, resolve view
    fn get_render_target<'a>(&self, ctx: &'a RenderContext) -> (&'a wgpu::TextureView, Option<&'a wgpu::TextureView>,) {
        let target_view = if self.output_to_screen {
            ctx.surface_view
        } else {
            &ctx.frame_resources.scene_color_view[0]
        };
        
        let is_msaa = ctx.wgpu_ctx.msaa_samples > 1;
    
        if is_msaa {
            let msaa_view = ctx.frame_resources.scene_msaa_view.as_ref().expect("MSAA view missing");
            ( msaa_view, Some(target_view) )
        } else {
            ( target_view, None )
        }
    }

    /// Prepare and sort render commands
    fn prepare_and_sort_commands(&mut self, ctx: &mut RenderContext) {

        let (render_target, resolve_target) = self.get_render_target(ctx);
        self.commands_bundle.color_view = Some(render_target.clone());
        self.commands_bundle.resolve_target = resolve_target.cloned();

        self.commands_bundle.clear();

        let Some(gpu_world) = ctx.resource_manager.get_global_state(ctx.render_state.id, ctx.extracted_scene.scene_id) else {
            error!("Render Environment missing for render_state_id {}, scene_id {}", ctx.render_state.id, ctx.extracted_scene.scene_id);
            return;
        };

        self.commands_bundle.gpu_global_bind_group_id =gpu_world.bind_group_id;
        self.commands_bundle.gpu_global_bind_group = Some(gpu_world.bind_group.clone());

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
            // Note: scene_id hash computation is optimized for caching, cost is low
            let fast_key = FastPipelineKey {
                material_handle: item.material,
                material_version: gpu_material.version,
                geometry_handle: item.geometry,
                geometry_version: geometry.layout_version(),
                instance_variants: item.item_variant_flags,
                global_state_id: gpu_world.id,
            };

            // ========== Hot Path Optimization: Check L1 Cache First ==========
            let (pipeline, pipeline_id) = if let Some(p) = ctx.pipeline_cache.get_pipeline_fast(fast_key) {
                // L1 cache hit: Directly use cached Pipeline, no need to compute shader_defines
                p.clone()
            } else {
                // L1 cache miss: Need full shader_defines computation to build/find Pipeline
                let geo_defines = geometry.shader_defines();

                let mat_defines = material.shader_defines();

                let options = ShaderCompilationOptions::from_merged(
                    &mat_defines,
                    &geo_defines,
                    &ctx.extracted_scene.scene_defines,
                    &item.item_shader_defines,
                );
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
                    // Reverse Z: Greater for depth test
                    depth_compare: if material.depth_test() { wgpu::CompareFunction::Greater } else { wgpu::CompareFunction::Always },
                    blend_state: if material.alpha_mode() == AlphaMode::Blend { Some(wgpu::BlendState::ALPHA_BLENDING) } else { None },
                    color_format: ctx.get_output_format(self.output_to_screen()),
                    depth_format: ctx.wgpu_ctx.depth_format,
                    sample_count: ctx.wgpu_ctx.msaa_samples,
                    front_face: if item.item_variant_flags & 0x1 != 0 { wgpu::FrontFace::Cw } else { wgpu::FrontFace::Ccw },
                };

                let use_transmission = material.use_transmission();

                let framme_resources = match use_transmission {
                    true => Some(ctx.frame_resources),
                    false => None,
                };

                let (pipeline, pipeline_id) = ctx.pipeline_cache.get_pipeline(
                    &ctx.wgpu_ctx.device,
                    material.shader_name(),
                    canonical_key,
                    &options,
                    &gpu_geometry.layout_info,
                    gpu_material,   // group 1: material_bind_group
                    object_bind_group,    // group 2: object_bind_group
                    &gpu_world,    // group 0: global_bind_group
                    framme_resources,    // group 3: screen_bind_group
                );

                ctx.pipeline_cache.insert_pipeline_fast(fast_key, (pipeline.clone(), pipeline_id));
                (pipeline, pipeline_id)
            };


            let mat_id = item.material.data().as_ffi() as u32;

            let is_transparent = material.alpha_mode() == AlphaMode::Blend;

            let sort_key = RenderKey::new(pipeline_id, mat_id, item.distance_sq, is_transparent);

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
                self.commands_bundle.insert_transparent(cmd);
            } else {
                self.commands_bundle.insert_opaque(cmd);
            }
        }



        self.commands_bundle.sort_commands();
    }

    /// Upload dynamic Uniform data
    fn upload_dynamic_uniforms(&mut self, ctx: &mut RenderContext) {

        if self.commands_bundle.is_empty() {
            return;
        }

        let opaque = &mut self.commands_bundle.opaque_commands;

        for cmd in opaque.iter_mut() {
            let world_matrix_inverse = cmd.model_matrix.inverse();
            let normal_matrix = Mat3Uniform::from_mat4(world_matrix_inverse.transpose());

            let offset = ctx.resource_manager.allocate_model_uniform(DynamicModelUniforms {
                world_matrix: cmd.model_matrix,
                world_matrix_inverse,
                normal_matrix,
                ..Default::default()
            });

            cmd.dynamic_offset = offset;
        }

        let transparent = &mut self.commands_bundle.transparent_commands;

        for cmd in transparent.iter_mut() {
            let world_matrix_inverse = cmd.model_matrix.inverse();
            let normal_matrix = Mat3Uniform::from_mat4(world_matrix_inverse.transpose());

            let offset = ctx.resource_manager.allocate_model_uniform(DynamicModelUniforms {
                world_matrix: cmd.model_matrix,
                world_matrix_inverse,
                normal_matrix,
                ..Default::default()
            });

            cmd.dynamic_offset = offset;
        }

        ctx.resource_manager.upload_model_buffer();
    }

    /// Execute draw list
    fn draw_list<'pass>(
        ctx: &'pass RenderContext,
        pass: &mut TrackedRenderPass<'pass>,
        cmds: &'pass [RenderCommand],
        // render_state_id: u32,
        // scene_id: u64,
    ) {
        if cmds.is_empty() { return; }

        for cmd in cmds {
            pass.set_pipeline(cmd.pipeline_id, &cmd.pipeline);

            if let Some(gpu_material) = ctx.resource_manager.get_material(cmd.material_handle) {
                pass.set_bind_group(1, gpu_material.bind_group_id, &gpu_material.bind_group, &[]);
            }

            pass.set_bind_group(
                2,
                cmd.object_bind_group.bind_group_id,
                &cmd.object_bind_group.bind_group,
                &[cmd.dynamic_offset],
            );

            if let Some(gpu_geometry) = ctx.resource_manager.get_geometry(cmd.geometry_handle) {
                for (slot, buffer) in gpu_geometry.vertex_buffers.iter().enumerate() {
                    pass.set_vertex_buffer(
                        slot as u32,
                        gpu_geometry.vertex_buffer_ids[slot],
                        buffer.slice(..),
                    );
                }

                if let Some((index_buffer, index_format, count, id)) = &gpu_geometry.index_buffer {
                    pass.set_index_buffer(*id, index_buffer.slice(..), *index_format);
                    pass.draw_indexed(0..*count, 0, gpu_geometry.instance_range.clone());
                } else {
                    pass.draw(gpu_geometry.draw_range.clone(), gpu_geometry.instance_range.clone());
                }
            }
        }
    }
}

impl RenderNode for ForwardRenderPass {
    fn name(&self) -> &str {
        "Forward Pass"
    }

    #[inline]
    fn output_to_screen(&self) -> bool {
        self.output_to_screen
    }

    fn prepare(&mut self, ctx: &mut RenderContext) {
        // 1. Prepare render commands (using RefCell for interior mutability)
        self.prepare_and_sort_commands(ctx);

        // 2. Upload dynamic Uniform
        self.upload_dynamic_uniforms(ctx);
    }

    fn run(&self, ctx: &mut RenderContext, encoder: &mut wgpu::CommandEncoder) {

        // 3. Get depth view
        let depth_view = &ctx.frame_resources.depth_view;

        let Some(color_view) = &self.commands_bundle.color_view else {
            log::error!("ForwardRenderPass: color_view missing");
            return;
        };

        let resolve_view = self.commands_bundle.resolve_target.as_ref();

        // 4. Begin render pass and execute drawing
        {
            // Get immutable borrow (must be declared before TrackedRenderPass to ensure lifetime)
            let commands_bundle = &self.commands_bundle;

            let pass_desc = wgpu::RenderPassDescriptor {
                label: Some("Forward Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: resolve_view,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        // Reverse Z: Clear to 0.0 (far clipping plane)
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };

            let pass = encoder.begin_render_pass(&pass_desc);
            let mut tracked = TrackedRenderPass::new(pass);

            if let Some(gpu_global_bind_group) = &commands_bundle.gpu_global_bind_group {
                tracked.set_bind_group(0, commands_bundle.gpu_global_bind_group_id, &gpu_global_bind_group, &[]);
            } else {
                return;
            }

            Self::draw_list(
                ctx,
                &mut tracked,
                &commands_bundle.opaque_commands,
            );
            Self::draw_list(
                ctx,
                &mut tracked,
                &commands_bundle.transparent_commands,
            );
        }
    }
}


struct RenderCommand {
    object_bind_group: crate::renderer::core::BindGroupContext,
    geometry_handle: crate::assets::GeometryHandle,
    material_handle: crate::assets::MaterialHandle,
    pipeline_id: u16,
    pipeline: wgpu::RenderPipeline,
    model_matrix: glam::Mat4,
    sort_key: RenderKey,
    dynamic_offset: u32,
}

struct CommandsBundle {
    opaque_commands: Vec<RenderCommand>,
    transparent_commands: Vec<RenderCommand>,
    gpu_global_bind_group_id: u64,
    gpu_global_bind_group: Option<wgpu::BindGroup>,
    color_view: Option<wgpu::TextureView>,
    resolve_target: Option<wgpu::TextureView>,
}

impl CommandsBundle {
    pub fn new() -> Self {
        Self {
            opaque_commands: Vec::with_capacity(512),
            transparent_commands: Vec::with_capacity(128),
            gpu_global_bind_group_id: 0,
            gpu_global_bind_group: None,
            color_view: None,
            resolve_target: None,
        }
    }

    pub fn clear(&mut self) {
        self.opaque_commands.clear();
        self.transparent_commands.clear();
    }

    pub fn insert_opaque(&mut self, cmd: RenderCommand) {
        self.opaque_commands.push(cmd);
    }
    pub fn insert_transparent(&mut self, cmd: RenderCommand) {
        self.transparent_commands.push(cmd);
    }

    pub fn sort_commands(&mut self) {
        self.opaque_commands.sort_unstable_by(|a, b| a.sort_key.cmp(&b.sort_key));
        self.transparent_commands.sort_unstable_by(|a, b| a.sort_key.cmp(&b.sort_key));
    }

    pub fn is_empty(&self) -> bool {
        self.opaque_commands.is_empty() && self.transparent_commands.is_empty()
    }
}