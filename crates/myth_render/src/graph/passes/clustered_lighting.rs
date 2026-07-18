//! Clustered lighting feature.
//!
//! This feature owns the persistent CPU-updated buffers required to drive the
//! clustered-lighting compute passes and injects the transient RDG buffer flow
//! that forwards clustered light lists into the forward scene passes.

use crate::core::gpu::Tracked;
use crate::graph::composer::GpuLightBuffers;
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    BufferDesc, BufferNodeId, ExecuteContext, ExtractContext, PassNode, PrepareContext,
};
use crate::graph::passes::light_merge::LightMergeFeature;
use crate::pipeline::{
    ComputePipelineId, ComputePipelineKey, ShaderCompilationOptions, ShaderSource,
};
use myth_resources::uniforms::{
    ClusterRecord, ClusteredLightingParams, GpuLightStorage, clustered_lighting_structs_wgsl,
    scene_lighting_structs_wgsl,
};
use myth_scene::light::LightKind;

/// Default cluster depth slices for forward clustered lighting.
pub const DEFAULT_CLUSTER_Z_SLICES: u32 = 24;
/// Approximate screen-space tile size in pixels.
pub const DEFAULT_CLUSTER_TILE_SIZE: u32 = 120;
/// Soft per-cluster light budget used for heatmap normalization and global
/// light-index capacity sizing.
pub const DEFAULT_MAX_LIGHTS_PER_CLUSTER: u32 = 256;
/// Workgroup width shared by the clustered light view transform pass.
pub const CLUSTER_LIGHT_VIEW_TRANSFORM_WG_SIZE: u32 = 64;
/// Fallback finite far depth when using an infinite reverse-Z camera.
pub const DEFAULT_CLUSTER_FAR_DEPTH_FALLBACK: f32 = 64.0;
/// Bit flag stored in `ClusteredLightingParams::budget.z`.
pub const CLUSTERED_LIGHTING_FLAG_ENABLED: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClusteredLightingConfig {
    pub tile_size_x: u32,
    pub tile_size_y: u32,
    pub slices_z: u32,
    pub max_lights_per_cluster: u32,
}

impl Default for ClusteredLightingConfig {
    fn default() -> Self {
        Self {
            tile_size_x: DEFAULT_CLUSTER_TILE_SIZE,
            tile_size_y: DEFAULT_CLUSTER_TILE_SIZE,
            slices_z: DEFAULT_CLUSTER_Z_SLICES,
            max_lights_per_cluster: DEFAULT_MAX_LIGHTS_PER_CLUSTER,
        }
    }
}

pub struct ClusteredLightingFeature {
    pub config: ClusteredLightingConfig,
    params_buffer: Option<Tracked<wgpu::Buffer>>,
    light_merge: LightMergeFeature,
    dispatch_args_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    light_view_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    cull_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    dispatch_args_pipeline: Option<ComputePipelineId>,
    light_view_pipeline: Option<ComputePipelineId>,
    cull_pipeline: Option<ComputePipelineId>,
    frame_params: ClusteredLightingParams,
}

#[derive(Clone, Copy)]
pub struct ClusteredLightingInputs {
    pub enabled: bool,
    pub cpu_light_metadata_buffer: BufferNodeId,
    pub cpu_light_data_buffer: BufferNodeId,
    pub injected_gpu_lights: Option<GpuLightBuffers>,
}

pub struct ClusteredLightingOutputs {
    pub final_light_metadata_buffer: BufferNodeId,
    pub final_light_data_buffer: BufferNodeId,
    pub final_indirect_count_buffer: Option<BufferNodeId>,
    pub params_buffer: BufferNodeId,
    pub cluster_records: Option<BufferNodeId>,
    pub light_indices: Option<BufferNodeId>,
}

impl Default for ClusteredLightingFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl ClusteredLightingFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ClusteredLightingConfig::default(),
            params_buffer: None,
            light_merge: LightMergeFeature::new(),
            dispatch_args_layout: None,
            light_view_layout: None,
            cull_layout: None,
            dispatch_args_pipeline: None,
            light_view_pipeline: None,
            cull_pipeline: None,
            frame_params: ClusteredLightingParams::default(),
        }
    }

    #[inline]
    #[must_use]
    pub fn params_desc(&self) -> BufferDesc {
        BufferDesc::new(
            std::mem::size_of::<ClusteredLightingParams>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        )
    }

    #[inline]
    #[must_use]
    pub fn cluster_record_desc(&self) -> BufferDesc {
        BufferDesc::new(
            u64::from(self.frame_params.grid_dimensions.y)
                * std::mem::size_of::<ClusterRecord>() as u64,
            wgpu::BufferUsages::STORAGE,
        )
    }

    #[inline]
    #[must_use]
    pub fn light_index_desc(&self) -> BufferDesc {
        BufferDesc::new(
            u64::from(self.frame_params.budget.y) * std::mem::size_of::<u32>() as u64,
            wgpu::BufferUsages::STORAGE,
        )
    }

    #[inline]
    #[must_use]
    pub fn light_index_allocator_desc(&self) -> BufferDesc {
        BufferDesc::new(
            std::mem::size_of::<u32>() as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )
    }

    #[inline]
    #[must_use]
    pub fn light_view_positions_desc(&self, light_slot_count: u32) -> BufferDesc {
        BufferDesc::new(
            u64::from(light_slot_count.max(1)) * std::mem::size_of::<[f32; 4]>() as u64,
            wgpu::BufferUsages::STORAGE,
        )
    }

    #[inline]
    #[must_use]
    pub fn dispatch_args_desc(&self) -> BufferDesc {
        BufferDesc::new(
            std::mem::size_of::<[u32; 4]>() as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        )
    }

    #[must_use]
    pub fn params_buffer(&self) -> Option<&Tracked<wgpu::Buffer>> {
        self.params_buffer.as_ref()
    }

    #[must_use]
    pub fn frame_params(&self) -> ClusteredLightingParams {
        self.frame_params
    }

    fn ensure_layouts(&mut self, device: &wgpu::Device) {
        if self.dispatch_args_layout.is_some()
            && self.light_view_layout.is_some()
            && self.cull_layout.is_some()
        {
            return;
        }

        self.dispatch_args_layout = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Cluster Dispatch Args Layout"),
                entries: &[storage_entry(0, true), storage_entry(1, false)],
            },
        )));

        self.light_view_layout = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Cluster Light View Layout"),
                entries: &[
                    uniform_entry(0),
                    storage_entry(1, true),
                    storage_entry(2, false),
                ],
            },
        )));

        self.cull_layout = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Cluster Cull Layout"),
                entries: &[
                    uniform_entry(0),
                    storage_entry(1, false),
                    storage_entry(2, false),
                    storage_entry(3, false),
                    uniform_entry(4),
                    storage_entry(5, true),
                ],
            },
        )));
    }

    fn ensure_pipelines(&mut self, ctx: &mut ExtractContext) {
        if self.dispatch_args_pipeline.is_some()
            && self.light_view_pipeline.is_some()
            && self.cull_pipeline.is_some()
        {
            return;
        }

        self.ensure_layouts(ctx.device);

        let global_state_key = (ctx.render_state.id, ctx.extracted_scene.scene_id);
        let gpu_world = ctx
            .resource_manager
            .get_global_state(global_state_key.0, global_state_key.1)
            .expect("Clustered lighting requires a prepared global bind group");

        let compilation_options = wgpu::PipelineCompilationOptions::default();
        let dispatch_args_layout = self
            .dispatch_args_layout
            .as_ref()
            .expect("cluster dispatch args layout missing");
        let light_view_layout = self
            .light_view_layout
            .as_ref()
            .expect("cluster light view layout missing");
        let cull_layout = self
            .cull_layout
            .as_ref()
            .expect("cluster cull layout missing");

        if self.dispatch_args_pipeline.is_none() {
            let (module, shader_hash) = ctx.shader_manager.get_or_compile(
                ctx.device,
                ShaderSource::File("entry/utility/clustered/cluster_dispatch_args"),
                &ShaderCompilationOptions::default(),
            );

            let layout = ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Cluster Dispatch Args Pipeline Layout"),
                    bind_group_layouts: &[Some(dispatch_args_layout)],
                    immediate_size: 0,
                });

            self.dispatch_args_pipeline = Some(
                ctx.pipeline_cache.get_or_create_compute(
                    ctx.device,
                    module,
                    &layout,
                    &ComputePipelineKey::new(shader_hash)
                        .with_compilation_options(&compilation_options),
                    &compilation_options,
                    "Cluster Dispatch Args Pipeline",
                ),
            );
        }

        if self.light_view_pipeline.is_none() {
            let mut options = ShaderCompilationOptions::default();
            options.inject_code("binding_code", &gpu_world.binding_wgsl);
            options.inject_code("scene_lighting_structs", scene_lighting_structs_wgsl());
            let (module, shader_hash) = ctx.shader_manager.get_or_compile(
                ctx.device,
                ShaderSource::File("entry/utility/clustered/cluster_light_view_transform"),
                &options,
            );

            let layout = ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Cluster Light View Pipeline Layout"),
                    bind_group_layouts: &[Some(&gpu_world.layout), Some(light_view_layout)],
                    immediate_size: 0,
                });

            self.light_view_pipeline = Some(
                ctx.pipeline_cache.get_or_create_compute(
                    ctx.device,
                    module,
                    &layout,
                    &ComputePipelineKey::new(shader_hash)
                        .with_compilation_options(&compilation_options),
                    &compilation_options,
                    "Cluster Light View Pipeline",
                ),
            );
        }

        if self.cull_pipeline.is_none() {
            let mut options = ShaderCompilationOptions::default();
            options.inject_code("binding_code", &gpu_world.binding_wgsl);
            options.inject_code("scene_lighting_structs", scene_lighting_structs_wgsl());
            options.inject_code(
                "clustered_lighting_structs",
                clustered_lighting_structs_wgsl(),
            );
            options.inject_code(
                "clustered_max_local_lights",
                format!("{}u", self.config.max_lights_per_cluster.max(1)),
            );
            let (module, shader_hash) = ctx.shader_manager.get_or_compile(
                ctx.device,
                ShaderSource::File("entry/utility/clustered/cluster_cull"),
                &options,
            );

            let layout = ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Cluster Cull Pipeline Layout"),
                    bind_group_layouts: &[Some(&gpu_world.layout), Some(cull_layout)],
                    immediate_size: 0,
                });

            self.cull_pipeline = Some(
                ctx.pipeline_cache.get_or_create_compute(
                    ctx.device,
                    module,
                    &layout,
                    &ComputePipelineKey::new(shader_hash)
                        .with_compilation_options(&compilation_options),
                    &compilation_options,
                    "Cluster Cull Pipeline",
                ),
            );
        }
    }

    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        enabled: bool,
        active_local_light_count: u32,
    ) {
        let render_uniforms = ctx.render_state.uniforms().read();
        let width = render_uniforms.viewport.x.max(1.0) as u32;
        let height = render_uniforms.viewport.y.max(1.0) as u32;
        let near = render_uniforms.camera_near.max(0.001);
        let camera_far = render_uniforms.camera_far;
        drop(render_uniforms);

        let far =
            resolve_cluster_far_depth(near, camera_far, estimate_cluster_light_max_depth(ctx));

        let cluster_x = width.div_ceil(self.config.tile_size_x.max(1));
        let cluster_y = height.div_ceil(self.config.tile_size_y.max(1));
        let cluster_z = self.config.slices_z.max(1);
        let total_clusters = cluster_x
            .saturating_mul(cluster_y)
            .saturating_mul(cluster_z)
            .max(1);
        let max_storage_bytes = ctx.device.limits().max_storage_buffer_binding_size;
        let max_indices_by_limit =
            (max_storage_bytes / std::mem::size_of::<u32>() as u64).max(u64::from(total_clusters));
        let requested_light_indices =
            u64::from(total_clusters) * u64::from(self.config.max_lights_per_cluster.max(1));
        let effective_light_indices = requested_light_indices.min(max_indices_by_limit);
        let effective_max_lights =
            (effective_light_indices / u64::from(total_clusters)).max(1) as u32;

        if effective_max_lights < self.config.max_lights_per_cluster {
            log::warn!(
                "Clustered lighting compact light-index capacity reduced from an average of {} to {} lights per cluster by maxStorageBufferBindingSize; extremely dense scenes may still overflow the global index list",
                self.config.max_lights_per_cluster,
                effective_max_lights,
            );
        }

        let log_ratio = (far / near).ln().max(0.0001);
        let slice_scale = cluster_z as f32 / log_ratio;
        let slice_bias = -(near.ln() * slice_scale);

        let frame_params = ClusteredLightingParams {
            screen_dimensions: glam::UVec4::new(width, height, cluster_x, cluster_y),
            grid_dimensions: glam::UVec4::new(
                cluster_z,
                total_clusters,
                self.config.tile_size_x.max(1),
                self.config.tile_size_y.max(1),
            ),
            budget: glam::UVec4::new(
                effective_max_lights,
                effective_light_indices.max(1) as u32,
                if enabled {
                    CLUSTERED_LIGHTING_FLAG_ENABLED
                } else {
                    0
                },
                active_local_light_count,
            ),
            depth_params: glam::Vec4::new(near, far, slice_scale, slice_bias),
        };
        let params_changed =
            bytemuck::bytes_of(&self.frame_params) != bytemuck::bytes_of(&frame_params);
        self.frame_params = frame_params;

        self.ensure_pipelines(ctx);
        self.light_merge.extract_and_prepare(ctx);

        let params_desc = self.params_desc();
        let params_buffer_missing = self.params_buffer.is_none();
        ensure_tracked_buffer(
            &mut self.params_buffer,
            ctx.device,
            params_desc,
            "Clustered Lighting Params",
        );

        if params_buffer_missing || params_changed {
            ctx.queue.write_buffer(
                self.params_buffer
                    .as_ref()
                    .expect("clustered params buffer must exist"),
                0,
                bytemuck::bytes_of(&self.frame_params),
            );
        }
    }

    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        inputs: ClusteredLightingInputs,
    ) -> ClusteredLightingOutputs {
        let params_buffer = self
            .params_buffer
            .as_ref()
            .expect("Clustered lighting params buffer must exist before graph build");
        let imported_params = ctx.graph.import_external_buffer(
            "Clustered_Params",
            BufferDesc::new(
                std::mem::size_of::<ClusteredLightingParams>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            params_buffer,
        );

        let cpu_local_light_count = ctx.scene_local_light_count();
        let cpu_light_capacity =
            light_capacity_from_desc(ctx.buffer_desc(inputs.cpu_light_data_buffer))
                .max(cpu_local_light_count);
        let final_lights = match inputs.injected_gpu_lights {
            Some(gpu_lights) if cpu_local_light_count == 0 => ClusteredLightingOutputs {
                final_light_metadata_buffer: gpu_lights.light_metadata,
                final_light_data_buffer: gpu_lights.light_storage,
                final_indirect_count_buffer: gpu_lights.indirect_count_buffer,
                params_buffer: imported_params,
                cluster_records: None,
                light_indices: None,
            },
            Some(gpu_lights) => {
                let merged = self.light_merge.add_to_graph(
                    ctx,
                    inputs.cpu_light_metadata_buffer,
                    inputs.cpu_light_data_buffer,
                    cpu_light_capacity,
                    gpu_lights,
                );
                ClusteredLightingOutputs {
                    final_light_metadata_buffer: merged.light_metadata,
                    final_light_data_buffer: merged.light_storage,
                    final_indirect_count_buffer: Some(merged.indirect_count_buffer),
                    params_buffer: imported_params,
                    cluster_records: None,
                    light_indices: None,
                }
            }
            None => ClusteredLightingOutputs {
                final_light_metadata_buffer: inputs.cpu_light_metadata_buffer,
                final_light_data_buffer: inputs.cpu_light_data_buffer,
                final_indirect_count_buffer: None,
                params_buffer: imported_params,
                cluster_records: None,
                light_indices: None,
            },
        };

        if !inputs.enabled {
            return final_lights;
        }

        let light_slots =
            light_capacity_from_desc(ctx.buffer_desc(final_lights.final_light_data_buffer));

        let dispatch_args_layout = self
            .dispatch_args_layout
            .as_ref()
            .expect("Cluster dispatch args layout must exist");
        let cull_layout = self
            .cull_layout
            .as_ref()
            .expect("Clustered lighting cull layout must exist");
        let light_view_layout = self
            .light_view_layout
            .as_ref()
            .expect("Clustered lighting light-view layout must exist");
        let dispatch_args_pipeline = ctx.pipeline_cache.get_compute_pipeline(
            self.dispatch_args_pipeline
                .expect("Cluster dispatch args pipeline missing"),
        );
        let light_view_pipeline = ctx.pipeline_cache.get_compute_pipeline(
            self.light_view_pipeline
                .expect("Cluster light view pipeline missing"),
        );
        let cull_pipeline = ctx
            .pipeline_cache
            .get_compute_pipeline(self.cull_pipeline.expect("Clustered cull pipeline missing"));

        let dispatch_args_buffer = final_lights
            .final_indirect_count_buffer
            .map(|count_buffer| {
                ctx.graph
                    .add_pass("Cluster_Light_Dispatch_Args", |builder| {
                        let count_buffer = builder.read_buffer(count_buffer);
                        let dispatch_args = builder.create_buffer(
                            "Cluster_Light_Dispatch_Args",
                            self.dispatch_args_desc(),
                        );

                        let node = ClusterDispatchArgsPassNode {
                            count_buffer,
                            dispatch_args,
                            dispatch_args_layout,
                            dispatch_args_pipeline,
                            bind_group: None,
                        };

                        (node, dispatch_args)
                    })
            });

        let light_view_positions = ctx.graph.add_pass("Cluster_Light_View_Pass", |builder| {
            let light_metadata_buffer =
                builder.read_buffer(final_lights.final_light_metadata_buffer);
            let light_data_buffer = builder.read_buffer(final_lights.final_light_data_buffer);
            if let Some(dispatch_args) = dispatch_args_buffer {
                builder.read_buffer(dispatch_args);
            }
            let light_view_positions = builder.create_buffer(
                "Cluster_Light_View_Positions",
                self.light_view_positions_desc(light_slots),
            );

            let node = ClusterLightViewPassNode {
                light_metadata_buffer,
                light_data_buffer,
                light_view_positions,
                indirect_dispatch_buffer: dispatch_args_buffer,
                light_view_layout,
                light_view_pipeline,
                bind_group: None,
                total_light_slots: light_slots,
            };

            (node, light_view_positions)
        });

        ctx.graph.add_pass("Cluster_Cull_Pass", |builder| {
            let params_buffer = builder.read_buffer(imported_params);
            let light_metadata_buffer =
                builder.read_buffer(final_lights.final_light_metadata_buffer);
            let light_view_positions = builder.read_buffer(light_view_positions);
            let cluster_records =
                builder.create_buffer("Cluster_Records", self.cluster_record_desc());
            let light_indices =
                builder.create_buffer("Cluster_Light_Indices", self.light_index_desc());
            let light_index_allocator = builder.create_buffer(
                "Cluster_Light_Index_Allocator",
                self.light_index_allocator_desc(),
            );

            let node = ClusterCullPassNode {
                params_buffer,
                light_metadata_buffer,
                light_view_positions,
                cluster_records,
                light_indices,
                light_index_allocator,
                cull_layout,
                cull_pipeline,
                bind_group: None,
                dispatch_grid_x: self.frame_params.screen_dimensions.z.max(1),
                dispatch_grid_y: self.frame_params.screen_dimensions.w.max(1),
                dispatch_grid_z: self.frame_params.grid_dimensions.x.max(1),
            };
            (
                node,
                ClusteredLightingOutputs {
                    final_light_metadata_buffer: final_lights.final_light_metadata_buffer,
                    final_light_data_buffer: final_lights.final_light_data_buffer,
                    final_indirect_count_buffer: final_lights.final_indirect_count_buffer,
                    params_buffer,
                    cluster_records: Some(cluster_records),
                    light_indices: Some(light_indices),
                },
            )
        })
    }
}

fn estimate_cluster_light_max_depth(ctx: &ExtractContext) -> Option<f32> {
    let view_matrix = ctx.render_camera.view_matrix;
    let mut max_depth = 0.0_f32;

    for light in &ctx.extracted_scene.lights {
        let range = match &light.kind {
            LightKind::Point(point) => point.range,
            LightKind::Spot(spot) => spot.range,
            LightKind::Directional(_) => 0.0,
        };

        if range <= 0.0 {
            continue;
        }

        let view_center = view_matrix * light.position.extend(1.0);
        let far_depth = -view_center.z + range;
        if far_depth.is_finite() {
            max_depth = max_depth.max(far_depth);
        }
    }

    (max_depth > 0.0).then_some(max_depth)
}

fn resolve_cluster_far_depth(
    near: f32,
    camera_far: f32,
    estimated_light_depth: Option<f32>,
) -> f32 {
    if camera_far.is_finite() {
        return camera_far.max(near + 0.001);
    }

    estimated_light_depth
        .unwrap_or(near + DEFAULT_CLUSTER_FAR_DEPTH_FALLBACK)
        .max(near + DEFAULT_CLUSTER_FAR_DEPTH_FALLBACK)
}

struct ClusterDispatchArgsPassNode<'a> {
    count_buffer: BufferNodeId,
    dispatch_args: BufferNodeId,
    dispatch_args_layout: &'a Tracked<wgpu::BindGroupLayout>,
    dispatch_args_pipeline: &'a wgpu::ComputePipeline,
    bind_group: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for ClusterDispatchArgsPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.bind_group = Some(
            ctx.build_bind_group(self.dispatch_args_layout, Some("Cluster Dispatch Args BG"))
                .bind_buffer(0, self.count_buffer)
                .bind_buffer(1, self.dispatch_args)
                .build(),
        );
    }

    fn execute(&self, _ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Cluster Dispatch Args Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(self.dispatch_args_pipeline);
        pass.set_bind_group(
            0,
            self.bind_group.expect("Cluster dispatch args BG missing"),
            &[],
        );
        pass.dispatch_workgroups(1, 1, 1);
    }
}

struct ClusterLightViewPassNode<'a> {
    light_metadata_buffer: BufferNodeId,
    light_data_buffer: BufferNodeId,
    light_view_positions: BufferNodeId,
    indirect_dispatch_buffer: Option<BufferNodeId>,
    light_view_layout: &'a Tracked<wgpu::BindGroupLayout>,
    light_view_pipeline: &'a wgpu::ComputePipeline,
    bind_group: Option<&'a wgpu::BindGroup>,
    total_light_slots: u32,
}

impl<'a> PassNode<'a> for ClusterLightViewPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.bind_group = Some(
            ctx.build_bind_group(self.light_view_layout, Some("Cluster Light View BG"))
                .bind_buffer(0, self.light_metadata_buffer)
                .bind_buffer(1, self.light_data_buffer)
                .bind_buffer(2, self.light_view_positions)
                .build(),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Cluster Light View Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(self.light_view_pipeline);
        pass.set_bind_group(0, ctx.baked_lists.global_bind_group, &[]);
        pass.set_bind_group(
            1,
            self.bind_group.expect("Cluster light view BG missing"),
            &[],
        );
        if let Some(indirect_dispatch_buffer) = self.indirect_dispatch_buffer {
            pass.dispatch_workgroups_indirect(ctx.get_buffer(indirect_dispatch_buffer), 0);
        } else {
            pass.dispatch_workgroups(
                self.total_light_slots
                    .div_ceil(CLUSTER_LIGHT_VIEW_TRANSFORM_WG_SIZE),
                1,
                1,
            );
        }
    }
}

struct ClusterCullPassNode<'a> {
    params_buffer: BufferNodeId,
    light_metadata_buffer: BufferNodeId,
    light_view_positions: BufferNodeId,
    cluster_records: BufferNodeId,
    light_indices: BufferNodeId,
    light_index_allocator: BufferNodeId,
    cull_layout: &'a Tracked<wgpu::BindGroupLayout>,
    cull_pipeline: &'a wgpu::ComputePipeline,
    bind_group: Option<&'a wgpu::BindGroup>,
    dispatch_grid_x: u32,
    dispatch_grid_y: u32,
    dispatch_grid_z: u32,
}

impl<'a> PassNode<'a> for ClusterCullPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        // ctx.queue.write_buffer(
        //     ctx.views.get_buffer(self.light_index_allocator),
        //     0,
        //     bytemuck::bytes_of(&0u32),
        // );

        self.bind_group = Some(
            ctx.build_bind_group(self.cull_layout, Some("Cluster Cull BG"))
                .bind_buffer(0, self.params_buffer)
                .bind_buffer(1, self.cluster_records)
                .bind_buffer(2, self.light_indices)
                .bind_buffer(3, self.light_index_allocator)
                .bind_buffer(4, self.light_metadata_buffer)
                .bind_buffer(5, self.light_view_positions)
                .build(),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        // Clear the light index allocator to zero before culling so that the first cluster starts with a clean slate;
        encoder.clear_buffer(ctx.get_buffer(self.light_index_allocator), 0, None);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Cluster Cull Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(self.cull_pipeline);
        pass.set_bind_group(0, ctx.baked_lists.global_bind_group, &[]);
        pass.set_bind_group(1, self.bind_group.expect("Cluster cull BG missing"), &[]);
        pass.dispatch_workgroups(
            self.dispatch_grid_x,
            self.dispatch_grid_y,
            self.dispatch_grid_z,
        );
    }
}

fn ensure_tracked_buffer(
    slot: &mut Option<Tracked<wgpu::Buffer>>,
    device: &wgpu::Device,
    desc: BufferDesc,
    label: &'static str,
) {
    let needs_recreate = slot
        .as_ref()
        .is_none_or(|buffer| buffer.size() != desc.logical_size);

    if needs_recreate {
        *slot = Some(Tracked::new(device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some(label),
                size: desc.logical_size.max(1),
                usage: desc.usage,
                mapped_at_creation: false,
            },
        )));
    }
}

fn light_capacity_from_desc(desc: BufferDesc) -> u32 {
    let bytes_per_light = std::mem::size_of::<GpuLightStorage>() as u64;
    ((desc.logical_size / bytes_per_light).max(1)).min(u64::from(u32::MAX)) as u32
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_CLUSTER_FAR_DEPTH_FALLBACK, resolve_cluster_far_depth};

    #[test]
    fn infinite_camera_far_resolves_to_finite_cluster_depth() {
        let far = resolve_cluster_far_depth(0.1, f32::INFINITY, Some(48.0));
        assert!(far.is_finite());
        assert!(far > 48.0);
    }

    #[test]
    fn infinite_camera_far_uses_fallback_when_scene_depth_missing() {
        let near = 0.25;
        let far = resolve_cluster_far_depth(near, f32::INFINITY, None);
        assert!(far.is_finite());
        assert!(far >= near + DEFAULT_CLUSTER_FAR_DEPTH_FALLBACK);
    }
}
