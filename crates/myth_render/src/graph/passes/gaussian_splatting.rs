//! RDG Gaussian Splatting Pass — 3D Gaussian Splatting Rendering
//!
//! Implements Myth's 3DGS path in three explicit stages:
//!
//! 1. Preprocess — project 3D Gaussians to 2D screen-space splats, evaluate
//!    SH colour, cull invisible points, and emit reverse-Z depth keys.
//! 2. GPU radix sort — ported from web-splat and adapted to Myth's RDG pass
//!    execution model. Sorting is back-to-front to match the blend equation.
//! 3. Render — draw storage-buffer-pulled triangle strips with back-to-front
//!    compositing and reverse-Z depth testing against opaque geometry.
//!
//! Multiple Gaussian clouds are supported simultaneously. Each cloud owns its
//! own preprocess buffers, sort buffers, and indirect draw buffer.

use std::sync::Arc;

use glam::Vec3A;

use crate::HDR_TEXTURE_FORMAT;
use crate::core::binding::BindGroupKey;
use crate::core::gpu::Tracked;
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    BufferDesc, BufferNodeId, ExecuteContext, ExtractContext, PassNode, PrepareContext,
    RenderTargetOps, TextureNodeId,
};
use crate::pipeline::{ShaderCompilationOptions, ShaderSource};
use myth_resources::GaussianCloudHandle;
use myth_resources::gaussian_splat::{
    GaussianCloud, GaussianSHCoefficients, GaussianSplat, Splat2D,
};
use myth_scene::camera::RenderCamera;

const PREPROCESS_WG_SIZE: u32 = 256;

const SORT_HISTOGRAM_WG_SIZE: u32 = 256;
const SORT_PREFIX_WG_SIZE: u32 = 128;
const SORT_SCATTER_WG_SIZE: u32 = 256;
const SORT_RADIX_LOG2: u32 = 8;
const SORT_RADIX_SIZE: usize = 1 << SORT_RADIX_LOG2;
const SORT_KEYVAL_PASSES: u32 = 4;
const SORT_HISTOGRAM_BLOCK_ROWS: usize = 15;
const SORT_SCATTER_BLOCK_ROWS: usize = SORT_HISTOGRAM_BLOCK_ROWS;
const SORT_KEYS_PER_WG: usize = SORT_HISTOGRAM_WG_SIZE as usize * SORT_HISTOGRAM_BLOCK_ROWS;

const SPLAT_VERTEX_COUNT: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCameraUniforms {
    view: [f32; 16],
    view_inv: [f32; 16],
    proj: [f32; 16],
    proj_inv: [f32; 16],
    viewport: [f32; 2],
    focal: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuRenderSettings {
    gaussian_scaling: f32,
    max_sh_deg: u32,
    mip_splatting: u32,
    kernel_size: f32,
    scene_extent: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSortInfos {
    keys_size: u32,
    padded_size: u32,
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuDispatchIndirect {
    dispatch_x: u32,
    dispatch_y: u32,
    dispatch_z: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuDrawIndirect {
    vertex_count: u32,
    instance_count: u32,
    base_vertex: u32,
    base_instance: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GaussianRenderPipelineKey {
    depth_format: wgpu::TextureFormat,
    msaa_samples: u32,
}

#[derive(Clone, Copy, Debug)]
struct SortBufferLayout {
    padded_key_capacity: usize,
    internal_buffer_words: usize,
}

fn padded_sort_key_capacity(key_count: usize) -> usize {
    let key_count = key_count.max(1);
    ((key_count + SORT_KEYS_PER_WG) / SORT_KEYS_PER_WG + 1) * SORT_KEYS_PER_WG
}

fn sort_scatter_blocks(key_count: usize) -> usize {
    let scatter_block_kvs = SORT_HISTOGRAM_WG_SIZE as usize * SORT_SCATTER_BLOCK_ROWS;
    (key_count.max(1) + scatter_block_kvs - 1) / scatter_block_kvs
}

fn sort_buffer_layout(key_count: usize) -> SortBufferLayout {
    let scatter_blocks_ru = sort_scatter_blocks(key_count).max(1);
    let padded_key_capacity = padded_sort_key_capacity(key_count);
    let internal_buffer_words = (SORT_KEYVAL_PASSES as usize + scatter_blocks_ru) * SORT_RADIX_SIZE;

    SortBufferLayout {
        padded_key_capacity,
        internal_buffer_words,
    }
}

fn detect_sort_subgroup_size(device: &wgpu::Device) -> u32 {
    let reported = device.adapter_info().subgroup_min_size;
    match reported {
        32.. => 32,
        16..=31 => 16,
        8..=15 => 8,
        _ => 32,
    }
}

fn build_sort_shader_options(subgroup_size: u32) -> ShaderCompilationOptions {
    let subgroup_size = match subgroup_size {
        32.. => 32,
        16..=31 => 16,
        8..=15 => 8,
        _ => 32,
    };

    let subgroup_size_usize = subgroup_size as usize;
    let rs_sweep_0_size = SORT_RADIX_SIZE / subgroup_size_usize;
    let rs_sweep_1_size = rs_sweep_0_size / subgroup_size_usize;
    let rs_mem_dwords = SORT_RADIX_SIZE + SORT_SCATTER_BLOCK_ROWS * SORT_SCATTER_WG_SIZE as usize;

    let mut options = ShaderCompilationOptions::default();
    options.add_define("HISTOGRAM_SG_SIZE", &subgroup_size.to_string());
    options.add_define("HISTOGRAM_WG_SIZE", &SORT_HISTOGRAM_WG_SIZE.to_string());
    options.add_define("PREFIX_WG_SIZE", &SORT_PREFIX_WG_SIZE.to_string());
    options.add_define("SCATTER_WG_SIZE", &SORT_SCATTER_WG_SIZE.to_string());
    options.add_define("RS_RADIX_LOG2", &SORT_RADIX_LOG2.to_string());
    options.add_define("RS_RADIX_SIZE", &SORT_RADIX_SIZE.to_string());
    options.add_define("RS_KEYVAL_SIZE", &SORT_KEYVAL_PASSES.to_string());
    options.add_define(
        "RS_HISTOGRAM_BLOCK_ROWS",
        &SORT_HISTOGRAM_BLOCK_ROWS.to_string(),
    );
    options.add_define(
        "RS_SCATTER_BLOCK_ROWS",
        &SORT_SCATTER_BLOCK_ROWS.to_string(),
    );
    options.add_define("RS_MEM_DWORDS", &rs_mem_dwords.to_string());
    options.add_define("RS_MEM_SWEEP_0_OFFSET", "0");
    options.add_define("RS_MEM_SWEEP_1_OFFSET", &rs_sweep_0_size.to_string());
    options.add_define(
        "RS_MEM_SWEEP_2_OFFSET",
        &(rs_sweep_0_size + rs_sweep_1_size).to_string(),
    );
    options
}

struct GaussianSortPipelines {
    zero_histograms: wgpu::ComputePipeline,
    calculate_histogram: wgpu::ComputePipeline,
    prefix_histogram: wgpu::ComputePipeline,
    scatter_even: wgpu::ComputePipeline,
    scatter_odd: wgpu::ComputePipeline,
}

struct CloudGpuData {
    num_points: u32,
    sort_layout: SortBufferLayout,

    gaussian_buf: Tracked<wgpu::Buffer>,
    sh_buf: Tracked<wgpu::Buffer>,
    camera_uniform_buf: Tracked<wgpu::Buffer>,
    render_settings_buf: Tracked<wgpu::Buffer>,
}

#[derive(Clone, Copy)]
struct CloudGraphBuffers {
    gaussian_buf: BufferNodeId,
    sh_buf: BufferNodeId,
    splat_2d_buf: BufferNodeId,
    sort_infos_buf: BufferNodeId,
    sort_dispatch_buf: BufferNodeId,
    sort_internal_buf: BufferNodeId,
    sort_depths_a_buf: BufferNodeId,
    sort_depths_b_buf: BufferNodeId,
    sort_indices_a_buf: BufferNodeId,
    sort_indices_b_buf: BufferNodeId,
    draw_indirect_buf: BufferNodeId,
    camera_uniform_buf: BufferNodeId,
    render_settings_buf: BufferNodeId,
    num_points: u32,
    sort_infos_init: GpuSortInfos,
    sort_dispatch_init: GpuDispatchIndirect,
    draw_indirect_init: GpuDrawIndirect,
}

#[derive(Clone, Copy)]
struct CloudComputeState<'a> {
    buffers: CloudGraphBuffers,
    preprocess_bg0: Option<&'a wgpu::BindGroup>,
    preprocess_bg1: Option<&'a wgpu::BindGroup>,
    preprocess_bg2: Option<&'a wgpu::BindGroup>,
    preprocess_bg3: Option<&'a wgpu::BindGroup>,
    sort_bg: Option<&'a wgpu::BindGroup>,
}

#[derive(Clone, Copy)]
struct CloudRenderState<'a> {
    buffers: CloudGraphBuffers,
    render_bg: Option<&'a wgpu::BindGroup>,
}

pub struct GaussianSplattingFeature {
    preprocess_pipeline: Option<wgpu::ComputePipeline>,
    sort_pipelines: Option<GaussianSortPipelines>,
    render_pipeline: Option<wgpu::RenderPipeline>,
    render_pipeline_key: Option<GaussianRenderPipelineKey>,

    preprocess_layout_g0: Option<Tracked<wgpu::BindGroupLayout>>,
    preprocess_layout_g1: Option<Tracked<wgpu::BindGroupLayout>>,
    preprocess_layout_g2: Option<Tracked<wgpu::BindGroupLayout>>,
    preprocess_layout_g3: Option<Tracked<wgpu::BindGroupLayout>>,
    sort_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    render_layout: Option<Tracked<wgpu::BindGroupLayout>>,

    clouds: Vec<(GaussianCloudHandle, u64, CloudGpuData)>,
    sorted_order: Vec<usize>,
    active: bool,
}

impl Default for GaussianSplattingFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl GaussianSplattingFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            preprocess_pipeline: None,
            sort_pipelines: None,
            render_pipeline: None,
            render_pipeline_key: None,
            preprocess_layout_g0: None,
            preprocess_layout_g1: None,
            preprocess_layout_g2: None,
            preprocess_layout_g3: None,
            sort_layout: None,
            render_layout: None,
            clouds: Vec::new(),
            sorted_order: Vec::new(),
            active: false,
        }
    }

    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        cloud_entries: &[(GaussianCloudHandle, Arc<GaussianCloud>, Vec3A)],
        camera: &RenderCamera,
        viewport_size: (u32, u32),
    ) {
        if cloud_entries.is_empty() {
            self.sorted_order.clear();
            self.active = false;
            return;
        }

        self.ensure_layouts(ctx.device);
        self.ensure_pipelines(ctx);

        let active_handles: Vec<GaussianCloudHandle> =
            cloud_entries.iter().map(|(handle, _, _)| *handle).collect();
        self.clouds
            .retain(|(handle, _, _)| active_handles.contains(handle));

        for (handle, cloud, _) in cloud_entries {
            let handle = *handle;
            let fingerprint = {
                let ptr = Arc::as_ptr(cloud) as u64;
                ptr ^ cloud.num_points as u64
            };

            match self
                .clouds
                .iter()
                .position(|(existing_handle, _, _)| *existing_handle == handle)
            {
                Some(index) if self.clouds[index].1 == fingerprint => {}
                Some(index) => {
                    let gpu_data = self.create_cloud_gpu_data(ctx.device, ctx.queue, cloud);
                    self.clouds[index] = (handle, fingerprint, gpu_data);
                }
                None => {
                    let gpu_data = self.create_cloud_gpu_data(ctx.device, ctx.queue, cloud);
                    self.clouds.push((handle, fingerprint, gpu_data));
                }
            }
        }

        for (handle, cloud, _) in cloud_entries {
            if let Some((_, _, gpu_data)) = self
                .clouds
                .iter()
                .find(|(existing_handle, _, _)| *existing_handle == *handle)
            {
                Self::update_cloud_uniforms(ctx.queue, gpu_data, camera, cloud, viewport_size);
            }
        }

        let camera_position = camera.position;
        let mut cloud_order: Vec<(usize, f32)> = cloud_entries
            .iter()
            .filter_map(|(handle, cloud, world_pos)| {
                let cloud_index = self
                    .clouds
                    .iter()
                    .position(|(existing_handle, _, _)| existing_handle == handle)?;
                let local_center = Vec3A::new(cloud.center.x, cloud.center.y, cloud.center.z);
                let world_center = *world_pos + local_center;
                let distance_sq = camera_position.distance_squared(world_center);
                Some((cloud_index, distance_sq))
            })
            .collect();

        cloud_order.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.sorted_order = cloud_order.into_iter().map(|(index, _)| index).collect();
        self.active = !self.sorted_order.is_empty();
    }

    fn ensure_layouts(&mut self, device: &wgpu::Device) {
        if self.preprocess_layout_g0.is_some() {
            return;
        }

        let uniform_entry =
            |binding: u32, visibility: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
                binding,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };

        let storage_ro_entry =
            |binding: u32, visibility: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
                binding,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };

        let storage_rw_entry =
            |binding: u32, visibility: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
                binding,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };

        let cs = wgpu::ShaderStages::COMPUTE;
        let vs = wgpu::ShaderStages::VERTEX;

        self.preprocess_layout_g0 = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GS Preprocess G0 (Camera)"),
                entries: &[uniform_entry(0, cs)],
            },
        )));

        self.preprocess_layout_g1 = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GS Preprocess G1 (Gaussians + SH + Splats)"),
                entries: &[
                    storage_ro_entry(0, cs),
                    storage_ro_entry(1, cs),
                    storage_rw_entry(2, cs),
                ],
            },
        )));

        self.preprocess_layout_g2 = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GS Preprocess G2 (Sort Front Buffers)"),
                entries: &[
                    storage_rw_entry(0, cs),
                    storage_rw_entry(1, cs),
                    storage_rw_entry(2, cs),
                    storage_rw_entry(3, cs),
                ],
            },
        )));

        self.preprocess_layout_g3 = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GS Preprocess G3 (Render Settings)"),
                entries: &[uniform_entry(0, cs)],
            },
        )));

        self.sort_layout = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GS Sort"),
                entries: &[
                    storage_rw_entry(0, cs),
                    storage_rw_entry(1, cs),
                    storage_rw_entry(2, cs),
                    storage_rw_entry(3, cs),
                    storage_rw_entry(4, cs),
                    storage_rw_entry(5, cs),
                ],
            },
        )));

        self.render_layout = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GS Render"),
                entries: &[storage_ro_entry(0, vs), storage_ro_entry(1, vs)],
            },
        )));
    }

    fn ensure_pipelines(&mut self, ctx: &mut ExtractContext) {
        let device = ctx.device;

        if self.preprocess_pipeline.is_none() {
            let shader_options = ShaderCompilationOptions::default();
            let (module, _) = ctx.shader_manager.get_or_compile(
                device,
                ShaderSource::File("entry/utility/gaussian_preprocess"),
                &shader_options,
            );

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GS Preprocess Pipeline Layout"),
                bind_group_layouts: &[
                    Some(self.preprocess_layout_g0.as_ref().unwrap()),
                    Some(self.preprocess_layout_g1.as_ref().unwrap()),
                    Some(self.preprocess_layout_g2.as_ref().unwrap()),
                    Some(self.preprocess_layout_g3.as_ref().unwrap()),
                ],
                immediate_size: 0,
            });

            self.preprocess_pipeline = Some(device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("GS Preprocess Pipeline"),
                    layout: Some(&layout),
                    module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                },
            ));
        }

        if self.sort_pipelines.is_none() {
            let subgroup_size = detect_sort_subgroup_size(device);
            let shader_options = build_sort_shader_options(subgroup_size);
            let (module, _) = ctx.shader_manager.get_or_compile(
                device,
                ShaderSource::File("entry/utility/gs_radix_sort"),
                &shader_options,
            );

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GS Sort Pipeline Layout"),
                bind_group_layouts: &[Some(self.sort_layout.as_ref().unwrap())],
                immediate_size: 0,
            });

            let make_pipeline = |entry_point: &str, label: &str| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    module,
                    entry_point: Some(entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
            };

            self.sort_pipelines = Some(GaussianSortPipelines {
                zero_histograms: make_pipeline("zero_histograms", "GS Sort Zero Histograms"),
                calculate_histogram: make_pipeline(
                    "calculate_histogram",
                    "GS Sort Calculate Histogram",
                ),
                prefix_histogram: make_pipeline("prefix_histogram", "GS Sort Prefix Histogram"),
                scatter_even: make_pipeline("scatter_even", "GS Sort Scatter Even"),
                scatter_odd: make_pipeline("scatter_odd", "GS Sort Scatter Odd"),
            });
        }

        let render_key = GaussianRenderPipelineKey {
            depth_format: ctx.wgpu_ctx.depth_format,
            msaa_samples: ctx.wgpu_ctx.msaa_samples,
        };

        if self.render_pipeline.is_none() || self.render_pipeline_key != Some(render_key) {
            let shader_options = ShaderCompilationOptions::default();
            let (module, _) = ctx.shader_manager.get_or_compile(
                device,
                ShaderSource::File("entry/utility/gaussian_render"),
                &shader_options,
            );

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GS Render Pipeline Layout"),
                bind_group_layouts: &[Some(self.render_layout.as_ref().unwrap())],
                immediate_size: 0,
            });

            self.render_pipeline = Some(device.create_render_pipeline(
                &wgpu::RenderPipelineDescriptor {
                    label: Some("GS Render Pipeline"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: HDR_TEXTURE_FORMAT,
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::One,
                                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                    operation: wgpu::BlendOperation::Add,
                                },
                                alpha: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::One,
                                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                    operation: wgpu::BlendOperation::Add,
                                },
                            }),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: None,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: render_key.depth_format,
                        depth_write_enabled: Some(false),
                        depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: render_key.msaa_samples,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview_mask: None,
                    cache: None,
                },
            ));

            self.render_pipeline_key = Some(render_key);
        }
    }

    fn create_cloud_gpu_data(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cloud: &GaussianCloud,
    ) -> CloudGpuData {
        let num_points =
            u32::try_from(cloud.num_points).expect("Gaussian cloud exceeds u32 capacity");
        let sort_layout = sort_buffer_layout(cloud.num_points);
        let upload_count = cloud.num_points.max(1);

        let gaussian_buf = Tracked::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS Gaussian Data"),
            size: (upload_count * std::mem::size_of::<GaussianSplat>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        queue.write_buffer(&gaussian_buf, 0, bytemuck::cast_slice(&cloud.gaussians));

        let sh_buf = Tracked::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS SH Coefficients"),
            size: (upload_count * std::mem::size_of::<GaussianSHCoefficients>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        queue.write_buffer(&sh_buf, 0, bytemuck::cast_slice(&cloud.sh_coefficients));

        let camera_uniform_buf = Tracked::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS Camera Uniform"),
            size: std::mem::size_of::<GpuCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let render_settings_buf = Tracked::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS Render Settings"),
            size: std::mem::size_of::<GpuRenderSettings>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        CloudGpuData {
            num_points,
            sort_layout,
            gaussian_buf,
            sh_buf,
            camera_uniform_buf,
            render_settings_buf,
        }
    }

    fn update_cloud_uniforms(
        queue: &wgpu::Queue,
        data: &CloudGpuData,
        camera: &RenderCamera,
        cloud: &GaussianCloud,
        viewport_size: (u32, u32),
    ) {
        let view = camera.view_matrix;
        let proj = camera.projection_matrix;
        let view_inv = view.inverse();
        let proj_inv = proj.inverse();

        let focal_x = (proj.x_axis.x * viewport_size.0 as f32 * 0.5).abs();
        let focal_y = (proj.y_axis.y * viewport_size.1 as f32 * 0.5).abs();

        let camera_uniforms = GpuCameraUniforms {
            view: view.to_cols_array(),
            view_inv: view_inv.to_cols_array(),
            proj: proj.to_cols_array(),
            proj_inv: proj_inv.to_cols_array(),
            viewport: [viewport_size.0 as f32, viewport_size.1 as f32],
            focal: [focal_x, focal_y],
        };
        queue.write_buffer(
            &data.camera_uniform_buf,
            0,
            bytemuck::bytes_of(&camera_uniforms),
        );

        let render_settings = GpuRenderSettings {
            gaussian_scaling: 1.0,
            max_sh_deg: cloud.sh_degree,
            mip_splatting: u32::from(cloud.mip_splatting),
            kernel_size: cloud.kernel_size,
            scene_extent: cloud.scene_extent().max(1e-5),
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        queue.write_buffer(
            &data.render_settings_buf,
            0,
            bytemuck::bytes_of(&render_settings),
        );
    }

    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        active_color: TextureNodeId,
        active_depth: TextureNodeId,
    ) -> TextureNodeId {
        if !self.active || self.sorted_order.is_empty() {
            return active_color;
        }

        let preprocess_pipeline = self.preprocess_pipeline.as_ref();
        let sort_pipelines = self.sort_pipelines.as_ref();
        let render_pipeline = self.render_pipeline.as_ref();

        let preprocess_layout_g0 = self
            .preprocess_layout_g0
            .as_ref()
            .expect("GS preprocess layout G0 missing");
        let preprocess_layout_g1 = self
            .preprocess_layout_g1
            .as_ref()
            .expect("GS preprocess layout G1 missing");
        let preprocess_layout_g2 = self
            .preprocess_layout_g2
            .as_ref()
            .expect("GS preprocess layout G2 missing");
        let preprocess_layout_g3 = self
            .preprocess_layout_g3
            .as_ref()
            .expect("GS preprocess layout G3 missing");
        let sort_layout = self.sort_layout.as_ref().expect("GS sort layout missing");
        let render_layout = self
            .render_layout
            .as_ref()
            .expect("GS render layout missing");

        let cloud_buffers = ctx.graph.add_pass("GS_Compute", |builder| {
            let mut graph_buffers = Vec::with_capacity(self.sorted_order.len());
            let mut compute_states = Vec::with_capacity(self.sorted_order.len());

            for &cloud_index in &self.sorted_order {
                let (_, _, gpu) = &self.clouds[cloud_index];
                let upload_count = usize::try_from(gpu.num_points.max(1))
                    .expect("Gaussian point count exceeds usize capacity");
                let sort_key_buffer_size =
                    (gpu.sort_layout.padded_key_capacity * std::mem::size_of::<u32>()) as u64;

                let gaussian_buf = builder.read_external_buffer(
                    "GS_Gaussian_Data",
                    BufferDesc::new(
                        (upload_count * std::mem::size_of::<GaussianSplat>()) as u64,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    ),
                    &gpu.gaussian_buf,
                );
                let sh_buf = builder.read_external_buffer(
                    "GS_SH_Coefficients",
                    BufferDesc::new(
                        (upload_count * std::mem::size_of::<GaussianSHCoefficients>()) as u64,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    ),
                    &gpu.sh_buf,
                );
                let camera_uniform_buf = builder.read_external_buffer(
                    "GS_Camera_Uniform",
                    BufferDesc::new(
                        std::mem::size_of::<GpuCameraUniforms>() as u64,
                        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    ),
                    &gpu.camera_uniform_buf,
                );
                let render_settings_buf = builder.read_external_buffer(
                    "GS_Render_Settings",
                    BufferDesc::new(
                        std::mem::size_of::<GpuRenderSettings>() as u64,
                        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    ),
                    &gpu.render_settings_buf,
                );

                let splat_2d_buf = builder.create_buffer(
                    "GS_Splat_2D",
                    BufferDesc::new(
                        (upload_count * std::mem::size_of::<Splat2D>()) as u64,
                        wgpu::BufferUsages::STORAGE,
                    ),
                );
                let sort_infos_buf = builder.create_buffer(
                    "GS_Sort_Infos",
                    BufferDesc::new(
                        std::mem::size_of::<GpuSortInfos>() as u64,
                        wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    ),
                );
                let sort_dispatch_buf = builder.create_buffer(
                    "GS_Sort_Dispatch",
                    BufferDesc::new(
                        std::mem::size_of::<GpuDispatchIndirect>() as u64,
                        wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::INDIRECT,
                    ),
                );
                let sort_internal_buf = builder.create_buffer(
                    "GS_Sort_Internal",
                    BufferDesc::new(
                        (gpu.sort_layout.internal_buffer_words * std::mem::size_of::<u32>()) as u64,
                        wgpu::BufferUsages::STORAGE,
                    ),
                );
                let sort_depths_a_buf = builder.create_buffer(
                    "GS_Sort_Depths_A",
                    BufferDesc::new(sort_key_buffer_size, wgpu::BufferUsages::STORAGE),
                );
                let sort_depths_b_buf = builder.create_buffer(
                    "GS_Sort_Depths_B",
                    BufferDesc::new(sort_key_buffer_size, wgpu::BufferUsages::STORAGE),
                );
                let sort_indices_a_buf = builder.create_buffer(
                    "GS_Sort_Indices_A",
                    BufferDesc::new(sort_key_buffer_size, wgpu::BufferUsages::STORAGE),
                );
                let sort_indices_b_buf = builder.create_buffer(
                    "GS_Sort_Indices_B",
                    BufferDesc::new(sort_key_buffer_size, wgpu::BufferUsages::STORAGE),
                );
                let draw_indirect_buf = builder.create_buffer(
                    "GS_Draw_Indirect",
                    BufferDesc::new(
                        std::mem::size_of::<GpuDrawIndirect>() as u64,
                        wgpu::BufferUsages::INDIRECT
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    ),
                );

                let buffers = CloudGraphBuffers {
                    gaussian_buf,
                    sh_buf,
                    splat_2d_buf,
                    sort_infos_buf,
                    sort_dispatch_buf,
                    sort_internal_buf,
                    sort_depths_a_buf,
                    sort_depths_b_buf,
                    sort_indices_a_buf,
                    sort_indices_b_buf,
                    draw_indirect_buf,
                    camera_uniform_buf,
                    render_settings_buf,
                    num_points: gpu.num_points,
                    sort_infos_init: GpuSortInfos {
                        keys_size: 0,
                        padded_size: gpu.sort_layout.padded_key_capacity as u32,
                        passes: SORT_KEYVAL_PASSES,
                        even_pass: 0,
                        odd_pass: 0,
                    },
                    sort_dispatch_init: GpuDispatchIndirect {
                        dispatch_x: 0,
                        dispatch_y: 1,
                        dispatch_z: 1,
                    },
                    draw_indirect_init: GpuDrawIndirect {
                        vertex_count: SPLAT_VERTEX_COUNT,
                        instance_count: 0,
                        base_vertex: 0,
                        base_instance: 0,
                    },
                };

                graph_buffers.push(buffers);
                compute_states.push(CloudComputeState {
                    buffers,
                    preprocess_bg0: None,
                    preprocess_bg1: None,
                    preprocess_bg2: None,
                    preprocess_bg3: None,
                    sort_bg: None,
                });
            }

            let graph_buffers = builder.graph.alloc_slice(&graph_buffers);
            let compute_states = builder.graph.alloc_slice_mut(&compute_states);

            (
                GaussianComputePassNode {
                    preprocess_pipeline,
                    sort_pipelines,
                    preprocess_layout_g0,
                    preprocess_layout_g1,
                    preprocess_layout_g2,
                    preprocess_layout_g3,
                    sort_layout,
                    clouds: compute_states,
                },
                graph_buffers,
            )
        });

        ctx.graph.add_pass("GS_Render", |builder| {
            for &cloud in cloud_buffers {
                builder.read_buffer(cloud.splat_2d_buf);
                builder.read_buffer(cloud.sort_indices_a_buf);
                builder.read_buffer(cloud.draw_indirect_buf);
            }

            let color_out = builder.mutate_texture(active_color, "GS_Color");
            let _depth_in = builder.read_texture(active_depth);

            let mut render_states = Vec::with_capacity(cloud_buffers.len());
            for &cloud in cloud_buffers {
                render_states.push(CloudRenderState {
                    buffers: cloud,
                    render_bg: None,
                });
            }
            let render_states = builder.graph.alloc_slice_mut(&render_states);

            (
                GaussianRenderPassNode {
                    render_pipeline,
                    render_layout,
                    clouds: render_states,
                    color_target: color_out,
                    depth_target: active_depth,
                },
                color_out,
            )
        })
    }
}

struct GaussianComputePassNode<'a> {
    preprocess_pipeline: Option<&'a wgpu::ComputePipeline>,
    sort_pipelines: Option<&'a GaussianSortPipelines>,
    preprocess_layout_g0: &'a Tracked<wgpu::BindGroupLayout>,
    preprocess_layout_g1: &'a Tracked<wgpu::BindGroupLayout>,
    preprocess_layout_g2: &'a Tracked<wgpu::BindGroupLayout>,
    preprocess_layout_g3: &'a Tracked<wgpu::BindGroupLayout>,
    sort_layout: &'a Tracked<wgpu::BindGroupLayout>,
    clouds: &'a mut [CloudComputeState<'a>],
}

impl<'a> PassNode<'a> for GaussianComputePassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let PrepareContext {
            views,
            global_bind_group_cache: cache,
            device,
            queue,
            ..
        } = ctx;
        let device = *device;

        for cloud in self.clouds.iter_mut() {
            queue.write_buffer(
                views.get_buffer(cloud.buffers.sort_infos_buf),
                0,
                bytemuck::bytes_of(&cloud.buffers.sort_infos_init),
            );
            queue.write_buffer(
                views.get_buffer(cloud.buffers.sort_dispatch_buf),
                0,
                bytemuck::bytes_of(&cloud.buffers.sort_dispatch_init),
            );
            queue.write_buffer(
                views.get_buffer(cloud.buffers.draw_indirect_buf),
                0,
                bytemuck::bytes_of(&cloud.buffers.draw_indirect_init),
            );

            let camera_uniform = views.get_tracked_buffer(cloud.buffers.camera_uniform_buf);
            let gaussian = views.get_tracked_buffer(cloud.buffers.gaussian_buf);
            let sh = views.get_tracked_buffer(cloud.buffers.sh_buf);
            let splat_2d = views.get_tracked_buffer(cloud.buffers.splat_2d_buf);
            let sort_infos = views.get_tracked_buffer(cloud.buffers.sort_infos_buf);
            let sort_dispatch = views.get_tracked_buffer(cloud.buffers.sort_dispatch_buf);
            let sort_internal = views.get_tracked_buffer(cloud.buffers.sort_internal_buf);
            let sort_depths_a = views.get_tracked_buffer(cloud.buffers.sort_depths_a_buf);
            let sort_depths_b = views.get_tracked_buffer(cloud.buffers.sort_depths_b_buf);
            let sort_indices_a = views.get_tracked_buffer(cloud.buffers.sort_indices_a_buf);
            let sort_indices_b = views.get_tracked_buffer(cloud.buffers.sort_indices_b_buf);
            let render_settings = views.get_tracked_buffer(cloud.buffers.render_settings_buf);

            let preprocess_bg0 = cache.get_or_create_bg(
                BindGroupKey::new(self.preprocess_layout_g0.id())
                    .with_resource(camera_uniform.id()),
                || {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("GS Preprocess BG0"),
                        layout: self.preprocess_layout_g0,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(
                                views.get_buffer_binding(cloud.buffers.camera_uniform_buf),
                            ),
                        }],
                    })
                },
            );

            let preprocess_bg1 = cache.get_or_create_bg(
                BindGroupKey::new(self.preprocess_layout_g1.id())
                    .with_resource(gaussian.id())
                    .with_resource(sh.id())
                    .with_resource(splat_2d.id()),
                || {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("GS Preprocess BG1"),
                        layout: self.preprocess_layout_g1,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(
                                    views.get_buffer_binding(cloud.buffers.gaussian_buf),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Buffer(
                                    views.get_buffer_binding(cloud.buffers.sh_buf),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Buffer(
                                    views.get_buffer_binding(cloud.buffers.splat_2d_buf),
                                ),
                            },
                        ],
                    })
                },
            );

            let preprocess_bg2 = cache.get_or_create_bg(
                BindGroupKey::new(self.preprocess_layout_g2.id())
                    .with_resource(sort_infos.id())
                    .with_resource(sort_depths_a.id())
                    .with_resource(sort_indices_a.id())
                    .with_resource(sort_dispatch.id()),
                || {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("GS Preprocess BG2"),
                        layout: self.preprocess_layout_g2,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(
                                    views.get_buffer_binding(cloud.buffers.sort_infos_buf),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Buffer(
                                    views.get_buffer_binding(cloud.buffers.sort_depths_a_buf),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Buffer(
                                    views.get_buffer_binding(cloud.buffers.sort_indices_a_buf),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::Buffer(
                                    views.get_buffer_binding(cloud.buffers.sort_dispatch_buf),
                                ),
                            },
                        ],
                    })
                },
            );

            let preprocess_bg3 = cache.get_or_create_bg(
                BindGroupKey::new(self.preprocess_layout_g3.id())
                    .with_resource(render_settings.id()),
                || {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("GS Preprocess BG3"),
                        layout: self.preprocess_layout_g3,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(
                                views.get_buffer_binding(cloud.buffers.render_settings_buf),
                            ),
                        }],
                    })
                },
            );

            let sort_bg = cache.get_or_create_bg(
                BindGroupKey::new(self.sort_layout.id())
                    .with_resource(sort_infos.id())
                    .with_resource(sort_internal.id())
                    .with_resource(sort_depths_a.id())
                    .with_resource(sort_depths_b.id())
                    .with_resource(sort_indices_a.id())
                    .with_resource(sort_indices_b.id()),
                || {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("GS Sort BG"),
                        layout: self.sort_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(
                                    views.get_buffer_binding(cloud.buffers.sort_infos_buf),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Buffer(
                                    views.get_buffer_binding(cloud.buffers.sort_internal_buf),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Buffer(
                                    views.get_buffer_binding(cloud.buffers.sort_depths_a_buf),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::Buffer(
                                    views.get_buffer_binding(cloud.buffers.sort_depths_b_buf),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::Buffer(
                                    views.get_buffer_binding(cloud.buffers.sort_indices_a_buf),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::Buffer(
                                    views.get_buffer_binding(cloud.buffers.sort_indices_b_buf),
                                ),
                            },
                        ],
                    })
                },
            );

            cloud.preprocess_bg0 = Some(preprocess_bg0);
            cloud.preprocess_bg1 = Some(preprocess_bg1);
            cloud.preprocess_bg2 = Some(preprocess_bg2);
            cloud.preprocess_bg3 = Some(preprocess_bg3);
            cloud.sort_bg = Some(sort_bg);
        }
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let preprocess_pipeline = self
            .preprocess_pipeline
            .expect("GS preprocess pipeline missing");
        let sort_pipelines = self.sort_pipelines.expect("GS sort pipelines missing");

        for cloud in self.clouds.iter() {
            let preprocess_workgroups =
                (cloud.buffers.num_points + PREPROCESS_WG_SIZE - 1) / PREPROCESS_WG_SIZE;

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("GS Preprocess"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(preprocess_pipeline);
                cpass.set_bind_group(
                    0,
                    cloud.preprocess_bg0.expect("GS preprocess BG0 missing"),
                    &[],
                );
                cpass.set_bind_group(
                    1,
                    cloud.preprocess_bg1.expect("GS preprocess BG1 missing"),
                    &[],
                );
                cpass.set_bind_group(
                    2,
                    cloud.preprocess_bg2.expect("GS preprocess BG2 missing"),
                    &[],
                );
                cpass.set_bind_group(
                    3,
                    cloud.preprocess_bg3.expect("GS preprocess BG3 missing"),
                    &[],
                );
                cpass.dispatch_workgroups(preprocess_workgroups, 1, 1);
            }

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("GS Sort Zero Histograms"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&sort_pipelines.zero_histograms);
                cpass.set_bind_group(0, cloud.sort_bg.expect("GS sort BG missing"), &[]);
                cpass.dispatch_workgroups_indirect(
                    ctx.get_buffer(cloud.buffers.sort_dispatch_buf),
                    0,
                );
            }

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("GS Sort Histogram"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&sort_pipelines.calculate_histogram);
                cpass.set_bind_group(0, cloud.sort_bg.expect("GS sort BG missing"), &[]);
                cpass.dispatch_workgroups_indirect(
                    ctx.get_buffer(cloud.buffers.sort_dispatch_buf),
                    0,
                );
            }

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("GS Sort Prefix"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&sort_pipelines.prefix_histogram);
                cpass.set_bind_group(0, cloud.sort_bg.expect("GS sort BG missing"), &[]);
                cpass.dispatch_workgroups(SORT_KEYVAL_PASSES, 1, 1);
            }

            for pass_index in 0..SORT_KEYVAL_PASSES {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("GS Sort Scatter"),
                    timestamp_writes: None,
                });
                cpass.set_bind_group(0, cloud.sort_bg.expect("GS sort BG missing"), &[]);

                if (pass_index % 2) == 0 {
                    cpass.set_pipeline(&sort_pipelines.scatter_even);
                } else {
                    cpass.set_pipeline(&sort_pipelines.scatter_odd);
                }

                cpass.dispatch_workgroups_indirect(
                    ctx.get_buffer(cloud.buffers.sort_dispatch_buf),
                    0,
                );
            }

            encoder.copy_buffer_to_buffer(
                ctx.get_buffer(cloud.buffers.sort_infos_buf),
                0,
                ctx.get_buffer(cloud.buffers.draw_indirect_buf),
                4,
                4,
            );
        }
    }
}

struct GaussianRenderPassNode<'a> {
    render_pipeline: Option<&'a wgpu::RenderPipeline>,
    render_layout: &'a Tracked<wgpu::BindGroupLayout>,
    clouds: &'a mut [CloudRenderState<'a>],
    color_target: TextureNodeId,
    depth_target: TextureNodeId,
}

impl<'a> PassNode<'a> for GaussianRenderPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        let PrepareContext {
            views,
            global_bind_group_cache: cache,
            device,
            ..
        } = ctx;
        let device = *device;

        for cloud in self.clouds.iter_mut() {
            let splat_2d = views.get_tracked_buffer(cloud.buffers.splat_2d_buf);
            let sort_indices = views.get_tracked_buffer(cloud.buffers.sort_indices_a_buf);

            let render_bg = cache.get_or_create_bg(
                BindGroupKey::new(self.render_layout.id())
                    .with_resource(splat_2d.id())
                    .with_resource(sort_indices.id()),
                || {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("GS Render BG"),
                        layout: self.render_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(
                                    views.get_buffer_binding(cloud.buffers.splat_2d_buf),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Buffer(
                                    views.get_buffer_binding(cloud.buffers.sort_indices_a_buf),
                                ),
                            },
                        ],
                    })
                },
            );

            cloud.render_bg = Some(render_bg);
        }
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let render_pipeline = self.render_pipeline.expect("GS render pipeline missing");

        let color_attachment = ctx
            .get_color_attachment(self.color_target, RenderTargetOps::Load, None)
            .expect("GS color target missing");
        let depth_attachment = ctx.get_depth_stencil_attachment(self.depth_target, 0.0);

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("GS Render"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: depth_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(render_pipeline);

        for cloud in self.clouds.iter() {
            rpass.set_bind_group(0, cloud.render_bg.expect("GS render BG missing"), &[]);
            rpass.draw_indirect(ctx.get_buffer(cloud.buffers.draw_indirect_buf), 0);
        }
    }
}
