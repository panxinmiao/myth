//! RDG Gaussian Splatting Pass — 3D Gaussian Splatting Rendering
//!
//! Implements Myth's 3DGS path in three explicit stages:
//!
//! 1. Preprocess — project 3D Gaussians to 2D screen-space splats, evaluate
//!    SH colour, cull invisible points, and emit reverse-Z depth keys.
//! 2. Sort preparation + GPU radix sort — pad the inactive tail of the active
//!    dispatch window with sentinels, then sort front-to-back to match the
//!    accumulation blend equation.
//! 3. Render — draw storage-buffer-pulled triangle strips into an isolated
//!    non-linear accumulation target, then composite that result back into
//!    Myth's linear HDR scene colour.
//!
//! Multiple Gaussian clouds are supported simultaneously. Each cloud owns its
//! own preprocess buffers, sort buffers, and indirect draw buffer.

use std::sync::Arc;

use glam::{Mat4, Vec3, Vec3A};

use crate::HDR_TEXTURE_FORMAT;
use crate::core::gpu::{CommonSampler, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    BufferDesc, BufferNodeId, ExecuteContext, ExtractContext, PassNode, PrepareContext,
    RenderTargetOps, TextureDesc, TextureNodeId,
};
use crate::pipeline::{ShaderCompilationOptions, ShaderSource};
use myth_resources::GaussianCloudHandle;
use myth_resources::gaussian_splat::{GaussianCloud, GaussianSHCoefficients};
use myth_resources::image::ColorSpace;

const PREPROCESS_WG_SIZE: u32 = 256;

const SORT_HISTOGRAM_WG_SIZE: u32 = 256;
const SORT_PREFIX_WG_SIZE: u32 = 128;
const SORT_SCATTER_WG_SIZE: u32 = 256;
const SORT_RADIX_LOG2: u32 = 8;
const SORT_RADIX_SIZE: usize = 1 << SORT_RADIX_LOG2;
const SORT_KEYVAL_PASSES: u32 = 4;
const SORT_MIN_BLOCK_ROWS: usize = 1;
const SORT_MAX_BLOCK_ROWS: usize = 15;

const SPLAT_VERTEX_COUNT: u32 = 4;
const SORT_DISPATCH_INDIRECT_OFFSET: u64 = std::mem::size_of::<[u32; 3]>() as u64;
const GS_ACCUMULATION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCompositeSettings {
    flags: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuRenderSettings {
    gaussian_scaling: f32,
    max_sh_deg: u32,
    mip_splatting: u32,
    kernel_size: f32,
    scene_extent: f32,
    color_space_flag: u32,
    opacity_compensation: f32,
    _pad0: u32,
    model_matrix: [f32; 16],
    model_inv_matrix: [f32; 16],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSortInfos {
    keys_size: u32,
    padded_size: u32,
    passes: u32,
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuGaussianCore {
    x: f32,
    y: f32,
    z: f32,
    opacity: u32,
    sh_idx: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuGaussianCovariance {
    cov: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSplatGeometry {
    pos: [f32; 2],
    v0: u32,
    v1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSplatAppearance {
    depth: f32,
    color_rg: u32,
    color_ba: u32,
    _pad: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GaussianRenderPipelineKey {
    depth_format: wgpu::TextureFormat,
    msaa_samples: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GaussianCompositePipelineKey {
    msaa_samples: u32,
}

#[derive(Clone, Copy, Debug)]
struct SortBufferLayout {
    padded_key_capacity: usize,
    internal_buffer_words: usize,
}

/// Device-specialized radix-sort parameters shared by both CPU-side buffer
/// sizing and WGSL pipeline overrides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GaussianSortConfig {
    subgroup_size: u32,
    block_rows: usize,
    keys_per_wg: usize,
    rs_mem_dwords: usize,
}

impl GaussianSortConfig {
    fn for_device(device: &wgpu::Device) -> Self {
        let subgroup_size = normalize_sort_subgroup_size(device.adapter_info().subgroup_min_size);
        let max_workgroup_storage_bytes =
            device.limits().max_compute_workgroup_storage_size as usize;
        let max_workgroup_storage_words = max_workgroup_storage_bytes / std::mem::size_of::<u32>();
        let fixed_workgroup_words = SORT_RADIX_SIZE * 2;
        let block_rows = max_workgroup_storage_words.saturating_sub(fixed_workgroup_words)
            / SORT_SCATTER_WG_SIZE as usize;
        let block_rows = block_rows.clamp(SORT_MIN_BLOCK_ROWS, SORT_MAX_BLOCK_ROWS);
        let keys_per_wg = SORT_HISTOGRAM_WG_SIZE as usize * block_rows;
        let rs_mem_dwords = SORT_RADIX_SIZE + SORT_SCATTER_WG_SIZE as usize * block_rows;

        debug_assert!(
            (rs_mem_dwords + SORT_RADIX_SIZE) * std::mem::size_of::<u32>()
                <= max_workgroup_storage_bytes
        );

        Self {
            subgroup_size,
            block_rows,
            keys_per_wg,
            rs_mem_dwords,
        }
    }

    fn buffer_layout(self, key_count: usize) -> SortBufferLayout {
        let scatter_blocks_ru = self.sort_scatter_blocks(key_count).max(1);
        let padded_key_capacity = self.padded_sort_key_capacity(key_count);
        let internal_buffer_words =
            (SORT_KEYVAL_PASSES as usize + scatter_blocks_ru) * SORT_RADIX_SIZE;

        SortBufferLayout {
            padded_key_capacity,
            internal_buffer_words,
        }
    }

    fn padded_sort_key_capacity(self, key_count: usize) -> usize {
        let key_count = key_count.max(1);
        ((key_count + self.keys_per_wg) / self.keys_per_wg + 1) * self.keys_per_wg
    }

    fn sort_scatter_blocks(self, key_count: usize) -> usize {
        let scatter_block_kvs = SORT_HISTOGRAM_WG_SIZE as usize * self.block_rows;
        (key_count.max(1) + scatter_block_kvs - 1) / scatter_block_kvs
    }

    fn block_rows_u32(self) -> u32 {
        self.block_rows as u32
    }

    fn keys_per_wg_u32(self) -> u32 {
        self.keys_per_wg as u32
    }

    fn rs_mem_dwords_u32(self) -> u32 {
        self.rs_mem_dwords as u32
    }
}

fn normalize_sort_subgroup_size(reported: u32) -> u32 {
    match reported {
        32.. => 32,
        16..=31 => 16,
        8..=15 => 8,
        _ => 32,
    }
}

fn build_sort_shader_options(sort_config: GaussianSortConfig) -> ShaderCompilationOptions {
    let subgroup_size = sort_config.subgroup_size;

    let mut options = ShaderCompilationOptions::default();
    options.add_define("HISTOGRAM_SG_SIZE", &subgroup_size.to_string());
    options.add_define("HISTOGRAM_WG_SIZE", &SORT_HISTOGRAM_WG_SIZE.to_string());
    options.add_define("PREFIX_WG_SIZE", &SORT_PREFIX_WG_SIZE.to_string());
    options.add_define("SCATTER_WG_SIZE", &SORT_SCATTER_WG_SIZE.to_string());
    options.add_define("RS_RADIX_LOG2", &SORT_RADIX_LOG2.to_string());
    options.add_define("RS_RADIX_SIZE", &SORT_RADIX_SIZE.to_string());
    options.add_define("RS_KEYVAL_SIZE", &SORT_KEYVAL_PASSES.to_string());
    options
}

struct GaussianSortPipelines {
    pad_keys: wgpu::ComputePipeline,
    zero_histograms: wgpu::ComputePipeline,
    calculate_histogram: wgpu::ComputePipeline,
    prefix_histogram: wgpu::ComputePipeline,
    scatter_0: wgpu::ComputePipeline,
    scatter_1: wgpu::ComputePipeline,
    scatter_2: wgpu::ComputePipeline,
    scatter_3: wgpu::ComputePipeline,
}

struct CloudGpuData {
    num_points: u32,
    num_sh_coefficients: u32,
    sort_layout: SortBufferLayout,

    gaussian_core_buf: Tracked<wgpu::Buffer>,
    gaussian_cov_buf: Tracked<wgpu::Buffer>,
    sh_buf: Tracked<wgpu::Buffer>,
    render_settings_buf: Tracked<wgpu::Buffer>,
}

#[derive(Clone, Copy)]
struct CloudGraphBuffers {
    gaussian_core_buf: BufferNodeId,
    gaussian_cov_buf: BufferNodeId,
    sh_buf: BufferNodeId,
    splat_geom_buf: BufferNodeId,
    splat_attr_buf: BufferNodeId,
    sort_infos_buf: BufferNodeId,
    sort_dispatch_buf: BufferNodeId,
    sort_internal_buf: BufferNodeId,
    sort_depths_a_buf: BufferNodeId,
    sort_depths_b_buf: BufferNodeId,
    sort_indices_a_buf: BufferNodeId,
    sort_indices_b_buf: BufferNodeId,
    draw_indirect_buf: BufferNodeId,
    render_settings_buf: BufferNodeId,
    num_points: u32,
    sort_infos_init: GpuSortInfos,
    draw_indirect_init: GpuDrawIndirect,
}

#[derive(Clone, Copy)]
struct CloudComputeState<'a> {
    buffers: CloudGraphBuffers,
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
    preprocess_global_layout_id: Option<u64>,
    sort_config: Option<GaussianSortConfig>,
    sort_pipelines: Option<GaussianSortPipelines>,
    render_pipeline: Option<wgpu::RenderPipeline>,
    render_pipeline_key: Option<GaussianRenderPipelineKey>,
    composite_pipeline: Option<wgpu::RenderPipeline>,
    composite_pipeline_key: Option<GaussianCompositePipelineKey>,

    preprocess_layout_g1: Option<Tracked<wgpu::BindGroupLayout>>,
    preprocess_layout_g2: Option<Tracked<wgpu::BindGroupLayout>>,
    preprocess_layout_g3: Option<Tracked<wgpu::BindGroupLayout>>,
    sort_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    render_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    composite_layout: Option<Tracked<wgpu::BindGroupLayout>>,
    composite_settings_buf: Option<Tracked<wgpu::Buffer>>,

    clouds: Vec<(GaussianCloudHandle, u64, CloudGpuData)>,
    sorted_order: Vec<usize>,
    active: bool,
    composite_input_is_srgb: bool,
    mixed_color_space_warned: bool,
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
            preprocess_global_layout_id: None,
            sort_config: None,
            sort_pipelines: None,
            render_pipeline: None,
            render_pipeline_key: None,
            composite_pipeline: None,
            composite_pipeline_key: None,
            preprocess_layout_g1: None,
            preprocess_layout_g2: None,
            preprocess_layout_g3: None,
            sort_layout: None,
            render_layout: None,
            composite_layout: None,
            composite_settings_buf: None,
            clouds: Vec::new(),
            sorted_order: Vec::new(),
            active: false,
            composite_input_is_srgb: true,
            mixed_color_space_warned: false,
        }
    }

    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        cloud_entries: &[(GaussianCloudHandle, Arc<GaussianCloud>, Mat4)],
    ) {
        if cloud_entries.is_empty() {
            self.sorted_order.clear();
            self.active = false;
            return;
        }

        self.ensure_layouts(ctx.device);
        self.ensure_pipelines(ctx);

        if self.composite_settings_buf.is_none() {
            self.composite_settings_buf = Some(Tracked::new(ctx.device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("GS Composite Settings"),
                    size: std::mem::size_of::<GpuCompositeSettings>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            )));
        }

        let first_color_space = cloud_entries[0].1.color_space;
        let mixed_color_space = cloud_entries
            .iter()
            .any(|(_, cloud, _)| cloud.color_space != first_color_space);
        if mixed_color_space && !self.mixed_color_space_warned {
            log::warn!(
                "Gaussian splatting accumulation expects a consistent cloud color space; mixed clouds will be composited using {:?}",
                first_color_space
            );
            self.mixed_color_space_warned = true;
        }
        self.composite_input_is_srgb = matches!(first_color_space, ColorSpace::Srgb);
        let composite_settings = GpuCompositeSettings {
            flags: [u32::from(self.composite_input_is_srgb), 0, 0, 0],
        };
        ctx.queue.write_buffer(
            self.composite_settings_buf
                .as_ref()
                .expect("GS composite settings buffer missing"),
            0,
            bytemuck::bytes_of(&composite_settings),
        );

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

        for (handle, cloud, model_matrix) in cloud_entries {
            if let Some((_, _, gpu_data)) = self
                .clouds
                .iter()
                .find(|(existing_handle, _, _)| *existing_handle == *handle)
            {
                Self::update_cloud_uniforms(ctx.queue, gpu_data, cloud, *model_matrix);
            }
        }

        let camera_position = ctx.render_camera.position;
        let mut cloud_order: Vec<(usize, f32)> = cloud_entries
            .iter()
            .filter_map(|(handle, cloud, model_matrix)| {
                let cloud_index = self
                    .clouds
                    .iter()
                    .position(|(existing_handle, _, _)| existing_handle == handle)?;
                let local_center = Vec3::new(cloud.center.x, cloud.center.y, cloud.center.z);
                let world_center = Vec3A::from(model_matrix.transform_point3(local_center));
                let distance_sq = camera_position.distance_squared(world_center);
                Some((cloud_index, distance_sq))
            })
            .collect();

        cloud_order.sort_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.sorted_order = cloud_order.into_iter().map(|(index, _)| index).collect();
        self.active = !self.sorted_order.is_empty();
    }

    fn ensure_layouts(&mut self, device: &wgpu::Device) {
        if self.preprocess_layout_g1.is_some() {
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

        self.preprocess_layout_g1 = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GS Preprocess G1 (Gaussians + SH + Split Splats)"),
                entries: &[
                    storage_ro_entry(0, cs),
                    storage_ro_entry(1, cs),
                    storage_ro_entry(2, cs),
                    storage_rw_entry(3, cs),
                    storage_rw_entry(4, cs),
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
                    // storage_rw_entry(3, cs),
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
                entries: &[
                    storage_ro_entry(0, vs),
                    storage_ro_entry(1, vs),
                    storage_ro_entry(2, vs),
                ],
            },
        )));

        self.composite_layout = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GS Composite"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    uniform_entry(2, wgpu::ShaderStages::FRAGMENT),
                ],
            },
        )));
    }

    fn ensure_pipelines(&mut self, ctx: &mut ExtractContext) {
        let device = ctx.device;
        let sort_config = *self
            .sort_config
            .get_or_insert_with(|| GaussianSortConfig::for_device(device));
        let global_state_key = (ctx.render_state.id, ctx.extracted_scene.scene_id);
        let gpu_world = ctx
            .resource_manager
            .get_global_state(global_state_key.0, global_state_key.1)
            .expect("GS preprocess requires a global render-state bind group");

        if self.preprocess_pipeline.is_none()
            || self.preprocess_global_layout_id != Some(gpu_world.layout_id)
        {
            let mut shader_options = ShaderCompilationOptions::default();
            shader_options.inject_code("binding_code", &gpu_world.binding_wgsl);
            let (module, _) = ctx.shader_manager.get_or_compile(
                device,
                ShaderSource::File("entry/utility/gaussian_preprocess"),
                &shader_options,
            );

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GS Preprocess Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&gpu_world.layout),
                    Some(self.preprocess_layout_g1.as_ref().unwrap()),
                    Some(self.preprocess_layout_g2.as_ref().unwrap()),
                    Some(self.preprocess_layout_g3.as_ref().unwrap()),
                ],
                immediate_size: 0,
            });

            let preprocess_constants = [(
                "GS_SORT_KEYS_PER_WG",
                f64::from(sort_config.keys_per_wg_u32()),
            )];
            let preprocess_compilation_options = wgpu::PipelineCompilationOptions {
                constants: &preprocess_constants,
                ..Default::default()
            };

            self.preprocess_pipeline = Some(device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("GS Preprocess Pipeline"),
                    layout: Some(&layout),
                    module,
                    entry_point: Some("main"),
                    compilation_options: preprocess_compilation_options,
                    cache: None,
                },
            ));

            self.preprocess_global_layout_id = Some(gpu_world.layout_id);
        }

        if self.sort_pipelines.is_none() {
            let shader_options = build_sort_shader_options(sort_config);

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GS Sort Pipeline Layout"),
                bind_group_layouts: &[Some(self.sort_layout.as_ref().unwrap())],
                immediate_size: 0,
            });

            let pad_constants = [(
                "RS_HISTOGRAM_BLOCK_ROWS",
                f64::from(sort_config.block_rows_u32()),
            )];
            let pad_compilation_options = wgpu::PipelineCompilationOptions {
                constants: &pad_constants,
                ..Default::default()
            };
            let sort_constants = [
                (
                    "RS_HISTOGRAM_BLOCK_ROWS",
                    f64::from(sort_config.block_rows_u32()),
                ),
                (
                    "RS_SCATTER_BLOCK_ROWS",
                    f64::from(sort_config.block_rows_u32()),
                ),
                ("RS_MEM_DWORDS", f64::from(sort_config.rs_mem_dwords_u32())),
            ];
            let sort_compilation_options = wgpu::PipelineCompilationOptions {
                constants: &sort_constants,
                ..Default::default()
            };

            let pad_keys = {
                let (pad_module, _) = ctx.shader_manager.get_or_compile(
                    device,
                    ShaderSource::File("entry/utility/gs_pad_sort_keys"),
                    &shader_options,
                );
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("GS Sort Pad Keys"),
                    layout: Some(&layout),
                    module: pad_module,
                    entry_point: Some("main"),
                    compilation_options: pad_compilation_options,
                    cache: None,
                })
            };

            let (
                zero_histograms,
                calculate_histogram,
                prefix_histogram,
                scatter_0,
                scatter_1,
                scatter_2,
                scatter_3,
            ) = {
                let (sort_module, _) = ctx.shader_manager.get_or_compile(
                    device,
                    ShaderSource::File("entry/utility/gs_radix_sort"),
                    &shader_options,
                );
                let make_pipeline = |entry_point: &str, label: &str| {
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout: Some(&layout),
                        module: sort_module,
                        entry_point: Some(entry_point),
                        compilation_options: sort_compilation_options.clone(),
                        cache: None,
                    })
                };

                (
                    make_pipeline("zero_histograms", "GS Sort Zero Histograms"),
                    make_pipeline("calculate_histogram", "GS Sort Calculate Histogram"),
                    make_pipeline("prefix_histogram", "GS Sort Prefix Histogram"),
                    make_pipeline("scatter_pass_0", "GS Sort Scatter Pass 0"),
                    make_pipeline("scatter_pass_1", "GS Sort Scatter Pass 1"),
                    make_pipeline("scatter_pass_2", "GS Sort Scatter Pass 2"),
                    make_pipeline("scatter_pass_3", "GS Sort Scatter Pass 3"),
                )
            };

            self.sort_pipelines = Some(GaussianSortPipelines {
                pad_keys,
                zero_histograms,
                calculate_histogram,
                prefix_histogram,
                scatter_0,
                scatter_1,
                scatter_2,
                scatter_3,
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
                            format: GS_ACCUMULATION_FORMAT,
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                                    dst_factor: wgpu::BlendFactor::One,
                                    operation: wgpu::BlendOperation::Add,
                                },
                                alpha: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                                    dst_factor: wgpu::BlendFactor::One,
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

        let composite_key = GaussianCompositePipelineKey {
            msaa_samples: ctx.wgpu_ctx.msaa_samples,
        };

        if self.composite_pipeline.is_none() || self.composite_pipeline_key != Some(composite_key) {
            let shader_options = ShaderCompilationOptions::default();
            let (module, _) = ctx.shader_manager.get_or_compile(
                device,
                ShaderSource::File("entry/utility/gs_composite"),
                &shader_options,
            );

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GS Composite Pipeline Layout"),
                bind_group_layouts: &[Some(self.composite_layout.as_ref().unwrap())],
                immediate_size: 0,
            });

            self.composite_pipeline = Some(device.create_render_pipeline(
                &wgpu::RenderPipelineDescriptor {
                    label: Some("GS Composite Pipeline"),
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
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: None,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: composite_key.msaa_samples,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview_mask: None,
                    cache: None,
                },
            ));

            self.composite_pipeline_key = Some(composite_key);
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
        let num_sh_coefficients = u32::try_from(cloud.sh_coefficients.len())
            .expect("Gaussian SH coefficient table exceeds u32 capacity");
        let sort_layout = self
            .sort_config
            .expect("GS sort config must exist before GPU upload")
            .buffer_layout(cloud.num_points);
        let upload_count = cloud.num_points.max(1);
        let sh_upload_count = cloud.sh_coefficients.len().max(1);

        let mut gaussian_cores = Vec::with_capacity(cloud.gaussians.len());
        let mut gaussian_covariances = Vec::with_capacity(cloud.gaussians.len());
        for gaussian in &cloud.gaussians {
            gaussian_cores.push(GpuGaussianCore {
                x: gaussian.x,
                y: gaussian.y,
                z: gaussian.z,
                opacity: gaussian.opacity,
                sh_idx: gaussian.sh_idx,
            });
            gaussian_covariances.push(GpuGaussianCovariance { cov: gaussian.cov });
        }

        let gaussian_core_buf = Tracked::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS Gaussian Core Data"),
            size: (upload_count * std::mem::size_of::<GpuGaussianCore>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        queue.write_buffer(&gaussian_core_buf, 0, bytemuck::cast_slice(&gaussian_cores));

        let gaussian_cov_buf = Tracked::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS Gaussian Covariance Data"),
            size: (upload_count * std::mem::size_of::<GpuGaussianCovariance>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        queue.write_buffer(
            &gaussian_cov_buf,
            0,
            bytemuck::cast_slice(&gaussian_covariances),
        );

        let sh_buf = Tracked::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS SH Coefficients"),
            size: (sh_upload_count * std::mem::size_of::<GaussianSHCoefficients>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        queue.write_buffer(&sh_buf, 0, bytemuck::cast_slice(&cloud.sh_coefficients));

        let render_settings_buf = Tracked::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS Render Settings"),
            size: std::mem::size_of::<GpuRenderSettings>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        CloudGpuData {
            num_points,
            num_sh_coefficients,
            sort_layout,
            gaussian_core_buf,
            gaussian_cov_buf,
            sh_buf,
            render_settings_buf,
        }
    }

    fn update_cloud_uniforms(
        queue: &wgpu::Queue,
        data: &CloudGpuData,
        cloud: &GaussianCloud,
        model_matrix: Mat4,
    ) {
        let model_inv_matrix = model_matrix.inverse();

        let render_settings = GpuRenderSettings {
            gaussian_scaling: 1.0,
            max_sh_deg: cloud.sh_degree,
            mip_splatting: u32::from(cloud.mip_splatting),
            kernel_size: cloud.kernel_size,
            scene_extent: cloud.scene_extent().max(1e-5),
            color_space_flag: match cloud.color_space {
                ColorSpace::Linear => 0,
                ColorSpace::Srgb => 1,
            },
            opacity_compensation: cloud.opacity_compensation,
            _pad0: 0,
            model_matrix: model_matrix.to_cols_array(),
            model_inv_matrix: model_inv_matrix.to_cols_array(),
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
        let composite_pipeline = self.composite_pipeline.as_ref();
        let composite_layout = self
            .composite_layout
            .as_ref()
            .expect("GS composite layout missing");
        let composite_settings_buf = self
            .composite_settings_buf
            .as_ref()
            .expect("GS composite settings buffer missing");

        let cloud_buffers = ctx.graph.add_pass("GS_Compute", |builder| {
            let mut graph_buffers = Vec::with_capacity(self.sorted_order.len());
            let mut compute_states = Vec::with_capacity(self.sorted_order.len());

            for &cloud_index in &self.sorted_order {
                let (_, _, gpu) = &self.clouds[cloud_index];
                let upload_count = usize::try_from(gpu.num_points.max(1))
                    .expect("Gaussian point count exceeds usize capacity");
                let sort_key_buffer_size =
                    (gpu.sort_layout.padded_key_capacity * std::mem::size_of::<u32>()) as u64;

                let gaussian_core_buf = builder.read_external_buffer(
                    "GS_Gaussian_Core_Data",
                    BufferDesc::new(
                        (upload_count * std::mem::size_of::<GpuGaussianCore>()) as u64,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    ),
                    &gpu.gaussian_core_buf,
                );
                let gaussian_cov_buf = builder.read_external_buffer(
                    "GS_Gaussian_Covariance_Data",
                    BufferDesc::new(
                        (upload_count * std::mem::size_of::<GpuGaussianCovariance>()) as u64,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    ),
                    &gpu.gaussian_cov_buf,
                );
                let sh_buf = builder.read_external_buffer(
                    "GS_SH_Coefficients",
                    BufferDesc::new(
                        (usize::try_from(gpu.num_sh_coefficients.max(1))
                            .expect("Gaussian SH count exceeds usize capacity")
                            * std::mem::size_of::<GaussianSHCoefficients>())
                            as u64,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    ),
                    &gpu.sh_buf,
                );
                let render_settings_buf = builder.read_external_buffer(
                    "GS_Render_Settings",
                    BufferDesc::new(
                        std::mem::size_of::<GpuRenderSettings>() as u64,
                        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    ),
                    &gpu.render_settings_buf,
                );

                let splat_geom_buf = builder.create_buffer(
                    "GS_Splat_Geometry",
                    BufferDesc::new(
                        (upload_count * std::mem::size_of::<GpuSplatGeometry>()) as u64,
                        wgpu::BufferUsages::STORAGE,
                    ),
                );
                let splat_attr_buf = builder.create_buffer(
                    "GS_Splat_Appearance",
                    BufferDesc::new(
                        (upload_count * std::mem::size_of::<GpuSplatAppearance>()) as u64,
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
                        std::mem::size_of::<[u32; 3]>() as u64,
                        wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC
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
                    gaussian_core_buf,
                    gaussian_cov_buf,
                    sh_buf,
                    splat_geom_buf,
                    splat_attr_buf,
                    sort_infos_buf,
                    sort_dispatch_buf,
                    sort_internal_buf,
                    sort_depths_a_buf,
                    sort_depths_b_buf,
                    sort_indices_a_buf,
                    sort_indices_b_buf,
                    draw_indirect_buf,
                    render_settings_buf,
                    num_points: gpu.num_points,
                    sort_infos_init: GpuSortInfos {
                        keys_size: 0,
                        padded_size: gpu.sort_layout.padded_key_capacity as u32,
                        passes: SORT_KEYVAL_PASSES,
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
                    preprocess_layout_g1,
                    preprocess_layout_g2,
                    preprocess_layout_g3,
                    sort_layout,
                    clouds: compute_states,
                },
                graph_buffers,
            )
        });

        let gs_accumulation = ctx.graph.add_pass("GS_Render", |builder| {
            for &cloud in cloud_buffers {
                builder.read_buffer(cloud.splat_geom_buf);
                builder.read_buffer(cloud.splat_attr_buf);
                builder.read_buffer(cloud.sort_indices_a_buf);
                builder.read_buffer(cloud.draw_indirect_buf);
            }

            let _depth_in = builder.read_texture(active_depth);

            let accumulation_desc = TextureDesc::new_2d(
                ctx.frame_config.width,
                ctx.frame_config.height,
                GS_ACCUMULATION_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            );
            let accumulation_resolved =
                builder.create_texture("GS_Accumulation", accumulation_desc);
            let accumulation_target = if ctx.frame_config.msaa_samples > 1 {
                builder.create_texture(
                    "GS_Accumulation_MSAA",
                    TextureDesc::new(
                        ctx.frame_config.width,
                        ctx.frame_config.height,
                        1,
                        1,
                        ctx.frame_config.msaa_samples,
                        wgpu::TextureDimension::D2,
                        GS_ACCUMULATION_FORMAT,
                        wgpu::TextureUsages::RENDER_ATTACHMENT,
                    ),
                )
            } else {
                accumulation_resolved
            };

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
                    color_target: accumulation_target,
                    resolve_target: (ctx.frame_config.msaa_samples > 1)
                        .then_some(accumulation_resolved),
                    depth_target: active_depth,
                },
                accumulation_resolved,
            )
        });

        ctx.graph.add_pass("GS_Composite", |builder| {
            builder.read_texture(gs_accumulation);
            let composite_settings = builder.read_external_buffer(
                "GS_Composite_Settings",
                BufferDesc::new(
                    std::mem::size_of::<GpuCompositeSettings>() as u64,
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                ),
                composite_settings_buf,
            );
            let color_out = builder.mutate_texture(active_color, "GS_Composite_Color");

            (
                GaussianCompositePassNode {
                    composite_pipeline,
                    composite_layout,
                    accumulation_tex: gs_accumulation,
                    composite_settings_buf: composite_settings,
                    color_target: color_out,
                    composite_bg: None,
                },
                color_out,
            )
        })
    }
}

struct GaussianComputePassNode<'a> {
    preprocess_pipeline: Option<&'a wgpu::ComputePipeline>,
    sort_pipelines: Option<&'a GaussianSortPipelines>,
    preprocess_layout_g1: &'a Tracked<wgpu::BindGroupLayout>,
    preprocess_layout_g2: &'a Tracked<wgpu::BindGroupLayout>,
    preprocess_layout_g3: &'a Tracked<wgpu::BindGroupLayout>,
    sort_layout: &'a Tracked<wgpu::BindGroupLayout>,
    clouds: &'a mut [CloudComputeState<'a>],
}

impl<'a> PassNode<'a> for GaussianComputePassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        for cloud in self.clouds.iter_mut() {
            ctx.queue.write_buffer(
                ctx.views.get_buffer(cloud.buffers.sort_infos_buf),
                0,
                bytemuck::bytes_of(&cloud.buffers.sort_infos_init),
            );
            ctx.queue.write_buffer(
                ctx.views.get_buffer(cloud.buffers.draw_indirect_buf),
                0,
                bytemuck::bytes_of(&cloud.buffers.draw_indirect_init),
            );

            let preprocess_bg1 = ctx
                .build_bind_group(self.preprocess_layout_g1, Some("GS Preprocess BG1"))
                .bind_buffer(0, cloud.buffers.gaussian_core_buf)
                .bind_buffer(1, cloud.buffers.gaussian_cov_buf)
                .bind_buffer(2, cloud.buffers.sh_buf)
                .bind_buffer(3, cloud.buffers.splat_geom_buf)
                .bind_buffer(4, cloud.buffers.splat_attr_buf)
                .build();

            let preprocess_bg2 = ctx
                .build_bind_group(self.preprocess_layout_g2, Some("GS Preprocess BG2"))
                .bind_buffer(0, cloud.buffers.sort_infos_buf)
                .bind_buffer(1, cloud.buffers.sort_depths_a_buf)
                .bind_buffer(2, cloud.buffers.sort_indices_a_buf)
                .build();

            let preprocess_bg3 = ctx
                .build_bind_group(self.preprocess_layout_g3, Some("GS Preprocess BG3"))
                .bind_buffer(0, cloud.buffers.render_settings_buf)
                .build();

            let sort_bg = ctx
                .build_bind_group(self.sort_layout, Some("GS Sort BG"))
                .bind_buffer(0, cloud.buffers.sort_infos_buf)
                .bind_buffer(1, cloud.buffers.sort_internal_buf)
                .bind_buffer(2, cloud.buffers.sort_depths_a_buf)
                .bind_buffer(3, cloud.buffers.sort_depths_b_buf)
                .bind_buffer(4, cloud.buffers.sort_indices_a_buf)
                .bind_buffer(5, cloud.buffers.sort_indices_b_buf)
                .build();

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
        let global_bind_group = ctx.baked_lists.global_bind_group;

        for cloud in self.clouds.iter() {
            let preprocess_workgroups =
                (cloud.buffers.num_points + PREPROCESS_WG_SIZE - 1) / PREPROCESS_WG_SIZE;

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("GS Preprocess"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(preprocess_pipeline);
                cpass.set_bind_group(0, global_bind_group, &[]);
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

            encoder.copy_buffer_to_buffer(
                ctx.get_buffer(cloud.buffers.sort_infos_buf),
                SORT_DISPATCH_INDIRECT_OFFSET,
                ctx.get_buffer(cloud.buffers.sort_dispatch_buf),
                0,
                std::mem::size_of::<[u32; 3]>() as u64,
            );

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("GS Sort Pad Keys"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&sort_pipelines.pad_keys);
                cpass.set_bind_group(0, cloud.sort_bg.expect("GS sort BG missing"), &[]);
                cpass.dispatch_workgroups_indirect(
                    ctx.get_buffer(cloud.buffers.sort_dispatch_buf),
                    0,
                );
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

                match pass_index {
                    0 => cpass.set_pipeline(&sort_pipelines.scatter_0),
                    1 => cpass.set_pipeline(&sort_pipelines.scatter_1),
                    2 => cpass.set_pipeline(&sort_pipelines.scatter_2),
                    3 => cpass.set_pipeline(&sort_pipelines.scatter_3),
                    _ => unreachable!(),
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
    resolve_target: Option<TextureNodeId>,
    depth_target: TextureNodeId,
}

impl<'a> PassNode<'a> for GaussianRenderPassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        for cloud in self.clouds.iter_mut() {
            cloud.render_bg = Some(
                ctx.build_bind_group(self.render_layout, Some("GS Render BG"))
                    .bind_buffer(0, cloud.buffers.splat_geom_buf)
                    .bind_buffer(1, cloud.buffers.splat_attr_buf)
                    .bind_buffer(2, cloud.buffers.sort_indices_a_buf)
                    .build(),
            );
        }
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let render_pipeline = self.render_pipeline.expect("GS render pipeline missing");

        let color_attachment = ctx
            .get_color_attachment(
                self.color_target,
                RenderTargetOps::Clear(wgpu::Color::TRANSPARENT),
                self.resolve_target,
            )
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

struct GaussianCompositePassNode<'a> {
    composite_pipeline: Option<&'a wgpu::RenderPipeline>,
    composite_layout: &'a Tracked<wgpu::BindGroupLayout>,
    accumulation_tex: TextureNodeId,
    composite_settings_buf: BufferNodeId,
    color_target: TextureNodeId,
    composite_bg: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for GaussianCompositePassNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.composite_bg = Some(
            ctx.build_bind_group(self.composite_layout, Some("GS Composite BG"))
                .bind_texture(0, self.accumulation_tex)
                .bind_common_sampler(1, CommonSampler::NearestClamp)
                .bind_buffer(2, self.composite_settings_buf)
                .build(),
        );
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let composite_pipeline = self
            .composite_pipeline
            .expect("GS composite pipeline missing");
        let color_attachment = ctx
            .get_color_attachment(self.color_target, RenderTargetOps::Load, None)
            .expect("GS composite color target missing");

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("GS Composite"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(composite_pipeline);
        rpass.set_bind_group(0, self.composite_bg.expect("GS composite BG missing"), &[]);
        rpass.draw(0..3, 0..1);
    }
}
