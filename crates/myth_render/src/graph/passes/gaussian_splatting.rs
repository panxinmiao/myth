//! RDG Gaussian Splatting Pass — 3D Gaussian Splatting Rendering
//!
//! Implements Myth's 3DGS path in three explicit stages:
//!
//! 1. Preprocess — project 3D Gaussians to 2D screen-space splats, evaluate
//!    SH colour, cull invisible points, and emit reverse-Z depth keys.
//! 2. Portable GPU radix sort — build per-workgroup histograms, scan them with
//!    a hierarchical Blelloch prefix sum, then stably scatter front-to-back.
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
use crate::pipeline::{ShaderCompilationOptions, ShaderManager, ShaderSource};
use myth_resources::GaussianCloudHandle;
use myth_resources::gaussian_splat::{
    GaussianCloud, GaussianSHCoefficients, GaussianSplat, Splat2D,
};
use myth_resources::image::ColorSpace;

const PREPROCESS_WG_SIZE: u32 = 256;

const SORT_WG_SIZE_X: u32 = 16;
const SORT_WG_SIZE_Y: u32 = 16;
const SORT_THREADS_PER_WG: usize = (SORT_WG_SIZE_X * SORT_WG_SIZE_Y) as usize;
const SORT_ITEMS_PER_THREAD: usize = 8;
const SORT_KEYS_PER_WG: usize = SORT_THREADS_PER_WG * SORT_ITEMS_PER_THREAD;
const SORT_RADIX_BITS: u32 = 4;
const SORT_RADIX_SIZE: usize = 1 << SORT_RADIX_BITS;
const SORT_PASSES: usize = 32 / SORT_RADIX_BITS as usize;
const SORT_SCAN_ITEMS_PER_WG: usize = SORT_THREADS_PER_WG * 2;
const SORT_MAX_SCAN_LEVELS: usize = 4;

const _: () = {
    assert!(SORT_SCAN_ITEMS_PER_WG == SORT_THREADS_PER_WG * 2);
    assert!(SORT_SCAN_ITEMS_PER_WG.is_power_of_two());
    assert!(SORT_THREADS_PER_WG.is_multiple_of(32));
    assert!(SORT_RADIX_SIZE <= SORT_THREADS_PER_WG);
    assert!(SORT_RADIX_SIZE.is_power_of_two());
    assert!(SORT_PASSES.is_multiple_of(2));
};

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
    max_workgroups: u32,
    scan_levels: u32,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GaussianCompositePipelineKey {
    msaa_samples: u32,
}

#[derive(Clone, Copy, Debug)]
struct SortBufferLayout {
    key_capacity: usize,
    max_workgroups: usize,
    histogram_words: usize,
    internal_buffer_words: usize,
    scan_level_count: usize,
    scan_workgroups: [u32; SORT_MAX_SCAN_LEVELS],
}

impl SortBufferLayout {
    fn for_key_count(key_count: usize) -> Self {
        let key_capacity = key_count.max(1);
        let max_workgroups = key_capacity.div_ceil(SORT_KEYS_PER_WG);
        let histogram_words = SORT_RADIX_SIZE
            .checked_mul(max_workgroups)
            .expect("Gaussian sort histogram size overflow");

        let mut scan_workgroups = [0; SORT_MAX_SCAN_LEVELS];
        let mut scan_level_count = 0;
        let mut level_words = histogram_words;
        let mut internal_buffer_words = histogram_words;

        loop {
            assert!(
                scan_level_count < SORT_MAX_SCAN_LEVELS,
                "Gaussian sort prefix scan exceeds supported hierarchy depth"
            );
            let workgroups = level_words.div_ceil(SORT_SCAN_ITEMS_PER_WG);
            scan_workgroups[scan_level_count] =
                u32::try_from(workgroups).expect("Gaussian sort scan dispatch exceeds u32");
            scan_level_count += 1;
            internal_buffer_words = internal_buffer_words
                .checked_add(workgroups)
                .expect("Gaussian sort scratch size overflow");

            if workgroups == 1 {
                break;
            }
            level_words = workgroups;
        }

        Self {
            key_capacity,
            max_workgroups,
            histogram_words,
            internal_buffer_words,
            scan_level_count,
            scan_workgroups,
        }
    }
}

fn build_sort_shader_options() -> ShaderCompilationOptions {
    let mut options = ShaderCompilationOptions::default();
    options.add_define("SORT_WG_SIZE_X", &SORT_WG_SIZE_X.to_string());
    options.add_define("SORT_WG_SIZE_Y", &SORT_WG_SIZE_Y.to_string());
    options.add_define("SORT_THREADS_PER_WG", &SORT_THREADS_PER_WG.to_string());
    options.add_define("SORT_ITEMS_PER_THREAD", &SORT_ITEMS_PER_THREAD.to_string());
    options.add_define("SORT_RADIX_BITS", &SORT_RADIX_BITS.to_string());
    options.add_define("SORT_RADIX_SIZE", &SORT_RADIX_SIZE.to_string());
    options.add_define(
        "SORT_SCAN_ITEMS_PER_WG",
        &SORT_SCAN_ITEMS_PER_WG.to_string(),
    );
    options
}

fn create_sort_pipelines(
    device: &wgpu::Device,
    shader_manager: &mut ShaderManager,
    sort_layout: &wgpu::BindGroupLayout,
) -> GaussianSortPipelines {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("GS Sort Pipeline Layout"),
        bind_group_layouts: &[Some(sort_layout)],
        immediate_size: 0,
    });
    let make_pipeline = |module: &wgpu::ShaderModule, entry_point: &str, label: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    };

    // Generate pass constants directly into distinct WGSL modules. In
    // particular, do not use pipeline override values here: Safari/WebKit can
    // incorrectly reuse a specialization across pipelines from one module.
    let mut radix_passes = Vec::with_capacity(SORT_PASSES);
    for pass in 0..SORT_PASSES {
        let mut options = build_sort_shader_options();
        options.add_define(
            "SORT_RADIX_SHIFT",
            &(u32::try_from(pass).expect("Gaussian radix pass exceeds u32") * SORT_RADIX_BITS)
                .to_string(),
        );
        options.add_define(
            "SORT_WRITE_KEYS",
            &u32::from(pass + 1 != SORT_PASSES).to_string(),
        );
        options.add_define("SORT_SCAN_LEVEL", "0");
        let (module, _) = shader_manager.get_or_compile(
            device,
            ShaderSource::File("entry/utility/3dgs/gs_radix_sort"),
            &options,
        );
        radix_passes.push(GaussianRadixPassPipelines {
            histogram: make_pipeline(
                module,
                "block_histogram",
                &format!("GS Sort Histogram {pass}"),
            ),
            scatter: make_pipeline(module, "ranked_scatter", &format!("GS Sort Scatter {pass}")),
        });
    }

    let mut prefix_scan = Vec::with_capacity(SORT_MAX_SCAN_LEVELS);
    let mut prefix_add = Vec::with_capacity(SORT_MAX_SCAN_LEVELS.saturating_sub(1));
    for level in 0..SORT_MAX_SCAN_LEVELS {
        let mut options = build_sort_shader_options();
        options.add_define("SORT_RADIX_SHIFT", "0");
        options.add_define("SORT_WRITE_KEYS", "1");
        options.add_define(
            "SORT_SCAN_LEVEL",
            &u32::try_from(level)
                .expect("Gaussian scan level exceeds u32")
                .to_string(),
        );
        let (module, _) = shader_manager.get_or_compile(
            device,
            ShaderSource::File("entry/utility/3dgs/gs_radix_sort"),
            &options,
        );
        prefix_scan.push(make_pipeline(
            module,
            "prefix_scan",
            &format!("GS Sort Prefix Scan {level}"),
        ));
        if level + 1 < SORT_MAX_SCAN_LEVELS {
            prefix_add.push(make_pipeline(
                module,
                "prefix_add",
                &format!("GS Sort Prefix Add {level}"),
            ));
        }
    }

    GaussianSortPipelines {
        radix_passes,
        prefix_scan,
        prefix_add,
    }
}

struct GaussianRadixPassPipelines {
    histogram: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
}

struct GaussianSortPipelines {
    radix_passes: Vec<GaussianRadixPassPipelines>,
    prefix_scan: Vec<wgpu::ComputePipeline>,
    prefix_add: Vec<wgpu::ComputePipeline>,
}

struct CloudGpuData {
    num_points: u32,
    num_sh_coefficients: u32,
    sort_layout: SortBufferLayout,

    gaussian_buf: Tracked<wgpu::Buffer>,
    sh_buf: Tracked<wgpu::Buffer>,
    render_settings_buf: Tracked<wgpu::Buffer>,
}

#[derive(Clone, Copy)]
struct CloudGraphBuffers {
    gaussian_buf: BufferNodeId,
    sh_buf: BufferNodeId,
    splat_buf: BufferNodeId,
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
    sort_layout: SortBufferLayout,
    sort_infos_init: GpuSortInfos,
    draw_indirect_init: GpuDrawIndirect,
}

#[derive(Clone, Copy)]
struct CloudComputeState<'a> {
    buffers: CloudGraphBuffers,
    preprocess_bg1: Option<&'a wgpu::BindGroup>,
    preprocess_bg2: Option<&'a wgpu::BindGroup>,
    preprocess_bg3: Option<&'a wgpu::BindGroup>,
    sort_bg_a_to_b: Option<&'a wgpu::BindGroup>,
    sort_bg_b_to_a: Option<&'a wgpu::BindGroup>,
}

#[derive(Clone, Copy)]
struct CloudRenderState<'a> {
    buffers: CloudGraphBuffers,
    render_bg: Option<&'a wgpu::BindGroup>,
}

pub struct GaussianSplattingFeature {
    preprocess_pipeline: Option<wgpu::ComputePipeline>,
    preprocess_global_layout_id: Option<u64>,
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
                label: Some("GS Preprocess G1 (Packed Gaussians + SH + Packed Splats)"),
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
                    storage_ro_entry(0, cs),
                    storage_rw_entry(1, cs),
                    storage_ro_entry(2, cs),
                    storage_rw_entry(3, cs),
                    storage_ro_entry(4, cs),
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
        let global_state_key = (ctx.render_state.id, ctx.extracted_scene.scene_id);
        let gpu_world = ctx
            .resource_manager
            .get_global_state(global_state_key.0, global_state_key.1)
            .expect("GS preprocess requires a global render-state bind group");

        if self.preprocess_pipeline.is_none()
            || self.preprocess_global_layout_id != Some(gpu_world.layout_id)
        {
            let mut shader_options = ShaderCompilationOptions::default();
            shader_options.add_define("GS_SORT_KEYS_PER_WG", &SORT_KEYS_PER_WG.to_string());
            shader_options.inject_code("binding_code", &gpu_world.binding_wgsl);
            shader_options.inject_code(
                "scene_lighting_structs",
                myth_resources::uniforms::scene_lighting_structs_wgsl(),
            );
            let (module, _) = ctx.shader_manager.get_or_compile(
                device,
                ShaderSource::File("entry/utility/3dgs/gaussian_preprocess"),
                &shader_options,
            );

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GS Preprocess Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&gpu_world.layout),
                    self.preprocess_layout_g1.as_deref(),
                    self.preprocess_layout_g2.as_deref(),
                    self.preprocess_layout_g3.as_deref(),
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

            self.preprocess_global_layout_id = Some(gpu_world.layout_id);
        }

        if self.sort_pipelines.is_none() {
            self.sort_pipelines = Some(create_sort_pipelines(
                device,
                ctx.shader_manager,
                self.sort_layout.as_ref().unwrap(),
            ));
        }

        let render_key = GaussianRenderPipelineKey {
            depth_format: ctx.wgpu_ctx.depth_format,
            msaa_samples: ctx.wgpu_ctx.msaa_samples,
        };

        if self.render_pipeline.is_none() || self.render_pipeline_key != Some(render_key) {
            let shader_options = ShaderCompilationOptions::default();
            let (module, _) = ctx.shader_manager.get_or_compile(
                device,
                ShaderSource::File("entry/utility/3dgs/gaussian_render"),
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
                ShaderSource::File("entry/utility/3dgs/gs_composite"),
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
        let sort_layout = SortBufferLayout::for_key_count(cloud.num_points);
        let limits = device.limits();
        let preprocess_workgroups = num_points.div_ceil(PREPROCESS_WG_SIZE);
        assert!(
            preprocess_workgroups <= limits.max_compute_workgroups_per_dimension,
            "Gaussian cloud requires {preprocess_workgroups} preprocess workgroups, exceeding the device limit of {}",
            limits.max_compute_workgroups_per_dimension
        );
        assert!(
            sort_layout.max_workgroups <= limits.max_compute_workgroups_per_dimension as usize,
            "Gaussian cloud requires {} sort workgroups, exceeding the device limit of {}",
            sort_layout.max_workgroups,
            limits.max_compute_workgroups_per_dimension
        );
        let key_buffer_size = sort_layout
            .key_capacity
            .checked_mul(std::mem::size_of::<u32>())
            .expect("Gaussian sort key buffer size overflow");
        let internal_buffer_size = sort_layout
            .internal_buffer_words
            .checked_mul(std::mem::size_of::<u32>())
            .expect("Gaussian sort scratch buffer size overflow");
        assert!(
            key_buffer_size <= limits.max_storage_buffer_binding_size as usize,
            "Gaussian sort key buffer ({key_buffer_size} bytes) exceeds the device storage binding limit ({})",
            limits.max_storage_buffer_binding_size
        );
        assert!(
            internal_buffer_size <= limits.max_storage_buffer_binding_size as usize,
            "Gaussian sort scratch buffer ({internal_buffer_size} bytes) exceeds the device storage binding limit ({})",
            limits.max_storage_buffer_binding_size
        );
        let upload_count = cloud.num_points.max(1);
        let sh_upload_count = cloud.sh_coefficients.len().max(1);

        let gaussian_buf = Tracked::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS Gaussian Data"),
            size: (upload_count * std::mem::size_of::<GaussianSplat>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        queue.write_buffer(&gaussian_buf, 0, bytemuck::cast_slice(&cloud.gaussians));

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
            gaussian_buf,
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
                    (gpu.sort_layout.key_capacity * std::mem::size_of::<u32>()) as u64;

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

                let splat_buf = builder.create_buffer(
                    "GS_Splats",
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
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
                    splat_buf,
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
                    sort_layout: gpu.sort_layout,
                    sort_infos_init: GpuSortInfos {
                        keys_size: 0,
                        max_workgroups: u32::try_from(gpu.sort_layout.max_workgroups)
                            .expect("Gaussian sort workgroup count exceeds u32"),
                        scan_levels: u32::try_from(gpu.sort_layout.scan_level_count)
                            .expect("Gaussian sort scan level count exceeds u32"),
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
                    sort_bg_a_to_b: None,
                    sort_bg_b_to_a: None,
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
                builder.read_buffer(cloud.splat_buf);
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
                .bind_buffer(0, cloud.buffers.gaussian_buf)
                .bind_buffer(1, cloud.buffers.sh_buf)
                .bind_buffer(2, cloud.buffers.splat_buf)
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

            let sort_bg_a_to_b = ctx
                .build_bind_group(self.sort_layout, Some("GS Sort BG A to B"))
                .bind_buffer(0, cloud.buffers.sort_infos_buf)
                .bind_buffer(1, cloud.buffers.sort_internal_buf)
                .bind_buffer(2, cloud.buffers.sort_depths_a_buf)
                .bind_buffer(3, cloud.buffers.sort_depths_b_buf)
                .bind_buffer(4, cloud.buffers.sort_indices_a_buf)
                .bind_buffer(5, cloud.buffers.sort_indices_b_buf)
                .build();
            let sort_bg_b_to_a = ctx
                .build_bind_group(self.sort_layout, Some("GS Sort BG B to A"))
                .bind_buffer(0, cloud.buffers.sort_infos_buf)
                .bind_buffer(1, cloud.buffers.sort_internal_buf)
                .bind_buffer(2, cloud.buffers.sort_depths_b_buf)
                .bind_buffer(3, cloud.buffers.sort_depths_a_buf)
                .bind_buffer(4, cloud.buffers.sort_indices_b_buf)
                .bind_buffer(5, cloud.buffers.sort_indices_a_buf)
                .build();

            cloud.preprocess_bg1 = Some(preprocess_bg1);
            cloud.preprocess_bg2 = Some(preprocess_bg2);
            cloud.preprocess_bg3 = Some(preprocess_bg3);
            cloud.sort_bg_a_to_b = Some(sort_bg_a_to_b);
            cloud.sort_bg_b_to_a = Some(sort_bg_b_to_a);
        }
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let preprocess_pipeline = self
            .preprocess_pipeline
            .expect("GS preprocess pipeline missing");
        let sort_pipelines = self.sort_pipelines.expect("GS sort pipelines missing");
        let global_bind_group = ctx.baked_lists.global_bind_group;

        for cloud in self.clouds.iter() {
            let preprocess_workgroups = cloud.buffers.num_points.div_ceil(PREPROCESS_WG_SIZE);

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

            let sort_bg_a_to_b = cloud
                .sort_bg_a_to_b
                .expect("GS sort A-to-B bind group missing");
            let sort_bg_b_to_a = cloud
                .sort_bg_b_to_a
                .expect("GS sort B-to-A bind group missing");
            let sort_dispatch = ctx.get_buffer(cloud.buffers.sort_dispatch_buf);
            let sort_internal = ctx.get_buffer(cloud.buffers.sort_internal_buf);
            let histogram_bytes = u64::try_from(
                cloud
                    .buffers
                    .sort_layout
                    .histogram_words
                    .checked_mul(std::mem::size_of::<u32>())
                    .expect("Gaussian sort histogram byte size overflow"),
            )
            .expect("Gaussian sort histogram byte size exceeds u64");

            {
                for (pass_index, radix_pass) in sort_pipelines.radix_passes.iter().enumerate() {
                    let sort_bg = if pass_index % 2 == 0 {
                        sort_bg_a_to_b
                    } else {
                        sort_bg_b_to_a
                    };

                    // The visible count is GPU-generated and can shrink between
                    // frames. Clear the whole fixed-capacity histogram so
                    // inactive workgroups never contribute stale counts.
                    encoder.clear_buffer(sort_internal, 0, Some(histogram_bytes));

                    {
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("GS Sort Histogram"),
                            timestamp_writes: None,
                        });
                        cpass.set_pipeline(&radix_pass.histogram);
                        cpass.set_bind_group(0, sort_bg, &[]);
                        cpass.dispatch_workgroups_indirect(sort_dispatch, 0);
                    }

                    for level in 0..cloud.buffers.sort_layout.scan_level_count {
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("GS Sort Prefix Scan"),
                            timestamp_writes: None,
                        });
                        cpass.set_pipeline(&sort_pipelines.prefix_scan[level]);
                        cpass.set_bind_group(0, sort_bg_a_to_b, &[]);
                        cpass.dispatch_workgroups(
                            cloud.buffers.sort_layout.scan_workgroups[level],
                            1,
                            1,
                        );
                    }

                    for level in
                        (0..cloud.buffers.sort_layout.scan_level_count.saturating_sub(1)).rev()
                    {
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("GS Sort Prefix Add"),
                            timestamp_writes: None,
                        });
                        cpass.set_pipeline(&sort_pipelines.prefix_add[level]);
                        cpass.set_bind_group(0, sort_bg_a_to_b, &[]);
                        cpass.dispatch_workgroups(
                            cloud.buffers.sort_layout.scan_workgroups[level],
                            1,
                            1,
                        );
                    }

                    {
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("GS Sort Scatter"),
                            timestamp_writes: None,
                        });
                        cpass.set_pipeline(&radix_pass.scatter);
                        cpass.set_bind_group(0, sort_bg, &[]);
                        cpass.dispatch_workgroups_indirect(sort_dispatch, 0);
                    }
                }
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
                    .bind_buffer(0, cloud.buffers.splat_buf)
                    .bind_buffer(1, cloud.buffers.sort_indices_a_buf)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

    fn test_buffer(
        device: &wgpu::Device,
        label: &str,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    async fn request_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            Ok(adapter) => adapter,
            Err(error) => {
                eprintln!("skipping Gaussian sort GPU test: no adapter available ({error})");
                return None;
            }
        };
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("GS Sort Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .map_err(|error| {
                eprintln!("skipping Gaussian sort GPU test: device request failed ({error})");
            })
            .ok()
    }

    #[test]
    fn sort_layout_handles_empty_and_workgroup_boundaries() {
        let empty = SortBufferLayout::for_key_count(0);
        assert_eq!(empty.key_capacity, 1);
        assert_eq!(empty.max_workgroups, 1);
        assert_eq!(empty.histogram_words, SORT_RADIX_SIZE);
        assert_eq!(empty.internal_buffer_words, SORT_RADIX_SIZE + 1);
        assert_eq!(empty.scan_level_count, 1);
        assert_eq!(empty.scan_workgroups[0], 1);

        let exact = SortBufferLayout::for_key_count(SORT_KEYS_PER_WG);
        assert_eq!(exact.key_capacity, SORT_KEYS_PER_WG);
        assert_eq!(exact.max_workgroups, 1);

        let overflow = SortBufferLayout::for_key_count(SORT_KEYS_PER_WG + 1);
        assert_eq!(overflow.max_workgroups, 2);
        assert_eq!(overflow.histogram_words, SORT_RADIX_SIZE * 2);
        assert_eq!(overflow.scan_workgroups[0], 1);
    }

    #[test]
    fn sort_layout_builds_hierarchical_scan_scratch() {
        let layout = SortBufferLayout::for_key_count(310_920);
        assert_eq!(layout.key_capacity, 310_920);
        assert_eq!(layout.max_workgroups, 152);
        assert_eq!(layout.histogram_words, 2_432);
        assert_eq!(layout.scan_level_count, 2);
        assert_eq!(&layout.scan_workgroups[..2], &[5, 1]);
        assert_eq!(layout.internal_buffer_words, 2_438);

        let three_levels = SortBufferLayout::for_key_count((16_384 + 1) * SORT_KEYS_PER_WG);
        assert_eq!(three_levels.scan_level_count, 3);
        assert_eq!(&three_levels.scan_workgroups[..3], &[513, 2, 1]);
    }

    #[test]
    fn even_radix_pass_count_leaves_payload_in_front_buffer() {
        assert_eq!(SORT_PASSES, 8);
        assert_eq!(SORT_PASSES % 2, 0);
    }

    #[test]
    fn portable_shaders_do_not_use_pipeline_overrides() {
        let sort_shader =
            include_str!("../../pipeline/shaders/entry/utility/3dgs/gs_radix_sort.wgsl");
        let preprocess_shader =
            include_str!("../../pipeline/shaders/entry/utility/3dgs/gaussian_preprocess.wgsl");
        for (name, source) in [
            ("portable sort", sort_shader),
            ("Gaussian preprocess", preprocess_shader),
        ] {
            assert!(
                !source
                    .lines()
                    .any(|line| line.trim_start().starts_with("override ")),
                "{name} must not depend on WebGPU pipeline override constants"
            );
        }
    }

    #[test]
    fn portable_gpu_sort_matches_stable_cpu_reference() {
        pollster::block_on(async {
            // More than 32 sort workgroups forces a two-level prefix scan.
            // Repeated buckets are deliberately separated in the input to
            // verify that the scatter remains stable.
            let key_count = 70_001usize;
            let buffer_capacity = 75_003usize;
            let keys: Vec<u32> = (0..key_count)
                .map(|index| {
                    let bucket = u32::try_from(index % 4_096).unwrap();
                    bucket.wrapping_mul(0x9e37_79b9).rotate_left(bucket & 31)
                })
                .collect();
            let payloads: Vec<u32> = (0..u32::try_from(key_count).unwrap()).collect();
            let mut expected = payloads.clone();
            expected.sort_by_key(|&index| keys[index as usize]);

            let layout = SortBufferLayout::for_key_count(buffer_capacity);
            assert_eq!(layout.scan_level_count, 2);

            let Some((device, queue)) = request_test_device().await else {
                return;
            };
            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("GS Sort Test Layout"),
                    entries: &[
                        test_storage_entry(0, true),
                        test_storage_entry(1, false),
                        test_storage_entry(2, true),
                        test_storage_entry(3, false),
                        test_storage_entry(4, true),
                        test_storage_entry(5, false),
                    ],
                });
            let mut shader_manager = ShaderManager::new();
            let pipelines = create_sort_pipelines(&device, &mut shader_manager, &bind_group_layout);

            let infos = GpuSortInfos {
                keys_size: u32::try_from(key_count).unwrap(),
                max_workgroups: u32::try_from(layout.max_workgroups).unwrap(),
                scan_levels: u32::try_from(layout.scan_level_count).unwrap(),
                dispatch_x: u32::try_from(key_count.div_ceil(SORT_KEYS_PER_WG)).unwrap(),
                dispatch_y: 1,
                dispatch_z: 1,
            };
            let infos_buffer = test_buffer(
                &device,
                "GS Sort Test Infos",
                std::mem::size_of::<GpuSortInfos>() as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            let internal_buffer = test_buffer(
                &device,
                "GS Sort Test Scratch",
                (layout.internal_buffer_words * std::mem::size_of::<u32>()) as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            let data_size = (buffer_capacity * std::mem::size_of::<u32>()) as u64;
            let active_data_size = (key_count * std::mem::size_of::<u32>()) as u64;
            let keys_a = test_buffer(
                &device,
                "GS Sort Test Keys A",
                data_size,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            let keys_b = test_buffer(
                &device,
                "GS Sort Test Keys B",
                data_size,
                wgpu::BufferUsages::STORAGE,
            );
            let payload_a = test_buffer(
                &device,
                "GS Sort Test Payload A",
                data_size,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            );
            let payload_b = test_buffer(
                &device,
                "GS Sort Test Payload B",
                data_size,
                wgpu::BufferUsages::STORAGE,
            );
            let readback = test_buffer(
                &device,
                "GS Sort Test Readback",
                active_data_size,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            );

            queue.write_buffer(&infos_buffer, 0, bytemuck::bytes_of(&infos));
            queue.write_buffer(&keys_a, 0, bytemuck::cast_slice(&keys));
            queue.write_buffer(&payload_a, 0, bytemuck::cast_slice(&payloads));

            let bind_group_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GS Sort Test Bind Group A to B"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: infos_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: internal_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: keys_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: keys_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: payload_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: payload_b.as_entire_binding(),
                    },
                ],
            });
            let bind_group_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GS Sort Test Bind Group B to A"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: infos_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: internal_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: keys_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: keys_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: payload_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: payload_a.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GS Sort Test Encoder"),
            });
            let histogram_bytes = (layout.histogram_words * std::mem::size_of::<u32>()) as u64;
            let sort_workgroups = u32::try_from(key_count.div_ceil(SORT_KEYS_PER_WG)).unwrap();

            for (pass_index, radix_pass) in pipelines.radix_passes.iter().enumerate() {
                let bind_group = if pass_index % 2 == 0 {
                    &bind_group_a_to_b
                } else {
                    &bind_group_b_to_a
                };
                encoder.clear_buffer(&internal_buffer, 0, Some(histogram_bytes));
                {
                    let mut pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&radix_pass.histogram);
                    pass.set_bind_group(0, bind_group, &[]);
                    pass.dispatch_workgroups(sort_workgroups, 1, 1);
                }
                for level in 0..layout.scan_level_count {
                    let mut pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&pipelines.prefix_scan[level]);
                    pass.set_bind_group(0, &bind_group_a_to_b, &[]);
                    pass.dispatch_workgroups(layout.scan_workgroups[level], 1, 1);
                }
                for level in (0..layout.scan_level_count.saturating_sub(1)).rev() {
                    let mut pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&pipelines.prefix_add[level]);
                    pass.set_bind_group(0, &bind_group_a_to_b, &[]);
                    pass.dispatch_workgroups(layout.scan_workgroups[level], 1, 1);
                }
                {
                    let mut pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&radix_pass.scatter);
                    pass.set_bind_group(0, bind_group, &[]);
                    pass.dispatch_workgroups(sort_workgroups, 1, 1);
                }
            }
            encoder.copy_buffer_to_buffer(&payload_a, 0, &readback, 0, active_data_size);
            queue.submit([encoder.finish()]);

            let slice = readback.slice(..);
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).ok();
            });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("Gaussian sort GPU poll failed");
            receiver
                .recv()
                .expect("Gaussian sort readback callback dropped")
                .expect("Gaussian sort readback mapping failed");

            let mapped = slice.get_mapped_range();
            let actual: Vec<u32> = bytemuck::cast_slice(&mapped).to_vec();
            drop(mapped);
            readback.unmap();

            if let Some((position, (&actual_index, &expected_index))) = actual
                .iter()
                .zip(&expected)
                .enumerate()
                .find(|(_, (actual_index, expected_index))| actual_index != expected_index)
            {
                panic!(
                    "portable GPU sort mismatch at {position}: actual index {actual_index}, expected {expected_index}, actual key {}, expected key {}",
                    keys[actual_index as usize], keys[expected_index as usize]
                );
            }
        });
    }
}
