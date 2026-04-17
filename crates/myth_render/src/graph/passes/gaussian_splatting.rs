//! RDG Gaussian Splatting Pass — 3D Gaussian Splatting Rendering
//!
//! Implements real-time 3DGS rendering within the RDG framework:
//!
//! 1. **Preprocess** — compute pass: frustum-cull 3D Gaussians, evaluate SH
//!    colour, project to 2D screen-space splats, emit sort keys.
//! 2. **Radix Sort** — 3 compute passes (histogram → prefix → scatter)
//!    repeated for 4 byte-passes to depth-sort visible splats.
//! 3. **Render** — render pass: vertex-pulled quad drawing with Gaussian
//!    kernel evaluation, front-to-back alpha blending.
//!
//! # Architecture
//!
//! - **[`GaussianSplattingFeature`]** — persistent GPU resource owner.
//!   Holds compute/render pipelines, bind group layouts, and per-cloud
//!   GPU buffers. Lazily initialised by [`extract_and_prepare`].
//!
//! - **[`GaussianSplattingPassNode`]** — ephemeral pass node.
//!   References Feature-owned bind groups and pipelines. Emits GPU
//!   commands during `execute()`.

use std::sync::Arc;

use crate::HDR_TEXTURE_FORMAT;
use crate::core::gpu::Tracked;
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    ExecuteContext, ExtractContext, PassNode, RenderTargetOps, TextureNodeId,
};
use crate::pipeline::{
    ColorTargetKey, ComputePipelineId, ComputePipelineKey, FullscreenPipelineKey,
    RenderPipelineId, ShaderCompilationOptions, ShaderSource,
};
use myth_resources::gaussian_splat::{GaussianCloud, Splat2D};
use myth_scene::camera::RenderCamera;

/// Workgroup size for the preprocess and sort compute shaders.
const WG_SIZE: u32 = 256;

/// Keys sorted per workgroup in the radix sort.
const SORT_KEYS_PER_WG: u32 = WG_SIZE * 15;

// =============================================================================
// GPU Uniform: Camera data for shaders
// =============================================================================

/// Camera uniform struct matching the WGSL `CameraUniforms`.
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

/// Render settings uniform matching the WGSL `RenderSettings`.
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

/// Sort info struct matching the WGSL `SortInfos` layout.
/// Note: `keys_size` is written atomically by the preprocess shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSortInfos {
    keys_size: u32,
    padded_size: u32,
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

// =============================================================================
// Per-Cloud GPU resources
// =============================================================================

/// GPU-side storage for a single Gaussian point cloud.
#[allow(dead_code)]
struct CloudGpuData {
    /// Number of Gaussians in this cloud.
    num_points: u32,

    // ─── Storage Buffers ───────────────────────────────────────────
    /// Gaussian 3D data (position + opacity + packed covariance).
    gaussian_buf: wgpu::Buffer,
    /// SH coefficients storage.
    sh_buf: wgpu::Buffer,
    /// 2D projected splats (written by preprocess, read by render).
    splat_2d_buf: wgpu::Buffer,

    // ─── Sort Buffers ──────────────────────────────────────────────
    sort_depths_buf: wgpu::Buffer,
    sort_indices_buf: wgpu::Buffer,
    sort_dispatch_buf: wgpu::Buffer,
    sort_infos_buf: wgpu::Buffer,
    /// Assist buffers for double-buffered radix sort.
    sort_assist_a_buf: wgpu::Buffer,
    sort_assist_b_buf: wgpu::Buffer,
    /// Histogram buffer (256 bins × num_workgroups × 4 passes).
    sort_histograms_buf: wgpu::Buffer,

    // ─── Uniform Buffers ───────────────────────────────────────────
    camera_uniform_buf: wgpu::Buffer,
    render_settings_buf: wgpu::Buffer,

    // ─── Bind Groups ──────────────────────────────────────────────
    /// Group 0 for preprocess: camera uniforms.
    preprocess_bg0: wgpu::BindGroup,
    /// Group 1 for preprocess: gaussians + SH + splats_2d.
    preprocess_bg1: wgpu::BindGroup,
    /// Group 2 for preprocess: sort_infos + depths + indices + dispatch.
    preprocess_bg2: wgpu::BindGroup,
    /// Group 3 for preprocess: render settings.
    preprocess_bg3: wgpu::BindGroup,

    /// Group 0 for sort: sort_infos + depths + indices + dispatch.
    sort_bg0: wgpu::BindGroup,
    /// Group 1 for sort: assist_a + assist_b + histograms.
    sort_bg1: wgpu::BindGroup,

    /// Group 0 for render: camera uniforms (same buffer).
    render_bg0: wgpu::BindGroup,
    /// Group 1 for render: splats_2d + sort_indices.
    render_bg1: wgpu::BindGroup,
}

// =============================================================================
// GaussianSplattingFeature — persistent GPU state
// =============================================================================

/// Persistent feature owning GPU pipelines, layouts, and per-cloud buffers
/// for 3D Gaussian Splatting rendering.
pub struct GaussianSplattingFeature {
    // ─── Compute Pipelines ─────────────────────────────────────────
    preprocess_pipeline: Option<ComputePipelineId>,
    sort_histogram_pipeline: Option<ComputePipelineId>,
    sort_prefix_pipeline: Option<ComputePipelineId>,
    sort_scatter_pipeline: Option<ComputePipelineId>,

    // ─── Render Pipeline ───────────────────────────────────────────
    render_pipeline: Option<RenderPipelineId>,

    // ─── Bind Group Layouts ────────────────────────────────────────
    // Preprocess
    preprocess_layout_g0: Option<Tracked<wgpu::BindGroupLayout>>,
    preprocess_layout_g1: Option<Tracked<wgpu::BindGroupLayout>>,
    preprocess_layout_g2: Option<Tracked<wgpu::BindGroupLayout>>,
    preprocess_layout_g3: Option<Tracked<wgpu::BindGroupLayout>>,
    // Sort
    sort_layout_g0: Option<Tracked<wgpu::BindGroupLayout>>,
    sort_layout_g1: Option<Tracked<wgpu::BindGroupLayout>>,
    // Render
    render_layout_g0: Option<Tracked<wgpu::BindGroupLayout>>,
    render_layout_g1: Option<Tracked<wgpu::BindGroupLayout>>,

    // ─── Per-Cloud GPU Data ────────────────────────────────────────
    /// Currently active cloud data. `None` when no cloud is loaded.
    cloud_data: Option<CloudGpuData>,

    /// Fingerprint of the currently uploaded cloud (pointer + num_points),
    /// used to detect when GPU re-upload is needed.
    cloud_fingerprint: u64,

    /// Whether we have an active cloud to render this frame.
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
            sort_histogram_pipeline: None,
            sort_prefix_pipeline: None,
            sort_scatter_pipeline: None,
            render_pipeline: None,

            preprocess_layout_g0: None,
            preprocess_layout_g1: None,
            preprocess_layout_g2: None,
            preprocess_layout_g3: None,
            sort_layout_g0: None,
            sort_layout_g1: None,
            render_layout_g0: None,
            render_layout_g1: None,

            cloud_data: None,
            cloud_fingerprint: 0,
            active: false,
        }
    }

    // =========================================================================
    // Extract & Prepare
    // =========================================================================

    /// Prepare GPU resources for Gaussian Splatting rendering.
    ///
    /// Called before the render graph is built. Ensures layouts, pipelines,
    /// and per-cloud GPU buffers are ready. Uploads cloud data if changed.
    pub fn extract_and_prepare(
        &mut self,
        ctx: &mut ExtractContext,
        cloud: &Arc<GaussianCloud>,
        camera: &RenderCamera,
        viewport_size: (u32, u32),
    ) {
        self.ensure_layouts(ctx.device);
        self.ensure_pipelines(ctx);
        self.ensure_cloud_data(ctx.device, ctx.queue, cloud);
        self.update_uniforms(ctx.queue, camera, cloud, viewport_size);
        self.active = true;
    }

    // =========================================================================
    // Lazy Initialization
    // =========================================================================

    fn ensure_layouts(&mut self, device: &wgpu::Device) {
        if self.preprocess_layout_g0.is_some() {
            return;
        }

        let uniform_entry = |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let storage_ro_entry =
            |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: vis,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };

        let storage_rw_entry =
            |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: vis,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };

        let cs = wgpu::ShaderStages::COMPUTE;
        let vs = wgpu::ShaderStages::VERTEX;
        let vs_fs = wgpu::ShaderStages::VERTEX_FRAGMENT;

        // ─── Preprocess Layouts ────────────────────────────────────
        self.preprocess_layout_g0 = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GS Preprocess G0 (Camera)"),
                entries: &[uniform_entry(0, cs)],
            },
        )));
        self.preprocess_layout_g1 = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GS Preprocess G1 (Gaussians+SH+Splats)"),
                entries: &[
                    storage_ro_entry(0, cs),  // gaussians
                    storage_ro_entry(1, cs),  // sh_coefs
                    storage_rw_entry(2, cs),  // points_2d
                ],
            },
        )));
        self.preprocess_layout_g2 = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GS Preprocess G2 (Sort)"),
                entries: &[
                    storage_rw_entry(0, cs),  // sort_infos
                    storage_rw_entry(1, cs),  // sort_depths
                    storage_rw_entry(2, cs),  // sort_indices
                    storage_rw_entry(3, cs),  // sort_dispatch
                ],
            },
        )));
        self.preprocess_layout_g3 = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GS Preprocess G3 (Settings)"),
                entries: &[uniform_entry(0, cs)],
            },
        )));

        // ─── Sort Layouts ──────────────────────────────────────────
        self.sort_layout_g0 = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GS Sort G0 (Infos+Depths+Indices+Dispatch)"),
                entries: &[
                    storage_rw_entry(0, cs),
                    storage_rw_entry(1, cs),
                    storage_rw_entry(2, cs),
                    storage_rw_entry(3, cs),
                ],
            },
        )));
        self.sort_layout_g1 = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GS Sort G1 (Assist+Histograms)"),
                entries: &[
                    storage_rw_entry(0, cs),
                    storage_rw_entry(1, cs),
                    storage_rw_entry(2, cs),
                ],
            },
        )));

        // ─── Render Layouts ────────────────────────────────────────
        self.render_layout_g0 = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GS Render G0 (Camera)"),
                entries: &[uniform_entry(0, vs_fs)],
            },
        )));
        self.render_layout_g1 = Some(Tracked::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("GS Render G1 (Splats+SortIndices)"),
                entries: &[
                    storage_ro_entry(0, vs),  // splats
                    storage_ro_entry(1, vs),  // sort_indices
                ],
            },
        )));
    }

    fn ensure_pipelines(&mut self, ctx: &mut ExtractContext) {
        if self.preprocess_pipeline.is_some() {
            return;
        }

        let device = ctx.device;
        let comp_opts = wgpu::PipelineCompilationOptions::default();
        let shader_opts = ShaderCompilationOptions::default();

        // ─── Preprocess Compute Pipeline ───────────────────────────
        {
            let (module, hash) = ctx.shader_manager.get_or_compile(
                device,
                ShaderSource::File("entry/utility/gaussian_preprocess"),
                &shader_opts,
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

            self.preprocess_pipeline = Some(ctx.pipeline_cache.get_or_create_compute(
                device,
                module,
                &layout,
                &ComputePipelineKey::new(hash).with_compilation_options(&comp_opts),
                &comp_opts,
                "GS Preprocess Pipeline",
            ));
        }

        // ─── Sort Histogram Pipeline ───────────────────────────────
        {
            let (module, hash) = ctx.shader_manager.get_or_compile(
                device,
                ShaderSource::File("entry/utility/gs_sort_histogram"),
                &shader_opts,
            );

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GS Sort Histogram Pipeline Layout"),
                bind_group_layouts: &[
                    Some(self.sort_layout_g0.as_ref().unwrap()),
                    Some(self.sort_layout_g1.as_ref().unwrap()),
                ],
                immediate_size: 0,
            });

            self.sort_histogram_pipeline = Some(ctx.pipeline_cache.get_or_create_compute(
                device,
                module,
                &layout,
                &ComputePipelineKey::new(hash).with_compilation_options(&comp_opts),
                &comp_opts,
                "GS Sort Histogram Pipeline",
            ));
        }

        // ─── Sort Prefix Pipeline ──────────────────────────────────
        {
            let (module, hash) = ctx.shader_manager.get_or_compile(
                device,
                ShaderSource::File("entry/utility/gs_sort_prefix"),
                &shader_opts,
            );

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GS Sort Prefix Pipeline Layout"),
                bind_group_layouts: &[
                    Some(self.sort_layout_g0.as_ref().unwrap()),
                    Some(self.sort_layout_g1.as_ref().unwrap()),
                ],
                immediate_size: 0,
            });

            self.sort_prefix_pipeline = Some(ctx.pipeline_cache.get_or_create_compute(
                device,
                module,
                &layout,
                &ComputePipelineKey::new(hash).with_compilation_options(&comp_opts),
                &comp_opts,
                "GS Sort Prefix Pipeline",
            ));
        }

        // ─── Sort Scatter Pipeline ─────────────────────────────────
        {
            let (module, hash) = ctx.shader_manager.get_or_compile(
                device,
                ShaderSource::File("entry/utility/gs_sort_scatter"),
                &shader_opts,
            );

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GS Sort Scatter Pipeline Layout"),
                bind_group_layouts: &[
                    Some(self.sort_layout_g0.as_ref().unwrap()),
                    Some(self.sort_layout_g1.as_ref().unwrap()),
                ],
                immediate_size: 0,
            });

            self.sort_scatter_pipeline = Some(ctx.pipeline_cache.get_or_create_compute(
                device,
                module,
                &layout,
                &ComputePipelineKey::new(hash).with_compilation_options(&comp_opts),
                &comp_opts,
                "GS Sort Scatter Pipeline",
            ));
        }

        // ─── Render Pipeline ───────────────────────────────────────
        {
            let (module, hash) = ctx.shader_manager.get_or_compile(
                device,
                ShaderSource::File("entry/utility/gaussian_render"),
                &shader_opts,
            );

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GS Render Pipeline Layout"),
                bind_group_layouts: &[
                    Some(self.render_layout_g0.as_ref().unwrap()),
                    Some(self.render_layout_g1.as_ref().unwrap()),
                ],
                immediate_size: 0,
            });

            // Front-to-back alpha blending:
            //   dst_color = src_alpha * src_color + (1 - src_alpha) * dst_color
            //   dst_alpha = src_alpha + (1 - src_alpha) * dst_alpha
            let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
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
            });

            let key = FullscreenPipelineKey::fullscreen(
                hash,
                smallvec::smallvec![color_target],
                None, // No depth test for Gaussian splatting (pre-sorted)
            );

            self.render_pipeline = Some(ctx.pipeline_cache.get_or_create_fullscreen(
                device,
                module,
                &layout,
                &key,
                "GS Render Pipeline",
            ));
        }
    }

    // =========================================================================
    // Per-Cloud GPU Buffer Management
    // =========================================================================

    fn ensure_cloud_data(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cloud: &Arc<GaussianCloud>,
    ) {
        // Check fingerprint to avoid re-upload
        let fingerprint = {
            let ptr = Arc::as_ptr(cloud) as u64;
            let mut h = ptr;
            h ^= cloud.num_points as u64;
            h
        };

        if self.cloud_fingerprint == fingerprint && self.cloud_data.is_some() {
            return;
        }

        let n = cloud.num_points as u64;
        let gaussian_size = std::mem::size_of::<myth_resources::gaussian_splat::GaussianSplat>() as u64;
        let sh_size = std::mem::size_of::<myth_resources::gaussian_splat::GaussianSHCoefficients>() as u64;
        let splat_2d_size = std::mem::size_of::<Splat2D>() as u64;

        // Compute sort buffer sizes
        let num_sort_wgs = (n + SORT_KEYS_PER_WG as u64 - 1) / SORT_KEYS_PER_WG as u64;
        let histogram_size = 256 * num_sort_wgs * 4 * 4; // 256 bins × wgs × 4 passes × u32

        // ─── Create Buffers ────────────────────────────────────────
        let gaussian_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS Gaussian Data"),
            size: n * gaussian_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&gaussian_buf, 0, bytemuck::cast_slice(&cloud.gaussians));

        let sh_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS SH Coefficients"),
            size: n * sh_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&sh_buf, 0, bytemuck::cast_slice(&cloud.sh_coefficients));

        let splat_2d_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS Splat 2D"),
            size: n * splat_2d_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let sort_depths_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS Sort Depths"),
            size: n * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let sort_indices_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS Sort Indices"),
            size: n * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let sort_dispatch_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS Sort Dispatch"),
            size: 12, // 3 × u32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sort_infos_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS Sort Infos"),
            size: std::mem::size_of::<GpuSortInfos>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sort_assist_a_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS Sort Assist A"),
            size: n * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let sort_assist_b_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS Sort Assist B"),
            size: n * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let sort_histograms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS Sort Histograms"),
            size: histogram_size.max(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let camera_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS Camera Uniform"),
            size: std::mem::size_of::<GpuCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let render_settings_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GS Render Settings"),
            size: std::mem::size_of::<GpuRenderSettings>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ─── Build Bind Groups ─────────────────────────────────────
        let preprocess_bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GS Preprocess BG0"),
            layout: self.preprocess_layout_g0.as_ref().unwrap(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_uniform_buf.as_entire_binding(),
            }],
        });

        let preprocess_bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GS Preprocess BG1"),
            layout: self.preprocess_layout_g1.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gaussian_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sh_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: splat_2d_buf.as_entire_binding(),
                },
            ],
        });

        let preprocess_bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GS Preprocess BG2"),
            layout: self.preprocess_layout_g2.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sort_infos_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sort_depths_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sort_indices_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sort_dispatch_buf.as_entire_binding(),
                },
            ],
        });

        let preprocess_bg3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GS Preprocess BG3"),
            layout: self.preprocess_layout_g3.as_ref().unwrap(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: render_settings_buf.as_entire_binding(),
            }],
        });

        let sort_bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GS Sort BG0"),
            layout: self.sort_layout_g0.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sort_infos_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sort_depths_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sort_indices_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sort_dispatch_buf.as_entire_binding(),
                },
            ],
        });

        let sort_bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GS Sort BG1"),
            layout: self.sort_layout_g1.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sort_assist_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sort_assist_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sort_histograms_buf.as_entire_binding(),
                },
            ],
        });

        let render_bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GS Render BG0"),
            layout: self.render_layout_g0.as_ref().unwrap(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_uniform_buf.as_entire_binding(),
            }],
        });

        let render_bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GS Render BG1"),
            layout: self.render_layout_g1.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: splat_2d_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sort_indices_buf.as_entire_binding(),
                },
            ],
        });

        self.cloud_data = Some(CloudGpuData {
            num_points: cloud.num_points as u32,
            gaussian_buf,
            sh_buf,
            splat_2d_buf,
            sort_depths_buf,
            sort_indices_buf,
            sort_dispatch_buf,
            sort_infos_buf,
            sort_assist_a_buf,
            sort_assist_b_buf,
            sort_histograms_buf,
            camera_uniform_buf,
            render_settings_buf,
            preprocess_bg0,
            preprocess_bg1,
            preprocess_bg2,
            preprocess_bg3,
            sort_bg0,
            sort_bg1,
            render_bg0,
            render_bg1,
        });

        self.cloud_fingerprint = fingerprint;
    }

    fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        camera: &RenderCamera,
        cloud: &GaussianCloud,
        viewport_size: (u32, u32),
    ) {
        let data = self.cloud_data.as_ref().unwrap();

        let view = camera.view_matrix;
        let proj = camera.projection_matrix;
        let view_inv = view.inverse();
        let proj_inv = proj.inverse();

        let fov_y = if camera.far.is_finite() {
            2.0 * (1.0 / proj.y_axis.y).atan()
        } else {
            // Infinite projection
            2.0 * (1.0 / proj.y_axis.y).atan()
        };
        let focal_y = viewport_size.1 as f32 / (2.0 * (fov_y / 2.0).tan());
        let focal_x = focal_y * (viewport_size.0 as f32 / viewport_size.1 as f32);

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

        let scene_extent = cloud.scene_extent();
        let render_settings = GpuRenderSettings {
            gaussian_scaling: 1.0,
            max_sh_deg: cloud.sh_degree,
            mip_splatting: u32::from(cloud.mip_splatting),
            kernel_size: cloud.kernel_size,
            scene_extent,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };

        queue.write_buffer(
            &data.render_settings_buf,
            0,
            bytemuck::bytes_of(&render_settings),
        );

        // Reset sort dispatch (will be incremented atomically by preprocess)
        queue.write_buffer(&data.sort_dispatch_buf, 0, bytemuck::bytes_of(&[0u32; 3]));

        // Reset sort infos (keys_size = 0, passes/even_pass set per-pass)
        let sort_infos = GpuSortInfos {
            keys_size: 0,
            padded_size: 0,
            passes: 0,
            even_pass: 0,
            odd_pass: 0,
        };
        queue.write_buffer(
            &data.sort_infos_buf,
            0,
            bytemuck::bytes_of(&sort_infos),
        );
    }

    // =========================================================================
    // Add to Render Graph
    // =========================================================================

    /// Emit the Gaussian Splatting passes into the RDG.
    ///
    /// Inserts:
    /// 1. A compute pass for preprocessing + radix sort
    /// 2. A render pass for final quad rendering
    ///
    /// The render pass writes to `active_color` with front-to-back blending.
    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        active_color: TextureNodeId,
        active_depth: TextureNodeId,
    ) -> TextureNodeId {
        if !self.active {
            return active_color;
        }

        let cloud_data = self.cloud_data.as_ref().unwrap();

        // ─── Compute Pass: Preprocess + Sort ───────────────────────
        let preprocess_pipeline = self
            .preprocess_pipeline
            .map(|id| ctx.pipeline_cache.get_compute_pipeline(id));
        let sort_histogram_pipeline = self
            .sort_histogram_pipeline
            .map(|id| ctx.pipeline_cache.get_compute_pipeline(id));
        let sort_prefix_pipeline = self
            .sort_prefix_pipeline
            .map(|id| ctx.pipeline_cache.get_compute_pipeline(id));
        let sort_scatter_pipeline = self
            .sort_scatter_pipeline
            .map(|id| ctx.pipeline_cache.get_compute_pipeline(id));

        let num_points = cloud_data.num_points;
        let preprocess_bg0 = &cloud_data.preprocess_bg0;
        let preprocess_bg1 = &cloud_data.preprocess_bg1;
        let preprocess_bg2 = &cloud_data.preprocess_bg2;
        let preprocess_bg3 = &cloud_data.preprocess_bg3;
        let sort_bg0 = &cloud_data.sort_bg0;
        let sort_bg1 = &cloud_data.sort_bg1;
        let sort_infos_buf = &cloud_data.sort_infos_buf;

        ctx.graph.add_pass("GS_Compute", |builder| {
            builder.mark_side_effect();
            (
                GaussianComputePassNode {
                    preprocess_pipeline,
                    sort_histogram_pipeline,
                    sort_prefix_pipeline,
                    sort_scatter_pipeline,
                    preprocess_bg0,
                    preprocess_bg1,
                    preprocess_bg2,
                    preprocess_bg3,
                    sort_bg0,
                    sort_bg1,
                    sort_infos_buf,
                    num_points,
                },
                (),
            )
        });

        // ─── Render Pass: Quad Drawing ─────────────────────────────
        let render_pipeline = self
            .render_pipeline
            .map(|id| ctx.pipeline_cache.get_render_pipeline(id));
        let render_bg0 = &cloud_data.render_bg0;
        let render_bg1 = &cloud_data.render_bg1;

        let gs_color = ctx.graph.add_pass("GS_Render", |builder| {
            let color_out = builder.mutate_texture(active_color, "GS_Color");
            let _depth_in = builder.read_texture(active_depth);
            (
                GaussianRenderPassNode {
                    render_pipeline,
                    render_bg0,
                    render_bg1,
                    color_target: color_out,
                    depth_target: active_depth,
                    num_points,
                },
                color_out,
            )
        });

        gs_color
    }
}

// =============================================================================
// Compute PassNode (Preprocess + Sort)
// =============================================================================

struct GaussianComputePassNode<'a> {
    preprocess_pipeline: Option<&'a wgpu::ComputePipeline>,
    sort_histogram_pipeline: Option<&'a wgpu::ComputePipeline>,
    sort_prefix_pipeline: Option<&'a wgpu::ComputePipeline>,
    sort_scatter_pipeline: Option<&'a wgpu::ComputePipeline>,
    preprocess_bg0: &'a wgpu::BindGroup,
    preprocess_bg1: &'a wgpu::BindGroup,
    preprocess_bg2: &'a wgpu::BindGroup,
    preprocess_bg3: &'a wgpu::BindGroup,
    sort_bg0: &'a wgpu::BindGroup,
    sort_bg1: &'a wgpu::BindGroup,
    sort_infos_buf: &'a wgpu::Buffer,
    num_points: u32,
}

impl PassNode<'_> for GaussianComputePassNode<'_> {
    fn execute(&self, _ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let preprocess = self.preprocess_pipeline.expect("GS preprocess pipeline missing");
        let sort_histogram = self.sort_histogram_pipeline.expect("GS sort histogram pipeline missing");
        let sort_prefix = self.sort_prefix_pipeline.expect("GS sort prefix pipeline missing");
        let sort_scatter = self.sort_scatter_pipeline.expect("GS sort scatter pipeline missing");

        let dispatch_wgs = (self.num_points + WG_SIZE - 1) / WG_SIZE;

        // ─── Step 1: Preprocess ────────────────────────────────────
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GS Preprocess"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(preprocess);
            cpass.set_bind_group(0, self.preprocess_bg0, &[]);
            cpass.set_bind_group(1, self.preprocess_bg1, &[]);
            cpass.set_bind_group(2, self.preprocess_bg2, &[]);
            cpass.set_bind_group(3, self.preprocess_bg3, &[]);
            cpass.dispatch_workgroups(dispatch_wgs, 1, 1);
        }

        // ─── Step 2: Radix Sort (4 byte passes) ───────────────────
        // For each of the 4 8-bit radix passes, we update the sort_infos
        // buffer with the current pass index and even/odd flags, then
        // dispatch histogram → prefix → scatter.
        //
        // Note: We must issue CPU-side buffer writes between passes.
        // In a more optimized version, this would be done via indirect
        // dispatch or push constants. For correctness, we use 4 separate
        // compute pass invocations per radix pass.
        let num_sort_wgs = (self.num_points as u64 + SORT_KEYS_PER_WG as u64 - 1) / SORT_KEYS_PER_WG as u64;
        let prefix_wgs = ((256 * num_sort_wgs) + 255) / 256;

        for pass_idx in 0u32..4 {
            let even_pass = if pass_idx % 2 == 0 { 1u32 } else { 0u32 };
            let odd_pass = if pass_idx % 2 == 1 { 1u32 } else { 0u32 };

            // Update sort_infos: passes = pass_idx, even_pass, odd_pass
            // We need to write the pass index into the buffer.
            // Since we can't write buffers during a compute pass, we write
            // before starting each sub-pass.
            let sort_infos_update = GpuSortInfos {
                keys_size: 0, // Will be kept from preprocess (atomic)
                padded_size: 0,
                passes: pass_idx,
                even_pass,
                odd_pass,
            };
            // Write only the non-atomic fields (offset past keys_size)
            _ctx.queue.write_buffer(
                self.sort_infos_buf,
                4, // skip keys_size (atomic, set by preprocess)
                bytemuck::bytes_of(&sort_infos_update.padded_size),
            );
            // Actually, we need to write padded_size, passes, even_pass, odd_pass
            // which is 4 * u32 = 16 bytes starting at offset 4
            let pass_data = [0u32, pass_idx, even_pass, odd_pass];
            _ctx.queue.write_buffer(self.sort_infos_buf, 4, bytemuck::cast_slice(&pass_data));

            // Histogram
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("GS Sort Histogram"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(sort_histogram);
                cpass.set_bind_group(0, self.sort_bg0, &[]);
                cpass.set_bind_group(1, self.sort_bg1, &[]);
                cpass.dispatch_workgroups(num_sort_wgs as u32, 1, 1);
            }

            // Prefix Sum
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("GS Sort Prefix"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(sort_prefix);
                cpass.set_bind_group(0, self.sort_bg0, &[]);
                cpass.set_bind_group(1, self.sort_bg1, &[]);
                cpass.dispatch_workgroups(prefix_wgs as u32, 1, 1);
            }

            // Scatter
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("GS Sort Scatter"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(sort_scatter);
                cpass.set_bind_group(0, self.sort_bg0, &[]);
                cpass.set_bind_group(1, self.sort_bg1, &[]);
                cpass.dispatch_workgroups(num_sort_wgs as u32, 1, 1);
            }
        }
    }
}

// =============================================================================
// Render PassNode
// =============================================================================

struct GaussianRenderPassNode<'a> {
    render_pipeline: Option<&'a wgpu::RenderPipeline>,
    render_bg0: &'a wgpu::BindGroup,
    render_bg1: &'a wgpu::BindGroup,
    color_target: TextureNodeId,
    depth_target: TextureNodeId,
    num_points: u32,
}

impl PassNode<'_> for GaussianRenderPassNode<'_> {
    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let pipeline = self.render_pipeline.expect("GS render pipeline missing");

        let color_attachment = ctx
            .get_color_attachment(self.color_target, RenderTargetOps::Load, None)
            .expect("GS color target missing");

        let depth_attachment = ctx
            .get_depth_stencil_attachment(self.depth_target, 0.0);

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("GS Render"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: depth_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(pipeline);
        rpass.set_bind_group(0, self.render_bg0, &[]);
        rpass.set_bind_group(1, self.render_bg1, &[]);
        // 4 vertices per quad × num_points quads
        rpass.draw(0..self.num_points * 4, 0..1);
    }
}
