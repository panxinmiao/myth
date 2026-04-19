//! SSSS Feature + Ephemeral PassNodes (Flattened)
//!
//! - **`SsssFeature`** (long-lived): owns pipeline cache, bind group layout,
//!   profiles storage buffer.  `extract_and_prepare()` compiles pipelines and
//!   uploads SSS profile data.
//! - **`SsssHorizontalNode`** / **`SsssVerticalNode`** (ephemeral per-frame):
//!   two independent RDG passes created by `SsssFeature::add_to_graph()`.
//!
//! Implements a separable Gaussian blur for SSS materials identified via
//! a **data-driven soft mask**: the `Feature_ID` colour attachment written
//! by the Prepass carries per-pixel `sss_id` / `ssr_id`.  The shader reads
//! this texture and performs an early-out for non-SSS pixels, replacing the
//! former hardware stencil test with zero cross-platform compatibility cost.
//!
//! # Data Flow (explicit wiring)
//!
//! ```text
//!  Opaque                            SsssPassNode
//!       |                 +-----------------------------------+
//! color_in  ------------>|  H Sub-Pass: Horizontal blur      |---> temp_blur
//! normal_in ----+------->|                                   |         |
//! depth_in  ----|        |  V Sub-Pass: Vertical blur        |<--------+
//! feature_id ---|        |  (writes back to color_in         |
//! specular_tex -+        |   in-place)                       |---> color_in
//!                        +-----------------------------------+
//! ```
//!
//! # Integration
//!
//! Must come **after** `OpaquePass` and **before** the Skybox/MSAA-Sync
//! stage in the `HighFidelity` render path.  The Feature only calls
//! `add_to_graph` when SSS is enabled — zero cost when disabled.
//!
//! # GPU Resources
//!
//! - **Profiles StorageBuffer**: 256 x `SssProfileData` (12 KB).
//!   Uploaded from `AssetServer.sss_registry`.
//! - **temp_blur TransientTexture**: `Rgba16Float`, same size as scene colour.

use crate::HDR_TEXTURE_FORMAT;
use crate::core::gpu::{CommonSampler, Tracked};
use crate::graph::composer::GraphBuilderContext;
use crate::graph::core::{
    ExecuteContext, ExtractContext, PassNode, PrepareContext, RenderTargetOps, TextureDesc,
    TextureNodeId,
};
use crate::pipeline::{
    ColorTargetKey, FullscreenPipelineKey, RenderPipelineId, ShaderCompilationOptions, ShaderSource,
};
use myth_resources::screen_space::SssProfileData;
use std::mem::size_of;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Feature (long-lived, stored in RenderFeatures)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Long-lived SSSS feature — owns persistent GPU resources (pipelines,
/// bind group layout, profiles buffer).
///
/// Produces an ephemeral [`SsssPassNode`] each frame via [`Self::add_to_graph`].
pub struct SsssFeature {
    // --- Pipelines -------------------------------------------------
    horizontal_pipeline: Option<RenderPipelineId>,
    vertical_pipeline: Option<RenderPipelineId>,
    bind_group_layout: Option<Tracked<wgpu::BindGroupLayout>>,

    // --- Persistent GPU Resources ----------------------------------
    profiles_buffer: Option<Tracked<wgpu::Buffer>>,
    last_registry_version: u64,
}

impl Default for SsssFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl SsssFeature {
    #[must_use]
    pub fn new() -> Self {
        Self {
            horizontal_pipeline: None,
            vertical_pipeline: None,
            bind_group_layout: None,
            profiles_buffer: None,
            last_registry_version: 0,
        }
    }

    /// Pre-RDG resource preparation: create layout, compile pipelines,
    /// upload SSS profile data.
    pub fn extract_and_prepare(&mut self, ctx: &mut ExtractContext) {
        let device = ctx.device;

        // -- 1. Lazy-create profiles storage buffer ---------------------
        if self.profiles_buffer.is_none() {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SSS Profiles Buffer"),
                size: (256 * size_of::<SssProfileData>()) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.profiles_buffer = Some(Tracked::new(buffer));
        }

        // -- 2. Diff-sync profiles data ---------------------------------
        let registry = ctx.assets.sss_registry.read();
        if self.last_registry_version != registry.version {
            ctx.queue.write_buffer(
                self.profiles_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&registry.buffer_data),
            );
            self.last_registry_version = registry.version;
        }

        // -- 3. Lazy-create pipelines + layout --------------------------
        if self.horizontal_pipeline.is_none() {
            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("SSSS Bind Group Layout"),
                    entries: &[
                        // binding 0: t_color
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
                        // binding 1: t_normal
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // binding 2: t_depth
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Depth,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // binding 3: u_profiles (storage)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 4: s_sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        // binding 5: t_feature_id
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Uint,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // binding 6: t_specular
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                    ],
                });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SSSS Pipeline Layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

            // -- Horizontal shader (no defines) -------------------------
            let shader_defines = ShaderCompilationOptions::default();

            let (hor_shader, hor_hash) = ctx.shader_manager.get_or_compile(
                device,
                ShaderSource::File("entry/post_process/ssss"),
                &shader_defines,
            );

            let color_target = ColorTargetKey::from(wgpu::ColorTargetState {
                format: HDR_TEXTURE_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            });

            // No depth-stencil state — pixel filtering uses the soft mask
            // (Feature_ID texture) inside the fragment shader.
            let hor_key = FullscreenPipelineKey::fullscreen(
                hor_hash,
                smallvec::smallvec![color_target.clone()],
                None,
            );

            self.horizontal_pipeline = Some(ctx.pipeline_cache.get_or_create_fullscreen(
                device,
                hor_shader,
                &pipeline_layout,
                &hor_key,
                "SSSS Horizontal Pipeline",
            ));

            // -- Vertical shader (SSSS_VERTICAL_PASS = 1) -------------
            let mut vert_defines = ShaderCompilationOptions::default();
            vert_defines.add_define("SSSS_VERTICAL_PASS", "1");

            let (vert_shader, vert_hash) = ctx.shader_manager.get_or_compile(
                device,
                ShaderSource::File("entry/post_process/ssss"),
                &vert_defines,
            );

            let vert_key = FullscreenPipelineKey::fullscreen(
                vert_hash,
                smallvec::smallvec![color_target],
                None,
            );

            self.vertical_pipeline = Some(ctx.pipeline_cache.get_or_create_fullscreen(
                device,
                vert_shader,
                &pipeline_layout,
                &vert_key,
                "SSSS Vertical Pipeline",
            ));

            self.bind_group_layout = Some(Tracked::new(bind_group_layout));
        }
    }

    /// Build the ephemeral pass nodes and insert them into the graph as
    /// two independent RDG passes within an `"SSSS_System"` group.
    ///
    /// All inputs are explicitly wired — no blackboard lookups.
    /// The pass modifies `scene_color` in-place (read + write via alias).
    ///
    /// # Flattened Pass Chain
    ///
    /// 1. **SSSS_Blur_H** — horizontal scatter: `scene_color` → `temp_blur`
    /// 2. **SSSS_Blur_V** — vertical scatter: `temp_blur` → `scene_color` alias
    pub fn add_to_graph<'a>(
        &'a self,
        ctx: &mut GraphBuilderContext<'a, '_>,
        scene_color: TextureNodeId,
        scene_depth: TextureNodeId,
        scene_normals: TextureNodeId,
        feature_id: TextureNodeId,
        specular_tex: TextureNodeId,
    ) -> TextureNodeId {
        let horizontal_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.horizontal_pipeline.expect("SsssFeature not prepared"));
        let vertical_pipeline = ctx
            .pipeline_cache
            .get_render_pipeline(self.vertical_pipeline.expect("SsssFeature not prepared"));
        let bind_group_layout = self.bind_group_layout.as_ref().unwrap();
        let profiles_buffer = self.profiles_buffer.as_ref().unwrap();

        ctx.with_group("SSSS_System", |ctx| {
            // ─── Pass 1: Horizontal blur ───────────────────────────
            let fc = ctx.frame_config;
            let (w, h) = (fc.width, fc.height);
            let hdr_format = fc.hdr_format;
            let temp_desc = TextureDesc::new_2d(
                w,
                h,
                hdr_format,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );

            let temp_blur: TextureNodeId = ctx.graph.add_pass("SSSS_Blur_H", |builder| {
                builder.read_texture(scene_color);
                builder.read_texture(scene_depth);
                builder.read_texture(scene_normals);
                builder.read_texture(feature_id);
                let out = builder.create_texture("SSSS_Temp", temp_desc);
                let node = SsssHorizontalNode {
                    scene_color_in: scene_color,
                    temp_blur: out,
                    depth_in: scene_depth,
                    normal_in: scene_normals,
                    feature_id,
                    specular_tex,
                    pipeline: horizontal_pipeline,
                    bind_group_layout,
                    profiles_buffer,
                    bind_group: None,
                };
                (node, out)
            });

            // ─── Pass 2: Vertical blur ─────────────────────────────
            let ssss_out: TextureNodeId = ctx.graph.add_pass("SSSS_Blur_V", |builder| {
                builder.read_texture(temp_blur);
                builder.read_texture(scene_depth);
                builder.read_texture(scene_normals);
                builder.read_texture(feature_id);
                builder.read_texture(specular_tex);
                let out = builder.mutate_texture(scene_color, "Scene_Color_SSSS");
                let node = SsssVerticalNode {
                    scene_color_out: out,
                    temp_blur,
                    depth_in: scene_depth,
                    normal_in: scene_normals,
                    feature_id,
                    specular_tex,
                    pipeline: vertical_pipeline,
                    bind_group_layout,
                    profiles_buffer,
                    bind_group: None,
                };
                (node, out)
            });

            ssss_out
        })
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Pass 1: SsssHorizontalNode (ephemeral, created per frame)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Ephemeral per-frame node for the horizontal SSS scatter pass.
///
/// Reads scene colour and writes to the scratch `temp_blur` texture.
/// Non-SSS pixels are skipped via shader early-out on `Feature_ID`.
struct SsssHorizontalNode<'a> {
    scene_color_in: TextureNodeId,
    temp_blur: TextureNodeId,
    depth_in: TextureNodeId,
    normal_in: TextureNodeId,
    feature_id: TextureNodeId,
    specular_tex: TextureNodeId,

    pipeline: &'a wgpu::RenderPipeline,
    bind_group_layout: &'a Tracked<wgpu::BindGroupLayout>,
    profiles_buffer: &'a Tracked<wgpu::Buffer>,

    bind_group: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for SsssHorizontalNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.bind_group = Some(crate::myth_bind_group!(ctx, self.bind_group_layout, Some("SSSS Horizontal Bind Group"), [
            0 => self.scene_color_in,
            1 => self.normal_in,
            2 => self.depth_in,
            3 => self.profiles_buffer,
            4 => CommonSampler::LinearClamp,
            5 => self.feature_id,
            6 => self.specular_tex,
        ]));
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let rtt = ctx.get_color_attachment(
            self.temp_blur,
            RenderTargetOps::Clear(wgpu::Color::TRANSPARENT),
            None,
        );

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSSS Horizontal"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        rpass.set_pipeline(self.pipeline);
        rpass.set_bind_group(0, self.bind_group.unwrap(), &[]);
        rpass.draw(0..3, 0..1);
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Pass 2: SsssVerticalNode (ephemeral, created per frame)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Ephemeral per-frame node for the vertical SSS scatter pass.
///
/// Reads the `temp_blur` scratch texture and writes back to the scene colour
/// alias (via `mutate_texture`).  Non-SSS pixels pass through unchanged
/// thanks to the shader soft-mask on `Feature_ID`.
struct SsssVerticalNode<'a> {
    scene_color_out: TextureNodeId,
    temp_blur: TextureNodeId,
    depth_in: TextureNodeId,
    normal_in: TextureNodeId,
    feature_id: TextureNodeId,
    specular_tex: TextureNodeId,

    pipeline: &'a wgpu::RenderPipeline,
    bind_group_layout: &'a Tracked<wgpu::BindGroupLayout>,
    profiles_buffer: &'a Tracked<wgpu::Buffer>,

    bind_group: Option<&'a wgpu::BindGroup>,
}

impl<'a> PassNode<'a> for SsssVerticalNode<'a> {
    fn prepare(&mut self, ctx: &mut PrepareContext<'a>) {
        self.bind_group = Some(crate::myth_bind_group!(ctx, self.bind_group_layout, Some("SSSS Vertical Bind Group"), [
            0 => self.temp_blur,
            1 => self.normal_in,
            2 => self.depth_in,
            3 => self.profiles_buffer,
            4 => CommonSampler::LinearClamp,
            5 => self.feature_id,
            6 => self.specular_tex,
        ]));
    }

    fn execute(&self, ctx: &ExecuteContext, encoder: &mut wgpu::CommandEncoder) {
        let rtt = ctx.get_color_attachment(self.scene_color_out, RenderTargetOps::Load, None);

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("SSSS Vertical"),
            color_attachments: &[rtt],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        rpass.set_pipeline(self.pipeline);
        rpass.set_bind_group(0, self.bind_group.unwrap(), &[]);
        rpass.draw(0..3, 0..1);
    }
}
