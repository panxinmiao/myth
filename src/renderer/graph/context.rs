//! Render Graph Context System
//!
//! Provides two phase-separated contexts for the render graph:
//!
//! - [`PrepareContext`]: Mutable context for the **prepare** phase. Owns exclusive
//!   write access to resource managers, pipeline caches and scene state. Passes
//!   allocate GPU resources, compile shaders and build bind groups here.
//!
//! - [`ExecuteContext`]: Immutable context for the **execute** phase. Provides
//!   read-only access to all rendering data. Passes record GPU commands here.
//!   Ping-pong state uses [`Cell`] for interior mutability.
//!
//! # Design Principles
//!
//! 1. **Strict Read/Write Separation**: `PrepareContext` is the only place where
//!    GPU resources may be created or mutated. `ExecuteContext` is purely read-only
//!    (except for the ping-pong counter via `Cell`).
//! 2. **Field-Level Borrow Splitting**: Both contexts store individual references
//!    to engine subsystems. The Rust borrow checker can split borrows across
//!    disjoint fields, enabling concurrent immutable access to e.g.
//!    `resource_manager` and `pipeline_cache` within the same call.

use crate::assets::AssetServer;
use crate::renderer::core::binding::{BindGroupKey, GlobalBindGroupCache};
use crate::renderer::core::resources::Tracked;
use crate::renderer::core::{ResourceManager, WgpuContext};
use crate::renderer::graph::frame::{FrameBlackboard, RenderLists};
use crate::renderer::graph::transient_pool::TransientTexturePool;
use crate::renderer::graph::{ExtractedScene, RenderState};
use crate::renderer::pipeline::PipelineCache;
use crate::scene::Scene;
use crate::scene::camera::RenderCamera;

// ─── Graph Resource Enum ──────────────────────────────────────────────────────

/// Logical identifier for graph-managed GPU texture resources.
///
/// Passes use this enum with [`PrepareContext::get_resource_view`] /
/// [`ExecuteContext::get_resource_view`] to request texture views by *semantic
/// role* rather than by hard-coded field paths. This decouples inter-pass data
/// flow from the physical storage layout in [`FrameResources`].
///
/// # Resource Categories
///
/// | Variant | Availability | Description |
/// |---------|-------------|-------------|
/// | `SceneColorInput` | HDR only | Current ping-pong **read** buffer |
/// | `SceneColorOutput` | HDR only | Current ping-pong **write** buffer |
/// | `SceneDepth` | Always | Main depth buffer (reverse-Z) |
/// | `SceneMsaa` | MSAA > 1 | Multi-sample intermediate color buffer |
/// | `SceneRenderTarget` | Always¹ | Primary scene color target |
/// | `Transmission` | On demand | Transmission copy buffer |
/// | `Surface` | Execute only | Swap-chain output view |
///
/// ¹ In HDR mode resolves to `scene_color_view[0]`; in LDR mode resolves to
///   the surface view (execute only) — calling from `PrepareContext` in LDR
///   mode will panic.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum GraphResource {
    /// Current ping-pong input for post-processing reads (HDR only).
    SceneColorInput,
    /// Current ping-pong output for post-processing writes (HDR only).
    SceneColorOutput,
    /// Scene depth buffer (reverse-Z, always available).
    // SceneDepth,
    DepthOnly,

    DepthStencil,
    /// MSAA intermediate color buffer (only when MSAA is enabled).
    SceneMsaa,
    /// Primary scene render target.
    ///
    /// - HDR: `scene_color_view[0]`
    /// - LDR + Execute: swap-chain surface view
    /// - LDR + Prepare: **panics** (surface not yet available)
    SceneRenderTarget,
    // /// Transmission copy texture (only when transmission effects are active).
    // Transmission,
    /// Swap-chain output surface view (execute phase only).
    Surface,
}

// ─── Prepare Context ──────────────────────────────────────────────────────────

/// Mutable context available during the **prepare** phase.
///
/// Passes use this to allocate GPU resources, create/update pipelines,
/// upload uniforms, perform visibility culling, and build bind groups.
///
/// # Ping-Pong State
///
/// The `color_view_flip_flop` counter tracks which scene color buffer is
/// currently the "input" and which is the "output" for post-processing
/// passes. Passes that write to the output buffer (e.g. Bloom) must call
/// [`flip_scene_color`](Self::flip_scene_color) at the end of `prepare`
/// so that subsequent passes see the correct input.
pub struct PrepareContext<'a> {
    /// WGPU core context (device, queue, surface configuration)
    pub wgpu_ctx: &'a WgpuContext,
    /// GPU resource manager (buffers, textures, bind groups)
    pub resource_manager: &'a mut ResourceManager,
    /// Pipeline cache (L1 fast cache + L2 canonical cache)
    pub pipeline_cache: &'a mut PipelineCache,
    /// Asset server (geometries, materials, textures)
    pub assets: &'a AssetServer,
    /// Current scene (mutable for light storage updates etc.)
    pub scene: &'a mut Scene,
    /// Active render camera
    pub camera: &'a RenderCamera,
    /// Per-frame render state (uniforms, version ID)
    pub render_state: &'a RenderState,
    /// Extracted scene data (render items, lights, defines)
    pub extracted_scene: &'a ExtractedScene,
    /// Render command lists (filled by SceneCullPass, consumed by draw passes)
    pub render_lists: &'a mut RenderLists,
    /// Frame blackboard for cross-pass transient data (SSAO / Transmission IDs)
    pub blackboard: &'a mut FrameBlackboard,
    /// Frame-persistent GPU resources (ping-pong buffers, depth, MSAA)
    pub frame_resources: &'a FrameResources,
    /// Transient texture pool (per-frame temporary allocations)
    pub transient_pool: &'a mut TransientTexturePool,
    /// Current time in seconds
    pub time: f32,
    /// Global bind group cache (cross-pass deduplication)
    pub global_bind_group_cache: &'a mut GlobalBindGroupCache,
    /// Ping-pong counter for post-processing I/O selection.
    pub(crate) color_view_flip_flop: usize,
}

impl PrepareContext<'_> {
    /// Returns the current "input" scene color texture (previous pass output).
    #[must_use]
    #[inline]
    pub fn get_scene_color_input(&self) -> &Tracked<wgpu::TextureView> {
        &self.frame_resources.scene_color_view[self.color_view_flip_flop]
    }

    /// Returns the current "output" scene color texture (current pass target).
    #[must_use]
    #[inline]
    pub fn get_scene_color_output(&self) -> &Tracked<wgpu::TextureView> {
        &self.frame_resources.scene_color_view[1 - self.color_view_flip_flop]
    }

    /// Flips the ping-pong state.
    ///
    /// Call this at the end of `prepare` in passes that write to
    /// `get_scene_color_output()` (e.g. Bloom), so that subsequent
    /// passes see the updated buffer as their input.
    #[inline]
    pub fn flip_scene_color(&mut self) {
        self.color_view_flip_flop = 1 - self.color_view_flip_flop;
    }

    /// Returns the appropriate render target format for scene geometry.
    #[must_use]
    #[inline]
    pub fn get_scene_render_target_format(&self) -> wgpu::TextureFormat {
        self.wgpu_ctx
            .render_path
            .main_color_format(self.wgpu_ctx.surface_view_format)
    }

    /// Returns the appropriate render target for scene geometry.
    ///
    /// In HighFidelity path returns `scene_color_view[0]`; in BasicForward path this is
    /// unavailable (no surface_view during prepare), so panics.
    #[must_use]
    #[inline]
    pub fn get_scene_render_target_view(&self) -> &wgpu::TextureView {
        debug_assert!(
            self.wgpu_ctx.render_path.supports_post_processing(),
            "get_scene_render_target_view() during prepare is only valid in HighFidelity path"
        );
        &self.frame_resources.scene_color_view[0]
    }

    // ── GraphResource API ──────────────────────────────────────────────────

    /// Retrieve a texture view by its logical [`GraphResource`] identifier.
    ///
    /// # Panics
    ///
    /// - `SceneMsaa` when MSAA is disabled.
    /// - `Transmission` when no transmission resource exists.
    /// - `Surface` (not available during prepare).
    /// - `SceneRenderTarget` in BasicForward path (no surface yet).
    #[must_use]
    #[inline]
    pub fn get_resource_view(&self, resource: GraphResource) -> &Tracked<wgpu::TextureView> {
        match resource {
            GraphResource::SceneColorInput => self.get_scene_color_input(),
            GraphResource::SceneColorOutput => self.get_scene_color_output(),
            // GraphResource::SceneDepth => &self.frame_resources.depth_view,
            GraphResource::DepthOnly => &self.frame_resources.depth_only_view,
            GraphResource::DepthStencil => &self.frame_resources.depth_view,
            GraphResource::SceneMsaa => self
                .frame_resources
                .scene_msaa_view
                .as_ref()
                .expect("SceneMsaa requested but MSAA is disabled"),
            GraphResource::SceneRenderTarget => {
                debug_assert!(
                    self.wgpu_ctx.render_path.supports_post_processing(),
                    "SceneRenderTarget during prepare is only valid in HighFidelity path"
                );
                &self.frame_resources.scene_color_view[0]
            }
            GraphResource::Surface => {
                panic!("GraphResource::Surface is not available during the prepare phase")
            }
        }
    }

    /// Try to retrieve a texture view, returning `None` for unavailable
    /// optional resources (`SceneMsaa`, `SceneNormal`, `Transmission`).
    #[must_use]
    #[inline]
    pub fn try_get_resource_view(
        &self,
        resource: GraphResource,
    ) -> Option<&Tracked<wgpu::TextureView>> {
        match resource {
            GraphResource::SceneMsaa => self.frame_resources.scene_msaa_view.as_ref(),
            GraphResource::Surface => None, // Transient; use blackboard.transmission_texture_id
            _ => Some(self.get_resource_view(resource)),
        }
    }
}

// ─── Execute Context ──────────────────────────────────────────────────────────

/// Read-only context available during the **execute** phase.
///
/// Passes use this to record GPU commands into a `CommandEncoder`. All
/// resource allocation should have been completed during the prepare phase;
/// this context provides only shared references to engine subsystems.
///
/// # Ping-Pong State
///
/// The `color_view_flip_flop` field uses [`Cell`] for interior mutability,
/// allowing passes to call [`flip_scene_color`](Self::flip_scene_color)
/// through `&self` without requiring `&mut self` on the context.
pub struct ExecuteContext<'a> {
    /// WGPU core context (device, queue, surface configuration)
    pub wgpu_ctx: &'a WgpuContext,
    /// GPU resource manager (read-only access during execute)
    pub resource_manager: &'a ResourceManager,
    /// Current frame's surface texture view
    pub surface_view: &'a wgpu::TextureView,
    /// Render command lists (read-only; filled during prepare)
    pub render_lists: &'a RenderLists,
    /// Frame blackboard (read-only during execute)
    pub blackboard: &'a FrameBlackboard,
    /// Frame-persistent GPU resources
    pub frame_resources: &'a FrameResources,
    /// Transient texture pool (read-only during execute)
    pub transient_pool: &'a TransientTexturePool,
}

impl<'a> ExecuteContext<'a> {
    /// Creates a new `ExecuteContext`.
    pub(crate) fn new(
        wgpu_ctx: &'a WgpuContext,
        resource_manager: &'a ResourceManager,
        surface_view: &'a wgpu::TextureView,
        render_lists: &'a RenderLists,
        blackboard: &'a FrameBlackboard,
        frame_resources: &'a FrameResources,
        transient_pool: &'a TransientTexturePool,
    ) -> Self {
        Self {
            wgpu_ctx,
            resource_manager,
            surface_view,
            render_lists,
            blackboard,
            frame_resources,
            transient_pool,
        }
    }
    /// Returns the appropriate render target for scene geometry.
    ///
    /// In HDR mode returns `scene_color_view[0]`; otherwise returns the surface.
    #[must_use]
    #[inline]
    pub fn get_scene_render_target_view(&self) -> &wgpu::TextureView {
        if self.wgpu_ctx.render_path.supports_post_processing() {
            &self.frame_resources.scene_color_view[0]
        } else {
            self.surface_view
        }
    }

    /// Returns the texture format of the scene render target.
    #[must_use]
    #[inline]
    pub fn get_scene_render_target_format(&self) -> wgpu::TextureFormat {
        self.wgpu_ctx
            .render_path
            .main_color_format(self.wgpu_ctx.surface_view_format)
    }

    // ── GraphResource API ──────────────────────────────────────────────────

    /// Retrieve a texture view by its logical [`GraphResource`] identifier.
    ///
    /// # Panics
    ///
    /// - `SceneMsaa` when MSAA is disabled.
    /// - `Transmission` when no transmission resource exists.
    #[must_use]
    #[inline]
    pub fn get_resource_view(&self, resource: GraphResource) -> &Tracked<wgpu::TextureView> {
        match resource {
            GraphResource::SceneColorInput | GraphResource::SceneColorOutput => {
                panic!(
                    "Ping-pong resources must be resolved and cached during the 'prepare' phase!"
                );
            }
            GraphResource::DepthStencil => &self.frame_resources.depth_view,
            GraphResource::DepthOnly => &self.frame_resources.depth_only_view,
            GraphResource::SceneMsaa => self
                .frame_resources
                .scene_msaa_view
                .as_ref()
                .expect("SceneMsaa requested but MSAA is disabled"),
            GraphResource::SceneRenderTarget => {
                if self.wgpu_ctx.render_path.supports_post_processing() {
                    &self.frame_resources.scene_color_view[0]
                } else {
                    // LDR mode: surface_view is not Tracked, use SceneRenderTarget
                    // via get_scene_render_target_view() instead.
                    panic!(
                        "Use get_scene_render_target_view() for LDR SceneRenderTarget \
                         (surface is not a Tracked resource)"
                    )
                }
            }
            GraphResource::Surface => {
                // surface_view is not Tracked — use `surface_view` field directly.
                panic!(
                    "GraphResource::Surface is not a Tracked resource; \
                     use ctx.surface_view directly"
                )
            }
        }
    }

    /// Try to retrieve a texture view, returning `None` for unavailable
    /// optional resources (`SceneMsaa`, `SceneNormal`, `Transmission`).
    ///
    /// Returns `None` for `Surface` and LDR `SceneRenderTarget` (not Tracked).
    #[must_use]
    #[inline]
    pub fn try_get_resource_view(
        &self,
        resource: GraphResource,
    ) -> Option<&Tracked<wgpu::TextureView>> {
        match resource {
            GraphResource::SceneMsaa => self.frame_resources.scene_msaa_view.as_ref(),
            GraphResource::Surface => None,
            GraphResource::SceneRenderTarget
                if !self.wgpu_ctx.render_path.supports_post_processing() =>
            {
                None
            }
            _ => Some(self.get_resource_view(resource)),
        }
    }
}

// ─── Frame Resources ──────────────────────────────────────────────────────────

pub struct FrameResources {
    // MSAA buffer (optional)
    pub scene_msaa_view: Option<Tracked<wgpu::TextureView>>,

    // Main scene color buffer (HDR)
    // Ping-pong mechanism: when not in straightforward mode, two alternating buffers serve as post-processing input/output
    pub scene_color_view: [Tracked<wgpu::TextureView>; 2],
    // Depth buffer
    pub depth_view: Tracked<wgpu::TextureView>,

    pub depth_only_view: Tracked<wgpu::TextureView>,
    /// BindGroupLayout for group 3 (transmission + sampler + SSAO).
    /// Persistent — created once, reused across frames.
    pub screen_bind_group_layout: Tracked<wgpu::BindGroupLayout>,

    /// Shared linear-clamp sampler for transmission / SSAO sampling.
    pub screen_sampler: Tracked<wgpu::Sampler>,

    /// 1×1 white (R8Unorm = 255) dummy texture for SSAO-disabled fallback.
    pub ssao_dummy_view: Tracked<wgpu::TextureView>,

    /// 1×1 placeholder texture (Rgba16Float) for transmission-disabled fallback.
    pub dummy_transmission_view: Tracked<wgpu::TextureView>,

    size: (u32, u32),
}

impl FrameResources {
    pub fn new(wgpu_ctx: &WgpuContext, size: (u32, u32)) -> Self {
        let device = &wgpu_ctx.device;

        // 1. Create layout (group 3: transmission + sampler + SSAO)
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Screen/Transmission Layout"),
            entries: &[
                // Binding 0: Transmission Texture
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
                // Binding 1: Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Binding 2: SSAO Texture (R8Unorm, always bound — white dummy when disabled)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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

        // 2. Create shared sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Transmission Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let placeholder_view = Self::create_texture_view(
            device,
            (1, 1),
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            1,
            1,
            "Placeholder Texture",
        );

        // SSAO dummy: 1×1 white R8Unorm texture (AO = 1.0 = fully lit)
        let ssao_dummy_view = Tracked::new(Self::create_initialized_r8_white(device));

        // Transmission dummy: 1×1 HDR placeholder (black)
        let dummy_transmission_view = Tracked::new(Self::create_texture_view(
            device,
            (1, 1),
            crate::renderer::HDR_TEXTURE_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING,
            1,
            1,
            "Transmission Dummy Texture",
        ));

        let mut resources = Self {
            size: (0, 0),

            depth_view: Tracked::new(placeholder_view.clone()),
            depth_only_view: Tracked::new(placeholder_view.clone()),
            scene_msaa_view: None,
            scene_color_view: [
                Tracked::new(placeholder_view.clone()),
                Tracked::new(placeholder_view.clone()),
            ],

            screen_bind_group_layout: Tracked::new(layout),
            screen_sampler: Tracked::new(sampler),
            ssao_dummy_view,
            dummy_transmission_view,
        };

        resources.resize(wgpu_ctx, size);
        resources
    }

    fn create_texture_view(
        device: &wgpu::Device,
        size: (u32, u32),
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        sample_count: u32,
        mip_level_count: u32,
        label: &str,
    ) -> wgpu::TextureView {
        let texture = Self::create_texture(
            device,
            size,
            format,
            usage,
            sample_count,
            mip_level_count,
            label,
        );

        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn create_texture(
        device: &wgpu::Device,
        size: (u32, u32),
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        sample_count: u32,
        mip_level_count: u32,
        label: &str,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }

    /// Creates a 1×1 R8Unorm texture initialized to 255 (white = no occlusion).
    fn create_initialized_r8_white(device: &wgpu::Device) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSAO White Dummy"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Build (or retrieve from cache) the screen bind group (group 3) for the
    /// given transmission and SSAO texture views.
    ///
    /// Returns `(bind_group, bind_group_id)` suitable for `TrackedRenderPass`.
    /// The `bind_group_id` is a composite of the resource IDs, ensuring that
    /// `TrackedRenderPass` skips redundant `set_bind_group` calls when the
    /// same views are reused across draw commands.
    pub fn build_screen_bind_group(
        &self,
        device: &wgpu::Device,
        cache: &mut GlobalBindGroupCache,
        transmission_view: &Tracked<wgpu::TextureView>,
        ssao_view: &Tracked<wgpu::TextureView>,
    ) -> (wgpu::BindGroup, u64) {
        let layout_id = self.screen_bind_group_layout.id();
        let sampler_id = self.screen_sampler.id();

        let key = BindGroupKey::new(layout_id)
            .with_resource(transmission_view.id())
            .with_resource(sampler_id)
            .with_resource(ssao_view.id());

        // Composite ID for TrackedRenderPass state tracking
        let bind_group_id = transmission_view
            .id()
            .wrapping_mul(6_364_136_223_846_793_005)
            ^ ssao_view.id().wrapping_mul(1_442_695_040_888_963_407)
            ^ sampler_id;

        let layout = &*self.screen_bind_group_layout;
        let sampler = &*self.screen_sampler;
        let tv = &**transmission_view;
        let sv = &**ssao_view;

        let bg = cache
            .get_or_create(key, || {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Screen BindGroup (Dynamic)"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(tv),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(sv),
                        },
                    ],
                })
            })
            .clone();

        (bg, bind_group_id)
    }

    /// Forces recreation of all frame resources regardless of size.
    ///
    /// Call this when render settings change (HDR mode, MSAA) that require
    /// texture format or sample count changes.
    pub fn force_recreate(&mut self, wgpu_ctx: &WgpuContext, size: (u32, u32)) {
        if size.0 == 0 || size.1 == 0 {
            return;
        }
        // Reset size to force recreation
        self.size = (0, 0);
        self.resize(wgpu_ctx, size);
    }

    #[allow(clippy::too_many_lines)]
    pub fn resize(&mut self, wgpu_ctx: &WgpuContext, size: (u32, u32)) {
        if self.size == size {
            return;
        }
        if size.0 == 0 || size.1 == 0 {
            return;
        }

        self.size = size;

        // Determine the actual sample count for rendering
        let render_sample_count = if wgpu_ctx.msaa_samples > 1 {
            wgpu_ctx.msaa_samples
        } else {
            1
        };

        // Depth Texture - sample count must match the render target
        let depth_view = Self::create_texture_view(
            &wgpu_ctx.device,
            size,
            wgpu_ctx.depth_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            render_sample_count,
            1,
            "Depth Texture",
        );

        self.depth_only_view = Tracked::new(depth_view.texture().create_view(
            &wgpu::TextureViewDescriptor {
                label: Some("Depth-Only View"),
                aspect: wgpu::TextureAspect::DepthOnly,
                ..Default::default()
            },
        ));

        self.depth_view = Tracked::new(depth_view);

        // Scene Color Texture(s) (ping-pong)
        if wgpu_ctx.render_path.supports_post_processing() {
            let ping_pong_texture_0 = Self::create_texture_view(
                &wgpu_ctx.device,
                size,
                crate::renderer::HDR_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                1,
                1,
                "Ping-Pong Texture 0",
            );

            let ping_pong_texture_1 = Self::create_texture_view(
                &wgpu_ctx.device,
                size,
                crate::renderer::HDR_TEXTURE_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                1,
                1,
                "Ping-Pong Texture 1",
            );
            self.scene_color_view = [
                Tracked::new(ping_pong_texture_0),
                Tracked::new(ping_pong_texture_1),
            ];
        }

        // MSAA Texture
        if wgpu_ctx.msaa_samples > 1 {
            let masaa_target_fromat = wgpu_ctx
                .render_path
                .main_color_format(wgpu_ctx.surface_view_format);

            let scene_msaa_view = Self::create_texture_view(
                &wgpu_ctx.device,
                size,
                masaa_target_fromat,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
                wgpu_ctx.msaa_samples,
                1,
                "Scene MSAA Color Texture",
            );
            self.scene_msaa_view = Some(Tracked::new(scene_msaa_view));
        } else {
            self.scene_msaa_view = None;
        }

        // Transmission and SSAO textures are now transient —
        // no per-resize allocation needed here.
    }
}
