//! Render frame management.
//!
//! `RenderFrame` is responsible for:
//! - Holding extracted scene data ([`ExtractedScene`]) and render state ([`RenderState`])
//! - Running the Extract and Prepare phases of the frame pipeline
//!
//! # Three-Phase Rendering Architecture
//!
//! The rendering pipeline is divided into three distinct phases:
//!
//! 1. **Prepare**: `extract_and_prepare()` — extract scene data and prepare GPU resources
//! 2. **Compose**: Chain render nodes via [`FrameComposer`]
//! 3. **Execute**: `FrameComposer::render()` — acquire the surface and submit GPU commands
//!
//! # Example
//!
//! ```ignore
//! renderer.begin_frame(scene, &camera, assets, time)?
//!     .add_node(RenderStage::UI, &ui_pass)
//!     .render();
//! ```

use glam::Mat4;
use rustc_hash::FxHashMap;

use crate::assets::{AssetServer, GeometryHandle, MaterialHandle};
use crate::renderer::core::{BindGroupContext, RenderView, ResourceManager};
use crate::renderer::graph::transient_pool::TransientTextureId;
use crate::scene::Scene;
use crate::scene::camera::RenderCamera;

use super::extracted::ExtractedScene;
use super::render_state::RenderState;

// ============================================================================
// FrameBlackboard
// ============================================================================

/// Frame blackboard for passing loose transient data between render passes.
///
/// `FrameBlackboard` serves as a per-frame communication bridge, storing
/// transient resource IDs (e.g. SSAO output, transmission copy) that are
/// shared across passes. Its lifetime is strictly limited to a single
/// frame — [`clear`](Self::clear) must be called at the start of each frame.
///
/// # Design Principles
///
/// - **Single Responsibility**: Decouples loose cross-pass transient IDs
///   from [`RenderLists`], keeping `RenderLists` focused on draw call management.
/// - **Declarative-Ready**: Lays the groundwork for future migration to a
///   declarative render graph.
/// - **Zero-Cost Abstraction**: Contains only `Option<TransientTextureId>` (`Copy`)
///   fields; clearing is a zero-overhead assignment.
#[derive(Default)]
pub struct FrameBlackboard {
    /// Transient texture ID of the current frame's SSAO blur output.
    ///
    /// Written by [`SsaoPass::prepare()`], read by `OpaquePass` / `TransparentPass`
    /// when building their group 3 bind groups. `None` means SSAO is disabled this frame.
    pub ssao_texture_id: Option<TransientTextureId>,

    /// Transient texture ID of the current frame's transmission copy.
    ///
    /// Written by [`TransmissionCopyPass::prepare()`], read by `TransparentPass`
    /// when building its group 3 bind group. `None` means no transmission effect this frame.
    pub transmission_texture_id: Option<TransientTextureId>,

    /// Transient texture ID for scene normals (HighFidelity path, consumed by SSAO).
    ///
    /// Written by `DepthNormalPrepass`, read by `SsaoPass`.
    pub scene_normal_texture_id: Option<TransientTextureId>,

    /// Transient texture ID for the screen-space feature ID texture.
    ///
    /// Written by `Prepass` in `rguint8` format; the R channel stores `sss_id`,
    /// the G channel stores `ssr_id`. Consumed by screen-space effect passes.
    pub feature_id_texture_id: Option<TransientTextureId>,

    /// SSSSS ping-pong texture ID.
    pub sssss_pingpong_texture_id: Option<TransientTextureId>,

    /// Specular texture ID (consumed by `OpaquePass` and `SssssPass`).
    pub specular_texture_id: Option<TransientTextureId>,
}

impl FrameBlackboard {
    /// Creates an empty frame blackboard.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears all fields at the start of each frame to prevent stale data.
    #[inline]
    pub fn clear(&mut self) {
        self.ssao_texture_id = None;
        self.transmission_texture_id = None;
        self.scene_normal_texture_id = None;
        self.feature_id_texture_id = None;
        self.sssss_pingpong_texture_id = None;
        self.specular_texture_id = None;
    }
}

// ============================================================================
// RenderCommand & RenderLists
// ============================================================================

/// A single render command.
///
/// Contains all information needed to draw one object. Produced by `CullPass`,
/// consumed by `OpaquePass` / `TransparentPass`.
///
/// # Performance Notes
/// - Pipeline is obtained via `clone` (`wgpu::RenderPipeline` is internally `Arc`)
/// - `dynamic_offset` supports dynamic uniform buffering
/// - `sort_key` enables efficient sorting (front-to-back / back-to-front)
pub struct RenderCommand {
    /// Per-object bind group (model matrix, skeleton, etc.)
    pub object_bind_group: BindGroupContext,
    /// Geometry handle
    pub geometry_handle: GeometryHandle,
    /// Material handle
    pub material_handle: MaterialHandle,
    /// Pipeline ID (used for state tracking to avoid redundant switches)
    pub pipeline_id: u16,
    /// Render pipeline (`wgpu::RenderPipeline` is internally `Arc`)
    pub pipeline: wgpu::RenderPipeline,
    /// Model-to-world matrix
    pub model_matrix: Mat4,
    /// Sort key
    pub sort_key: RenderKey,
    /// Dynamic uniform offset
    pub dynamic_offset: u32,
    /// Screen Space Feature Mask (for stencil writing in Prepass)
    pub ss_feature_mask: u32,
}

pub struct ShadowRenderCommand {
    pub object_bind_group: BindGroupContext,
    pub geometry_handle: GeometryHandle,
    pub material_handle: MaterialHandle,
    pub pipeline: wgpu::RenderPipeline,
    pub dynamic_offset: u32,
}

pub struct ShadowLightInstance {
    pub light_id: u64,
    pub layer_index: u32,
    pub light_buffer_index: usize,
    pub light_view_projection: Mat4,
}

/// Prepared skybox draw state for inline rendering (LDR path).
///
/// Populated by [`SkyboxPass::prepare()`] and consumed by
/// [`SimpleForwardPass::run()`] to draw the skybox between
/// opaque and transparent objects within a single render pass.
pub struct PreparedSkyboxDraw {
    /// The skybox render pipeline (variant-specific).
    pub pipeline: wgpu::RenderPipeline,
    /// The skybox bind group (uniforms + optional texture/sampler).
    pub bind_group: wgpu::BindGroup,
}

impl PreparedSkyboxDraw {
    pub fn draw<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        global_bind_group: &'a wgpu::BindGroup,
    ) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, global_bind_group, &[]);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Render lists.
///
/// Stores culled and sorted render commands. Populated by `SceneCullPass`,
/// consumed by `OpaquePass`, `TransparentPass`, and `SimpleForwardPass`.
///
/// # Design Principles
/// - **Data-logic separation**: stores data only, contains no rendering logic
/// - **Inter-frame reuse**: pre-allocated memory, cleared via `clear()` each frame
/// - **Extensible**: can add `alpha_test`, `shadow_casters`, etc. in the future
pub struct RenderLists {
    /// Opaque command list (front-to-back sorted)
    pub opaque: Vec<RenderCommand>,
    /// Transparent command list (back-to-front sorted)
    pub transparent: Vec<RenderCommand>,
    /// Shadow command queues, keyed by `(light_id, layer_index)` for per-view culling.
    ///
    /// Each cascade of a directional light (or each spot light) gets its own queue.
    pub shadow_queues: FxHashMap<(u64, u32), Vec<ShadowRenderCommand>>,
    pub shadow_lights: Vec<ShadowLightInstance>,

    /// All active render views for the current frame.
    ///
    /// Populated by `SceneCullPass`, consumed by `ShadowPass` and other passes.
    /// Contains main camera view + all shadow views.
    pub active_views: Vec<RenderView>,

    /// Global bind group ID (used for state tracking)
    pub gpu_global_bind_group_id: u64,
    /// Global bind group (camera, lighting, environment, etc.)
    pub gpu_global_bind_group: Option<wgpu::BindGroup>,

    /// Whether a transmission copy is needed this frame
    pub use_transmission: bool,

    /// Prepared skybox draw state for inline rendering in the LDR path.
    ///
    /// Set by `SkyboxPass::prepare()`, consumed by `SimpleForwardPass::run()`.
    pub prepared_skybox: Option<PreparedSkyboxDraw>,
}

impl RenderLists {
    /// Creates empty render lists with pre-allocated default capacity.
    #[must_use]
    pub fn new() -> Self {
        Self {
            opaque: Vec::with_capacity(512),
            transparent: Vec::with_capacity(128),
            shadow_queues: FxHashMap::default(),
            shadow_lights: Vec::with_capacity(16),
            active_views: Vec::with_capacity(16),
            gpu_global_bind_group_id: 0,
            gpu_global_bind_group: None,
            use_transmission: false,
            prepared_skybox: None,
        }
    }

    /// Clears all lists (retains capacity for memory reuse).
    #[inline]
    pub fn clear(&mut self) {
        self.opaque.clear();
        self.transparent.clear();
        self.shadow_queues.clear();
        self.shadow_lights.clear();
        self.active_views.clear();
        self.gpu_global_bind_group = None;
        self.use_transmission = false;
        self.prepared_skybox = None;
    }

    /// Inserts an opaque render command.
    #[inline]
    pub fn insert_opaque(&mut self, cmd: RenderCommand) {
        self.opaque.push(cmd);
    }

    /// Inserts a transparent render command.
    #[inline]
    pub fn insert_transparent(&mut self, cmd: RenderCommand) {
        self.transparent.push(cmd);
    }

    /// Sorts command lists.
    ///
    /// - Opaque: by Pipeline > Material > Depth (front-to-back)
    /// - Transparent: by Depth (back-to-front) > Pipeline > Material
    pub fn sort(&mut self) {
        self.opaque
            .sort_unstable_by(|a, b| a.sort_key.cmp(&b.sort_key));
        self.transparent
            .sort_unstable_by(|a, b| a.sort_key.cmp(&b.sort_key));
    }

    /// Returns `true` if both lists are empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.opaque.is_empty() && self.transparent.is_empty()
    }
}

impl Default for RenderLists {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// RenderKey — Sort Key
// ============================================================================

/// Render sort key (Pipeline ID + Material ID + Depth).
///
/// Encodes sorting information in a 64-bit integer for efficient radix sorting.
///
/// # Sorting Strategy
/// - **Opaque objects**: Pipeline > Material > Depth (front-to-back)
///   - Minimizes pipeline state switches
///   - Front-to-back leverages Early-Z for performance
/// - **Transparent objects**: Depth (back-to-front) > Pipeline > Material
///   - Ensures correct alpha blending order
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RenderKey(u64);

impl RenderKey {
    /// Constructs a sort key.
    ///
    /// # Parameters
    /// - `pipeline_id`: Pipeline index (14 bits)
    /// - `material_index`: Material index (20 bits)
    /// - `depth`: Squared distance to the camera
    /// - `transparent`: Whether the object is transparent
    #[must_use]
    pub fn new(pipeline_id: u16, material_index: u32, depth: f32, transparent: bool) -> Self {
        // 1. Compress depth into 30 bits.
        // Note: assumes depth >= 0.0. Clamping negative values to 0 is safe.
        let d_u32 = if depth.is_sign_negative() {
            0
        } else {
            depth.to_bits() >> 2
        };
        let raw_d_bits = u64::from(d_u32) & 0x3FFF_FFFF;

        // 2. Prepare other fields
        let p_bits = u64::from(pipeline_id & 0x3FFF); // 14 bits
        let m_bits = u64::from(material_index & 0xFFFFF); // 20 bits

        if transparent {
            // [Transparent]: Sort by Depth (far→near) > Pipeline > Material

            // 1. Invert depth so farther objects (larger depth) get smaller values, sorting first.
            let d_bits = raw_d_bits ^ 0x3FFF_FFFF;

            // 2. Bit layout: Depth (30) << 34 | Pipeline (14) << 20 | Material (20)
            Self((d_bits << 34) | (p_bits << 20) | m_bits)
        } else {
            // [Opaque]: Sort by Pipeline > Material > Depth (near→far)

            // Depth in ascending order (smaller depth first = front-to-back)
            let d_bits = raw_d_bits;

            // Pipeline (14) << 50 | Material (20) << 30 | Depth (30)
            Self((p_bits << 50) | (m_bits << 30) | d_bits)
        }
    }
}

// ============================================================================
// RenderFrame
// ============================================================================

/// Render frame manager.
///
/// Uses a render graph architecture:
/// 1. **Extract**: Pull rendering data from the scene
/// 2. **Prepare**: Prepare GPU resources
/// 3. **Execute**: Run render passes via [`FrameComposer`]
///
/// # Performance Notes
/// - `ExtractedScene` is persistent to reuse memory across frames
/// - `FrameComposer` is created per-frame but extremely cheap (just pointer ops)
///
/// # Note
/// `RenderLists` is stored in `RendererState` rather than here
/// to avoid borrow-checker limitations.
pub struct RenderFrame {
    pub(crate) render_state: RenderState,
    pub(crate) extracted_scene: ExtractedScene,
}

impl Default for RenderFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderFrame {
    #[must_use]
    pub fn new() -> Self {
        Self {
            render_state: RenderState::new(),
            extracted_scene: ExtractedScene::with_capacity(1024),
        }
    }

    /// Returns a reference to the render state.
    #[inline]
    pub fn render_state(&self) -> &RenderState {
        &self.render_state
    }

    /// Returns a reference to the extracted scene data.
    #[inline]
    pub fn extracted_scene(&self) -> &ExtractedScene {
        &self.extracted_scene
    }

    /// Phase 1: Extract scene data and prepare global resources.
    ///
    /// Runs the Extract and Prepare phases to set up for Compose and Execute.
    ///
    /// # Phases
    ///
    /// 1. **Extract**: Pull rendering data from the [`Scene`] into [`ExtractedScene`]
    /// 2. **Prepare**: Prepare global GPU resources (camera uniforms, lighting data, etc.)
    ///
    /// # Note
    ///
    /// This method does **not** acquire the surface texture. Surface acquisition is
    /// deferred to `FrameComposer::render()` to minimize swap-chain buffer hold time.
    pub fn extract_and_prepare(
        &mut self,
        resource_manager: &mut ResourceManager,
        scene: &mut Scene,
        camera: &RenderCamera,
        assets: &AssetServer,
        time: f32,
    ) {
        resource_manager.next_frame();

        // 1. Extract: reuse memory, avoid per-frame allocation
        self.extracted_scene
            .extract_into(scene, camera, assets, resource_manager);

        // 2. Resolve GPU environment and BRDF LUT before prepare_global.
        //    This creates textures in the cache and determines env_map_max_mip_level.
        let env_max_mip = resource_manager.resolve_gpu_environment(assets, &scene.environment);
        resource_manager.ensure_brdf_lut();

        // Patch env_map_max_mip_level into the scene uniform buffer
        {
            let current = scene.uniforms_buffer.read().env_map_max_mip_level;
            if (current - env_max_mip).abs() > f32::EPSILON {
                scene.uniforms_buffer.write().env_map_max_mip_level = env_max_mip;
            }
        }

        // 3. Prepare: global GPU resources
        self.render_state.update(camera, time);
        resource_manager.prepare_global(assets, scene, &self.render_state);
    }

    /// Periodically prune stale resources.
    pub fn maybe_prune(&self, resource_manager: &mut ResourceManager) {
        // Periodic cleanup (TODO: LRU eviction strategy)
        if resource_manager.frame_index().is_multiple_of(600) {
            resource_manager.prune(6000);
        }
    }
}
