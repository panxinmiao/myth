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
use crate::renderer::pipeline::{PipelineCache, RenderPipelineId};
use crate::scene::Scene;
use crate::scene::camera::RenderCamera;

use super::extracted::ExtractedScene;
use super::render_state::RenderState;
use super::shadow_utils;


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
    /// Pipeline handle (index into [`PipelineCache`] storage).
    ///
    /// Resolve to a `&wgpu::RenderPipeline` via
    /// [`PipelineCache::get_render_pipeline`] during the execute phase.
    pub pipeline_id: RenderPipelineId,
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
    /// Pipeline handle (index into [`PipelineCache`] storage).
    pub pipeline_id: RenderPipelineId,
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
    /// The skybox render pipeline ID (variant-specific, resolved via `PipelineCache`).
    pub pipeline_id: RenderPipelineId,
    /// The skybox bind group (uniforms + optional texture/sampler).
    pub bind_group: wgpu::BindGroup,
}

impl PreparedSkyboxDraw {
    pub fn draw<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        global_bind_group: &'a wgpu::BindGroup,
        pipeline_cache: &'a PipelineCache,
    ) {
        let pipeline = pipeline_cache.get_render_pipeline(self.pipeline_id);
        pass.set_pipeline(pipeline);
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
    /// - `pipeline_id`: Pipeline handle
    /// - `material_index`: Material index (20 bits)
    /// - `depth`: Squared distance to the camera
    /// - `transparent`: Whether the object is transparent
    #[must_use]
    pub fn new(
        pipeline_id: RenderPipelineId,
        material_index: u32,
        depth: f32,
        transparent: bool,
    ) -> Self {
        // 1. Compress depth into 30 bits.
        // Note: assumes depth >= 0.0. Clamping negative values to 0 is safe.
        let d_u32 = if depth.is_sign_negative() {
            0
        } else {
            depth.to_bits() >> 2
        };
        let raw_d_bits = u64::from(d_u32) & 0x3FFF_FFFF;

        // 2. Prepare other fields
        let p_bits = u64::from(pipeline_id.0 & 0x3FFF); // 14 bits
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

    /// Extract scene data, build shadow views, and prepare global GPU resources.
    ///
    /// # Phases
    ///
    /// 1. **Extract** — Pull rendering data from the [`Scene`] into
    ///    [`ExtractedScene`].
    /// 2. **Shadow View Generation** — Build [`RenderView`]s for all
    ///    shadow-casting lights (pure math, no GPU work).
    /// 3. **Shadow Metadata** — Write per-light shadow layer indices,
    ///    cascade matrices and split distances into the light storage
    ///    buffer so the global bind group already contains correct data.
    /// 4. **Global Prepare** — Upload camera / scene / light uniforms and
    ///    create the global bind group (Group 0).
    ///
    /// # Note
    ///
    /// Surface acquisition is deferred to `FrameComposer::render()` to
    /// minimise swap-chain buffer hold time.
    pub fn extract_and_prepare(
        &mut self,
        resource_manager: &mut ResourceManager,
        scene: &mut Scene,
        camera: &RenderCamera,
        assets: &AssetServer,
        time: f32,
        render_lists: &mut RenderLists,
        surface_size: (u32, u32),
    ) {
        use crate::renderer::core::view::RenderView;

        resource_manager.next_frame();

        // ── 1. Extract ─────────────────────────────────────────────────
        self.extracted_scene
            .extract_into(scene, camera, assets, resource_manager);

        // ── 2. Resolve GPU environment + BRDF LUT ─────────────────────
        let env_max_mip = resource_manager.resolve_gpu_environment(assets, &scene.environment);
        resource_manager.ensure_brdf_lut();

        {
            let current = scene.uniforms_buffer.read().env_map_max_mip_level;
            if (current - env_max_mip).abs() > f32::EPSILON {
                scene.uniforms_buffer.write().env_map_max_mip_level = env_max_mip;
            }
        }

        // ── 3. Build shadow views (pure math) ──────────────────────────
        render_lists.clear();
        render_lists.active_views.push(RenderView::new_main_camera(
            camera.view_projection_matrix,
            camera.frustum,
            surface_size,
        ));

        let shadow_views = Self::build_shadow_views(
            &self.extracted_scene,
            camera,
            render_lists.active_views.len(),
        );

        // ── 4. Ensure shadow map resources ─────────────────────────────
        let total_shadow_layers = shadow_views.len() as u32;
        let max_shadow_map_size = shadow_views
            .iter()
            .map(|v| v.viewport_size.0)
            .max()
            .unwrap_or(1);
        resource_manager.ensure_shadow_maps(total_shadow_layers, max_shadow_map_size);

        render_lists.active_views.extend(shadow_views);

        // ── 5. Update light storage buffer with shadow metadata ────────
        Self::update_light_shadow_metadata(
            &render_lists.active_views,
            &self.extracted_scene.lights,
            scene,
            resource_manager,
        );

        // ── 6. Global GPU resources ────────────────────────────────────
        self.render_state.update(camera, time);
        resource_manager.prepare_global(assets, scene, &self.render_state);
    }

    /// Build [`RenderView`]s for all shadow-casting lights.
    ///
    /// Returns a `Vec` of shadow views. Each directional light may produce
    /// multiple cascade views; spot lights produce a single view; point
    /// lights are reserved for future cubemap shadows.
    fn build_shadow_views(
        extracted_scene: &ExtractedScene,
        camera: &RenderCamera,
        _existing_view_count: usize,
    ) -> Vec<crate::renderer::core::view::RenderView> {
        use crate::scene::light::LightKind;

        // Compute scene caster extent (for CSM Z extension)
        let camera_pos: glam::Vec3 = camera.position.to_array().into();
        let mut max_distance = 0.0f32;
        for item in &extracted_scene.render_items {
            if !item.cast_shadows {
                continue;
            }
            let aabb = item.world_aabb;
            let effective_radius = if aabb.is_finite() {
                aabb.size().length() * 0.5
            } else {
                0.0
            };
            let center_ws = aabb.center();
            let distance = camera_pos.distance(center_ws) + effective_radius;
            max_distance = max_distance.max(distance);
        }
        let scene_caster_extent = max_distance.max(50.0);

        let mut shadow_views = Vec::with_capacity(16);

        for (light_buffer_index, light) in extracted_scene.lights.iter().enumerate() {
            if !light.cast_shadows {
                continue;
            }
            let shadow_cfg = light.shadow.clone().unwrap_or_default();

            match &light.kind {
                LightKind::Directional(_) => {
                    let cam_far = if camera.far.is_finite() {
                        camera.far
                    } else {
                        shadow_cfg.max_shadow_distance
                    };
                    let shadow_far = shadow_cfg.max_shadow_distance.min(cam_far);
                    let caster_extension =
                        scene_caster_extent.max(shadow_cfg.max_shadow_distance);
                    let base_layer = shadow_views.len() as u32;

                    let (views, _splits) = shadow_utils::build_directional_views(
                        light.id,
                        light.direction,
                        light_buffer_index,
                        camera,
                        &shadow_cfg,
                        shadow_far,
                        caster_extension,
                        base_layer,
                    );
                    shadow_views.extend(views);
                }
                LightKind::Spot(spot) => {
                    let base_layer = shadow_views.len() as u32;
                    shadow_views.push(shadow_utils::build_spot_view(
                        light.id,
                        light_buffer_index,
                        light.position,
                        light.direction,
                        spot,
                        &shadow_cfg,
                        base_layer,
                    ));
                }
                LightKind::Point(_) => {
                    // Future: build 6 cubemap face views
                }
            }
        }

        shadow_views
    }

    /// Write per-light shadow metadata (layer indices, cascade matrices,
    /// bias values) into the scene's light storage buffer.
    ///
    /// Also populates `render_lists.shadow_lights` for the GPU shadow pass.
    fn update_light_shadow_metadata(
        active_views: &[crate::renderer::core::view::RenderView],
        extracted_lights: &[crate::renderer::graph::extracted::ExtractedLight],
        scene: &mut Scene,
        resource_manager: &mut ResourceManager,
    ) {
        use crate::renderer::core::view::ViewTarget;
        use crate::renderer::graph::shadow_utils::MAX_CASCADES;
        use glam::{Mat4, Vec4};

        // Reset shadow fields
        {
            let mut light_storage = scene.light_storage_buffer.write();
            for light in light_storage.iter_mut() {
                light.shadow_layer_index = -1;
                light.shadow_matrices.0 = [Mat4::IDENTITY; 4];
                light.cascade_count = 0;
                light.cascade_splits = Vec4::ZERO;
            }
        }

        let total_layers = active_views
            .iter()
            .filter(|v| v.is_shadow())
            .count() as u32;

        if total_layers == 0 {
            resource_manager.ensure_buffer(&scene.light_storage_buffer);
            return;
        }

        // Aggregate per-light shadow metadata
        {
            let mut light_storage = scene.light_storage_buffer.write();

            for (light_buffer_index, light) in extracted_lights.iter().enumerate() {
                if !light.cast_shadows {
                    continue;
                }
                let shadow_cfg = light.shadow.clone().unwrap_or_default();

                let mut base_layer = u32::MAX;
                let mut cascade_count = 0u32;
                let mut cascade_matrices = [Mat4::IDENTITY; MAX_CASCADES as usize];
                let mut cascade_splits_arr = [0.0f32; MAX_CASCADES as usize];

                for view in active_views {
                    let ViewTarget::ShadowLight {
                        light_id,
                        layer_index,
                    } = view.target
                    else {
                        continue;
                    };
                    if light_id != light.id {
                        continue;
                    }
                    if layer_index < base_layer {
                        base_layer = layer_index;
                    }
                    cascade_count += 1;
                }

                if cascade_count == 0 {
                    continue;
                }

                for view in active_views {
                    let ViewTarget::ShadowLight {
                        light_id,
                        layer_index,
                    } = view.target
                    else {
                        continue;
                    };
                    if light_id != light.id {
                        continue;
                    }
                    let cascade_idx = (layer_index - base_layer) as usize;
                    if cascade_idx < MAX_CASCADES as usize {
                        cascade_matrices[cascade_idx] = view.view_projection;
                        if let Some(split) = view.csm_split {
                            cascade_splits_arr[cascade_idx] = split;
                        }
                    }
                }

                if let Some(gpu_light) = light_storage.get_mut(light_buffer_index) {
                    gpu_light.shadow_layer_index = base_layer.cast_signed();
                    gpu_light.shadow_matrices.0 = cascade_matrices;
                    gpu_light.cascade_count = cascade_count;
                    gpu_light.cascade_splits = Vec4::new(
                        cascade_splits_arr[0],
                        cascade_splits_arr[1.min(cascade_count as usize - 1)],
                        cascade_splits_arr[2.min(cascade_count as usize - 1)],
                        cascade_splits_arr[3.min(cascade_count as usize - 1)],
                    );
                    gpu_light.shadow_bias = shadow_cfg.bias;
                    gpu_light.shadow_normal_bias = shadow_cfg.normal_bias;
                }
            }
        }

        resource_manager.ensure_buffer(&scene.light_storage_buffer);
    }

    /// Periodically prune stale resources.
    pub fn maybe_prune(&self, resource_manager: &mut ResourceManager) {
        // Periodic cleanup (TODO: LRU eviction strategy)
        if resource_manager.frame_index().is_multiple_of(600) {
            resource_manager.prune(6000);
        }
    }
}
