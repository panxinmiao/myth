//! Render View Abstraction
//!
//! Defines [`RenderView`], a unified abstraction for any rendering viewpoint.
//!
//! # "Everything is a View" Architecture
//!
//! All rendering tasks are expressed as `RenderView` instances:
//!
//! - **Main Camera**: 1 view
//! - **Spot Light Shadow**: 1 view
//! - **Directional Light (CSM)**: N views (one per cascade)
//! - **Point Light (future)**: 6 views (one per cubemap face)
//!
//! This enables unified culling: each view carries a frustum, and the
//! cull pass tests every renderable against every view's frustum.
//!
//! # Data Flow
//!
//! ```text
//! SceneCullPass::prepare()
//!     ├── build main camera RenderView
//!     ├── build shadow RenderViews (via shadow_utils)
//!     └── per-view frustum culling → RenderLists
//!
//! ShadowPass::prepare()
//!     └── read active_views → upload VP matrices to GPU
//!
//! ShadowPass::run()
//!     └── iterate shadow views → draw per-view commands
//! ```

use glam::Mat4;

use crate::scene::camera::Frustum;

/// Identifies what a [`RenderView`] is rendering for.
///
/// Used as a key to index per-view command queues in [`RenderLists`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViewTarget {
    /// The main camera view.
    MainCamera,
    /// A shadow map view for a specific light and layer.
    ///
    /// - For directional lights (CSM): `layer_index` = cascade index (0..N).
    /// - For spot lights: `layer_index` = 0.
    /// - For point lights (future): `layer_index` = cubemap face (0..5).
    ShadowLight { light_id: u64, layer_index: u32 },
}

/// A unified rendering viewpoint.
///
/// Contains all data needed for CPU-side culling and GPU-side rendering
/// from a single viewpoint.
pub struct RenderView {
    /// Identifies this view's purpose (main camera, shadow cascade, etc.).
    pub target: ViewTarget,

    /// Debug name (e.g. `"DirLight_42_Cascade_2"`).
    pub name: String,

    /// View-Projection matrix, uploaded to the GPU for vertex transformation.
    pub view_projection: Mat4,

    /// Frustum extracted from `view_projection`, used for CPU frustum culling.
    pub frustum: Frustum,

    /// Size of the render target for this view (width, height).
    ///
    /// For shadow atlas: the region within the atlas.
    /// For shadow array: typically `(map_size, map_size)`.
    pub viewport_size: (u32, u32),

    /// For CSM cascades: the far split distance in view space.
    /// `None` for non-CSM views.
    pub csm_split: Option<f32>,

    /// Index into the GPU light storage buffer.
    /// Only meaningful for shadow views.
    pub light_buffer_index: usize,
}

impl RenderView {
    /// Create a new render view with the given parameters.
    ///
    /// The frustum is automatically extracted from `view_projection`.
    #[must_use]
    pub fn new_main_camera(vp: Mat4, frustum: Frustum, size: (u32, u32)) -> Self {
        Self {
            target: ViewTarget::MainCamera,
            name: "MainCamera".to_string(),
            view_projection: vp,
            frustum,
            viewport_size: size,
            csm_split: None,
            light_buffer_index: 0,
        }
    }

    /// Create a shadow view from pre-computed view-projection matrix and frustum.
    #[must_use]
    pub fn new_shadow(
        light_id: u64,
        layer_index: u32,
        light_buffer_index: usize,
        name: String,
        vp: Mat4,
        frustum: Frustum,
        size: (u32, u32),
        csm_split: Option<f32>,
    ) -> Self {
        Self {
            target: ViewTarget::ShadowLight {
                light_id,
                layer_index,
            },
            name,
            view_projection: vp,
            frustum,
            viewport_size: size,
            csm_split,
            light_buffer_index,
        }
    }

    /// Returns `true` if this is a shadow view.
    #[inline]
    #[must_use]
    pub fn is_shadow(&self) -> bool {
        matches!(self.target, ViewTarget::ShadowLight { .. })
    }
}
