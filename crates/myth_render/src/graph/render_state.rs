//! Render State
//!
//! Manages per-frame render state (camera, time, and other global uniforms).

use std::sync::atomic::{AtomicU32, Ordering};

use myth_resources::buffer::CpuBuffer;
use myth_resources::uniforms::RenderStateUniforms;
use myth_scene::camera::RenderCamera;

use crate::renderer::FrameTime;

// ─── Debug View Target (compile-time gated) ─────────────────────────────────

/// Semantic identifier for an intermediate render texture to visualise.
///
/// This enum lives in the **render state layer** — it carries no frame-specific
/// physical IDs (`TextureNodeId`).  The [`FrameComposer`] resolves it each
/// frame into a concrete RDG resource, safely handling cases where the
/// target texture was not produced (e.g. SSAO disabled).
///
/// Derived from [`DebugViewMode`](myth_scene::camera::DebugViewMode) during
/// the extract phase.  Material-override modes (Albedo, Roughness, Metalness)
/// do not use this target resolution — they are handled via shader defines.
#[cfg(feature = "debug_view")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DebugViewTarget {
    #[default]
    None,
    SceneDepth,
    SceneNormal,
    Velocity,
    SsaoRaw,
}

#[cfg(feature = "debug_view")]
impl DebugViewTarget {
    /// Maps a logical [`DebugViewMode`] to the render-layer target.
    ///
    /// Material-override modes return `None` since they bypass the
    /// post-process debug overlay entirely.
    #[must_use]
    pub fn from_mode(mode: myth_scene::camera::DebugViewMode) -> Self {
        use myth_scene::camera::DebugViewMode;
        match mode {
            DebugViewMode::SSAO     => Self::SsaoRaw,
            DebugViewMode::Normal   => Self::SceneNormal,
            DebugViewMode::Velocity => Self::Velocity,
            DebugViewMode::Depth    => Self::SceneDepth,
            _                       => Self::None,
        }
    }

    /// WGSL `view_mode` uniform value for the debug shader.
    ///
    /// | Mode | Mapping |
    /// |------|---------|    /// | 1    | SSAO → single-channel grayscale |
    /// | 2    | Normal → signed vector remap |
    /// | 3    | Velocity → directional colour |
    /// | 4    | Depth → linearised reverse-Z |
    #[must_use]
    pub const fn view_mode(self) -> u32 {
        match self {
            Self::None        => 0,
            Self::SsaoRaw     => 1,
            Self::SceneNormal => 2,
            Self::Velocity    => 3,
            Self::SceneDepth  => 4,
        }
    }
}

static NEXT_RENDER_STATE_ID: AtomicU32 = AtomicU32::new(0);

pub struct RenderState {
    pub id: u32,
    uniforms: CpuBuffer<RenderStateUniforms>,
    /// Previous frame's view-projection matrix (for TAA reprojection).
    prev_view_projection: glam::Mat4,
    /// Previous frame's jitter (for TAA de-jitter).
    prev_jitter: glam::Vec2,
    /// Previous frame's jitter-free VP matrix (for velocity calculation).
    prev_unjittered_vp: glam::Mat4,
    /// Active debug-view mode (from camera settings, resolved per-frame).
    #[cfg(feature = "debug_view")]
    pub debug_view_mode: myth_scene::camera::DebugViewMode,
    /// Scale factor for debug view (e.g. velocity amplification).
    #[cfg(feature = "debug_view")]
    pub debug_view_scale: f32,
}

impl Default for RenderState {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderState {
    pub fn new() -> Self {
        Self {
            id: NEXT_RENDER_STATE_ID.fetch_add(1, Ordering::Relaxed),
            uniforms: CpuBuffer::new(
                RenderStateUniforms::default(),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                Some("RenderState Uniforms"),
            ),
            prev_view_projection: glam::Mat4::IDENTITY,
            prev_jitter: glam::Vec2::ZERO,
            prev_unjittered_vp: glam::Mat4::IDENTITY,
            #[cfg(feature = "debug_view")]
            debug_view_mode: myth_scene::camera::DebugViewMode::None,
            #[cfg(feature = "debug_view")]
            debug_view_scale: 100.0,
        }
    }

    pub fn uniforms(&self) -> &CpuBuffer<RenderStateUniforms> {
        &self.uniforms
    }

    pub fn uniforms_mut(&mut self) -> myth_resources::buffer::BufferGuard<'_, RenderStateUniforms> {
        self.uniforms.write()
    }

    pub fn update(&mut self, camera: &RenderCamera, frame_time: FrameTime) {
        let prev_vp = self.prev_view_projection;
        let prev_j = self.prev_jitter;
        let prev_unjittered_vp = self.prev_unjittered_vp;

        let unjittered_vp = camera.unjittered_projection * camera.view_matrix;

        let mut u = self.uniforms_mut();
        u.view_projection = camera.view_projection_matrix;
        u.view_projection_inverse = camera.view_projection_matrix.inverse();
        u.projection_matrix = camera.projection_matrix;
        u.projection_inverse = camera.projection_matrix.inverse();
        u.view_matrix = camera.view_matrix;
        u.prev_view_projection = prev_vp;
        u.unjittered_view_projection = unjittered_vp;
        u.prev_unjittered_view_projection = prev_unjittered_vp;
        u.camera_position = camera.position.into();
        u.time = frame_time.time % 7200.0; // Wrap time to avoid precision issues in shaders
        u.time_cycle_2pi = frame_time.time % std::f32::consts::PI * 2.0;
        u.delta_time = frame_time.delta_time;
        u.jitter = camera.jitter;
        u.prev_jitter = prev_j;
        u.camera_near = camera.near;
        u.camera_far = camera.far;
        drop(u);

        // Latch current values for next frame.
        self.prev_view_projection = camera.view_projection_matrix;
        self.prev_jitter = camera.jitter;
        self.prev_unjittered_vp = unjittered_vp;
    }
}
