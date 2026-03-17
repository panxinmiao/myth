//! Render State
//!
//! Manages per-frame render state (camera, time, and other global uniforms).

use std::sync::atomic::{AtomicU32, Ordering};

use myth_resources::buffer::CpuBuffer;
use myth_resources::uniforms::RenderStateUniforms;
use myth_scene::camera::RenderCamera;

// ─── Debug View Target (compile-time gated) ─────────────────────────────────

/// Semantic identifier for an intermediate render texture to visualise.
///
/// This enum lives in the **state layer** — it carries no frame-specific
/// physical IDs (`TextureNodeId`).  The [`FrameComposer`] resolves it each
/// frame into a concrete RDG resource, safely handling cases where the
/// target texture was not produced (e.g. SSAO disabled).
#[cfg(feature = "debug_view")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugViewTarget {
    /// No debug overlay — show the final tonemapped image.
    None,
    /// Main scene depth buffer (reverse-Z, linearised for display).
    SceneDepth,
    /// View-space normals from the geometry prepass.
    SceneNormal,
    /// Screen-space velocity buffer (TAA reprojection vectors).
    Velocity,
    /// Raw SSAO term before spatial blur.
    SsaoRaw,
    /// First mip level of the Bloom downsample chain.
    BloomMip0,
}

#[cfg(feature = "debug_view")]
impl Default for DebugViewTarget {
    fn default() -> Self {
        Self::None
    }
}

#[cfg(feature = "debug_view")]
impl DebugViewTarget {
    /// Display label for the UI combo box.
    pub const fn label(self) -> &'static str {
        match self {
            Self::None => "Final Image",
            Self::SceneDepth => "Scene Depth",
            Self::SceneNormal => "Scene Normal",
            Self::Velocity => "Velocity Buffer",
            Self::SsaoRaw => "SSAO Raw",
            Self::BloomMip0 => "Bloom Mip 0",
        }
    }

    /// WGSL `view_mode` uniform value for the debug shader.
    ///
    /// | Mode | Mapping |
    /// |------|---------|
    /// | 0    | RGB pass-through |
    /// | 1    | Single-channel R → grayscale |
    /// | 2    | Signed vector `[-1,1]` → `[0,1]` |
    /// | 3    | Linear depth visualisation |
    pub const fn view_mode(self) -> u32 {
        match self {
            Self::None => 0,
            Self::SceneDepth => 3,
            Self::SceneNormal => 2,
            Self::Velocity => 2,
            Self::SsaoRaw => 1,
            Self::BloomMip0 => 0,
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
    /// Active debug-view target (semantic intent, resolved per-frame).
    #[cfg(feature = "debug_view")]
    pub debug_view_target: DebugViewTarget,
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
            debug_view_target: DebugViewTarget::None,
        }
    }

    pub fn uniforms(&self) -> &CpuBuffer<RenderStateUniforms> {
        &self.uniforms
    }

    pub fn uniforms_mut(
        &mut self,
    ) -> myth_resources::buffer::BufferGuard<'_, RenderStateUniforms> {
        self.uniforms.write()
    }

    pub fn update(&mut self, camera: &RenderCamera, time: f32) {
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
        u.time = time;
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
