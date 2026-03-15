//! Render State
//!
//! Manages per-frame render state (camera, time, and other global uniforms).

use std::sync::atomic::{AtomicU32, Ordering};

use crate::resources::buffer::CpuBuffer;
use crate::resources::uniforms::RenderStateUniforms;
use crate::scene::camera::RenderCamera;

static NEXT_RENDER_STATE_ID: AtomicU32 = AtomicU32::new(0);

pub struct RenderState {
    pub id: u32,
    uniforms: CpuBuffer<RenderStateUniforms>,
    /// Previous frame's view-projection matrix (for TAA reprojection).
    prev_view_projection: glam::Mat4,
    /// Previous frame's jitter (for TAA de-jitter).
    prev_jitter: glam::Vec2,
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
        }
    }

    pub fn uniforms(&self) -> &CpuBuffer<RenderStateUniforms> {
        &self.uniforms
    }

    pub fn uniforms_mut(
        &mut self,
    ) -> crate::resources::buffer::BufferGuard<'_, RenderStateUniforms> {
        self.uniforms.write()
    }

    pub fn update(&mut self, camera: &RenderCamera, time: f32) {
        let prev_vp = self.prev_view_projection;
        let prev_j = self.prev_jitter;

        let mut u = self.uniforms_mut();
        u.view_projection = camera.view_projection_matrix;
        u.view_projection_inverse = camera.view_projection_matrix.inverse();
        u.projection_matrix = camera.projection_matrix;
        u.projection_inverse = camera.projection_matrix.inverse();
        u.view_matrix = camera.view_matrix;
        u.prev_view_projection = prev_vp;
        u.camera_position = camera.position.into();
        u.time = time;
        u.jitter = camera.jitter;
        u.prev_jitter = prev_j;
        drop(u);

        // Latch current values for next frame.
        self.prev_view_projection = camera.view_projection_matrix;
        self.prev_jitter = camera.jitter;
    }
}
