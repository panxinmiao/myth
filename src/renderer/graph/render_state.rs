//! 渲染状态
//!
//! 管理每帧的渲染状态（相机、时间等全局 Uniform）

use std::sync::atomic::{AtomicU32, Ordering};

use crate::resources::uniforms::RenderStateUniforms;
use crate::resources::buffer::CpuBuffer;
use crate::scene::camera::Camera;

static NEXT_RENDER_STATE_ID: AtomicU32 = AtomicU32::new(0);

pub struct RenderState {
    pub id: u32,
    uniforms: CpuBuffer<RenderStateUniforms>,
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
                Some("RenderState Uniforms")
            ),
        }
    }

    pub fn uniforms(&self) -> &CpuBuffer<RenderStateUniforms> {
        &self.uniforms
    }

    pub fn uniforms_mut(&mut self) -> crate::resources::buffer::BufferGuard<'_, RenderStateUniforms> {
        self.uniforms.write()
    }

    pub fn update(&mut self, camera: &Camera, time: f32) {
        let view_matrix = camera.view_matrix;
        let vp_matrix = camera.view_projection_matrix;
        let camera_position = camera.world_matrix.translation.to_vec3();

        let mut u = self.uniforms_mut();
        u.view_projection = vp_matrix;
        u.view_projection_inverse = vp_matrix.inverse();
        u.view_matrix = view_matrix;
        u.camera_position = camera_position;
        u.time = time;
    }
}
