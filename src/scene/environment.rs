use std::sync::atomic::{AtomicU32, Ordering};

use crate::resources::buffer::{BufferRef};
use crate::resources::uniforms::{GpuLightStorage};
use crate::resources::uniforms::{EnvironmentUniforms};

static NEXT_WORLD_ID: AtomicU32 = AtomicU32::new(0);

pub struct Environment {
    pub id: u32,
    pub(crate) light_storage_buffer: BufferRef,
    pub(crate) uniform_buffer: BufferRef,
    pub uniforms: EnvironmentUniforms,
}

impl Environment {
    pub fn new() -> Self {
        Self {
            id: NEXT_WORLD_ID.fetch_add(1, Ordering::Relaxed),
            light_storage_buffer: BufferRef::new_with_capacity(std::mem::size_of::<GpuLightStorage>() * 32, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, Some("Light Storage Buffer")), 
            uniform_buffer: BufferRef::new(&[EnvironmentUniforms::default()], wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, Some("Environment Uniform Buffer")),
            uniforms: EnvironmentUniforms {
                num_lights: 0,
            },
        }
    }

    pub fn update_lights(&mut self, gpu_lights: Vec<GpuLightStorage>) {
        if gpu_lights.is_empty() {
            return;
        }

        // 更新 Light Storage Buffer
        self.light_storage_buffer.update(&gpu_lights);
        self.uniforms.num_lights = gpu_lights.len() as u32;
        self.uniform_buffer.update(&[self.uniforms]);
    }
}