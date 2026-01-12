use std::sync::atomic::{AtomicU32, Ordering};

use crate::resources::buffer::{BufferRef};
use crate::resources::uniforms::{GpuLightStorage};

static NEXT_WORLD_ID: AtomicU32 = AtomicU32::new(0);

pub struct Environment {
    pub id: u32,
    pub(crate) light_storage_buffer: Option<BufferRef>,

}

impl Environment {
    pub fn new() -> Self {
        Self {
            id: NEXT_WORLD_ID.fetch_add(1, Ordering::Relaxed),
            // frame_uniform_buffer: frame_buffer,
            light_storage_buffer: None,
        }
    }

    pub fn update_lights(&mut self, gpu_lights: Vec<GpuLightStorage>) {
            
        if gpu_lights.is_empty() {
            self.light_storage_buffer = None;
            self.id = NEXT_WORLD_ID.fetch_add(1, Ordering::Relaxed);
            return;
        }
        // 更新 Light Storage Buffer
        if let Some(light_storage_buffer) = &self.light_storage_buffer {
            light_storage_buffer.update(&gpu_lights);
        }
        else {
            let light_storage_buffer = BufferRef::new(
                &gpu_lights,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Light Storage Buffer")
            );
            self.light_storage_buffer = Some(light_storage_buffer);
            self.id = NEXT_WORLD_ID.fetch_add(1, Ordering::Relaxed);
        }
        
    }
}