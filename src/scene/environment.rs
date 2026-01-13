use std::sync::atomic::{AtomicU32, Ordering};

use crate::resources::buffer::{BufferRef};
use crate::resources::uniform_slot::UniformSlot;
use crate::resources::uniforms::{GpuLightStorage};
use crate::resources::uniforms::{EnvironmentUniforms};

static NEXT_WORLD_ID: AtomicU32 = AtomicU32::new(0);

pub struct Environment {
    pub id: u32,
    // Storage Buffer 保留 BufferRef（动态数组，需要CPU副本）
    pub(crate) light_storage_buffer: BufferRef,
    // Uniform 改为 UniformSlot（小数据，无需中间层）
    pub(crate) uniforms: UniformSlot<EnvironmentUniforms>,
}

impl Environment {
    pub fn new() -> Self {
        Self {
            id: NEXT_WORLD_ID.fetch_add(1, Ordering::Relaxed),
            light_storage_buffer: BufferRef::new_with_capacity(
                std::mem::size_of::<GpuLightStorage>() * 32, 
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, 
                Some("Light Storage Buffer")
            ), 
            uniforms: UniformSlot::new(
                EnvironmentUniforms::default(),
                "Environment Uniforms"
            ),
        }
    }

    pub fn update_lights(&mut self, gpu_lights: Vec<GpuLightStorage>) {
        if gpu_lights.is_empty() {
            return;
        }

        // Storage Buffer 依然使用 BufferRef（需要CPU副本）
        self.light_storage_buffer.update(&gpu_lights);
        
        // Uniform 直接修改（无需手动拷贝到BufferRef）
        self.uniforms.get_mut().num_lights = gpu_lights.len() as u32;
        self.uniforms.mark_dirty();
    }
}