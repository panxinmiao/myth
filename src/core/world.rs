// src/core/world.rs

use uuid::Uuid;
use glam::Vec3;

use crate::core::buffer::{DataBuffer, BufferRef};
use crate::core::uniforms::{GlobalFrameUniforms, GlobalLightUniforms};
use crate::core::camera::Camera;
use crate::core::scene::Scene;
/// WorldEnvironment 管理场景的全局渲染环境
pub struct WorldEnvironment {
    pub id: Uuid,
    
    // 使用 BufferRef (内部封装了 Arc<RwLock<DataBuffer>>)
    pub frame_uniforms: BufferRef,
    pub light_uniforms: BufferRef,
}

impl WorldEnvironment {
    pub fn new() -> Self {
        // 1. 初始化 GlobalFrameUniforms Buffer
        // DataBuffer::new 需要初始数据，我们传入 default
        let frame_default = GlobalFrameUniforms::default();
        let frame_buffer = DataBuffer::new(
            &[frame_default], 
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, 
            Some("Global Frame Uniforms")
        );
        
        // 2. 初始化 GlobalLightUniforms Buffer
        let light_default = GlobalLightUniforms::default();
        let light_buffer = DataBuffer::new(
            &[light_default],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            Some("Global Light Uniforms")
        );

        Self {
            id: Uuid::new_v4(),
            frame_uniforms: BufferRef::new(frame_buffer),
            light_uniforms: BufferRef::new(light_buffer),
        }
    }

    /// 每帧调用：从 Camera 和 Scene 更新 CPU Buffer 数据
    pub fn update(&self, camera: &Camera, scene: &Scene) {
        // 1. 更新 Frame Uniforms
        {
            let mut uniforms = GlobalFrameUniforms::default();
            
            let view_matrix = camera.get_view_matrix(Some(scene));
            let proj_matrix = camera.get_projection_matrix();
            let vp_matrix = proj_matrix * view_matrix;

            uniforms.view_projection = vp_matrix;
            uniforms.view_projection_inverse = vp_matrix.inverse();
            uniforms.view_matrix = view_matrix;

            // 获取写锁并更新数据
            self.frame_uniforms.write().update(&[uniforms]);
        }

        // 2. 更新 Light Uniforms (临时硬编码，后续接入 Light 组件)
        {
            let mut uniforms = GlobalLightUniforms::default();
            // 测试：平行光
            uniforms.directional_light_direction = Vec3::new(-1.0, -1.0, -0.5).normalize();
            uniforms.directional_light_color = Vec3::new(1.0, 1.0, 1.0);
            
            self.light_uniforms.write().update(&[uniforms]);
        }
    }
}