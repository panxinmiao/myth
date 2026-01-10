// src/core/world.rs

use uuid::Uuid;
use glam::Vec3;

use crate::core::buffer::{DataBuffer, BufferRef};
use crate::core::uniforms::{GlobalFrameUniforms, GlobalLightUniforms};
use crate::core::camera::Camera;
use crate::core::scene::Scene;
use crate::core::light::{LightType};
use crate::core::uniforms::{GpuLightData, MAX_DIR_LIGHTS, MAX_POINT_LIGHTS, MAX_SPOT_LIGHTS};
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
    pub fn update(&self, camera: &mut Camera, scene: &Scene) {
        // 1. 更新 Frame Uniforms
        {
            let mut uniforms = GlobalFrameUniforms::default();
            camera.update_matrix_world(scene);
            
            let view_matrix = camera.view_matrix();
            let vp_matrix =  camera.view_projection_matrix();

            uniforms.view_projection = *vp_matrix;
            uniforms.view_projection_inverse = vp_matrix.inverse();
            uniforms.view_matrix = *view_matrix;

            // 获取写锁并更新数据
            self.frame_uniforms.write().update(&[uniforms]);
        }

        // 2. 更新 Light Uniforms (临时硬编码，后续接入 Light 组件)
        {
            let mut uniforms = GlobalLightUniforms::default();
            
            let mut dir_count = 0;
            let mut point_count = 0;
            let mut spot_count = 0;

            for (_id, node) in scene.nodes.iter() {
                if let Some(light_idx) = node.light {
                    if let Some(light) = scene.lights.get(light_idx) {
                        
                        // 获取灯光的世界变换
                        let world_mat = node.world_matrix(); 
                        let pos = world_mat.translation;
                        // 从旋转中提取方向 (-Z)
                        let dir = world_mat.transform_vector3(-Vec3::Z).normalize();

                        match light.light_type {
                            LightType::Directional => {
                                if dir_count < MAX_DIR_LIGHTS {
                                    uniforms.dir_lights[dir_count] = GpuLightData {
                                        color: [light.color.x, light.color.y, light.color.z, light.intensity],
                                        direction: [dir.x, dir.y, dir.z, 0.0],
                                        ..Default::default()
                                    };
                                    dir_count += 1;
                                }
                            },
                            LightType::Point => {
                                if point_count < MAX_POINT_LIGHTS {
                                    uniforms.point_lights[point_count] = GpuLightData {
                                        color: [light.color.x, light.color.y, light.color.z, light.intensity],
                                        position: [pos.x, pos.y, pos.z, light.range],
                                        ..Default::default()
                                    };
                                    point_count += 1;
                                }
                            },
                            LightType::Spot => {
                                if spot_count < MAX_SPOT_LIGHTS {
                                    uniforms.spot_lights[spot_count] = GpuLightData {
                                        color: [light.color.x, light.color.y, light.color.z, light.intensity],
                                        position: [pos.x, pos.y, pos.z, light.range],
                                        direction: [dir.x, dir.y, dir.z, 0.0],
                                        info: [light.inner_cone.cos(), light.outer_cone.cos(), 0.0, 0.0],
                                        ..Default::default()
                                    };
                                    spot_count += 1;
                                }
                            },
                        }
                    }
                }
            }
            
            uniforms.num_dir_lights = dir_count as u32;
            uniforms.num_point_lights = point_count as u32;
            uniforms.num_spot_lights = spot_count as u32;
            
            self.light_uniforms.write().update(&[uniforms]);
        }
    }
}