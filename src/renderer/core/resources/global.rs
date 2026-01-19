//! 全局渲染资源管理
//!
//! 负责管理 Light Storage Buffer、Environment Uniforms 等全局 GPU 资源。
//! 这是 Render-Logic Separation 的核心模块，将原本在 Scene 中的 GPU 资源
//! 移至渲染层统一管理。
//!
//! # 设计理念
//! - Scene 只负责存储纯数据（灯光参数、环境配置）
//! - GlobalResources 负责将这些数据转换为 GPU 可用的格式
//! - 通过版本比对避免每帧重复上传
//!
//! # 性能考虑
//! - 使用预分配的 Vec 避免每帧内存分配
//! - 版本号追踪实现 Dirty Check，只在数据变化时上传
//! - 灯光数据使用连续内存布局，对 CPU 缓存友好

use glam::Vec3;

use crate::resources::uniforms::{EnvironmentUniforms, GpuLightStorage};
use crate::scene::light::LightKind;
use crate::scene::Scene;

/// 全局渲染资源
/// 
/// 持有从 Scene 收集并转换的 GPU 数据，包括：
/// - 灯光存储 Buffer（Storage Buffer，支持动态数量）
/// - 环境 Uniforms（Uniform Buffer，固定大小）
/// 
/// # 版本追踪
/// 使用 generation 字段追踪数据变化，ResourceManager 通过比对
/// generation 决定是否需要重新上传数据到 GPU。
pub struct GlobalResources {
    /// 灯光数据（CPU 端缓存）
    /// 
    /// 预分配 16 个灯光的空间，避免小场景的内存分配。
    /// 当灯光数量超过容量时会自动扩容。
    light_data: Vec<GpuLightStorage>,
    
    /// 环境 Uniforms（CPU 端缓存）
    environment_uniforms: EnvironmentUniforms,
    
    /// 数据版本号，每次数据变化时递增
    /// 用于 Dirty Check，避免每帧重复上传
    generation: u64,
    
    /// 上次同步时的灯光数量
    /// 用于检测灯光数量变化
    last_light_count: usize,
}

impl Default for GlobalResources {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalResources {
    /// 创建新的全局资源管理器
    /// 
    /// 预分配 16 个灯光的存储空间
    pub fn new() -> Self {
        Self {
            light_data: Vec::with_capacity(16),
            environment_uniforms: EnvironmentUniforms::default(),
            generation: 0,
            last_light_count: 0,
        }
    }
    
    /// 从 Scene 收集并同步全局资源
    /// 
    /// 这是每帧调用的核心方法，负责：
    /// 1. 遍历场景中的活跃灯光
    /// 2. 将灯光数据转换为 GPU 格式
    /// 3. 更新环境 Uniforms
    /// 4. 检测数据变化并更新版本号
    /// 
    /// # 返回值
    /// 返回 `true` 表示数据有变化，需要上传到 GPU
    /// 
    /// # 性能说明
    /// - 复用 light_data Vec，避免每帧分配
    /// - 通过比对避免不必要的数据拷贝
    /// 
    /// # TODO: 性能优化点
    /// 当前实现每帧都会遍历所有灯光节点。对于大量静态灯光的场景，
    /// 可以考虑：
    /// 1. 增量更新：只处理标记为 dirty 的灯光
    /// 2. 空间分区：使用 BVH 或 Grid 剔除不可见灯光
    /// 3. 双缓冲：减少 CPU-GPU 同步开销
    pub fn sync_from_scene(&mut self, scene: &Scene) -> bool {
        let mut changed = false;
        
        // 1. 收集灯光数据
        let new_light_data: Vec<GpuLightStorage> = scene
            .iter_active_lights()
            .map(|(light, world_matrix)| {
                let pos = world_matrix.translation.to_vec3();
                let dir = world_matrix.transform_vector3(-Vec3::Z).normalize();
                
                let mut gpu_light = GpuLightStorage {
                    color: light.color,
                    intensity: light.intensity,
                    position: pos,
                    direction: dir,
                    ..Default::default()
                };
                
                match &light.kind {
                    LightKind::Point(point) => {
                        gpu_light.range = point.range;
                    }
                    LightKind::Spot(spot) => {
                        gpu_light.range = spot.range;
                        gpu_light.inner_cone_cos = spot.inner_cone.cos();
                        gpu_light.outer_cone_cos = spot.outer_cone.cos();
                    }
                    LightKind::Directional(_) => {}
                }
                
                gpu_light
            })
            .collect();
        
        // 2. 检测灯光数据变化
        // 注意：这里使用简单的长度+内容比对
        // 对于大量灯光，可以考虑使用 hash 或增量标记
        let light_count = new_light_data.len();
        if light_count != self.last_light_count || self.light_data != new_light_data {
            self.light_data = new_light_data;
            self.last_light_count = light_count;
            changed = true;
        }
        
        // 3. 确保至少有一个占位灯光（Shader 要求）
        if self.light_data.is_empty() {
            if self.last_light_count != 0 {
                self.light_data.push(GpuLightStorage::default());
                self.last_light_count = 0;
                changed = true;
            }
        }
        
        // 4. 更新环境 Uniforms
        let env = &scene.environment;
        let new_env_uniforms = EnvironmentUniforms {
            ambient_light: env.ambient_color,
            num_lights: light_count as u32,
            env_map_intensity: env.intensity,
            env_map_max_mip_level: env.env_map_max_mip_level,
            ..Default::default()
        };
        
        if self.environment_uniforms != new_env_uniforms {
            self.environment_uniforms = new_env_uniforms;
            changed = true;
        }
        
        // 5. 更新版本号
        if changed {
            self.generation += 1;
        }
        
        changed
    }
    
    /// 获取灯光数据的字节表示
    #[inline]
    pub fn light_data_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.light_data)
    }
    
    /// 获取环境 Uniforms 的字节表示
    #[inline]
    pub fn environment_uniforms_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(&self.environment_uniforms)
    }
    
    /// 获取当前数据版本号
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }
    
    /// 获取灯光数量
    #[inline]
    pub fn light_count(&self) -> usize {
        self.light_data.len()
    }
    
    /// 获取灯光数据引用
    #[inline]
    pub fn light_data(&self) -> &[GpuLightStorage] {
        &self.light_data
    }
    
    /// 获取环境 Uniforms 引用
    #[inline]
    pub fn environment_uniforms(&self) -> &EnvironmentUniforms {
        &self.environment_uniforms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_global_resources_default() {
        let resources = GlobalResources::new();
        assert_eq!(resources.generation(), 0);
        assert_eq!(resources.light_count(), 0);
    }
}
