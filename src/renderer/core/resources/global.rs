//! 全局渲染资源管理
//!
//! 负责管理 Light Storage Buffer、Environment Uniforms 等全局 GPU 资源。
//! 这是 Render-Logic Separation 的核心模块，将原本在 Scene 中的 GPU 资源
//! 移至渲染层统一管理。

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
/// # 版本追踪策略
/// 
/// 使用两级版本号系统：
/// - `data_version`: 数据内容变化时递增，用于触发 `write_buffer`
/// - `structure_version`: Buffer 结构变化时递增，用于触发 BindGroup 重建
/// 
/// 只有以下情况需要重建 BindGroup：
/// 1. Light Buffer 需要扩容（灯光数量增加超过当前容量）
/// 2. 其他资源 ID 变化（由外部追踪）
pub struct GlobalResources {
    /// 灯光数据（CPU 端缓存）
    light_data: Vec<GpuLightStorage>,
    
    /// 环境 Uniforms（CPU 端缓存）
    environment_uniforms: EnvironmentUniforms,
    
    /// 数据版本号：每次数据内容变化时递增
    /// 用于判断是否需要 write_buffer
    data_version: u64,
    
    /// 结构版本号：Buffer 结构（大小/ID）变化时递增
    /// 用于判断是否需要重建 BindGroup
    structure_version: u64,
    
    /// Light Buffer 当前已分配的容量（灯光数量）
    /// 用于判断是否需要扩容
    light_buffer_capacity: usize,
    
    /// 上次同步时的灯光数量
    last_light_count: usize,

    scratch_light_data: Vec<GpuLightStorage>,
    
    /// 标记数据是否有变化（本帧）
    data_dirty: bool,
}

impl Default for GlobalResources {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalResources {
    /// 初始灯光 Buffer 容量
    const INITIAL_LIGHT_CAPACITY: usize = 16;
    
    /// 创建新的全局资源管理器
    pub fn new() -> Self {
        Self {
            light_data: Vec::with_capacity(Self::INITIAL_LIGHT_CAPACITY),
            environment_uniforms: EnvironmentUniforms::default(),
            data_version: 0,
            structure_version: 0,
            light_buffer_capacity: Self::INITIAL_LIGHT_CAPACITY,
            last_light_count: 0,

            scratch_light_data: Vec::with_capacity(Self::INITIAL_LIGHT_CAPACITY),
            data_dirty: false,
        }
    }
    
    /// 从 Scene 收集并同步全局资源
    /// 
    /// # 返回值
    /// 返回 `true` 表示数据有变化，需要上传到 GPU
    pub fn sync_from_scene(&mut self, scene: &Scene) -> bool {
        self.data_dirty = false;

        self.scratch_light_data.clear();

        for (light, world_matrix) in scene.iter_active_lights() {
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

            self.scratch_light_data.push(gpu_light);

        }
        
        // 2. 检测灯光数据变化
        let light_count = self.scratch_light_data.len();
        let light_count_for_buffer = light_count.max(1); // 至少1个（Shader要求）
        
        // 检查是否需要扩容 Buffer（结构变化）
        if light_count_for_buffer > self.light_buffer_capacity {
            // 扩容策略：翻倍增长，减少频繁扩容
            self.light_buffer_capacity = (light_count_for_buffer * 2).max(Self::INITIAL_LIGHT_CAPACITY);
            self.structure_version += 1;
        }

        // 3. 比较暂存区和当前数据
        if self.light_data != self.scratch_light_data {
            // 只有数据不同时才交换
            std::mem::swap(&mut self.light_data, &mut self.scratch_light_data);
            self.last_light_count = self.scratch_light_data.len();
            self.data_dirty = true;
        }
        
        // 3. 确保至少有一个占位灯光（Shader 要求）
        if self.light_data.is_empty() {
            self.light_data.push(GpuLightStorage::default());
            self.data_dirty = true;
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
            self.data_dirty = true;
        }
        
        // 5. 更新数据版本号
        if self.data_dirty {
            self.data_version += 1;
        }
        
        self.data_dirty
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
    
    /// 获取数据版本号（内容变化）
    #[inline]
    pub fn data_version(&self) -> u64 {
        self.data_version
    }
    
    /// 获取结构版本号（Buffer 结构变化）
    #[inline]
    pub fn structure_version(&self) -> u64 {
        self.structure_version
    }
    
    /// 本帧数据是否有变化
    #[inline]
    pub fn is_data_dirty(&self) -> bool {
        self.data_dirty
    }
    
    /// 获取灯光数量
    #[inline]
    pub fn light_count(&self) -> usize {
        self.light_data.len()
    }
    
    /// 获取 Light Buffer 当前容量
    #[inline]
    pub fn light_buffer_capacity(&self) -> usize {
        self.light_buffer_capacity
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