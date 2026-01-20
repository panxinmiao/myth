use thunderdome::Index;
use crate::assets::{GeometryHandle, MaterialHandle};
use crate::resources::buffer::CpuBuffer;
use crate::resources::uniforms::MorphUniforms;

/// 最大激活的 Morph Target 数量
pub const MAX_MORPH_TARGETS: usize = 32;

/// 权重阈值，低于此值的权重将被忽略
pub const MORPH_WEIGHT_THRESHOLD: f32 = 0.001;

pub type MeshHandle = Index;

/// 渲染代理缓存
/// 
/// 存储 Mesh 实例的渲染缓存数据，避免每帧重复查找
#[derive(Debug, Clone, Default)]
pub struct RenderCache {
    /// 缓存的 BindGroup ID（用于快速路径）
    pub bind_group_id: Option<u64>,
    /// 缓存的 Pipeline ID
    pub pipeline_id: Option<u16>,

    /// 缓存版本号（用于失效检测）
    pub(crate) geometry_id: Option<GeometryHandle>,
    pub(crate) geometry_version: u64, // 结构版本
    pub(crate) material_id: Option<MaterialHandle>,
    /// Material 的 BindGroupLayout ID（GPU 物理资源 ID）
    pub(crate) material_layout_id: u64,
    pub(crate) model_buffer_id: u64,  // 检查 Allocator 是否重置/Resize
    pub(crate) skeleton_id: Option<u64>, // joint_matrices buffer id
}

impl RenderCache {
    /// 使缓存失效
    // #[inline]
    // pub fn invalidate(&mut self) {
    //     self.bind_group_id = None;
    //     self.pipeline_id = None;
    // }

    // 检查缓存是否依然有效
    pub fn is_valid(
        &self, 
        geo_handle: GeometryHandle,
        geo_version: u64,
        mat_handle: MaterialHandle,
        mat_layout_id: u64,
        current_model_buffer_id: u64,
        skeleton_buffer_id: Option<u64>
    ) -> bool {
        self.bind_group_id.is_some()
            && self.geometry_id == Some(geo_handle)
            && self.geometry_version == geo_version
            && self.material_id == Some(mat_handle)
            && self.material_layout_id == mat_layout_id
            && self.model_buffer_id == current_model_buffer_id
            && self.skeleton_id == skeleton_buffer_id
    }
}

#[derive(Debug, Clone)]
pub struct Mesh {
    pub name: String,
    
    // === 场景图节点 ===
    pub node_id: Option<Index>,
    
    // === 资源引用 ===
    pub geometry: GeometryHandle,
    pub material: MaterialHandle,
    
    // === 实例特定的渲染设置 ===
    pub visible: bool, 
    
    // 绘制顺序 (Render Order)
    pub render_order: i32,

    /// Morph Target 原始权重 (所有 target 的权重，CPU 逻辑用)
    pub morph_target_influences: Vec<f32>,
    
    /// Morph Uniform Buffer (GPU 用，每帧更新)
    pub(crate) morph_uniforms: CpuBuffer<MorphUniforms>,

    // pub(crate) active_joint_indices: Vec<u16>,
    
    // === 渲染缓存 ===
    /// 渲染代理缓存，避免每帧重复查找
    pub render_cache: RenderCache,
}

impl Mesh {
    pub fn new(
        geometry: GeometryHandle, 
        material: MaterialHandle
    ) -> Self {
        Self {
            name: "Mesh".to_string(),
            node_id : None,
            geometry,
            material,
            visible: true,
            render_order: 0,
            morph_target_influences: Vec::new(),
            morph_uniforms: CpuBuffer::new_uniform( Some("Mesh Morph Uniforms")),
            // active_joint_indices: Vec::new(),
            render_cache: RenderCache::default(),
        }
    }

    pub fn morph_uniforms(&self) -> &MorphUniforms {
        self.morph_uniforms.read()
    }

    pub fn morph_uniforms_mut(&mut self) -> crate::resources::buffer::BufferGuard<'_, MorphUniforms> {
        self.morph_uniforms.write()
    }
    
    /// 初始化 morph target influences 数组
    /// 应在加载 geometry 后调用
    pub fn init_morph_targets(&mut self, target_count: u32, vertex_count: u32) {
        self.morph_target_influences = vec![0.0; target_count as usize];
        
        // 初始化 uniform buffer
        let mut uniforms = self.morph_uniforms.write();
        uniforms.vertex_count = vertex_count;
        uniforms.count = 0;
        uniforms.flags = 0;
    }
    
    /// 设置单个 morph target 的权重
    pub fn set_morph_target_influence(&mut self, index: usize, weight: f32) {
        if index < self.morph_target_influences.len() {
            self.morph_target_influences[index] = weight;
        }
    }
    
    /// 批量设置 morph target 权重
    pub fn set_morph_target_influences(&mut self, weights: &[f32]) {
        let len = weights.len().min(self.morph_target_influences.len());
        self.morph_target_influences[..len].copy_from_slice(&weights[..len]);
    }
    
    /// 更新 Morph Uniforms (排序剔除并填充 GPU buffer)
    /// 应在每帧渲染前调用
    pub fn update_morph_uniforms(&mut self) {
        if self.morph_target_influences.is_empty() {
            return;
        }
        
        // 1. 收集激活的权重 (过滤掉极小值)
        let mut active_targets: Vec<(usize, f32)> = self.morph_target_influences
            .iter()
            .enumerate()
            .filter(|(_, w)| w.abs() > MORPH_WEIGHT_THRESHOLD)
            .map(|(i, w)| (i, *w))
            .collect();
        
        // 2. 按权重从大到小排序
        active_targets.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
        
        // 3. 截断到最大容量
        active_targets.truncate(MAX_MORPH_TARGETS);
        
        // 4. 填充 Uniform Buffer (打包到 Vec4/UVec4 中)
        let mut uniforms = self.morph_uniforms.write();
        uniforms.count = active_targets.len() as u32;
        
        // 清空数组 (8 个 Vec4，每个包含 4 个值)
        for i in 0..8 {
            uniforms.weights[i] = glam::Vec4::ZERO;
            uniforms.indices[i] = glam::UVec4::ZERO;
        }
        
        // 填充激活的 targets (将索引 i 映射到 Vec4[i/4][i%4])
        for (i, (target_idx, weight)) in active_targets.iter().enumerate() {
            let vec_idx = i / 4;  // 哪个 Vec4
            let component = i % 4; // Vec4 中的哪个分量
            uniforms.weights[vec_idx][component] = *weight;
            uniforms.indices[vec_idx][component] = *target_idx as u32;
        }
    }
    
    // 使渲染缓存失效（当 geometry 或 material 改变时调用）
    // pub fn invalidate_cache(&mut self) {
    //     self.render_cache.invalidate();
    // }
}