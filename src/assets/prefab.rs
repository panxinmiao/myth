use std::sync::Arc;

use glam::Affine3A;

use crate::animation::clip::AnimationClip;
use crate::resources::mesh::Mesh;
use crate::scene::transform::Transform;

/// 预制体节点：只包含数据，使用索引引用子节点
#[derive(Debug, Clone)]
pub struct PrefabNode {
    pub name: Option<String>,
    pub transform: Transform,
    /// 子节点在 Prefab.nodes 中的索引
    pub children_indices: Vec<usize>,
    /// Mesh 组件（如有）
    pub mesh: Option<Mesh>,
    /// 引用的骨骼索引（在 Prefab.skeletons 中）
    pub skin_index: Option<usize>,
    /// 形变权重（如有）
    pub morph_weights: Option<Vec<f32>>,
}

impl PrefabNode {
    pub fn new() -> Self {
        Self {
            name: None,
            transform: Transform::new(),
            children_indices: Vec::new(),
            mesh: None,
            skin_index: None,
            morph_weights: None,
        }
    }
}

impl Default for PrefabNode {
    fn default() -> Self {
        Self::new()
    }
}

/// 预制体骨架数据
#[derive(Debug, Clone)]
pub struct PrefabSkeleton {
    pub name: String,
    /// 根骨骼在 bones 中的索引
    pub root_bone_index: usize,
    /// 骨骼节点索引（指向 Prefab.nodes 的索引）
    pub bone_indices: Vec<usize>,
    /// 逆绑定矩阵
    pub inverse_bind_matrices: Vec<Affine3A>,
}

/// 预制体：从 glTF 等资源文件解析出的中间数据结构
/// 
/// Prefab 是线程安全的纯数据结构，不包含任何 NodeHandle 或 Scene 引用。
/// 通过 `Scene::instantiate()` 方法可以将 Prefab 实例化为场景节点。
#[derive(Debug, Clone)]
pub struct Prefab {
    /// 扁平化存储的所有节点
    pub nodes: Vec<PrefabNode>,
    /// 根节点在 nodes 中的索引
    pub root_indices: Vec<usize>,
    /// 骨骼数据
    pub skeletons: Vec<PrefabSkeleton>,
    /// 动画数据
    pub animations: Vec<AnimationClip>,
}

impl Prefab {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root_indices: Vec::new(),
            skeletons: Vec::new(),
            animations: Vec::new(),
        }
    }
}

impl Default for Prefab {
    fn default() -> Self {
        Self::new()
    }
}

/// 线程安全的 Prefab 引用
pub type SharedPrefab = Arc<Prefab>;
