use glam::Affine3A;
use crate::scene::NodeHandle;
use crate::scene::transform::Transform;

/// 精简的场景节点 (只包含核心热数据)
/// 
/// 设计原则：
/// - 只保留每帧必须遍历的数据（层级关系和变换）
/// - 其他属性（Mesh, Camera, Light, Skin等）移至 Scene 的组件存储
/// - 提高 CPU 缓存命中率
#[derive(Debug, Clone)]
pub struct Node {
    // === 核心层级关系 (Hierarchy) ===
    pub parent: Option<NodeHandle>,
    pub children: Vec<NodeHandle>,

    // === 核心空间数据 ===
    // 这些是每帧必须访问的热数据 (Hot Data)
    pub transform: Transform,

    // === 核心状态 ===
    pub visible: bool,
}

impl Node {
    pub fn new() -> Self {
        Self {
            parent: None,
            children: Vec::new(),
            transform: Transform::new(),
            visible: true,
        }
    }

    /// 获取世界矩阵
    #[inline]
    pub fn world_matrix(&self) -> &Affine3A {
        &self.transform.world_matrix
    }
}

impl Default for Node {
    fn default() -> Self {
        Self::new()
    }
}