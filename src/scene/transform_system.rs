//! 变换系统 (Transform System)
//!
//! 负责场景图的矩阵层级更新，与 Scene 解耦以避免借用冲突。
//! 这是一个独立的系统，只需要借用 nodes SlotMap 和 root_nodes 列表。
//!
//! # 并行化准备
//! 
//! 场景图更新可以按层级（BFS顺序）分批并行化：
//! 1. 第一层：所有根节点（无依赖，可并行）
//! 2. 第二层：根节点的所有直接子节点（依赖第一层结果，本层内可并行）
//! 3. ...以此类推
//!
//! 使用 `build_level_order_batches` 预计算层级批次，然后可以：
//! - 单线程顺序执行各批次
//! - 使用 rayon 对每批次内的节点并行处理

use glam::Affine3A;
use slotmap::{SlotMap, SparseSecondaryMap};

use crate::scene::node::Node;
use crate::scene::camera::Camera;
use crate::scene::{NodeHandle, CameraKey};

/// 层级批次信息，用于并行化处理
#[derive(Debug, Default)]
pub struct LevelOrderBatches {
    /// 每一层的节点列表，batches[0] 是根节点层，batches[1] 是第一层子节点...
    pub batches: Vec<Vec<NodeHandle>>,
    /// 每个节点对应的父节点索引（用于查找父世界矩阵）
    pub parent_indices: Vec<Option<NodeHandle>>,
}

impl LevelOrderBatches {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// 清空并复用内存
    pub fn clear(&mut self) {
        for batch in &mut self.batches {
            batch.clear();
        }
        self.parent_indices.clear();
    }
    
    /// 获取总节点数
    pub fn total_nodes(&self) -> usize {
        self.batches.iter().map(|b| b.len()).sum()
    }
    
    /// 获取层级深度
    pub fn depth(&self) -> usize {
        self.batches.len()
    }
}

/// 构建按层级排序的节点批次（BFS顺序）
/// 
/// 这个函数将场景图按层级展开，使得：
/// - 同一层级的节点在同一个批次中，可以并行处理
/// - 不同层级按顺序处理，确保父节点先于子节点更新
/// 
/// # 并行化策略
/// ```ignore
/// for batch in batches.iter() {
///     // 每个 batch 内的节点可以并行处理
///     batch.par_iter().for_each(|node_handle| {
///         // 安全：同层级节点互不依赖
///         update_single_node(...);
///     });
/// }
/// ```
pub fn build_level_order_batches(
    nodes: &SlotMap<NodeHandle, Node>,
    roots: &[NodeHandle],
    output: &mut LevelOrderBatches,
) {
    output.clear();
    
    if roots.is_empty() {
        return;
    }
    
    // 第一层：根节点
    let mut current_level: Vec<NodeHandle> = roots.to_vec();
    
    while !current_level.is_empty() {
        // 收集下一层节点
        let mut next_level = Vec::new();
        
        for &node_handle in &current_level {
            if let Some(node) = nodes.get(node_handle) {
                for &child_handle in &node.children {
                    next_level.push(child_handle);
                }
            }
        }
        
        // 保存当前层
        if output.batches.len() <= output.depth() {
            output.batches.push(current_level);
        } else {
            let idx = output.depth();
            output.batches[idx] = current_level;
        }
        
        current_level = next_level;
    }
    
    // 移除空批次
    output.batches.retain(|b| !b.is_empty());
}

/// 使用预计算的批次进行层级更新（为并行化准备）
/// 
/// 当前是单线程版本，但结构已经为 rayon 并行做好准备。
/// 启用并行只需将内层循环改为 `par_iter`。
pub fn update_hierarchy_batched(
    nodes: &mut SlotMap<NodeHandle, Node>,
    cameras: &mut SlotMap<CameraKey, Camera>,
    camera_components: &SparseSecondaryMap<NodeHandle, CameraKey>,
    batches: &LevelOrderBatches,
) {
    for (level, batch) in batches.batches.iter().enumerate() {
        for &node_handle in batch {
            // 获取父世界矩阵
            let parent_world = if let Some(node) = nodes.get(node_handle) {
                if let Some(parent_handle) = node.parent {
                    nodes.get(parent_handle)
                        .map(|p| p.transform.world_matrix)
                        .unwrap_or(Affine3A::IDENTITY)
                } else {
                    Affine3A::IDENTITY
                }
            } else {
                continue;
            };
            
            // 更新当前节点
            if let Some(node) = nodes.get_mut(node_handle) {
                let local_changed = node.transform.update_local_matrix();
                let parent_changed = level > 0;
                
                if local_changed || parent_changed {
                    let new_world = parent_world * *node.transform.local_matrix();
                    node.transform.set_world_matrix(new_world);
                    
                    // 同步更新相机
                    if let Some(&camera_key) = camera_components.get(node_handle) {
                        if let Some(camera) = cameras.get_mut(camera_key) {
                            camera.update_view_projection(&new_world);
                        }
                    }
                }
            }
        }
    }
}

/// 更新整个场景层级的世界矩阵
/// 
/// # 参数
/// * `nodes` - 节点 SlotMap 的可变引用
/// * `cameras` - 相机 SlotMap 的可变引用（用于同步更新相机的视图投影矩阵）
/// * `camera_components` - 节点到相机的组件映射
/// * `roots` - 根节点句柄列表
/// 
/// # 设计说明
/// 这个函数只借用必要的数据结构，而不是整个 Scene，
/// 从而避免了 "上帝对象" 导致的借用冲突问题。
pub fn update_hierarchy(
    nodes: &mut SlotMap<NodeHandle, Node>,
    cameras: &mut SlotMap<CameraKey, Camera>,
    camera_components: &SparseSecondaryMap<NodeHandle, CameraKey>,
    roots: &[NodeHandle],
) {
    for &root_handle in roots {
        update_transform_recursive(nodes, cameras, camera_components, root_handle, Affine3A::IDENTITY, false);
    }
}

/// 递归更新节点变换（迭代优化版本）
/// 
/// 使用显式栈替代递归调用，避免深层级场景的栈溢出风险，
/// 同时减少重复借用开销。
pub fn update_hierarchy_iterative(
    nodes: &mut SlotMap<NodeHandle, Node>,
    cameras: &mut SlotMap<CameraKey, Camera>,
    camera_components: &SparseSecondaryMap<NodeHandle, CameraKey>,
    roots: &[NodeHandle],
) {
    // 工作栈：(节点句柄, 父世界矩阵, 父是否变化)
    let mut stack: Vec<(NodeHandle, Affine3A, bool)> = Vec::with_capacity(64);
    
    // 初始化：所有根节点入栈
    for &root_handle in roots.iter().rev() {
        stack.push((root_handle, Affine3A::IDENTITY, false));
    }
    
    while let Some((node_handle, parent_world_matrix, parent_changed)) = stack.pop() {
        // 获取当前节点
        let Some(node) = nodes.get_mut(node_handle) else {
            continue;
        };
        
        // 1. 更新局部矩阵
        let local_changed = node.transform.update_local_matrix();
        let world_needs_update = local_changed || parent_changed;
        
        // 2. 更新世界矩阵
        if world_needs_update {
            let new_world = parent_world_matrix * *node.transform.local_matrix();
            node.transform.set_world_matrix(new_world);
            
            // 同步更新相机
            if let Some(&camera_key) = camera_components.get(node_handle) {
                if let Some(camera) = cameras.get_mut(camera_key) {
                    camera.update_view_projection(&new_world);
                }
            }
        }
        
        // 3. 收集子节点信息（避免二次借用）
        let current_world = node.transform.world_matrix;
        let children_count = node.children.len();
        
        // 4. 将子节点压入栈（逆序以保持处理顺序）
        for i in (0..children_count).rev() {
            if let Some(node) = nodes.get(node_handle) {
                if let Some(&child_handle) = node.children.get(i) {
                    stack.push((child_handle, current_world, world_needs_update));
                }
            }
        }
    }
}

/// 递归更新单个节点及其子树（保留原始递归版本作为参考）
fn update_transform_recursive(
    nodes: &mut SlotMap<NodeHandle, Node>,
    cameras: &mut SlotMap<CameraKey, Camera>,
    camera_components: &SparseSecondaryMap<NodeHandle, CameraKey>,
    node_handle: NodeHandle,
    parent_world_matrix: Affine3A,
    parent_changed: bool,
) {
    // 阶段 1: 处理当前节点
    let (current_world_matrix, children_handles, world_needs_update) = {
        let Some(node) = nodes.get_mut(node_handle) else {
            return;
        };
        
        // 1. 智能更新局部矩阵
        let local_changed = node.transform.update_local_matrix();
        
        // 2. 决定是否更新世界矩阵
        let world_needs_update = local_changed || parent_changed;
        
        if world_needs_update {
            let new_world = parent_world_matrix * *node.transform.local_matrix();
            node.transform.set_world_matrix(new_world);
            
            // 同步更新相机
            if let Some(&camera_key) = camera_components.get(node_handle) {
                if let Some(camera) = cameras.get_mut(camera_key) {
                    camera.update_view_projection(&new_world);
                }
            }
        }
        
        // 收集必要信息，避免后续借用冲突
        let world = node.transform.world_matrix;
        let children: Vec<NodeHandle> = node.children.clone();
        
        (world, children, world_needs_update)
    };
    
    // 阶段 2: 递归处理子节点
    for child_handle in children_handles {
        update_transform_recursive(nodes, cameras, camera_components, child_handle, current_world_matrix, world_needs_update);
    }
}

/// 仅更新单个节点的局部矩阵（不递归）
/// 用于需要立即刷新单个节点的场景
#[inline]
pub fn update_single_node_local(node: &mut Node) -> bool {
    node.transform.update_local_matrix()
}

/// 从指定节点开始向下更新子树
/// 用于局部更新场景图的一部分
pub fn update_subtree(
    nodes: &mut SlotMap<NodeHandle, Node>,
    cameras: &mut SlotMap<CameraKey, Camera>,
    camera_components: &SparseSecondaryMap<NodeHandle, CameraKey>,
    root_handle: NodeHandle,
) {
    // 获取父节点的世界矩阵（如果有的话）
    let parent_world = if let Some(node) = nodes.get(root_handle) {
        if let Some(parent_handle) = node.parent {
            nodes.get(parent_handle)
                .map(|p| p.transform.world_matrix)
                .unwrap_or(Affine3A::IDENTITY)
        } else {
            Affine3A::IDENTITY
        }
    } else {
        return;
    };
    
    update_transform_recursive(nodes, cameras, camera_components, root_handle, parent_world, true);
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    
    #[test]
    fn test_hierarchy_update() {
        let mut nodes: SlotMap<NodeHandle, Node> = SlotMap::with_key();
        let mut cameras: SlotMap<CameraKey, Camera> = SlotMap::with_key();
        let camera_components: SparseSecondaryMap<NodeHandle, CameraKey> = SparseSecondaryMap::new();
        
        // 创建简单的父子层级
        let mut parent = Node::new();
        parent.transform.position = Vec3::new(1.0, 0.0, 0.0);
        let parent_handle = nodes.insert(parent);
        
        let mut child = Node::new();
        child.transform.position = Vec3::new(0.0, 1.0, 0.0);
        child.parent = Some(parent_handle);
        let child_handle = nodes.insert(child);
        
        // 建立父子关系
        nodes.get_mut(parent_handle).unwrap().children.push(child_handle);
        
        let roots = vec![parent_handle];
        
        // 执行更新
        update_hierarchy(&mut nodes, &mut cameras, &camera_components, &roots);
        
        // 验证子节点的世界位置
        let child_world_pos = nodes.get(child_handle).unwrap().transform.world_matrix.translation;
        assert!((child_world_pos.x - 1.0).abs() < 1e-5);
        assert!((child_world_pos.y - 1.0).abs() < 1e-5);
    }
}
