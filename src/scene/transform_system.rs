//! 变换系统 (Transform System)
//!
//! 负责场景图的矩阵层级更新，与 Scene 解耦以避免借用冲突。
//! 这是一个独立的系统，只需要借用 nodes Arena 和 root_nodes 列表。

use glam::Affine3A;
use thunderdome::Arena;
use slotmap::SlotMap;

use crate::scene::node::Node;
use crate::scene::camera::Camera;
use crate::scene::{NodeIndex, CameraKey};

/// 更新整个场景层级的世界矩阵
/// 
/// # 参数
/// * `nodes` - 节点 Arena 的可变引用
/// * `cameras` - 相机 SlotMap 的可变引用（用于同步更新相机的视图投影矩阵）
/// * `roots` - 根节点索引列表
/// 
/// # 设计说明
/// 这个函数只借用必要的数据结构，而不是整个 Scene，
/// 从而避免了 "上帝对象" 导致的借用冲突问题。
pub fn update_hierarchy(
    nodes: &mut Arena<Node>,
    cameras: &mut SlotMap<CameraKey, Camera>,
    roots: &[NodeIndex],
) {
    for &root_idx in roots {
        update_transform_recursive(nodes, cameras, root_idx, Affine3A::IDENTITY, false);
    }
}

/// 递归更新节点变换（迭代优化版本）
/// 
/// 使用显式栈替代递归调用，避免深层级场景的栈溢出风险，
/// 同时减少重复借用开销。
pub fn update_hierarchy_iterative(
    nodes: &mut Arena<Node>,
    cameras: &mut SlotMap<CameraKey, Camera>,
    roots: &[NodeIndex],
) {
    // 工作栈：(节点索引, 父世界矩阵, 父是否变化)
    let mut stack: Vec<(NodeIndex, Affine3A, bool)> = Vec::with_capacity(64);
    
    // 初始化：所有根节点入栈
    for &root_idx in roots.iter().rev() {
        stack.push((root_idx, Affine3A::IDENTITY, false));
    }
    
    while let Some((node_idx, parent_world_matrix, parent_changed)) = stack.pop() {
        // 获取当前节点
        let Some(node) = nodes.get_mut(node_idx) else {
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
            if let Some(camera_idx) = node.camera {
                if let Some(camera) = cameras.get_mut(camera_idx) {
                    camera.update_view_projection(&new_world);
                }
            }
        }
        
        // 3. 收集子节点信息（避免二次借用）
        let current_world = node.transform.world_matrix;
        let children_count = node.children.len();
        
        // 4. 将子节点压入栈（逆序以保持处理顺序）
        for i in (0..children_count).rev() {
            if let Some(node) = nodes.get(node_idx) {
                if let Some(&child_idx) = node.children.get(i) {
                    stack.push((child_idx, current_world, world_needs_update));
                }
            }
        }
    }
}

/// 递归更新单个节点及其子树（保留原始递归版本作为参考）
fn update_transform_recursive(
    nodes: &mut Arena<Node>,
    cameras: &mut SlotMap<CameraKey, Camera>,
    node_idx: NodeIndex,
    parent_world_matrix: Affine3A,
    parent_changed: bool,
) {
    // 阶段 1: 处理当前节点
    let (current_world_matrix, children_indices, world_needs_update) = {
        let Some(node) = nodes.get_mut(node_idx) else {
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
            if let Some(camera_idx) = node.camera {
                if let Some(camera) = cameras.get_mut(camera_idx) {
                    camera.update_view_projection(&new_world);
                }
            }
        }
        
        // 收集必要信息，避免后续借用冲突
        let world = node.transform.world_matrix;
        let children: Vec<NodeIndex> = node.children.clone();
        
        (world, children, world_needs_update)
    };
    
    // 阶段 2: 递归处理子节点
    for child_idx in children_indices {
        update_transform_recursive(nodes, cameras, child_idx, current_world_matrix, world_needs_update);
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
    nodes: &mut Arena<Node>,
    cameras: &mut SlotMap<CameraKey, Camera>,
    root_idx: NodeIndex,
) {
    // 获取父节点的世界矩阵（如果有的话）
    let parent_world = if let Some(node) = nodes.get(root_idx) {
        if let Some(parent_idx) = node.parent {
            nodes.get(parent_idx)
                .map(|p| p.transform.world_matrix)
                .unwrap_or(Affine3A::IDENTITY)
        } else {
            Affine3A::IDENTITY
        }
    } else {
        return;
    };
    
    update_transform_recursive(nodes, cameras, root_idx, parent_world, true);
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    
    #[test]
    fn test_hierarchy_update() {
        let mut nodes = Arena::new();
        let mut cameras = SlotMap::with_key();
        
        // 创建简单的父子层级
        let mut parent = Node::new("parent");
        parent.transform.position = Vec3::new(1.0, 0.0, 0.0);
        let parent_idx = nodes.insert(parent);
        
        let mut child = Node::new("child");
        child.transform.position = Vec3::new(0.0, 1.0, 0.0);
        child.parent = Some(parent_idx);
        let child_idx = nodes.insert(child);
        
        // 建立父子关系
        nodes.get_mut(parent_idx).unwrap().children.push(child_idx);
        
        let roots = vec![parent_idx];
        
        // 执行更新
        update_hierarchy(&mut nodes, &mut cameras, &roots);
        
        // 验证子节点的世界位置
        let child_world_pos = nodes.get(child_idx).unwrap().transform.world_matrix.translation;
        assert!((child_world_pos.x - 1.0).abs() < 1e-5);
        assert!((child_world_pos.y - 1.0).abs() < 1e-5);
    }
}
