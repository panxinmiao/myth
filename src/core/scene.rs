use thunderdome::{Arena, Index};
use std::sync::{Arc, RwLock};
use glam::{Affine3A, Vec4}; 
use crate::core::node::Node;
use crate::core::mesh::Mesh;
use crate::core::camera::Camera;
use crate::core::material::Material;
use crate::core::geometry::Geometry;

pub struct Scene {
    pub nodes: Arena<Node>,
    pub root_nodes: Vec<Index>,

    // ====组件/资源池====
    pub meshes: Arena<Mesh>,
    pub cameras: Arena<Camera>,

    // pub lights: Arena<Light>, // 未来可扩展

    // 暂时简单用 RGBA，后面可以用 Texture
    pub background: Option<Vec4>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            nodes: Arena::new(),
            root_nodes: Vec::new(),
            meshes: Arena::new(),
            cameras: Arena::new(),
            background: Some(Vec4::new(0.0, 0.0, 0.0, 1.0)),
        }
    }

    /// 开始构建一个节点
    pub fn build_node(&mut self, name: &str) -> NodeBuilder {
        NodeBuilder::new(self, name)
    }

    /// 添加一个节点到场景 (默认放在根节点)
    pub fn add_node(&mut self, node: Node, parent: Option<Index>) -> Index {
        let idx = self.nodes.insert(node);

        if let Some(parent_idx) = parent {
            // 如果指定了父节点，建立父子关系
            if let Some(p) = self.nodes.get_mut(parent_idx) {
                p.children.push(idx);
            }
            if let Some(c) = self.nodes.get_mut(idx) {
                c.parent = Some(parent_idx);
            }
        } else {
            // 否则作为根节点
            self.root_nodes.push(idx);
        }
        idx
    }

    /// 移除节点 (递归移除所有子节点)
    pub fn remove_node(&mut self, idx: Index) {
        // 1. 先把它的 children 列表拿出来，避免借用冲突
        let children = if let Some(node) = self.nodes.get(idx) {
            node.children.clone()
        } else {
            return;
        };

        // 2. 递归移除子节点
        for child in children {
            self.remove_node(child);
        }

        // 3. 处理父节点关系
        // 获取该节点的 parent ID
        let parent_opt = self.nodes.get(idx).and_then(|n| n.parent);

        if let Some(parent_idx) = parent_opt {
            // 从父节点的 children 列表中移除自己
            if let Some(parent) = self.nodes.get_mut(parent_idx) {
                if let Some(pos) = parent.children.iter().position(|&x| x == idx) {
                    parent.children.remove(pos);
                }
            }
        } else {
            // 如果是根节点，从 root_nodes 移除
            if let Some(pos) = self.root_nodes.iter().position(|&x| x == idx) {
                self.root_nodes.remove(pos);
            }
        }

        // 3. === 清理组件 ===
        self.meshes.remove(idx);
        self.cameras.remove(idx);

        // 4. 彻底删除数据
        self.nodes.remove(idx);
    }

    /// 核心逻辑：建立父子关系 (Attach)
    /// 解决 Rust 中同时借用两个 Node 的难题
    pub fn attach(&mut self, child_idx: Index, parent_idx: Index) {
        if child_idx == parent_idx {{
            log::warn!("Cannot attach node to itself!");
            return;
        }}
        // 1. Detach from old
        let old_parent = self.nodes.get(child_idx).and_then(|n| n.parent);
        if let Some(p) = old_parent {
            if let Some(n) = self.nodes.get_mut(p) {
                if let Some(i) = n.children.iter().position(|&x| x == child_idx) {
                    n.children.remove(i);
                }
            }
        } else {
             if let Some(i) = self.root_nodes.iter().position(|&x| x == child_idx) {
                self.root_nodes.remove(i);
            }
        }

        // 2. Attach to new
        if let Some(p) = self.nodes.get_mut(parent_idx) {
            p.children.push(child_idx);
        } else {
            log::error!("Parent node not found during attach!");
            // 恢复 child 到 root_nodes 防止数据丢失（可选策略）
            self.root_nodes.push(child_idx);
            return;
        }
        
        // 3. Update child
        if let Some(c) = self.nodes.get_mut(child_idx) {
            c.parent = Some(parent_idx);
            c.mark_dirty(); // 强制标记脏，确保矩阵更新
        }
    }

    /// 获取只读引用
    pub fn get_node(&self, idx: Index) -> Option<&Node> {
        self.nodes.get(idx)
    }

    /// 获取可变引用 (用于修改 TRS)
    pub fn get_node_mut(&mut self, idx: Index) -> Option<&mut Node> {
        self.nodes.get_mut(idx)
    }

    // 基础 API：单纯添加 Mesh 数据，返回句柄
    // 仅供内部或高级用户使用
    pub fn add_mesh(&mut self, mesh: Mesh) -> Index {
        self.meshes.insert(mesh)
    }

    // =========================================================
    // ✨ 推荐的高级 API：一步到位创建物体
    // =========================================================
    pub fn create_mesh(&mut self, name: &str, geometry: Arc<RwLock<Geometry>>, material: Arc<RwLock<Material>>, parent: Option<Index>) -> Index {
        // 1. 先创建 Node (为了拿到 node_id)
        let node = Node::new(name);
        let node_id = self.add_node(node, parent);

        // 2. 再创建 Mesh (填入 node_id)
        let mesh = Mesh::new(Some(node_id), geometry, material);
        let mesh_id = self.meshes.insert(mesh);

        // 3. 回填：把 mesh_id 填回 Node
        // 这一步建立了双向链接：Node -> Mesh, Mesh -> Node
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.mesh = Some(mesh_id);
        }

        // 返回 Node ID，因为用户通常操作的是 Node (移动、旋转)
        node_id
    }

    // ========================================================================
    // 矩阵更新流水线 (Affine3A 版)
    // ========================================================================

    /// 更新整个场景的世界矩阵
    /// 这是每帧渲染前必须调用的
    pub fn update_matrix_world(&mut self) {
        let roots = self.root_nodes.clone();
        for root_idx in roots {
            // 根节点的父矩阵是 Identity (Affine3A::IDENTITY)
            self.update_transform_recursive(root_idx, Affine3A::IDENTITY, false);
        }
    }

    fn update_transform_recursive(
        &mut self, 
        node_idx: Index, 
        parent_world_matrix: Affine3A, // 注意类型变化
        parent_changed: bool
    ) {
        // 1. 借用 Node 数据
        let (current_world_matrix, children, changed) = {
            let node = self.nodes.get_mut(node_idx).unwrap();
            
            // 1. 智能更新局部矩阵 (Shadow State Check)
            let local_changed = node.update_local_matrix();

            // 2. 决定是否更新世界矩阵
            let world_needs_update = local_changed || parent_changed;

            if world_needs_update {
                // Affine3A 的乘法比 Mat4 更快，忽略最后一行 0,0,0,1 的计算
                let new_world = parent_world_matrix * *node.local_matrix();
                node.set_world_matrix(new_world);
            }

            (*node.world_matrix(), node.children.clone(), world_needs_update)
        }; 

        for child_idx in children {
            self.update_transform_recursive(child_idx, current_world_matrix, changed);
        }
    }
}


pub struct NodeBuilder<'a> {
    scene: &'a mut Scene,
    node: Node, // 暂存正在构建的 Node 数据
    parent: Option<Index>, // 暂存父节点 ID
}

impl<'a> NodeBuilder<'a> {
    pub fn new(scene: &'a mut Scene, name: &str) -> Self {
        Self {
            scene,
            node: Node::new(name),
            parent: None,
        }
    }

    // === 链式配置方法 ===

    pub fn with_position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.node.position = glam::Vec3::new(x, y, z);
        self
    }

    pub fn with_scale(mut self, s: f32) -> Self {
        self.node.scale = glam::Vec3::splat(s);
        self
    }

    /// 设置父节点
    pub fn with_parent(mut self, parent: Index) -> Self {
        self.parent = Some(parent);
        self
    }

    /// 关联 Mesh (传入 Mesh 句柄)
    pub fn with_mesh(mut self, mesh_handle: Index) -> Self {
        self.node.mesh = Some(mesh_handle);
        self
    }

    // === 终结方法 ===

    /// 完成构建，将 Node 插入 Scene，返回 Index
    pub fn build(self) -> Index {
        // 1. 插入 Node Arena
        let node_idx = self.scene.nodes.insert(self.node);

        // 2. 处理父子关系
        if let Some(parent_idx) = self.parent {
            self.scene.attach(node_idx, parent_idx);
        } else {
            // 如果没父节点，默认加入 root
            self.scene.root_nodes.push(node_idx);
        }

        node_idx
    }
}