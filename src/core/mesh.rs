use std::sync::{Arc, RwLock};
use uuid::Uuid;
use thunderdome::Index;
use crate::core::geometry::Geometry;
use crate::core::material::Material;

#[derive(Debug, Clone)]
pub struct Mesh {
    // === 标识 ===
    pub id: Uuid,
    pub name: String,
    
    // === 场景图节点 ===
    // 这是一个弱引用 (Index)，具体的 Node 数据在 Scene.nodes 里
    pub node_id: Option<Index>,
    
    // === 资源引用 (共享所有权) ===
    // 使用 RwLock 允许我们在 Mesh 存在时修改 Geometry (比如做变形动画)
    pub geometry: Arc<RwLock<Geometry>>,
    pub material: Arc<RwLock<Material>>,
    
    // === 实例特定的渲染设置 ===
    // 有时候我们想隐藏某个 Mesh，但 Node 还在 (比如 LOD 切换)
    pub visible: bool, 
    
    // 绘制顺序 (Render Order)
    // 对于透明物体排序很有用
    pub render_order: i32, 
}

impl Mesh {
    pub fn new(
        node_id: Option<Index>, 
        geometry: Arc<RwLock<Geometry>>, 
        material: Arc<RwLock<Material>>
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: "Mesh".to_string(),
            node_id,
            geometry,
            material,
            visible: true,
            render_order: 0,
        }
    }

    pub fn from_resource(
        geometry: Geometry, 
        material: Material
    ) -> Self {
        Self::new(
            None,
            Arc::new(RwLock::new(geometry)),
            Arc::new(RwLock::new(material)),
        )
    }
}