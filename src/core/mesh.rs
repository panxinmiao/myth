use std::sync::{Arc};
use uuid::Uuid;
use thunderdome::Index;
use crate::core::geometry::Geometry;
use crate::core::material::{Material};

#[derive(Debug, Clone)]
pub struct Mesh {
    // === 标识 ===
    pub id: Uuid,
    pub name: String,
    
    // === 场景图节点 ===
    pub node_id: Option<Index>,
    
    // === 资源引用 ===
    pub geometry: Arc<Geometry>,
    pub material: Arc<Material>,
    
    // === 实例特定的渲染设置 ===
    pub visible: bool, 
    
    // 绘制顺序 (Render Order)
    pub render_order: i32, 
}

impl Mesh {
    pub fn new(
        node_id: Option<Index>, 
        geometry: Arc<Geometry>, 
        material: Arc<Material>
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
            Arc::new(geometry),
            Arc::new(material),
        )
    }
}