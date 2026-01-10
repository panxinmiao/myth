use uuid::Uuid;
use thunderdome::Index;
use crate::core::assets::{GeometryHandle, MaterialHandle};

pub type MeshHandle = Index;

#[derive(Debug, Clone)]
pub struct Mesh {
    // === 标识 ===
    pub id: Uuid,
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
}

impl Mesh {
    pub fn new(
        geometry: GeometryHandle, 
        material: MaterialHandle
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: "Mesh".to_string(),
            node_id : None,
            geometry,
            material,
            visible: true,
            render_order: 0,
        }
    }
}