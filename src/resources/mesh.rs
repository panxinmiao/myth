use thunderdome::Index;
use crate::assets::{GeometryHandle, MaterialHandle};

pub type MeshHandle = Index;

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
        }
    }
}