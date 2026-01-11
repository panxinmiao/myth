//! 场景图系统模块
//! 
//! 管理场景层级结构和组件：
//! - Node: 场景节点（支持父子关系和变换）
//! - Scene: 场景容器
//! - Camera: 相机组件
//! - Light: 光源组件

pub mod node;
pub mod scene;
pub mod camera;
pub mod light;
pub mod environment;

// 重新导出常用类型
pub use node::Node;
pub use scene::Scene;
pub use camera::{Camera, ProjectionType};
pub use light::{Light, LightType};


use thunderdome::Index;
pub type NodeIndex = Index;

use slotmap::new_key_type;

new_key_type! {
    pub struct MeshKey;
    pub struct CameraKey;
    pub struct LightKey;
}