//! 核心资源定义模块
//! 
//! 包含渲染所需的核心数据结构，不依赖于 GPU 实现：
//! - Mesh: 网格对象
//! - Material: 材质定义
//! - Texture: 纹理配置
//! - Image: 图像数据
//! - Geometry: 几何数据
//! - Buffer: 通用缓冲区数据
//! - Uniforms: 着色器统一变量

pub mod mesh;
pub mod material;
pub mod material_builder;
pub mod texture;
pub mod image;
pub mod geometry;
pub mod buffer;
pub mod uniforms;
pub mod uniform_slot;
pub mod version_tracker;

// 重新导出常用类型
pub use mesh::{Mesh, MeshHandle};
pub use material::{
    Material, MaterialData, MaterialFeatures,
    MeshBasicMaterial, MeshStandardMaterial, MeshPhongMaterial,
};
pub use material_builder::{MeshBasicMaterialBuilder, MeshStandardMaterialBuilder};
pub use texture::{Texture, TextureSampler};
pub use image::{Image, ImageDescriptor};
pub use geometry::{
    Geometry, Attribute, 
    BoundingBox, BoundingSphere,
    GeometryFeatures,
};
pub use buffer::{BufferRef};
pub use uniforms::{WgslType};
pub use uniform_slot::{UniformSlot};