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
pub mod material_builder;
pub mod texture;
pub mod image;
pub mod geometry;
pub mod primitives;
pub mod buffer;
pub mod uniforms;
pub mod version_tracker;
pub mod material;

pub use mesh::{Mesh, MeshHandle};
pub use material::{
    Material, MaterialData, MaterialFeatures, MaterialTrait,
    MeshBasicMaterial, MeshStandardMaterial, MeshPhongMaterial, MeshPhysicalMaterial, 
    MaterialSettings, MaterialBindings, Side,
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