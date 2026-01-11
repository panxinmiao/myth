//! Three - 基于 wgpu 的 Rust 渲染引擎
//! 
//! # 模块组织
//! 
//! - `resources/`: 核心资源定义（Mesh, Material, Texture, Image, Geometry）
//! - `assets/`: 资产管理系统（AssetServer, Handle）
//! - `scene/`: 场景图系统（Node, Scene, Camera, Light）
//! - `render/`: 渲染器（GPU 资源管理、管线、数据、Pass）
//! - `app/`: 应用层（窗口管理、事件循环）
//! - `core/`: 核心工具和类型（保留兼容）

pub mod resources;
pub mod assets;
pub mod scene;
pub mod render;
pub mod app;

// 重新导出常用类型
pub use resources::{Mesh, Material, Texture, Image, Geometry};
pub use assets::AssetServer;
pub use scene::{Node, Scene, Camera, Light};
pub use render::Renderer;