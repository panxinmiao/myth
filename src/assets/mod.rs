//! 资产管理系统模块
//! 
//! 提供资产的加载、存储和管理功能：
//! - AssetServer: 资产服务器（原 core/assets.rs）
//! - Handle: 资产句柄类型

pub mod server;

// 重新导出 AssetServer 及相关类型
pub use server::{
    AssetServer,
    GeometryHandle, MaterialHandle, TextureHandle,
};
