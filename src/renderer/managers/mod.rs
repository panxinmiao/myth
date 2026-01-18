//! GPU 数据管理器
//!
//! 已废弃模块 - 功能已整合到 ResourceManager 中
//! 保留此模块仅为向后兼容，新代码请使用 crate::renderer::core::ResourceManager

// 重新导出 ResourceManager 中的类型以保持向后兼容
pub use crate::renderer::core::{ObjectBindingData, CachedBindGroupId};
