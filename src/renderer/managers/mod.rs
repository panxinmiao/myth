//! GPU 数据管理器
//!
//! 各类数据的 GPU 映射管理器：
//! - ModelManager: 模型变换矩阵管理
//! - SkeletonManager: 骨骼动画管理

pub mod model;
pub mod skeleton;

pub use model::{ModelManager, ObjectBindingData};
pub use skeleton::SkeletonManager;
