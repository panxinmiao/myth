//! 帧级动态数据管理子模块
//! 
//! 管理每帧变化的数据（如模型变换矩阵）：
//! - ModelBufferManager: 模型缓冲区管理器

pub mod model_manager;
pub mod skeleton_manager;

// 重新导出常用类型
pub use model_manager::{ModelManager, ObjectBindingData};
pub use skeleton_manager::SkeletonManager;