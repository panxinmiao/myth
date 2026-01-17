//! 渲染管线模块
//!
//! 管理着色器编译和管线状态：
//! - PipelineCache: 管线缓存（L1快速缓存 + L2规范缓存）
//! - vertex: 顶点布局生成
//! - shader_gen: 着色器代码生成
//! - shader_manager: 着色器模板管理

pub mod cache;
pub mod vertex;
pub mod shader_manager;
pub mod shader_gen;

pub use cache::{PipelineCache, PipelineKey, FastPipelineKey};
