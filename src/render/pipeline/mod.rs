//! 渲染管线子模块
//! 
//! 管理着色器编译和管线状态：
//! - PipelineCache: 管线缓存
//! - VertexLayout: 顶点布局
//! - ShaderManager: 着色器管理器
//! - ShaderGenerator: 着色器生成器

pub mod cache;
pub mod vertex;
pub mod shader_manager;
pub mod shader_gen;

// 重新导出常用类型
pub use cache::PipelineCache;
pub use vertex::{GeneratedVertexLayout, OwnedVertexBufferDesc};
pub use shader_gen::ShaderGenerator;
