//! GPU 资源管理子模块
//! 
//! 负责 GPU 端资源的创建、更新和管理：
//! - ResourceManager: 资源管理器（核心大管家）
//! - GpuImage: GPU 图像资源
//! - GpuTexture: GPU 纹理资源
//! - GpuBuffer: GPU 缓冲区资源
//! - Binding: BindGroup 管理
//! - ResourceBuilder: 资源构建器

pub mod manager;
pub mod image;
pub mod texture;
pub mod buffer;
pub mod binding;
pub mod builder;

// 重新导出常用类型
pub use manager::ResourceManager;
pub use image::GpuImage;
pub use texture::GpuTexture;
pub use buffer::GpuBuffer;
pub use binding::{BindingResource, Bindings};
pub use builder::ResourceBuilder;
