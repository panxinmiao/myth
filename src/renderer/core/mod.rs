//! WGPU 核心上下文封装
//!
//! 提供：
//! - WgpuContext: 只持有 device, queue, surface, config，负责 Resize 和 Present
//! - ResourceManager: GPU 资源管理（Buffer, Texture, BindGroup）
//! - 绑定相关工具

pub mod context;
pub mod resources;
pub mod binding;
pub mod builder;

pub use context::WgpuContext;
pub use resources::{ResourceManager, BindGroupContext};
pub use binding::{BindingResource, Bindings};
pub use builder::ResourceBuilder;
