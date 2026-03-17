//! wgpu Core Context and Resource Management
//!
//! This module provides low-level GPU abstractions:
//!
//! - [`WgpuContext`]: Core GPU context (device, queue, surface, config)
//! - [`ResourceManager`]: GPU resource lifecycle management
//! - [`Bindings`]: Shader resource binding trait
//!
//! [`ResourceBuilder`](myth_resources::ResourceBuilder) and
//! [`BindingResource`](myth_resources::BindingResource) are defined in
//! `myth_resources` and re-exported here for convenience.

pub mod binding;
pub mod context;
pub mod gpu;
pub mod view;

pub use binding::{Bindings, GlobalBindGroupCache};
pub use context::WgpuContext;
pub use gpu::{BindGroupContext, ResourceManager};
pub use view::{RenderView, ViewTarget};

// Re-export core resource types from myth_resources.
pub use myth_resources::{BindingResource, ResourceBuilder};
