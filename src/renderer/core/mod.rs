//! wgpu Core Context and Resource Management
//!
//! This module provides low-level GPU abstractions:
//!
//! - [`WgpuContext`]: Core GPU context (device, queue, surface, config)
//! - [`ResourceManager`]: GPU resource lifecycle management (buffers, textures, bind groups)
//! - [`Bindings`]: Shader resource binding utilities
//! - [`ResourceBuilder`]: Declarative resource binding builder
//!
//! # Architecture
//!
//! The core module sits between the raw wgpu API and the higher-level
//! rendering abstractions. It handles:
//!
//! - Device and queue management
//! - Surface configuration and resize
//! - GPU resource creation and caching
//! - Bind group layout management
//!
//! # Thread Safety
//!
//! Most types are designed to be used from a single thread, though
//! the underlying wgpu types are Send + Sync.

pub mod binding;
pub mod builder;
pub mod context;
pub mod resources;
pub mod view;

pub use binding::{BindingResource, Bindings};
pub use builder::ResourceBuilder;
pub use context::WgpuContext;
pub use resources::{BindGroupContext, ResourceManager};
pub use view::{RenderView, ViewTarget};
