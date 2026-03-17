//! Myth Render — Core rendering system for the Myth engine.
//!
//! This crate provides the complete GPU rendering pipeline, including:
//!
//! - **[`core`]**: wgpu context wrapper ([`WgpuContext`](core::WgpuContext)),
//!   resource management, and bind-group utilities.
//! - **[`graph`]**: Declarative Render Graph (RDG) for frame organization,
//!   scene extraction, pass scheduling, and compositing.
//! - **[`pipeline`]**: Shader compilation, template preprocessing, and
//!   two-level pipeline cache (L1/L2).
//! - **[`settings`]**: Render path configuration and quality knobs.

pub mod core;
pub mod graph;
pub mod pipeline;
pub mod renderer;
pub mod settings;

pub use renderer::Renderer;
pub use settings::{RenderPath, RendererSettings};

/// HDR texture format used for high dynamic range render targets.
pub const HDR_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

