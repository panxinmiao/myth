//! Pipeline Management and Shader Compilation
//!
//! This module manages shader compilation and render pipeline state:
//!
//! - [`PipelineCache`]: Two-level pipeline cache (L1 fast lookup + L2 canonical cache)
//! - [`PipelineKey`]: Unique identifier for pipeline configurations
//! - Vertex layout generation
//! - Shader template compilation with macro preprocessing
//!
//! # Caching Strategy
//!
//! The pipeline cache uses a two-level strategy:
//!
//! - **L1 Cache**: Fast hash-based lookup for common pipeline configurations
//! - **L2 Cache**: Canonical storage indexed by full pipeline descriptor
//!
//! This approach minimizes shader recompilation while supporting dynamic
//! shader macro combinations from materials.
//!
//! # Shader System
//!
//! Shaders use a template system with Jinja2-style macros:
//!
//! ```wgsl
//! {% if HAS_NORMAL_MAP %}
//! let normal = sample_normal_map(...);
//! {% endif %}
//! ```
//!
//! Materials declare their required macros via `shader_defines()`, which
//! are passed to the shader compiler.

pub mod cache;
pub mod shader_gen;
pub mod shader_manager;
pub mod vertex;

pub use cache::{
	FastPipelineKey, FastShadowPipelineKey, PipelineCache, PipelineKey, ShadowPipelineKey,
};
pub use shader_gen::ShaderCompilationOptions;
