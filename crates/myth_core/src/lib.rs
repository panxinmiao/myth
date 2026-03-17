//! # Myth Core
//!
//! Foundational types and utilities for the Myth engine.
//!
//! This crate provides:
//! - Error types used across the engine
//! - Utility modules (timer, FPS counter, string interning)
//! - Core handle types (`NodeHandle`, `SkeletonKey`)
//! - The `Transform` component

pub mod errors;
pub mod handles;
pub mod transform;
pub mod utils;

pub use errors::{AssetError, Error, PlatformError, RenderError, Result};
pub use handles::{NodeHandle, SkeletonKey};
pub use transform::Transform;
pub use utils::interner::Symbol;

/// Maximum number of morph targets supported per mesh.
pub const MAX_MORPH_TARGETS: usize = 128;
