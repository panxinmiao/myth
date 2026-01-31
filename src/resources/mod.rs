//! Core Resource Definitions
//!
//! This module contains CPU-side data structures for rendering resources.
//! These types are GPU-agnostic and define the logical representation of
//! rendering data before it's uploaded to the GPU.
//!
//! # Module Structure
//!
//! - [`mesh`] - Mesh objects combining geometry and materials
//! - [`geometry`] - Vertex data and attributes (positions, normals, UVs, etc.)
//! - [`material`] - Material definitions (Standard, Physical, Phong, Basic)
//! - [`texture`] - Texture configuration and sampling parameters
//! - [`image`] - Raw image data storage
//! - [`buffer`] - Generic CPU buffer with version tracking
//! - [`uniforms`] - Shader uniform data structures
//! - [`shader_defines`] - Dynamic shader macro system
//! - [`primitives`] - Built-in geometry primitives (Box, Sphere, Plane, etc.)
//!
//! # Design Principles
//!
//! - **CPU-side only**: No direct GPU dependencies in this module
//! - **Version tracking**: Resources track changes for efficient GPU sync
//! - **Handle-based references**: Use SlotMap handles for safe resource references
//! - **Shared ownership**: Use `Arc` for data that may be shared across objects

pub mod mesh;
pub mod texture;
pub mod image;
pub mod geometry;
pub mod primitives;
pub mod buffer;
pub mod uniforms;
pub mod version_tracker;
pub mod material;
pub mod input;
pub mod shader_defines;

pub use mesh::Mesh;
pub use material::{
    Material, MaterialType, MaterialTrait, RenderableMaterialTrait,
    MeshBasicMaterial, MeshStandardMaterial, MeshPhongMaterial, MeshPhysicalMaterial, 
    Side, TextureSlot, TextureTransform, AlphaMode,
};
pub use buffer::BufferRef;
pub use geometry::{Attribute, BoundingBox, BoundingSphere, Geometry};
pub use image::{Image, ImageDescriptor};
pub use input::{ButtonState, Input, Key, MouseButton};
pub use shader_defines::ShaderDefines;
pub use texture::{Texture, TextureSampler};
pub use uniforms::WgslType;