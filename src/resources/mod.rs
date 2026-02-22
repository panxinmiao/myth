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
//! - **Handle-based references**: Use `SlotMap` handles for safe resource references
//! - **Shared ownership**: Use `Arc` for data that may be shared across objects

pub mod bloom;
pub mod buffer;
pub mod fxaa;
pub mod geometry;
pub mod image;
pub mod input;
pub mod material;
pub mod mesh;
pub mod primitives;
pub mod shader_defines;
pub mod texture;
pub mod tone_mapping;
pub mod uniforms;
pub mod version_tracker;

pub use material::{
    AlphaMode, Material, MaterialTrait, MaterialType, MeshBasicMaterial, MeshPhongMaterial,
    MeshPhysicalMaterial, PhysicalFeatures, RenderableMaterialTrait, Side, TextureSlot,
    TextureTransform,
};
pub use mesh::Mesh;

pub use bloom::BloomSettings;
pub use buffer::BufferRef;
pub use fxaa::{FxaaQuality, FxaaSettings};
pub use geometry::{Attribute, BoundingBox, BoundingSphere, Geometry};
pub use image::{Image, ImageDescriptor};
pub use input::{ButtonState, Input, Key, MouseButton};
pub use shader_defines::ShaderDefines;
pub use texture::{Texture, TextureSampler};
pub use tone_mapping::{ToneMappingMode, ToneMappingSettings};
pub use uniforms::{Mat3Uniform, WgslType};
