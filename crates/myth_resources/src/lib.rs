//! # Myth Resources
//!
//! CPU-side data structures for rendering resources. These types define
//! the logical representation of rendering data before GPU upload.
//!
//! # Module Overview
//!
//! - [`mesh`] - Mesh objects combining geometry and materials
//! - [`geometry`] - Vertex data and attributes
//! - [`material`] - Material definitions (Physical, Phong, Unlit)
//! - [`texture`] - Texture configuration and sampling parameters
//! - [`image`] - Raw image data storage
//! - [`buffer`] - CPU buffer with version tracking
//! - [`uniforms`] - Shader uniform data structures
//! - [`shader_defines`] - Dynamic shader macro system
//! - [`primitives`] - Built-in geometry primitives
//! - [`handles`] - Strongly-typed resource handles
//! - [`binding`] - GPU binding resource descriptions

pub mod anti_aliasing;
pub mod binding;
pub mod bloom;
pub mod buffer;
pub mod builder;
pub mod fxaa;
pub mod geometry;
pub mod handles;
pub mod image;
pub mod input;
pub mod material;
pub mod mesh;
pub mod primitives;
pub mod screen_space;
pub mod shader_defines;
pub mod ssao;
pub mod taa;
pub mod texture;
pub mod tone_mapping;
pub mod uniforms;
pub mod version_tracker;

// Re-export handle types
pub use handles::{
    GeometryHandle, GpuBufferHandle, ImageHandle, MaterialHandle, PrefabHandle, TextureHandle,
};

// Re-export common resource types
pub use material::{
    AlphaMode, Material, MaterialTrait, MaterialType, PhongMaterial, PhysicalFeatures,
    PhysicalMaterial, RenderableMaterialTrait, Side, TextureSlot, TextureTransform, UnlitMaterial,
};
pub use mesh::Mesh;

// Re-export the material definition macro
pub use myth_macros::myth_material;

// Re-export the GPU struct macro
pub use myth_macros::gpu_struct;

pub use anti_aliasing::AntiAliasingMode;
pub use bloom::BloomSettings;
pub use buffer::BufferRef;
pub use fxaa::{FxaaQuality, FxaaSettings};
pub use geometry::{
    Attribute, BoundingBox, BoundingSphere, Geometry, IndexAttribute, IndexFormat, VertexFormat,
};
pub use image::Image;
pub use image::{ColorSpace, ImageDimension, PixelFormat};
pub use input::{ButtonState, Input, Key, MouseButton};
pub use shader_defines::ShaderDefines;
pub use ssao::SsaoSettings;
pub use taa::TaaSettings;
pub use texture::{Texture, TextureSampler};
pub use tone_mapping::{AgxLook, ToneMappingMode, ToneMappingSettings};
pub use uniforms::{Mat3Uniform, WgslType};

// Re-export binding/builder types for myth_render
pub use binding::BindingResource;
pub use builder::{Binding, BindingDesc, ResourceBuilder, WgslStructName};
