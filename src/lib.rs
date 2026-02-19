//! # Myth - A High-Performance WebGPU Rendering Engine
//!
//! Myth-Engine is a modern 3D rendering engine built with Rust and wgpu, inspired by Three.js.
//! It provides a flexible, high-performance foundation for real-time graphics applications.
//!
//! ## Quick Start
//!
//! For the simplest way to get started, use the [`prelude`] module:
//!
//! ```rust,ignore
//! use myth::prelude::*;
//!
//! struct MyApp;
//!
//! impl AppHandler for MyApp {
//!     fn init(engine: &mut Engine, window: &dyn Window) -> Self {
//!         // Create a scene with a mesh
//!         let scene = engine.scene_manager.create_active();
//!         
//!         // Add a cube
//!         let geometry = Geometry::new_box(1.0, 1.0, 1.0);
//!         let material = Material::new_basic(Vec4::new(1.0, 0.5, 0.2, 1.0));
//!         let mesh = Mesh::new(
//!             engine.assets.geometries.add(geometry),
//!             engine.assets.materials.add(material),
//!         );
//!         scene.add_mesh(mesh);
//!         
//!         // Setup camera
//!         let camera = Camera::new_perspective(60.0, 16.0/9.0, 0.1);
//!         let cam_node = scene.add_camera(camera);
//!         scene.active_camera = Some(cam_node);
//!         
//!         MyApp
//!     }
//!     
//!     fn update(&mut self, engine: &mut Engine, _: &dyn Window, frame: &FrameState) {
//!         // Update logic here
//!     }
//! }
//!
//! fn main() -> myth::errors::Result<()> {
//!     App::new().with_title("My 3D App").run::<MyApp>()
//! }
//! ```
//!
//! ## Architecture Overview
//!
//! The engine follows a layered architecture designed for modularity and performance:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                         App Layer                          │
//! │               (Winit integration, event loop)              │
//! ├─────────────────────────────────────────────────────────────┤
//! │                       Engine Layer                         │
//! │              (Engine, SceneManager)                   │
//! ├─────────────────────────────────────────────────────────────┤
//! │   Scene Graph   │    Renderer     │      Animation        │
//! │  (Node, Camera, │  (Core, Graph,  │  (Mixer, Clip, Track) │
//! │   Light, etc.)  │    Pipeline)    │                       │
//! ├─────────────────────────────────────────────────────────────┤
//! │                      Resources Layer                       │
//! │         (Geometry, Material, Texture, Mesh, etc.)          │
//! ├─────────────────────────────────────────────────────────────┤
//! │                       wgpu / WebGPU                        │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Module Organization
//!
//! This crate uses a **progressive disclosure** pattern:
//!
//! - **High-level API**: Use [`prelude`] for common types and traits
//! - **Domain modules**: Access organized APIs via [`scene`], [`resources`], [`assets`], etc.
//! - **Low-level access**: Advanced users can access [`renderer`] internals
//!
//! ### Core Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`app`] | Application lifecycle and window management |
//! | [`engine`] | Core engine instance ([`Engine`]) |
//! | [`scene`] | Scene graph with nodes, cameras, and lights |
//! | [`resources`] | Resource definitions (geometry, material, texture) |
//! | [`assets`] | Asset loading and management (glTF, images) |
//! | [`animation`] | Skeletal and morph target animation system |
//! | [`math`] | Re-exported math types from `glam` |
//! | [`render`] | Rendering configuration and advanced APIs |
//!
//! ## Feature Flags
//!
//! - `winit` (default) - Window management via winit
//! - `gltf` (default) - glTF 2.0 model loading
//! - `http` (default) - HTTP asset loading (native only)
//!
//! ## Design Principles
//!
//! - **Performance First**: Cache-friendly data layouts, GPU resource pooling
//! - **Progressive Disclosure**: Simple API for beginners, full control for experts
//! - **Type Safety**: `SlotMap` handles for safe resource references
//! - **Cross-Platform**: WebGPU/Vulkan/Metal/DX12 via wgpu

// ============================================================================
// Internal Module Declarations
// ============================================================================

pub mod animation;
pub mod app;
pub mod assets;
pub mod engine;
pub mod errors;
pub mod renderer;
pub mod resources;
pub mod scene;
pub mod utils;

// ============================================================================
// Prelude - Common imports for everyday use
// ============================================================================

/// Prelude module for convenient imports.
///
/// Import everything you need for basic usage with a single line:
///
/// ```rust,ignore
/// use myth::prelude::*;
/// ```
///
/// This includes:
/// - Application types: [`App`], [`AppHandler`], [`Engine`], [`FrameState`]
/// - Scene types: [`Scene`], [`Node`], [`NodeHandle`], [`Camera`], [`Light`]
/// - Resource types: [`Mesh`], [`Geometry`], [`Material`], [`Texture`]
/// - Common math types from `glam`
/// - Asset loading: [`AssetServer`]
pub mod prelude {
    // Application
    #[cfg(feature = "winit")]
    pub use crate::app::winit::App;
    pub use crate::app::{AppHandler, Window};
    pub use crate::engine::{Engine, FrameState};

    // Scene graph
    pub use crate::scene::{
        BackgroundMapping, BackgroundMode, Camera, Light, LightKind, Node, NodeHandle,
        ProjectionType, ResolveGeometry, ResolveMaterial, Scene, SceneLogic, SceneNode,
        SkeletonKey, Transform,
    };

    // Resources
    pub use crate::resources::{
        AlphaMode, BloomSettings, Geometry, Image, Material, MaterialType, Mesh, MeshBasicMaterial,
        MeshPhongMaterial, MeshPhysicalMaterial, Side, Texture, TextureSlot,
    };

    // Assets
    pub use crate::assets::GltfLoader;
    pub use crate::assets::{
        AssetServer, ColorSpace, GeometryHandle, MaterialHandle, TextureHandle,
    };

    // Animation
    pub use crate::animation::{AnimationAction, AnimationClip, AnimationMixer, LoopMode};

    // Math (re-export common glam types)
    pub use glam::{Affine3A, EulerRot, Mat3, Mat4, Quat, Vec2, Vec3, Vec4};

    // Utilities
    pub use crate::utils::orbit_control::OrbitControls;

    // Renderer (limited exposure)
    pub use crate::renderer::graph::{FrameComposer, RenderStage};
    pub use crate::renderer::settings::RenderSettings;
}

// ============================================================================
// Math Module - Re-exported glam types
// ============================================================================

/// Math types re-exported from the `glam` crate.
///
/// Using this module ensures version compatibility between your code
/// and the engine's internal math operations.
///
/// # Example
///
/// ```rust,ignore
/// use myth::math::{Vec3, Quat, Mat4};
///
/// let position = Vec3::new(1.0, 2.0, 3.0);
/// let rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
/// ```
///
/// # Available Types
///
/// ## Vectors
/// - [`Vec2`], [`Vec3`], [`Vec4`] - Float vectors
/// - [`IVec2`], [`IVec3`], [`IVec4`] - Integer vectors
/// - [`UVec2`], [`UVec3`], [`UVec4`] - Unsigned integer vectors
///
/// ## Matrices
/// - [`Mat2`], [`Mat3`], [`Mat4`] - Square matrices
/// - [`Affine2`], [`Affine3A`] - Affine transformation matrices
///
/// ## Quaternions
/// - [`Quat`] - Unit quaternion for rotations
///
/// ## Other
/// - [`EulerRot`] - Euler rotation order enumeration
pub mod math {
    pub use glam::*;
}

// ============================================================================
// Render Module - Advanced rendering API (alias for renderer)
// ============================================================================

/// Rendering system configuration and advanced APIs.
///
/// Most users only need [`RenderSettings`](renderer::settings::RenderSettings)
/// to configure basic options. Advanced users can access the render graph
/// system for custom passes.
///
/// # Basic Usage
///
/// ```rust,ignore
/// use myth::render::RenderSettings;
///
/// let settings = RenderSettings {
///     vsync: true,
///     clear_color: wgpu::Color::BLACK,
///     ..Default::default()
/// };
///
/// App::new()
///     .with_settings(settings)
///     .run::<MyApp>()?;
/// ```
///
/// # Advanced: Custom Render Passes
///
/// Implement [`RenderNode`](renderer::graph::RenderNode) to add custom
/// rendering passes:
///
/// ```rust,ignore
/// use myth::render::{FrameComposer, RenderStage, RenderNode};
///
/// impl AppHandler for MyApp {
///     fn compose_frame<'a>(&'a self, composer: FrameComposer<'a>) {
///         composer
///             .add_node(RenderStage::UI, &self.ui_pass)
///             .render();
///     }
/// }
/// ```
pub mod render {
    pub use crate::renderer::Renderer;
    pub use crate::renderer::graph::{
        FrameBuilder, FrameComposer, RenderContext, RenderNode, RenderStage, RenderState,
        TrackedRenderPass,
    };
    pub use crate::renderer::settings::RenderSettings;

    /// Low-level GPU context access.
    ///
    /// These types are for advanced users who need direct GPU access.
    /// Most applications should not need to use these directly.
    pub mod core {
        pub use crate::renderer::core::ResourceManager;
        pub use crate::renderer::core::WgpuContext;
        // Advanced: GPU binding system
        pub use crate::renderer::core::{BindingResource, Bindings, ResourceBuilder};
    }
}

// ============================================================================
// Top-Level Re-exports for Convenience
// ============================================================================

// Application
#[cfg(feature = "winit")]
pub use app::winit::App;
pub use app::{AppHandler, Window};
pub use engine::{Engine, FrameState};

// Scene (most common types)
pub use scene::{
    BackgroundMapping, BackgroundMode, Camera, Light, Node, NodeHandle, Scene, Transform,
};

// Resources (most common types)
pub use resources::{
    AlphaMode,
    Attribute,
    Geometry,
    Image,
    Material,
    // Advanced: Material trait for custom materials
    MaterialTrait,
    MaterialType,
    Mesh,
    MeshBasicMaterial,
    MeshPhongMaterial,
    MeshPhysicalMaterial,
    RenderableMaterialTrait,
    // Geometry primitives
    ShaderDefines,
    Side,
    Texture,
    TextureSlot,
    TextureTransform,
    // Tone mapping
    ToneMappingMode,
    ToneMappingSettings,
};

// Primitives - Geometry creation functions
pub use resources::primitives::{
    PlaneOptions, SphereOptions, create_box, create_plane, create_sphere,
};

// Assets
pub use assets::{AssetServer, ColorSpace, GeometryHandle, MaterialHandle, TextureHandle};

// Animation
pub use animation::{
    AnimationAction, AnimationClip, AnimationMixer, AnimationSystem, Binder, InterpolationMode,
    LoopMode, Track, TrackData, TrackMeta,
};

// Renderer
pub use renderer::Renderer;
pub use renderer::graph::{FrameBuilder, FrameComposer, RenderStage};
pub use renderer::settings::RenderSettings;

// Errors
pub use errors::{AssetError, Error, PlatformError, RenderError, Result};

// Utilities
pub use utils::interner;
pub use utils::orbit_control::OrbitControls;
