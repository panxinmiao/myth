//! # Myth - A High-Performance WebGPU Rendering Engine
//!
//! Myth-Engine is a modern 3D rendering engine built with Rust and wgpu, inspired by Three.js.
//! It provides a flexible, high-performance foundation for real-time graphics applications.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use myth::prelude::*;
//!
//! struct MyApp;
//!
//! impl AppHandler for MyApp {
//!     fn init(engine: &mut Engine, window: &dyn Window) -> Self {
//!         let scene = engine.scene_manager.create_active();
//!         let geometry = Geometry::new_box(1.0, 1.0, 1.0);
//!         let material = Material::new_unlit(Vec4::new(1.0, 0.5, 0.2, 1.0));
//!         let mesh = Mesh::new(
//!             engine.assets.geometries.add(geometry),
//!             engine.assets.materials.add(material),
//!         );
//!         scene.add_mesh(mesh);
//!         let camera = Camera::new_perspective(60.0, 16.0/9.0, 0.1);
//!         let cam_node = scene.add_camera(camera);
//!         scene.active_camera = Some(cam_node);
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
//! ## Feature Flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `winit` | **yes** | Window management via winit |
//! | `gltf` | **yes** | glTF 2.0 model loading |
//! | `http` | **yes** | HTTP asset loading (native only) |
//! | `gltf-meshopt` | no | Meshopt decompression for glTF |

// ============================================================================
// Sub-crate re-exports (facade modules matching the old monolith paths)
// ============================================================================

/// Error types and `Result` alias.
pub mod errors {
    pub use myth_core::errors::*;
}

/// Scene graph – nodes, cameras, lights, transforms.
pub use myth_scene as scene;

/// GPU resource definitions – geometry, material, texture, mesh, etc.
pub use myth_resources as resources;

/// Animation system – clips, mixers, tracks, skeletal / morph-target.
pub use myth_animation as animation;

/// Asset loading – server, storage, glTF loaders.
pub use myth_assets as assets;

/// Renderer internals – core, graph, pipeline.
pub use myth_render as renderer;

/// Application framework – engine, handlers, windowing.
#[cfg(feature = "winit")]
pub use myth_app as app;

/// Engine core without windowing (always available even without `winit`).
pub mod engine {
    pub use myth_app::engine::*;
}

// ============================================================================
// Local utilities (re-exports from sub-crates)
// ============================================================================

pub mod utils {
    pub use myth_app::OrbitControls;
    pub use myth_core::utils::FpsCounter;
    pub mod fps_counter {
        pub use myth_core::utils::FpsCounter;
    }
}

// ============================================================================
// Math module – re-exported glam types
// ============================================================================

pub mod math {
    pub use glam::*;
}

// ============================================================================
// Render module – high-level rendering API alias
// ============================================================================

pub mod render {
    pub use myth_render::graph::{FrameComposer, RenderState};
    pub use myth_render::renderer::Renderer;
    pub use myth_render::settings::{RenderPath, RendererSettings};

    #[doc(hidden)]
    #[deprecated(since = "0.2.0", note = "Renamed to `RendererSettings`")]
    pub type RenderSettings = RendererSettings;

    /// Low-level GPU context access.
    pub mod core {
        pub use myth_render::core::ResourceManager;
        pub use myth_render::core::WgpuContext;
        pub use myth_render::core::{BindingResource, Bindings, ResourceBuilder};
    }
}

// ============================================================================
// Prelude – common imports for everyday use
// ============================================================================

pub mod prelude {
    // Application
    #[cfg(feature = "winit")]
    pub use myth_app::winit::App;
    pub use myth_app::{AppHandler, Window};
    pub use myth_app::{Engine, FrameState};

    // Scene graph
    pub use myth_core::{NodeHandle, SkeletonKey, Transform};
    pub use myth_scene::camera::ProjectionType;
    pub use myth_scene::{
        BackgroundMapping, BackgroundMode, BackgroundSettings, Camera, Light, LightKind, Node,
        Scene, SceneLogic, SceneNode,
    };

    // Resources
    pub use myth_resources::{
        AlphaMode, BloomSettings, FxaaQuality, FxaaSettings, Geometry, Image, Material,
        MaterialType, Mesh, PhongMaterial, PhysicalMaterial, Side, SsaoSettings, TaaSettings,
        Texture, TextureSlot, UnlitMaterial,
    };

    // Assets
    pub use myth_assets::ColorSpace;
    pub use myth_assets::SceneExt;
    #[cfg(feature = "gltf")]
    pub use myth_assets::loaders::gltf::GltfLoader;
    pub use myth_assets::{AssetServer, GeometryHandle, MaterialHandle, TextureHandle};

    // Animation
    pub use myth_animation::{
        AnimationAction, AnimationClip, AnimationEvent, AnimationMixer, ClipBinding, FiredEvent,
        LoopMode, Rig,
    };

    // Math
    pub use glam::{Affine3A, EulerRot, Mat3, Mat4, Quat, Vec2, Vec3, Vec4};

    // Utilities
    pub use myth_app::OrbitControls;

    // Renderer
    #[cfg(feature = "debug_view")]
    pub use myth_render::graph::DebugViewTarget;
    pub use myth_render::graph::FrameComposer;
    pub use myth_render::settings::{AntiAliasingMode, RenderPath, RendererSettings};

    #[doc(hidden)]
    #[deprecated(since = "0.2.0", note = "Renamed to `RendererSettings`")]
    pub type RenderSettings = RendererSettings;
}

// ============================================================================
// Top-level re-exports for convenience
// ============================================================================

// Application
#[cfg(feature = "winit")]
pub use myth_app::winit::App;
pub use myth_app::{AppHandler, Window};
pub use myth_app::{Engine, FrameState};

// Scene
pub use myth_core::{NodeHandle, Transform};
pub use myth_scene::{
    BackgroundMapping, BackgroundMode, BackgroundSettings, Camera, Light, Node, Scene,
};

// Resources
pub use myth_resources::primitives::{
    PlaneOptions, SphereOptions, create_box, create_plane, create_sphere,
};
pub use myth_resources::{
    AlphaMode, Attribute, FxaaQuality, FxaaSettings, Geometry, Image, Material, MaterialTrait,
    MaterialType, Mesh, PhongMaterial, PhysicalMaterial, RenderableMaterialTrait, ShaderDefines,
    Side, TaaSettings, Texture, TextureSlot, TextureTransform, ToneMappingMode,
    ToneMappingSettings, UnlitMaterial,
};

// Assets
pub use myth_assets::{AssetServer, GeometryHandle, MaterialHandle, TextureHandle};
pub use myth_assets::{ColorSpace, GeometryQuery, ResolveGeometry, ResolveMaterial, SceneExt};

// Animation
pub use myth_animation::{
    AnimationAction, AnimationClip, AnimationEvent, AnimationMixer, AnimationSystem, Binder,
    ClipBinding, FiredEvent, InterpolationMode, LoopMode, Rig, Track, TrackBinding, TrackData,
    TrackMeta,
};

// Renderer
pub use myth_render::Renderer;
pub use myth_render::graph::FrameComposer;
pub use myth_render::settings::{RenderPath, RendererSettings};

#[doc(hidden)]
#[deprecated(since = "0.2.0", note = "Renamed to `RendererSettings`")]
pub type RenderSettings = RendererSettings;

// Errors
pub use myth_core::{AssetError, Error, PlatformError, RenderError, Result};

// Utilities
pub use myth_app::OrbitControls;
pub use myth_core::utils::interner;
