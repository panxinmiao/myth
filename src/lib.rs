//! # Myth - A High-Performance, WGPU-Based Rendering Engine for Rust.
//!
//! Myth-Engine is a modern 3D rendering engine built with Rust and wgpu.
//! It provides a flexible, high-performance foundation for real-time graphics applications.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use myth::prelude::*;

//! struct MyApp;

//! impl AppHandler for MyApp {
//!    fn init(engine: &mut Engine, _: &dyn Window) -> Self {
//!        // 0. Create a Scene
//!        let scene = engine.scene_manager.create_active();
//!
//!        // 1. Create a cube mesh with a checkerboard texture
//!        let tex_handle = engine.assets.checkerboard(512, 64);
//!        let mesh_handle = scene.spawn_box(
//!            1.0, 1.0, 1.0,
//!            PhongMaterial::new(Vec4::new(1.0, 0.76, 0.33, 1.0)).with_map(tex_handle),
//!            &engine.assets,
//!        );
//!        // 2. Setup Camera
//!        let cam_node_id = scene.add_camera(Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1));
//!        scene.node(&cam_node_id).set_position(0.0, 0.0, 5.0).look_at(Vec3::ZERO);
//!        scene.active_camera = Some(cam_node_id);
//!        // 3. Add Light
//!        scene.add_light(Light::new_directional(Vec3::ONE, 5.0));
//!
//!        // 4. Setup update callback to rotate the cube
//!        scene.on_update(move |scene, _input, _dt| {
//!            if let Some(node) = scene.get_node_mut(mesh_handle) {
//!                let rot_y = Quat::from_rotation_y(0.02);
//!                let rot_x = Quat::from_rotation_x(0.01);
//!                node.transform.rotation = node.transform.rotation * rot_y * rot_x;
//!            }
//!        });
//!        Self {}
//!    }
//! }

//! fn main() -> myth::Result<()> {
//!     App::new().with_title("Myth-Engine Demo").run::<MyApp>()
//! }
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `winit` | **yes** | Window management via winit |
//! | `gltf` | **yes** | glTF 2.0 model loading |
//! | `http` | **yes** | HTTP asset loading |
//! | `gltf-meshopt` | no | Meshopt decompression for glTF |
//! | `debug_view` | no | Render graph debug view targets |
//! | `rdg_inspector` | no | Render graph inspector |
//!

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
    pub use myth_render::settings::{RenderPath, RendererInitConfig, RendererSettings};

    /// Low-level GPU context access.
    pub mod core {
        pub use myth_render::core::ResourceManager;
        pub use myth_render::core::WgpuContext;
        pub use myth_render::core::{BindingResource, Bindings, ResourceBuilder};
        pub use myth_render::core::{ReadbackError, ReadbackFrame, ReadbackStream};
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
        BackgroundMapping, BackgroundMode, BackgroundSettings, Camera, DayNightCycle, Light,
        LightKind, Node, ProceduralSkyParams, Scene, SceneLogic, SceneNode,
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
    pub use myth_assets::{
        AssetServer, GeometryHandle, ImageHandle, MaterialHandle, PrefabHandle, TextureHandle,
    };

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
    pub use myth_render::graph::FrameComposer;
    pub use myth_render::settings::{
        AntiAliasingMode, RenderPath, RendererInitConfig, RendererSettings,
    };
    #[cfg(feature = "debug_view")]
    pub use myth_scene::{DebugViewMode, DebugViewSettings};
}

// ============================================================================
// Top-level re-exports for convenience
// ============================================================================

// Application
#[cfg(feature = "winit")]
pub use myth_app::winit::App;
pub use myth_app::{AppHandler, Window};
pub use myth_app::{Engine, FrameState};
pub use myth_macros::main;

// Scene
pub use myth_core::{NodeHandle, Transform};
pub use myth_scene::{
    BackgroundMapping, BackgroundMode, BackgroundSettings, Camera, DayNightCycle, Light, Node,
    ProceduralSkyParams, Scene,
};

// Resources
pub use myth_resources::primitives::{
    PlaneOptions, SphereOptions, create_box, create_plane, create_sphere,
};
pub use myth_resources::{
    AgxLook, AlphaMode, AntiAliasingMode, Attribute, FxaaQuality, FxaaSettings, Geometry, Image,
    IndexFormat, Material, MaterialTrait, MaterialType, Mesh, PhongMaterial, PhysicalMaterial,
    RenderableMaterialTrait, ShaderDefines, Side, TaaSettings, Texture, TextureSlot,
    TextureTransform, ToneMappingMode, ToneMappingSettings, UnlitMaterial, VertexFormat,
};

// Assets
pub use myth_assets::{AssetServer, GeometryHandle, ImageHandle, MaterialHandle, TextureHandle};
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
pub use myth_render::settings::{RenderPath, RendererInitConfig, RendererSettings};

// Errors
pub use myth_core::{AssetError, Error, PlatformError, RenderError, Result};

// Utilities
pub use myth_app::OrbitControls;
pub use myth_core::utils::interner;
