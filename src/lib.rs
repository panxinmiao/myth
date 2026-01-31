//! # Three - A High-Performance WebGPU Rendering Engine
//!
//! Three is a modern 3D rendering engine built with Rust and wgpu, inspired by Three.js.
//! It provides a flexible, high-performance foundation for real-time graphics applications.
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
//! │              (ThreeEngine, SceneManager)                   │
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
//! ## Key Modules
//!
//! - [`app`] - Application lifecycle and window management (Winit integration)
//! - [`engine`] - Core engine instance that orchestrates all subsystems
//! - [`scene`] - Scene graph system with nodes, cameras, lights, and transforms
//! - [`renderer`] - Rendering pipeline with Core, Graph, and Pipeline layers
//! - [`resources`] - CPU-side resource definitions (Geometry, Material, Texture, etc.)
//! - [`animation`] - Skeletal and morph target animation system
//! - [`assets`] - Asset loading and management (glTF, textures, etc.)
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use three::{App, AppHandler, ThreeEngine, Scene, Camera, Mesh};
//! use three::app::winit::AppHandler;
//!
//! struct MyApp;
//!
//! impl AppHandler for MyApp {
//!     fn init(engine: &mut ThreeEngine, window: &Arc<Window>) -> Self {
//!         // Create a scene with a camera and some objects
//!         let scene = Scene::new();
//!         // ... setup scene ...
//!         MyApp
//!     }
//!
//!     fn update(&mut self, engine: &mut ThreeEngine, window: &Arc<Window>, frame: &FrameState) {
//!         // Update logic here
//!     }
//! }
//!
//! fn main() -> anyhow::Result<()> {
//!     App::new()
//!         .with_title("My 3D App")
//!         .run::<MyApp>()
//! }
//! ```
//!
//! ## Feature Flags
//!
//! - `winit` (default) - Enables window management via winit
//! - `gltf` (default) - Enables glTF model loading
//! - `http` (default) - Enables HTTP asset loading
//!
//! ## Design Principles
//!
//! - **Performance First**: Cache-friendly data layouts, GPU resource pooling, and efficient batching
//! - **Modular Design**: Each subsystem is independent and can be used separately
//! - **Type Safety**: Strong typing with SlotMap handles for resource references
//! - **Modern Graphics**: Built on wgpu for cross-platform WebGPU/Vulkan/Metal/DX12 support

#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::too_many_arguments)]

pub mod animation;
pub mod app;
pub mod assets;
pub mod engine;
pub mod errors;
pub mod renderer;
pub mod resources;
pub mod scene;
pub mod utils;

pub use animation::{AnimationAction, AnimationClip, AnimationMixer, AnimationSystem, Binder, LoopMode};
pub use assets::{AssetServer, ColorSpace};
pub use engine::ThreeEngine;
pub use errors::ThreeError;
pub use renderer::core::WgpuContext;
pub use renderer::graph::{FrameBuilder, FrameComposer, RenderStage};
pub use renderer::Renderer;
pub use resources::primitives::*;
pub use resources::{
    Geometry, Image, Material, MaterialTrait, MaterialType, Mesh, RenderableMaterialTrait,
    ShaderDefines, Side, Texture,
};
pub use resources::material::*;
pub use scene::{Camera, Light, Node, Scene};
pub use utils::interner;
pub use utils::orbit_control::OrbitControls;
