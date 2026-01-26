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

pub use animation::{AnimationAction, AnimationClip, AnimationMixer, Binder, LoopMode};
pub use assets::{AssetServer, ColorSpace};
pub use engine::ThreeEngine;
pub use errors::ThreeError;
pub use renderer::core::WgpuContext;
pub use renderer::Renderer;
pub use resources::primitives::*;
pub use resources::{
    Geometry, Image, Material, MaterialTrait, MaterialType, Mesh, RenderableMaterialTrait,
    ShaderDefines, Side, Texture,
};
pub use scene::{Camera, Light, Node, Scene};
pub use utils::interner;
pub use utils::orbit_control::OrbitControls;
