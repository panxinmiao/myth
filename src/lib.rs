#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::too_many_arguments)]

pub mod resources;
pub mod assets;
pub mod scene;
pub mod renderer;
pub mod app;
pub mod errors;
pub mod utils;
pub mod animation;

pub use resources::{Mesh, Material, MaterialTrait, RenderableMaterialTrait, MaterialType, Texture, Image, Geometry, Side, ShaderDefines};
pub use assets::{AssetServer, ColorSpace};
pub use scene::{Node, Scene, Camera, Light};
pub use resources::primitives::*;
pub use renderer::Renderer;
pub use renderer::core::WgpuContext;
pub use app::App;
pub use errors::ThreeError;
pub use utils::orbit_control::OrbitControls;
pub use utils::interner;
pub use animation::{AnimationClip, AnimationAction, AnimationMixer, Binder, LoopMode};
