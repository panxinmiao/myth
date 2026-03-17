//! # Myth Scene
//!
//! Hierarchical scene graph for the Myth engine.
//!
//! Provides [`Scene`] — the core data container that stores nodes,
//! components (meshes, cameras, lights, skins, morph weights, animations),
//! environment/post-processing settings, and GPU synchronisation buffers.

pub mod background;
pub mod camera;
pub mod environment;
pub mod light;
pub mod node;
pub mod scene;
pub mod skeleton;
pub mod transform_system;
pub mod wrapper;

// Re-exports from dependencies
pub use myth_core::{NodeHandle, SkeletonKey, Transform};

// Re-exports from this crate
pub use background::{BackgroundMapping, BackgroundMode, BackgroundSettings};
pub use camera::{Camera, ProjectionType, Frustum, RenderCamera};
pub use environment::Environment;
pub use light::{DirectionalLight, Light, LightKind, PointLight, ShadowConfig, SpotLight};
pub use node::Node;
pub use scene::{CallbackLogic, NodeBuilder, Scene, SceneLogic, SplitPrimitiveTag};
pub use skeleton::{BindMode, Skeleton, SkinBinding};
pub use wrapper::SceneNode;

