//! Scene Graph System
//!
//! This module implements a hierarchical scene graph for organizing 3D objects,
//! cameras, and lights. It uses a component-based architecture with `SlotMap`
//! handles for safe, efficient entity references.
//!
//! # Architecture
//!
//! The scene system consists of:
//!
//! - [`Node`] - Core scene node with hierarchy and transform data
//! - [`Transform`] - Position, rotation, and scale component
//! - [`Scene`] - Container for all scene objects and components
//! - [`Camera`] - Perspective and orthographic camera component
//! - [`Light`] - Directional, point, and spot light components
//! - [`Environment`] - Skybox and IBL (Image-Based Lighting) settings
//!
//! # Data Layout
//!
//! The scene uses a hybrid ECS-style storage:
//!
//! - **Hot data** (hierarchy, transforms) stored directly in [`Node`]
//! - **Dense components** (names) use `SecondaryMap`
//! - **Sparse components** (cameras, lights, meshes) use `SparseSecondaryMap`
//!
//! This layout optimizes for common access patterns while minimizing memory overhead.
//!
//! # Example
//!
//! ```rust,ignore
//! use myth_engine::scene::{Scene, Node, Camera, Light};
//!
//! let mut scene = Scene::new();
//!
//! // Create a camera
//! let camera_node = scene.create_node_with_name("MainCamera");
//! scene.set_camera(camera_node, Camera::new_perspective(60.0, 16.0/9.0, 0.1));
//! scene.active_camera = Some(camera_node);
//!
//! // Create a light
//! let light_node = scene.create_node_with_name("Sun");
//! scene.set_light(light_node, Light::new_directional(Vec3::ONE, 1.0));
//! ```

pub mod camera;
pub mod environment;
pub mod light;
pub mod manager;
pub mod node;
pub mod scene;
pub mod skeleton;
pub mod transform;
pub mod transform_system;

// Re-export common types
pub use camera::{Camera, ProjectionType};
pub use light::{Light, LightKind};
pub use node::Node;
pub use scene::{Scene, SceneLogic};
pub use transform::Transform;
pub use transform_system::{
    LevelOrderBatches, build_level_order_batches, update_hierarchy_batched,
};

use slotmap::new_key_type;

new_key_type! {
    /// Strongly-typed handle for scene nodes.
    ///
    /// This is a SlotMap key that provides:
    /// - Generation tracking for safe handle reuse
    /// - Lightweight (8 bytes) for efficient storage
    /// - Copy semantics for easy sharing
    pub struct NodeHandle;

    /// Strongly-typed handle for skeleton resources.
    ///
    /// Skeletons are shared resources that can be referenced
    /// by multiple skinned mesh instances.
    pub struct SkeletonKey;
}
