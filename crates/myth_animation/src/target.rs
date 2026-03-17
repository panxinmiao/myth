use glam::{Quat, Vec3};
use myth_core::{NodeHandle, Transform};

/// Trait abstracting scene operations needed by the animation system.
///
/// This allows the animation crate to manipulate scene nodes without
/// depending on the scene crate directly, breaking the circular dependency.
pub trait AnimationTarget {
    /// Get the children of a node.
    fn node_children(&self, handle: NodeHandle) -> Option<Vec<NodeHandle>>;

    /// Get a node's name.
    fn node_name(&self, handle: NodeHandle) -> Option<String>;

    /// Get a node's transform.
    fn node_transform(&self, handle: NodeHandle) -> Option<Transform>;

    /// Get a mutable reference and apply a transform mutation.
    fn set_node_position(&mut self, handle: NodeHandle, position: Vec3);

    /// Set a node's rotation.
    fn set_node_rotation(&mut self, handle: NodeHandle, rotation: Quat);

    /// Set a node's scale.
    fn set_node_scale(&mut self, handle: NodeHandle, scale: Vec3);

    /// Mark a node's transform as dirty (needs hierarchy update).
    fn mark_node_dirty(&mut self, handle: NodeHandle);

    /// Check if a rest transform exists for a node.
    fn has_rest_transform(&self, handle: NodeHandle) -> bool;

    /// Get the rest transform for a node.
    fn rest_transform(&self, handle: NodeHandle) -> Option<Transform>;

    /// Store a rest transform for a node.
    fn store_rest_transform(&mut self, handle: NodeHandle, transform: Transform);

    /// Get or create a mutable morph weights vector for a node.
    fn morph_weights_mut(&mut self, handle: NodeHandle) -> &mut Vec<f32>;
}
