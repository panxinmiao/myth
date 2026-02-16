//! Chainable node operation wrapper.
//!
//! [`SceneNode`] borrows a [`Scene`] mutably and provides a fluent API
//! for transforming nodes without needing `get_node_mut().unwrap()`.
//!
//! All methods silently no-op when the handle is stale, so users never
//! encounter panics from dangling handles.
//!
//! # Example
//!
//! ```rust,ignore
//! scene.node(&handle)
//!     .set_position(0.0, 3.0, 0.0)
//!     .set_scale(2.0)
//!     .look_at(Vec3::ZERO)
//!     .set_visible(false);
//! ```
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::must_use_candidate)]
use glam::{Quat, Vec3};

use crate::scene::NodeHandle;
use crate::scene::scene::Scene;

/// Temporary mutable borrow of a scene node for chainable operations.
pub struct SceneNode<'a> {
    scene: &'a mut Scene,
    handle: NodeHandle,
}

impl<'a> SceneNode<'a> {
    #[inline]
    pub fn new(scene: &'a mut Scene, handle: NodeHandle) -> Self {
        Self { scene, handle }
    }

    /// Returns the underlying handle.
    #[inline]
    #[must_use]
    pub fn handle(&self) -> NodeHandle {
        self.handle
    }

    // -- Transform setters (chainable) --

    /// Sets the node's local position.
    #[inline]
    pub fn set_position(self, x: f32, y: f32, z: f32) -> Self {
        if let Some(node) = self.scene.get_node_mut(self.handle) {
            node.transform.position = Vec3::new(x, y, z);
        }
        self
    }

    /// Sets the node's local position from a Vec3.
    #[inline]
    pub fn set_position_vec(self, pos: Vec3) -> Self {
        if let Some(node) = self.scene.get_node_mut(self.handle) {
            node.transform.position = pos;
        }
        self
    }

    /// Sets uniform scale.
    #[inline]
    pub fn set_scale(self, s: f32) -> Self {
        if let Some(node) = self.scene.get_node_mut(self.handle) {
            node.transform.scale = Vec3::splat(s);
        }
        self
    }

    /// Sets non-uniform scale.
    #[inline]
    pub fn set_scale_xyz(self, x: f32, y: f32, z: f32) -> Self {
        if let Some(node) = self.scene.get_node_mut(self.handle) {
            node.transform.scale = Vec3::new(x, y, z);
        }
        self
    }

    /// Sets rotation from a quaternion.
    #[inline]
    pub fn set_rotation(self, quat: Quat) -> Self {
        if let Some(node) = self.scene.get_node_mut(self.handle) {
            node.transform.rotation = quat;
        }
        self
    }

    /// Sets rotation from Euler angles (XYZ intrinsic order, radians).
    #[inline]
    pub fn set_rotation_euler(self, x: f32, y: f32, z: f32) -> Self {
        if let Some(node) = self.scene.get_node_mut(self.handle) {
            node.transform.set_rotation_euler(x, y, z);
        }
        self
    }

    /// Rotates around the Y axis by `angle` radians (cumulative).
    #[inline]
    pub fn rotate_y(self, angle: f32) -> Self {
        if let Some(node) = self.scene.get_node_mut(self.handle) {
            node.transform.rotation *= Quat::from_rotation_y(angle);
        }
        self
    }

    /// Rotates around the X axis by `angle` radians (cumulative).
    #[inline]
    pub fn rotate_x(self, angle: f32) -> Self {
        if let Some(node) = self.scene.get_node_mut(self.handle) {
            node.transform.rotation *= Quat::from_rotation_x(angle);
        }
        self
    }

    /// Orients the node to face `target` (in parent-local space).
    #[inline]
    pub fn look_at(self, target: Vec3) -> Self {
        if let Some(node) = self.scene.get_node_mut(self.handle) {
            node.transform.look_at(target, Vec3::Y);
        }
        self
    }

    /// Sets node visibility.
    #[inline]
    pub fn set_visible(self, visible: bool) -> Self {
        if let Some(node) = self.scene.get_node_mut(self.handle) {
            node.visible = visible;
        }
        self
    }

    /// Sets the mesh cast_shadows flag (no-op if node has no mesh).
    #[inline]
    pub fn set_cast_shadows(self, cast: bool) -> Self {
        if let Some(mesh) = self.scene.get_mesh_mut(self.handle) {
            mesh.cast_shadows = cast;
        }
        self
    }

    /// Sets the mesh receive_shadows flag (no-op if node has no mesh).
    #[inline]
    pub fn set_receive_shadows(self, receive: bool) -> Self {
        if let Some(mesh) = self.scene.get_mesh_mut(self.handle) {
            mesh.receive_shadows = receive;
        }
        self
    }

    /// Convenience: enable/disable both cast and receive shadows.
    #[inline]
    pub fn set_shadows(self, cast: bool, receive: bool) -> Self {
        self.set_cast_shadows(cast).set_receive_shadows(receive)
    }
}
