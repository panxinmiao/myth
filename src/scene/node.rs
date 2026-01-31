use glam::Affine3A;
use crate::scene::NodeHandle;
use crate::scene::transform::Transform;

/// A minimal scene node containing only essential hot data.
///
/// # Design Principles
///
/// - Only keeps data that must be traversed every frame (hierarchy and transform)
/// - Other attributes (Mesh, Camera, Light, Skin, etc.) are stored in Scene's component maps
/// - Improves CPU cache hit rate by keeping nodes small and contiguous
///
/// # Hierarchy
///
/// Nodes form a tree structure through parent-child relationships:
/// - `parent`: Optional handle to parent node (None for root nodes)
/// - `children`: List of child node handles
///
/// # Transform
///
/// Each node has a [`Transform`] component that manages:
/// - Local position, rotation, and scale
/// - Cached local and world matrices
/// - Dirty flag for efficient updates
#[derive(Debug, Clone)]
pub struct Node {
    // === Core Hierarchy ===
    /// Parent node handle (None for root nodes)
    pub parent: Option<NodeHandle>,
    /// Child node handles
    pub children: Vec<NodeHandle>,

    // === Core Spatial Data ===
    /// Transform component (hot data accessed every frame)
    pub transform: Transform,

    // === Core State ===
    /// Visibility flag for culling
    pub visible: bool,
}

impl Node {
    /// Creates a new node with default transform and visibility.
    pub fn new() -> Self {
        Self {
            parent: None,
            children: Vec::new(),
            transform: Transform::new(),
            visible: true,
        }
    }

    /// Returns a reference to the world transformation matrix.
    ///
    /// This matrix transforms local coordinates to world coordinates.
    /// It is automatically updated by the transform system each frame.
    #[inline]
    pub fn world_matrix(&self) -> &Affine3A {
        &self.transform.world_matrix
    }
}

impl Default for Node {
    fn default() -> Self {
        Self::new()
    }
}