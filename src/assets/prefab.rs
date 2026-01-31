use std::sync::Arc;

use glam::Affine3A;

use crate::animation::clip::AnimationClip;
use crate::resources::mesh::Mesh;
use crate::scene::transform::Transform;

/// Prefab node: contains only data, uses indices to reference child nodes
#[derive(Debug, Clone)]
pub struct PrefabNode {
    pub name: Option<String>,
    pub transform: Transform,
    /// Child node indices in Prefab.nodes
    pub children_indices: Vec<usize>,
    /// Mesh component (if any)
    pub mesh: Option<Mesh>,
    /// Referenced skeleton index (in Prefab.skeletons)
    pub skin_index: Option<usize>,
    /// Morph weights (if any)
    pub morph_weights: Option<Vec<f32>>,
}

impl PrefabNode {
    pub fn new() -> Self {
        Self {
            name: None,
            transform: Transform::new(),
            children_indices: Vec::new(),
            mesh: None,
            skin_index: None,
            morph_weights: None,
        }
    }
}

impl Default for PrefabNode {
    fn default() -> Self {
        Self::new()
    }
}

/// Prefab skeleton data
#[derive(Debug, Clone)]
pub struct PrefabSkeleton {
    pub name: String,
    /// Root bone index in bones
    pub root_bone_index: usize,
    /// Bone node indices (pointing to Prefab.nodes indices)
    pub bone_indices: Vec<usize>,
    /// Inverse bind matrices
    pub inverse_bind_matrices: Vec<Affine3A>,
}

/// Prefab: intermediate data structure parsed from resource files like glTF
/// 
/// Prefab is a thread-safe pure data structure that doesn't contain any NodeHandle or Scene references.
/// Use `Scene::instantiate()` method to instantiate a Prefab into scene nodes.
#[derive(Debug, Clone)]
pub struct Prefab {
    /// All nodes stored in a flat array
    pub nodes: Vec<PrefabNode>,
    /// Root node indices in nodes
    pub root_indices: Vec<usize>,
    /// Skeleton data
    pub skeletons: Vec<PrefabSkeleton>,
    /// Animation data
    pub animations: Vec<AnimationClip>,
}

impl Prefab {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root_indices: Vec::new(),
            skeletons: Vec::new(),
            animations: Vec::new(),
        }
    }
}

impl Default for Prefab {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe Prefab reference
pub type SharedPrefab = Arc<Prefab>;
