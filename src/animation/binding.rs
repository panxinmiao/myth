use crate::scene::NodeHandle;

/// Defines the target property for animation data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetPath {
    Translation, // Maps to transform.position
    Rotation,    // Maps to transform.rotation
    Scale,       // Maps to transform.scale
    Weights,     // Maps to Morph Target weights
}

/// Binding relationship: maps track `track_index` from a Clip to the target property
/// of `node_handle` in the scene.
#[derive(Debug, Clone)]
pub struct PropertyBinding {
    pub track_index: usize,
    pub node_handle: NodeHandle,
    pub target: TargetPath,
}
