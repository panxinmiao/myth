use crate::scene::NodeHandle;

/// Defines the target property for animation data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetPath {
    Translation, // Maps to transform.position
    Rotation,    // Maps to transform.rotation
    Scale,       // Maps to transform.scale
    Weights,     // Maps to Morph Target weights
}

// ============================================================================
// Rig — logical skeleton attached to an animated entity
// ============================================================================

/// Logical skeleton describing the bone topology of an animated entity.
///
/// Attached to the animation root node at instantiation time. The `bones`
/// array provides O(1) access from a logical bone index to the concrete
/// scene [`NodeHandle`], while `bone_paths` stores the corresponding
/// hierarchical paths used for clip binding.
#[derive(Debug, Clone)]
pub struct Rig {
    /// Bone node handles ordered by logical bone index.
    pub bones: Vec<NodeHandle>,
    /// Hierarchical path segments for each bone, parallel to `bones`.
    /// Used during [`ClipBinding`] construction to match track paths.
    pub bone_paths: Vec<Vec<String>>,
}

// ============================================================================
// ClipBinding — precomputed track-to-bone mapping
// ============================================================================

/// Maps a single animation track to a logical bone in a [`Rig`].
#[derive(Debug, Clone)]
pub struct TrackBinding {
    /// Index into [`AnimationClip::tracks`].
    pub track_index: usize,
    /// Index into [`Rig::bones`], giving O(1) node handle lookup at runtime.
    pub bone_index: usize,
    /// Which transform property this track drives.
    pub target: TargetPath,
}

/// Precomputed mapping from an [`AnimationClip`] to a [`Rig`].
///
/// Built once when a clip is first associated with a rig (see
/// [`Binder::build_clip_binding`]). The mapping is valid for any entity
/// that shares the same rig topology, enabling cross-model animation
/// retargeting without per-frame string comparisons.
#[derive(Debug, Clone)]
pub struct ClipBinding {
    pub bindings: Vec<TrackBinding>,
}
