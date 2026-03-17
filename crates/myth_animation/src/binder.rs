use crate::binding::{ClipBinding, Rig, TrackBinding};
use crate::clip::AnimationClip;
use crate::target::AnimationTarget;
use myth_core::NodeHandle;

/// Resolves animation clip track targets to logical bone indices via a [`Rig`].
///
/// Two main workflows:
///
/// 1. **Rig construction** — [`build_rig`](Self::build_rig) walks the scene
///    subtree to produce a [`Rig`] with ordered bone handles and paths.
/// 2. **Clip binding** — [`build_clip_binding`](Self::build_clip_binding) maps
///    each track's path to a bone index, lazily recording rest poses for
///    newly animated nodes.
pub struct Binder;

impl Binder {
    /// Builds a [`Rig`] by walking the subtree rooted at `root_node`.
    ///
    /// Every named descendant (and the root itself) becomes a bone entry.
    /// Bone paths are recorded relative to `root_node` so that the same
    /// rig topology can be matched against multiple clips.
    pub fn build_rig(target: &dyn AnimationTarget, root_node: NodeHandle) -> Rig {
        let mut bones = Vec::new();
        let mut bone_paths = Vec::new();

        // Depth-first walk collecting (handle, path) pairs.
        let mut stack: Vec<(NodeHandle, Vec<String>)> = Vec::new();

        // Seed with root's direct children.
        if let Some(children) = target.node_children(root_node) {
            for child in children {
                if let Some(name) = target.node_name(child) {
                    stack.push((child, vec![name]));
                }
            }
        }

        // Also include root itself if it has a name.
        if let Some(name) = target.node_name(root_node) {
            bones.push(root_node);
            bone_paths.push(vec![name]);
        }

        while let Some((handle, path)) = stack.pop() {
            // Only register named nodes (unnamed internal nodes are skipped).
            if target.node_name(handle).is_some() {
                bones.push(handle);
                bone_paths.push(path.clone());
            }

            if let Some(children) = target.node_children(handle) {
                for child in children {
                    if let Some(child_name) = target.node_name(child) {
                        let mut child_path = path.clone();
                        child_path.push(child_name);
                        stack.push((child, child_path));
                    }
                }
            }
        }

        Rig { bones, bone_paths }
    }

    /// Builds a [`ClipBinding`] mapping clip tracks to rig bone indices.
    ///
    /// For every track whose [`TrackMeta::path`] matches a bone path in the
    /// rig, a [`TrackBinding`] is emitted. Nodes receiving their first
    /// animation binding get their current transform lazily recorded as
    /// the rest pose.
    pub fn build_clip_binding(
        target: &mut dyn AnimationTarget,
        rig: &Rig,
        clip: &AnimationClip,
    ) -> ClipBinding {
        let mut bindings = Vec::with_capacity(clip.tracks.len());

        for (track_idx, track) in clip.tracks.iter().enumerate() {
            let t = track.meta.target;

            // Match track path against rig bone paths.
            let bone_index = rig.bone_paths.iter().position(|p| *p == track.meta.path);

            if let Some(bone_index) = bone_index {
                let node_handle = rig.bones[bone_index];

                // Lazy record rest pose on first encounter.
                if !target.has_rest_transform(node_handle)
                    && let Some(transform) = target.node_transform(node_handle)
                {
                    target.store_rest_transform(node_handle, transform);
                }

                bindings.push(TrackBinding {
                    track_index: track_idx,
                    bone_index,
                    target: t,
                });
            }
        }

        ClipBinding { bindings }
    }

    /// Convenience: builds a rig *and* a clip binding in one call.
    ///
    /// Equivalent to calling [`build_rig`](Self::build_rig) followed by
    /// [`build_clip_binding`](Self::build_clip_binding).
    pub fn bind(
        target: &mut dyn AnimationTarget,
        root_node: NodeHandle,
        clip: &AnimationClip,
    ) -> (Rig, ClipBinding) {
        let rig = Self::build_rig(target, root_node);
        let binding = Self::build_clip_binding(target, &rig, clip);
        (rig, binding)
    }
}
