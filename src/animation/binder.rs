use crate::animation::binding::PropertyBinding;
use crate::animation::clip::AnimationClip;
use crate::scene::{NodeHandle, Scene};

/// Resolves animation clip track targets to concrete scene node handles.
///
/// Binding is scoped to a specific subtree rooted at `root_node`, preventing
/// name collisions when multiple instances of the same prefab exist in a scene.
pub struct Binder;

impl Binder {
    /// Binds animation tracks to scene nodes within the `root_node` subtree.
    ///
    /// Each track's `node_name` is matched against node names reachable from
    /// `root_node` (inclusive). Only the first match within the subtree is used.
    pub fn bind(
        scene: &Scene,
        root_node: NodeHandle,
        clip: &AnimationClip,
    ) -> Vec<PropertyBinding> {
        let mut bindings = Vec::with_capacity(clip.tracks.len());

        for (track_idx, track) in clip.tracks.iter().enumerate() {
            let node_name = &track.meta.node_name;
            let target = track.meta.target;

            if let Some(node_handle) = find_node_in_descendants(scene, root_node, node_name) {
                bindings.push(PropertyBinding {
                    track_index: track_idx,
                    node_handle,
                    target,
                });
            }
        }

        bindings
    }
}

/// Searches for a node by name within the subtree rooted at `root`.
///
/// Performs a depth-first traversal starting from `root` (inclusive).
/// Returns the first matching node handle, or `None` if not found.
fn find_node_in_descendants(scene: &Scene, root: NodeHandle, name: &str) -> Option<NodeHandle> {
    // Check the root node itself
    if scene.get_name(root).is_some_and(|n| n == name) {
        return Some(root);
    }

    // Depth-first search through descendants
    let mut stack = Vec::new();
    if let Some(node) = scene.get_node(root) {
        stack.extend_from_slice(node.children());
    }

    while let Some(handle) = stack.pop() {
        if scene.get_name(handle).is_some_and(|n| n == name) {
            return Some(handle);
        }
        if let Some(node) = scene.get_node(handle) {
            stack.extend_from_slice(node.children());
        }
    }

    None
}
