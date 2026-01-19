use crate::scene::{Scene, NodeIndex};
use crate::animation::clip::AnimationClip;
use crate::animation::binding::PropertyBinding;

pub struct Binder;

impl Binder {
    /// 解析动画片段，将轨道绑定到场景中的实际 NodeIndex
    pub fn bind(scene: &Scene, root_node: NodeIndex, clip: &AnimationClip) -> Vec<PropertyBinding> {
        let mut bindings = Vec::with_capacity(clip.tracks.len());
        
        for (track_idx, track) in clip.tracks.iter().enumerate() {
            let node_name = &track.meta.node_name;
            let target = track.meta.target;

            if let Some(node_id) = find_node_by_name(scene, root_node, node_name) {
                bindings.push(PropertyBinding {
                    track_index: track_idx,
                    node_id,
                    target,
                });
            }
        }

        bindings
    }
}

fn find_node_by_name(scene: &Scene, current: NodeIndex, name: &str) -> Option<NodeIndex> {
    if let Some(node) = scene.get_node(current) {
        if node.name == name {
            return Some(current);
        }
        for &child in &node.children {
            if let Some(found) = find_node_by_name(scene, child, name) {
                return Some(found);
            }
        }
    }
    None
}