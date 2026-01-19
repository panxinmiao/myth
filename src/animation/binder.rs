use crate::scene::{Scene, NodeIndex};
use crate::animation::clip::AnimationClip;
use crate::animation::binding::PropertyBinding;

pub struct Binder;

impl Binder {
    /// 解析动画片段，将轨道绑定到场景中的实际 NodeIndex
    /// 搜索整个场景中的节点
    pub fn bind(scene: &Scene, _root_node: NodeIndex, clip: &AnimationClip) -> Vec<PropertyBinding> {
        let mut bindings = Vec::with_capacity(clip.tracks.len());
        
        for (track_idx, track) in clip.tracks.iter().enumerate() {
            let node_name = &track.meta.node_name;
            let target = track.meta.target;

            if let Some(node_id) = find_node_in_scene(scene, node_name) {
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

fn find_node_in_scene(scene: &Scene, name: &str) -> Option<NodeIndex> {
    for (idx, node) in scene.nodes.iter() {
        if node.name == name {
            return Some(idx);
        }
    }
    None
}