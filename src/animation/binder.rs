use crate::animation::binding::PropertyBinding;
use crate::animation::clip::AnimationClip;
use crate::scene::{NodeHandle, Scene};

pub struct Binder;

impl Binder {
    /// 解析动画片段，将轨道绑定到场景中的实际 `NodeHandle`
    /// 搜索整个场景中的节点
    pub fn bind(
        scene: &Scene,
        _root_node: NodeHandle,
        clip: &AnimationClip,
    ) -> Vec<PropertyBinding> {
        let mut bindings = Vec::with_capacity(clip.tracks.len());

        for (track_idx, track) in clip.tracks.iter().enumerate() {
            let node_name = &track.meta.node_name;
            let target = track.meta.target;

            if let Some(node_handle) = find_node_in_scene(scene, node_name) {
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

fn find_node_in_scene(scene: &Scene, name: &str) -> Option<NodeHandle> {
    scene.find_node_by_name(name)
}
