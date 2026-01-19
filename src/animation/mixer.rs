use crate::scene::Scene;
use crate::animation::action::AnimationAction;
use crate::animation::binding::TargetPath;
use crate::animation::clip::TrackData;

pub struct AnimationMixer {
    actions: Vec<AnimationAction>,
}

impl AnimationMixer {
    pub fn new() -> Self {
        Self { 
            actions: Vec::new(),
        }
    }

    pub fn add_action(&mut self, action: AnimationAction) {
        self.actions.push(action);
    }

    pub fn update(&mut self, dt: f32, scene: &mut Scene) {
        for action in &mut self.actions {
            action.update(dt);
        }

        for action in &mut self.actions {
            if action.paused || !action.enabled || action.weight <= 0.0 {
                continue;
            }

            for binding in &action.bindings {
                let track = &action.clip().tracks[binding.track_index];
                let cursor = &mut action.track_cursors[binding.track_index].clone();

                match (&track.data, binding.target) {
                    (TrackData::Vector3(t), TargetPath::Translation) => {
                        if let Some(node) = scene.get_node_mut(binding.node_id) {
                            let val = t.sample_with_cursor(action.time, cursor);
                            node.transform.position = val; 
                            node.transform.mark_dirty();
                        }
                    },
                    (TrackData::Vector3(t), TargetPath::Scale) => {
                        if let Some(node) = scene.get_node_mut(binding.node_id) {
                            let val = t.sample_with_cursor(action.time, cursor);
                            node.transform.scale = val;
                            node.transform.mark_dirty();
                        }
                    },
                    (TrackData::Quaternion(t), TargetPath::Rotation) => {
                        if let Some(node) = scene.get_node_mut(binding.node_id) {
                            let val = t.sample_with_cursor(action.time, cursor);
                            node.transform.rotation = val;
                            node.transform.mark_dirty();
                        }
                    },  
                    (TrackData::MorphWeights(t), TargetPath::Weights) => {
                        let weights_pod = t.sample_with_cursor(action.time, cursor);
                        
                        let mesh_key = scene.get_node(binding.node_id).and_then(|n| n.mesh);
                        let target_count = mesh_key
                            .and_then(|key| scene.meshes.get(key))
                            .map(|mesh| mesh.morph_target_influences.len())
                            .unwrap_or(0);
                        
                        if target_count > 0 {
                            if let Some(node) = scene.get_node_mut(binding.node_id) {
                                node.set_morph_weights_from_pod(&weights_pod, target_count);
                            }
                        }
                    },
                    _ => {}
                }

                action.track_cursors[binding.track_index] = cursor.clone();
            }
        }
    }
}