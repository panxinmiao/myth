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
            let clip = action.clip().clone();

            let bindings = &action.bindings;
            let cursors = &mut action.track_cursors;
            let time = action.time;

            for binding in bindings {
                let track_index = binding.track_index;
                let track = &clip.tracks[track_index];

                let cursor = &mut cursors[track_index];

                 match (&track.data, binding.target) {
                    (TrackData::Vector3(t), TargetPath::Translation) => {
                        if let Some(node) = scene.get_node_mut(binding.node_handle) {
                            let val = t.sample_with_cursor(time, cursor);
                            node.transform.position = val; 
                            node.transform.mark_dirty();
                        }
                    },
                    (TrackData::Vector3(t), TargetPath::Scale) => {
                        if let Some(node) = scene.get_node_mut(binding.node_handle) {
                            let val = t.sample_with_cursor(time, cursor);
                            node.transform.scale = val;
                            node.transform.mark_dirty();
                        }
                    },
                    (TrackData::Quaternion(t), TargetPath::Rotation) => {
                        if let Some(node) = scene.get_node_mut(binding.node_handle) {
                            let val = t.sample_with_cursor(time, cursor);
                            node.transform.rotation = val;
                            node.transform.mark_dirty();
                        }
                    },  
                    (TrackData::MorphWeights(t), TargetPath::Weights) => {
                        let weights_pod = t.sample_with_cursor(time, cursor);
                        scene.set_morph_weights_from_pod(binding.node_handle, &weights_pod);
                    },
                    _ => {}
                }
            }
        }
    }
}