use crate::scene::Scene;
use crate::animation::action::AnimationAction;
use crate::animation::binding::TargetPath;
use crate::animation::clip::TrackData;

pub struct AnimationMixer {
    actions: Vec<AnimationAction>,
    morph_weight_buffer: Vec<f32>,
}

impl AnimationMixer {
    pub fn new() -> Self {
        Self { 
            actions: Vec::new(),
            morph_weight_buffer: vec![0.0; 64],
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
                        let mesh_key = if let Some(node) = scene.get_node(binding.node_id) {
                            node.mesh
                        } else {
                            None
                        };
                        
                        if let Some(mesh_key) = mesh_key {
                            if let Some(mesh) = scene.meshes.get_mut(mesh_key) {
                                let num_targets = mesh.morph_target_influences.len();
                                if num_targets > 0 {
                                    if self.morph_weight_buffer.len() < num_targets {
                                        self.morph_weight_buffer.resize(num_targets, 0.0);
                                    }
                                    t.sample(action.time, &mut self.morph_weight_buffer[..num_targets]);
                                    mesh.set_morph_target_influences(&self.morph_weight_buffer[..num_targets]);
                                }
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