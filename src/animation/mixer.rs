use crate::scene::Scene;
use crate::animation::action::AnimationAction;
use crate::animation::binding::TargetPath;
use crate::animation::clip::TrackData;

pub struct AnimationMixer {
    actions: Vec<AnimationAction>,
}

impl AnimationMixer {
    pub fn new() -> Self {
        Self { actions: Vec::new() }
    }

    pub fn add_action(&mut self, action: AnimationAction) {
        self.actions.push(action);
    }

    /// 核心更新循环
    /// 传入 &mut Scene，因为我们要修改 Node 的 Transform
    pub fn update(&mut self, dt: f32, scene: &mut Scene) {
        
        // 1. 更新所有 Action 的时间
        for action in &mut self.actions {
            action.update(dt);
        }

        // 2. 应用动画值

        for action in &mut self.actions {
            if action.paused || !action.enabled || action.weight <= 0.0 {
                continue;
            }

            // 遍历该动作的所有绑定
            for binding in &action.bindings {
                // 获取 Node (如果 Node 被删了，这里会忽略)
                if let Some(node) = scene.get_node_mut(binding.node_id) {
                    
                    let cursor = &mut action.track_cursors[binding.track_index].clone();

                    let track = &action.clip().tracks[binding.track_index];

                    match (&track.data, binding.target) {
                        (TrackData::Vector3(t), TargetPath::Translation) => {
                            let val = t.sample_with_cursor(action.time, cursor);
                            node.transform.position = val; 
                            node.transform.mark_dirty();
                        },
                        (TrackData::Vector3(t), TargetPath::Scale) => {
                            let val = t.sample_with_cursor(action.time, cursor);
                            node.transform.scale = val;
                            node.transform.mark_dirty();
                        },
                        (TrackData::Quaternion(t), TargetPath::Rotation) => {
                            let val = t.sample_with_cursor(action.time, cursor);
                            node.transform.rotation = val;
                            node.transform.mark_dirty();
                        },  
                        (TrackData::Scalar(_t), TargetPath::Weights) => {
                            // TODO: Morph target weights support
                        },
                        _ => {}
                    }

                    action.track_cursors[binding.track_index] = cursor.clone();

                }
            }
        }
    }
}