use rustc_hash::FxHashMap;
use slotmap::{new_key_type, SlotMap};

use crate::scene::Scene;
use crate::animation::action::AnimationAction;
use crate::animation::binding::TargetPath;
use crate::animation::clip::TrackData;


new_key_type! {
    pub struct ActionHandle;
}

pub struct AnimationMixer {
    actions: SlotMap<ActionHandle, AnimationAction>,

    name_map: FxHashMap<String, ActionHandle>,

    active_handles: Vec<ActionHandle>,

    pub time: f32,
    pub time_scale: f32,
}

impl Default for AnimationMixer {
    fn default() -> Self {
        Self::new()
    }
}

impl AnimationMixer {
    pub fn new() -> Self {
        Self { 
            actions: SlotMap::with_key(),
            name_map: FxHashMap::default(),
            active_handles: Vec::new(),
            time: 0.0,
            time_scale: 1.0,
        }
    }

    pub fn list_animations(&self) -> Vec<String> {
        self.name_map.keys().cloned().collect()
    }

    pub fn add_action(&mut self, action: AnimationAction) -> ActionHandle {
        let name = action.clip().name.clone();
        
        let handle = self.actions.insert(action);
        // 建立索引
        self.name_map.insert(name, handle);
        
        handle
    }

    /// 只读访问
    pub fn get_action(&self, name: &str) -> Option<&AnimationAction> {
        let handle = *self.name_map.get(name)?;
        self.actions.get(handle)
    }

    /// 只读访问
    pub fn get_action_by_handle(&self, handle: ActionHandle) -> Option<&AnimationAction> {
        self.actions.get(handle)
    }

    // 获取动画控制器
    pub fn action(&mut self, name: &str) -> Option<ActionControl<'_>> {
        let handle = *self.name_map.get(name)?;
        Some(ActionControl {
            mixer: self,
            handle,
        })
    }

    pub fn any_action(&mut self) -> Option<ActionControl<'_>> {
        if let Some((handle, _)) = self.actions.iter().next() {
            Some(ActionControl {
                mixer: self,
                handle,
            })
        } else {
            None
        }
    }

    /// 如果用户已经有了 Handle
    pub fn get_control(&mut self, handle: ActionHandle) -> Option<ActionControl<'_>> {
        if self.actions.contains_key(handle) {
             Some(ActionControl {
                mixer: self,
                handle,
            })
        } else {
            None
        }
    }

    /// 播放指定动画
    pub fn play(&mut self, name: &str) {
        if let Some(&handle) = self.name_map.get(name) {
            // 1. 如果不在激活列表中，加入
            if !self.active_handles.contains(&handle) {
                self.active_handles.push(handle);
            }

            // 2. 重置并启用该动画
            if let Some(action) = self.actions.get_mut(handle) {
                action.enabled = true;
                action.weight = 1.0;
                action.time = 0.0;
                action.paused = false;
            }

            // 3. (可选) 可以在这里把 active_handles 里的其他动画停掉
            // self.stop_others(handle);
        } else {
            log::warn!("Animation not found: {}", name);
        }
    }

    /// 停止指定动画
    pub fn stop(&mut self, name: &str) {
        if let Some(&handle) = self.name_map.get(name) {
            if let Some(action) = self.actions.get_mut(handle) {
                action.enabled = false;
                action.weight = 0.0;
            }
            // 从激活列表中移除
            self.active_handles.retain(|&h| h != handle);
        }
    }

    /// 停止所有动画
    pub fn stop_all(&mut self) {
        for handle in &self.active_handles {
            if let Some(action) = self.actions.get_mut(*handle) {
                action.enabled = false;
                action.weight = 0.0;
            }
        }
        self.active_handles.clear();
    }

    pub fn update(&mut self, dt: f32, scene: &mut Scene) {

        let dt = dt * self.time_scale;
        self.time += dt;

        for &handle in &self.active_handles {
            if let Some(action) = self.actions.get_mut(handle) {
                action.update(dt);
            }
        }

        for &handle in &self.active_handles {

            let action = match self.actions.get_mut(handle) {
                Some(a) if a.enabled && !a.paused && a.weight > 0.0 => a,
                _ => continue,
            };

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




pub struct ActionControl<'a> {
    mixer: &'a mut AnimationMixer,
    handle: ActionHandle,
}

impl<'a> ActionControl<'a> {
    /// 核心逻辑：播放
    pub fn play(self) -> Self {
        // 1. 确保加入激活列表
        if !self.mixer.active_handles.contains(&self.handle) {
            self.mixer.active_handles.push(self.handle);
        }
        
        // 2. 修改 Action 自身状态
        if let Some(action) = self.mixer.actions.get_mut(self.handle) {
            action.enabled = true;
            action.paused = false;
            action.weight = 1.0;
            action.time = 0.0; // 从头开始播放
        }
        self
    }

    pub fn set_loop_mode(self, mode: crate::animation::action::LoopMode) -> Self {
        if let Some(action) = self.mixer.actions.get_mut(self.handle) {
            action.loop_mode = mode;
        }
        self
    }

    pub fn set_time_scale(self, scale: f32) -> Self {
        if let Some(action) = self.mixer.actions.get_mut(self.handle) {
            action.time_scale = scale;
        }
        self
    }

    pub fn set_weight(self, weight: f32) -> Self {
        if let Some(action) = self.mixer.actions.get_mut(self.handle) {
            action.weight = weight;
        }
        self
    }

    pub fn set_time(self, time: f32) -> Self {
        if let Some(action) = self.mixer.actions.get_mut(self.handle) {
            action.time = time;
        }
        self
    }

    pub fn resume(self) -> Self {
        if let Some(action) = self.mixer.actions.get_mut(self.handle) {
            action.paused = false;
        }
        self
    }

    pub fn pause(self) -> Self {
        if let Some(action) = self.mixer.actions.get_mut(self.handle) {
            action.paused = true;
        }
        self
    }

    /// 核心逻辑：停止
    pub fn stop(self) {
         if let Some(action) = self.mixer.actions.get_mut(self.handle) {
            action.enabled = false;
            action.weight = 0.0;
        }
        // 从激活列表移除（或者留给 update 清理，这里立刻移除比较干净）
        self.mixer.active_handles.retain(|&h| h != self.handle);
    }
    
    /// 核心逻辑：淡入
    pub fn fade_in(self, _duration: f32) -> Self {
        // 实现淡入逻辑...
        self.play() // 链式调用
    }
}


impl<'a> std::ops::Deref for ActionControl<'a> {
    type Target = AnimationAction;
    fn deref(&self) -> &Self::Target {
        //由于 handle 必定有效（内部逻辑保证），这里可以用 unwrap 或者安全处理
        self.mixer.actions.get(self.handle).unwrap()
    }
}

impl<'a> std::ops::DerefMut for ActionControl<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.mixer.actions.get_mut(self.handle).unwrap()
    }
}