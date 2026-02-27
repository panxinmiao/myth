use rustc_hash::FxHashMap;
use slotmap::{SlotMap, new_key_type};

use crate::animation::action::AnimationAction;
use crate::animation::binding::TargetPath;
use crate::animation::clip::TrackData;
use crate::scene::Scene;

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
    #[must_use]
    pub fn new() -> Self {
        Self {
            actions: SlotMap::with_key(),
            name_map: FxHashMap::default(),
            active_handles: Vec::new(),
            time: 0.0,
            time_scale: 1.0,
        }
    }

    #[must_use]
    pub fn list_animations(&self) -> Vec<String> {
        self.name_map.keys().cloned().collect()
    }

    pub fn add_action(&mut self, action: AnimationAction) -> ActionHandle {
        let name = action.clip().name.clone();

        let handle = self.actions.insert(action);
        // Build name-to-handle index
        self.name_map.insert(name, handle);

        handle
    }

    /// Read-only access
    #[must_use]
    pub fn get_action(&self, name: &str) -> Option<&AnimationAction> {
        let handle = *self.name_map.get(name)?;
        self.actions.get(handle)
    }

    /// Read-only access
    #[must_use]
    pub fn get_action_by_handle(&self, handle: ActionHandle) -> Option<&AnimationAction> {
        self.actions.get(handle)
    }

    // Get animation controller
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

    /// Returns a control wrapper if the user already has a Handle.
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

    /// Plays the specified animation.
    pub fn play(&mut self, name: &str) {
        if let Some(&handle) = self.name_map.get(name) {
            // 1. Add to active list if not already present
            if !self.active_handles.contains(&handle) {
                self.active_handles.push(handle);
            }

            // 2. Reset and enable the animation
            if let Some(action) = self.actions.get_mut(handle) {
                action.enabled = true;
                action.weight = 1.0;
                action.paused = false;
            }

            // 3. (Optional) Could stop other animations in active_handles here
        } else {
            log::warn!("Animation not found: {name}");
        }
    }

    /// Stops the specified animation.
    pub fn stop(&mut self, name: &str) {
        if let Some(&handle) = self.name_map.get(name) {
            if let Some(action) = self.actions.get_mut(handle) {
                action.enabled = false;
                action.weight = 0.0;
            }
            // Remove from active list
            self.active_handles.retain(|&h| h != handle);
        }
    }

    /// Stops all animations.
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
                    }
                    (TrackData::Vector3(t), TargetPath::Scale) => {
                        if let Some(node) = scene.get_node_mut(binding.node_handle) {
                            let val = t.sample_with_cursor(time, cursor);
                            node.transform.scale = val;
                            node.transform.mark_dirty();
                        }
                    }
                    (TrackData::Quaternion(t), TargetPath::Rotation) => {
                        if let Some(node) = scene.get_node_mut(binding.node_handle) {
                            let val = t.sample_with_cursor(time, cursor);
                            node.transform.rotation = val;
                            node.transform.mark_dirty();
                        }
                    }
                    (TrackData::MorphWeights(t), TargetPath::Weights) => {
                        let weights_pod = t.sample_with_cursor(time, cursor);
                        scene.set_morph_weights_from_pod(binding.node_handle, &weights_pod);
                    }
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

impl ActionControl<'_> {
    /// Core logic: play.
    #[must_use]
    pub fn play(self) -> Self {
        // 1. Ensure added to active list
        if !self.mixer.active_handles.contains(&self.handle) {
            self.mixer.active_handles.push(self.handle);
        }

        // 2. Modify the Action's own state
        if let Some(action) = self.mixer.actions.get_mut(self.handle) {
            action.enabled = true;
            action.paused = false;
            action.weight = 1.0;
            action.time = 0.0; // Start playback from the beginning
        }
        self
    }

    #[must_use]
    pub fn set_loop_mode(self, mode: crate::animation::action::LoopMode) -> Self {
        if let Some(action) = self.mixer.actions.get_mut(self.handle) {
            action.loop_mode = mode;
        }
        self
    }

    #[must_use]
    pub fn set_time_scale(self, scale: f32) -> Self {
        if let Some(action) = self.mixer.actions.get_mut(self.handle) {
            action.time_scale = scale;
        }
        self
    }

    #[must_use]
    pub fn set_weight(self, weight: f32) -> Self {
        if let Some(action) = self.mixer.actions.get_mut(self.handle) {
            action.weight = weight;
        }
        self
    }

    #[must_use]
    pub fn set_time(self, time: f32) -> Self {
        if let Some(action) = self.mixer.actions.get_mut(self.handle) {
            action.time = time;
        }
        self
    }

    #[must_use]
    pub fn resume(self) -> Self {
        if let Some(action) = self.mixer.actions.get_mut(self.handle) {
            action.paused = false;
        }
        self
    }

    #[must_use]
    pub fn pause(self) -> Self {
        if let Some(action) = self.mixer.actions.get_mut(self.handle) {
            action.paused = true;
        }
        self
    }

    /// Core logic: stop.
    pub fn stop(self) {
        if let Some(action) = self.mixer.actions.get_mut(self.handle) {
            action.enabled = false;
            action.weight = 0.0;
        }
        // Remove from active list (could leave for update to clean, but immediate removal is cleaner)
        self.mixer.active_handles.retain(|&h| h != self.handle);
    }

    /// Core logic: fade in.
    #[must_use]
    pub fn fade_in(self, _duration: f32) -> Self {
        // Implement fade-in logic...
        self.play() // Chain call
    }
}

impl std::ops::Deref for ActionControl<'_> {
    type Target = AnimationAction;
    fn deref(&self) -> &Self::Target {
        // Since handle is guaranteed valid (ensured by internal logic), unwrap is safe here
        self.mixer.actions.get(self.handle).unwrap()
    }
}

impl std::ops::DerefMut for ActionControl<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.mixer.actions.get_mut(self.handle).unwrap()
    }
}
