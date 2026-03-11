use rustc_hash::FxHashMap;
use slotmap::{SlotMap, new_key_type};

use crate::animation::action::AnimationAction;
use crate::animation::binding::TargetPath;
use crate::animation::blending::{BlendEntry, FrameBlendState};
use crate::animation::clip::TrackData;
use crate::animation::events::{self, FiredEvent};
use crate::scene::Scene;

new_key_type! {
    pub struct ActionHandle;
}

/// Manages playback and blending of multiple animation actions.
///
/// The mixer drives time advancement for all active actions, accumulates
/// sampled animation data into per-node blend buffers, and applies the
/// final blended result to scene nodes once per frame.
///
/// # Blending
///
/// When multiple actions are active simultaneously, their contributions
/// are combined using weight-based accumulation (see [`FrameBlendState`]).
/// If the total accumulated weight for a property is less than 1.0, the
/// original (rest pose) value fills the remainder.
///
/// # Events
///
/// Animation events fired during the frame are collected and can be
/// consumed via [`drain_events`](Self::drain_events).
pub struct AnimationMixer {
    actions: SlotMap<ActionHandle, AnimationAction>,
    name_map: FxHashMap<String, ActionHandle>,
    active_handles: Vec<ActionHandle>,

    /// Global mixer time in seconds.
    pub time: f32,
    /// Global time scale multiplier applied to all actions.
    pub time_scale: f32,

    /// Per-frame blend accumulator (reused across frames to avoid allocation).
    blend_state: FrameBlendState,
    /// Events fired during the most recent update.
    fired_events: Vec<FiredEvent>,
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
            blend_state: FrameBlendState::new(),
            fired_events: Vec::new(),
        }
    }

    /// Returns a list of all registered animation clip names.
    #[must_use]
    pub fn list_animations(&self) -> Vec<String> {
        self.name_map.keys().cloned().collect()
    }

    /// Registers an action and returns its handle.
    pub fn add_action(&mut self, action: AnimationAction) -> ActionHandle {
        let name = action.clip().name.clone();
        let handle = self.actions.insert(action);
        self.name_map.insert(name, handle);
        handle
    }

    /// Read-only access to an action by clip name.
    #[must_use]
    pub fn get_action(&self, name: &str) -> Option<&AnimationAction> {
        let handle = *self.name_map.get(name)?;
        self.actions.get(handle)
    }

    /// Read-only access to an action by handle.
    #[must_use]
    pub fn get_action_by_handle(&self, handle: ActionHandle) -> Option<&AnimationAction> {
        self.actions.get(handle)
    }

    /// Returns a chainable control wrapper for the named action.
    pub fn action(&mut self, name: &str) -> Option<ActionControl<'_>> {
        let handle = *self.name_map.get(name)?;
        Some(ActionControl {
            mixer: self,
            handle,
        })
    }

    /// Returns a control wrapper for the first registered action.
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

    /// Returns a control wrapper for an existing handle.
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

    /// Plays the named animation, adding it to the active set.
    pub fn play(&mut self, name: &str) {
        if let Some(&handle) = self.name_map.get(name) {
            if !self.active_handles.contains(&handle) {
                self.active_handles.push(handle);
            }
            if let Some(action) = self.actions.get_mut(handle) {
                action.enabled = true;
                action.weight = 1.0;
                action.paused = false;
            }
        } else {
            log::warn!("Animation not found: {name}");
        }
    }

    /// Stops the named animation and removes it from the active set.
    pub fn stop(&mut self, name: &str) {
        if let Some(&handle) = self.name_map.get(name) {
            if let Some(action) = self.actions.get_mut(handle) {
                action.enabled = false;
                action.weight = 0.0;
            }
            self.active_handles.retain(|&h| h != handle);
        }
    }

    /// Stops all active animations.
    pub fn stop_all(&mut self) {
        for handle in &self.active_handles {
            if let Some(action) = self.actions.get_mut(*handle) {
                action.enabled = false;
                action.weight = 0.0;
            }
        }
        self.active_handles.clear();
    }

    /// Drains all events fired during the most recent update.
    pub fn drain_events(&mut self) -> Vec<FiredEvent> {
        std::mem::take(&mut self.fired_events)
    }

    /// Returns a read-only slice of events fired during the most recent update.
    #[must_use]
    pub fn events(&self) -> &[FiredEvent] {
        &self.fired_events
    }

    /// Advances all active actions and applies blended results to the scene.
    ///
    /// This is the core per-frame entry point. The update proceeds in three phases:
    ///
    /// 1. **Time advancement**: Each active action's time is advanced. Animation
    ///    events that fall within the `[t_prev, t_curr]` window are collected.
    /// 2. **Sampling & accumulation**: Active actions sample their tracks and
    ///    accumulate weighted results into the blend buffer.
    /// 3. **Application**: Blended values are written to scene nodes. For transform
    ///    properties (translation, rotation, scale), `mark_dirty()` is called
    ///    exactly once per affected node regardless of how many properties changed.
    pub fn update(&mut self, dt: f32, scene: &mut Scene) {
        let dt = dt * self.time_scale;
        self.time += dt;

        // Clear per-frame state
        self.blend_state.clear();
        self.fired_events.clear();

        // Phase 1: Advance time and collect events
        for &handle in &self.active_handles {
            if let Some(action) = self.actions.get_mut(handle) {
                let t_prev = action.time;
                action.update(dt);
                let t_curr = action.time;

                // Collect animation events
                let clip = action.clip();
                events::collect_events(
                    &clip.events,
                    t_prev,
                    t_curr,
                    clip.duration,
                    &clip.name,
                    &mut self.fired_events,
                );
            }
        }

        // Phase 2: Sample tracks and accumulate into blend buffer
        for &handle in &self.active_handles {
            let action = match self.actions.get_mut(handle) {
                Some(a) if a.enabled && !a.paused && a.weight > 0.0 => a,
                _ => continue,
            };

            let clip = action.clip().clone();
            let weight = action.weight;
            let time = action.time;
            let bindings = &action.bindings;
            let cursors = &mut action.track_cursors;

            for binding in bindings {
                let track_index = binding.track_index;
                let track = &clip.tracks[track_index];
                let cursor = &mut cursors[track_index];
                let node = binding.node_handle;

                match (&track.data, binding.target) {
                    (TrackData::Vector3(t), TargetPath::Translation) => {
                        let val = t.sample_with_cursor(time, cursor);
                        self.blend_state.accumulate_translation(node, val, weight);
                    }
                    (TrackData::Vector3(t), TargetPath::Scale) => {
                        let val = t.sample_with_cursor(time, cursor);
                        self.blend_state.accumulate_scale(node, val, weight);
                    }
                    (TrackData::Quaternion(t), TargetPath::Rotation) => {
                        let val = t.sample_with_cursor(time, cursor);
                        self.blend_state.accumulate_rotation(node, val, weight);
                    }
                    (TrackData::MorphWeights(t), TargetPath::Weights) => {
                        let val = t.sample_with_cursor(time, cursor);
                        self.blend_state
                            .accumulate_morph_weights(node, &val, weight);
                    }
                    _ => {}
                }
            }
        }

        // Phase 3: Apply blended results to scene nodes
        if self.blend_state.is_empty() {
            return;
        }

        for (&node_handle, props) in self.blend_state.iter_nodes() {
            let mut transform_dirty = false;

            for (target, entry) in props {
                match (target, entry) {
                    (TargetPath::Translation, BlendEntry::Translation { value, weight }) => {
                        if let Some(node) = scene.get_node_mut(node_handle) {
                            if *weight < 1.0 {
                                let rest = node.transform.position;
                                node.transform.position = rest.lerp(*value, *weight);
                            } else {
                                node.transform.position = *value;
                            }
                            transform_dirty = true;
                        }
                    }
                    (TargetPath::Rotation, BlendEntry::Rotation { value, weight }) => {
                        if let Some(node) = scene.get_node_mut(node_handle) {
                            if *weight < 1.0 {
                                let rest = node.transform.rotation;
                                let corrected = if rest.dot(*value) < 0.0 {
                                    -*value
                                } else {
                                    *value
                                };
                                node.transform.rotation =
                                    rest.lerp(corrected, *weight).normalize();
                            } else {
                                node.transform.rotation = *value;
                            }
                            transform_dirty = true;
                        }
                    }
                    (TargetPath::Scale, BlendEntry::Scale { value, weight }) => {
                        if let Some(node) = scene.get_node_mut(node_handle) {
                            if *weight < 1.0 {
                                let rest = node.transform.scale;
                                node.transform.scale = rest.lerp(*value, *weight);
                            } else {
                                node.transform.scale = *value;
                            }
                            transform_dirty = true;
                        }
                    }
                    (
                        TargetPath::Weights,
                        BlendEntry::MorphWeights {
                            weights,
                            total_weight,
                        },
                    ) => {
                        apply_morph_weights(scene, node_handle, weights, *total_weight);
                    }
                    _ => {}
                }
            }

            // Single mark_dirty per node, regardless of how many properties changed
            if transform_dirty {
                if let Some(node) = scene.get_node_mut(node_handle) {
                    node.transform.mark_dirty();
                }
            }
        }
    }
}

/// Applies blended morph weights to the scene, mixing with the rest pose
/// when the total accumulated weight is below 1.0.
fn apply_morph_weights(scene: &mut Scene, node: crate::scene::NodeHandle, weights: &[f32], total_weight: f32) {
    let target = scene.morph_weights.entry(node).unwrap().or_default();
    if target.len() < weights.len() {
        target.resize(weights.len(), 0.0);
    }
    if total_weight >= 1.0 {
        target[..weights.len()].copy_from_slice(weights);
    } else {
        for (dst, &src) in target.iter_mut().zip(weights.iter()) {
            *dst = *dst * (1.0 - total_weight) + src * total_weight;
        }
    }
}

// ============================================================================
// ActionControl — chainable builder for action state manipulation
// ============================================================================

/// Chainable wrapper for mutating an action within a mixer.
///
/// Obtained from [`AnimationMixer::action`] or [`AnimationMixer::get_control`].
/// All setter methods return `self` to support method chaining.
pub struct ActionControl<'a> {
    mixer: &'a mut AnimationMixer,
    handle: ActionHandle,
}

impl ActionControl<'_> {
    /// Starts or restarts playback from the beginning.
    #[must_use]
    pub fn play(self) -> Self {
        if !self.mixer.active_handles.contains(&self.handle) {
            self.mixer.active_handles.push(self.handle);
        }
        if let Some(action) = self.mixer.actions.get_mut(self.handle) {
            action.enabled = true;
            action.paused = false;
            action.weight = 1.0;
            action.time = 0.0;
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

    /// Stops playback and removes the action from the active set.
    pub fn stop(self) {
        if let Some(action) = self.mixer.actions.get_mut(self.handle) {
            action.enabled = false;
            action.weight = 0.0;
        }
        self.mixer.active_handles.retain(|&h| h != self.handle);
    }

    /// Starts playback with a fade-in effect over the given duration.
    #[must_use]
    pub fn fade_in(self, _duration: f32) -> Self {
        // TODO: Implement gradual weight interpolation
        self.play()
    }
}

impl std::ops::Deref for ActionControl<'_> {
    type Target = AnimationAction;
    fn deref(&self) -> &Self::Target {
        self.mixer.actions.get(self.handle).unwrap()
    }
}

impl std::ops::DerefMut for ActionControl<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.mixer.actions.get_mut(self.handle).unwrap()
    }
}
