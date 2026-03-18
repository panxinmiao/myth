use rustc_hash::{FxHashMap, FxHashSet};
use slotmap::{SlotMap, new_key_type};

use crate::action::AnimationAction;
use crate::binding::{Rig, TargetPath};
use crate::blending::{BlendEntry, FrameBlendState};
use crate::clip::TrackData;
use crate::events::{self, FiredEvent};
use crate::target::AnimationTarget;
use myth_core::NodeHandle;

new_key_type! {
    pub struct ActionHandle;
}

/// Manages playback and blending of multiple animation actions.
///
/// The mixer drives time advancement for all active actions, accumulates
/// sampled animation data into per-node blend buffers, and applies the
/// final blended result to scene nodes once per frame.
///
/// # Rest Pose & State Restoration
///
/// The mixer tracks which nodes were animated in the previous frame. When
/// a node loses all animation influence (e.g. an action is stopped), it is
/// automatically restored to its rest pose.
///
/// # Blending
///
/// When multiple actions are active simultaneously, their contributions
/// are combined using weight-based accumulation.
/// If the total accumulated weight for a property is less than 1.0, the
/// rest pose value fills the remainder.
///
/// # Events
///
/// Animation events fired during the frame are collected and can be
/// consumed via [`drain_events`](Self::drain_events).
pub struct AnimationMixer {
    actions: SlotMap<ActionHandle, AnimationAction>,
    name_map: FxHashMap<String, ActionHandle>,
    active_handles: Vec<ActionHandle>,

    /// Logical skeleton for this entity, providing O(1) bone-index → node-handle lookup.
    rig: Rig,

    /// Global mixer time in seconds.
    pub time: f32,
    /// Global time scale multiplier applied to all actions.
    pub time_scale: f32,

    /// Per-frame blend accumulator (reused across frames to avoid allocation).
    blend_state: FrameBlendState,
    /// Events fired during the most recent update.
    fired_events: Vec<FiredEvent>,
    /// Node handles that were animated in the previous frame.
    /// Used for rest-pose restoration when animation influence is lost.
    animated_last_frame: FxHashSet<NodeHandle>,

    // Temporary buffer for blended morph weights during application phase.
    morph_buffer: crate::values::MorphWeightData,

    pub enabled: bool,
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
            rig: Rig {
                bones: Vec::new(),
                bone_paths: Vec::new(),
            },
            time: 0.0,
            time_scale: 1.0,
            blend_state: FrameBlendState::new(),
            fired_events: Vec::new(),
            animated_last_frame: FxHashSet::default(),
            morph_buffer: crate::values::MorphWeightData::default(),
            enabled: true,
        }
    }

    /// Sets the logical skeleton used for bone-index → node-handle lookup.
    pub fn set_rig(&mut self, rig: Rig) {
        self.rig = rig;
    }

    /// Returns a read-only reference to the mixer's rig.
    #[must_use]
    pub fn rig(&self) -> &Rig {
        &self.rig
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

    pub fn get_control_by_name(&mut self, name: &str) -> Option<ActionControl<'_>> {
        let handle = *self.name_map.get(name)?;
        self.get_control(handle)
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
                action.stop();
            }
            self.active_handles.retain(|&h| h != handle);
        }
    }

    /// Stops all active animations.
    pub fn stop_all(&mut self) {
        for handle in &self.active_handles {
            if let Some(action) = self.actions.get_mut(*handle) {
                action.stop();
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

    /// Advances all active actions and applies blended results to the target.
    ///
    /// This is the core per-frame entry point. The update proceeds in four phases:
    ///
    /// 1. **Time advancement**: Each active action's time is advanced. Animation
    ///    events that fall within the `[t_prev, t_curr]` window are collected.
    /// 2. **Sampling & accumulation**: Active actions sample their tracks and
    ///    accumulate weighted results into the blend buffer. Track-to-node
    ///    mapping uses [`crate::binding::ClipBinding`] + [`Rig`] for O(1) lookup.
    /// 3. **Application**: Blended values are mixed with the rest pose and
    ///    written to scene nodes.
    /// 4. **Restoration**: Nodes that were animated last frame but received no
    ///    contributions this frame are reset to their rest pose.
    pub fn update(&mut self, dt: f32, target: &mut dyn AnimationTarget) {
        if !self.enabled {
            return;
        }

        // phase 0: Restore all nodes that were animated in the previous frame to their rest pose.
        for &prev_handle in &self.animated_last_frame {
            if let Some(rest) = target.rest_transform(prev_handle) {
                target.set_node_position(prev_handle, rest.position);
                target.set_node_rotation(prev_handle, rest.rotation);
                target.set_node_scale(prev_handle, rest.scale);
                target.mark_node_dirty(prev_handle);
            }
        }

        let dt = dt * self.time_scale;
        self.time += dt;

        // Clear per-frame state
        self.blend_state.clear();
        self.fired_events.clear();

        self.animated_last_frame.clear();

        // Phase 1: Advance time and collect events
        for &handle in &self.active_handles {
            if let Some(action) = self.actions.get_mut(handle) {
                let t_prev = action.time;
                action.update(dt);
                let t_curr = action.time;

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

        // Phase 2: Sample tracks and accumulate into blend buffer (O(1) per track)
        for &handle in &self.active_handles {
            let action = match self.actions.get_mut(handle) {
                Some(a) if a.enabled && !a.paused && a.weight > 0.0 => a,
                _ => continue,
            };

            let clip = action.clip().clone();
            let weight = action.weight;
            let time = action.time;
            let cursors = &mut action.track_cursors;

            for tb in &action.clip_binding.bindings {
                let track = &clip.tracks[tb.track_index];
                let cursor = &mut cursors[tb.track_index];
                let node_handle = self.rig.bones[tb.bone_index];

                match (&track.data, tb.target) {
                    (TrackData::Vector3(t), TargetPath::Translation) => {
                        let val = t.sample_with_cursor(time, cursor);
                        self.blend_state
                            .accumulate_translation(node_handle, val, weight);
                    }
                    (TrackData::Vector3(t), TargetPath::Scale) => {
                        let val = t.sample_with_cursor(time, cursor);
                        self.blend_state.accumulate_scale(node_handle, val, weight);
                    }
                    (TrackData::Quaternion(t), TargetPath::Rotation) => {
                        let val = t.sample_with_cursor(time, cursor);
                        self.blend_state
                            .accumulate_rotation(node_handle, val, weight);
                    }
                    (TrackData::MorphWeights(t), TargetPath::Weights) => {
                        t.sample_with_cursor_into(time, cursor, &mut self.morph_buffer);
                        self.blend_state.accumulate_morph_weights(
                            node_handle,
                            &self.morph_buffer,
                            weight,
                        );
                    }
                    _ => {}
                }
            }
        }

        // Phase 3: Apply blended results to scene nodes using rest pose as base
        for (&node_handle, props) in self.blend_state.iter_nodes() {
            self.animated_last_frame.insert(node_handle);

            let rest_transform = target.rest_transform(node_handle).unwrap_or_default();

            for (t, entry) in props {
                match (t, entry) {
                    (TargetPath::Translation, BlendEntry::Translation { value, weight }) => {
                        if *weight < 1.0 {
                            target.set_node_position(
                                node_handle,
                                rest_transform.position.lerp(*value, *weight),
                            );
                        } else {
                            target.set_node_position(node_handle, *value);
                        }
                        target.mark_node_dirty(node_handle);
                    }
                    (TargetPath::Rotation, BlendEntry::Rotation { value, weight }) => {
                        if *weight < 1.0 {
                            let corrected = if rest_transform.rotation.dot(*value) < 0.0 {
                                -*value
                            } else {
                                *value
                            };
                            target.set_node_rotation(
                                node_handle,
                                rest_transform.rotation.lerp(corrected, *weight).normalize(),
                            );
                        } else {
                            target.set_node_rotation(node_handle, *value);
                        }
                        target.mark_node_dirty(node_handle);
                    }
                    (TargetPath::Scale, BlendEntry::Scale { value, weight }) => {
                        if *weight < 1.0 {
                            target.set_node_scale(
                                node_handle,
                                rest_transform.scale.lerp(*value, *weight),
                            );
                        } else {
                            target.set_node_scale(node_handle, *value);
                        }
                        target.mark_node_dirty(node_handle);
                    }
                    (
                        TargetPath::Weights,
                        BlendEntry::MorphWeights {
                            weights,
                            total_weight,
                        },
                    ) => {
                        apply_morph_weights(target, node_handle, weights, *total_weight);
                    }
                    _ => {}
                }
            }
        }
    }
}

/// Applies blended morph weights to the target, mixing with the rest pose
/// when the total accumulated weight is below 1.0.
fn apply_morph_weights(
    target: &mut dyn AnimationTarget,
    node: NodeHandle,
    weights: &[f32],
    total_weight: f32,
) {
    let dst = target.morph_weights_mut(node);
    if dst.len() < weights.len() {
        dst.resize(weights.len(), 0.0);
    }
    if total_weight >= 1.0 {
        dst[..weights.len()].copy_from_slice(weights);
    } else {
        for (d, &src) in dst.iter_mut().zip(weights.iter()) {
            *d = src * total_weight;
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
    pub fn set_loop_mode(self, mode: crate::action::LoopMode) -> Self {
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
    #[must_use]
    pub fn stop(self) -> Self {
        if let Some(action) = self.mixer.actions.get_mut(self.handle) {
            action.enabled = false;
            action.weight = 0.0;
        }
        self.mixer.active_handles.retain(|&h| h != self.handle);
        self
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
