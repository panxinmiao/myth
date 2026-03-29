//! Animation Event System
//!
//! Provides frame-synchronized event triggering for animation playback.
//! Events are defined at specific times within an animation clip and fire
//! when playback crosses their timestamp.
//!
//! # Usage
//!
//! Events are consumed by external gameplay systems after each mixer update.
//! The event queue is cleared at the start of every frame.
//!
//! ```rust,ignore
//! // Define events on a clip
//! clip.events.push(AnimationEvent::new(0.5, "footstep_left"));
//! clip.events.push(AnimationEvent::new(1.2, "footstep_right"));
//!
//! // After mixer.update(), consume fired events
//! for event in mixer.drain_events() {
//!     match event.name.as_str() {
//!         "footstep_left" => play_sound("step_l.wav"),
//!         _ => {}
//!     }
//! }
//! ```

/// A single animation event defined at a specific time within a clip.
#[derive(Debug, Clone)]
pub struct AnimationEvent {
    /// Time (in seconds) within the clip at which this event fires.
    pub time: f32,
    /// User-defined event identifier.
    pub name: String,
}

impl AnimationEvent {
    #[must_use]
    pub fn new(time: f32, name: impl Into<String>) -> Self {
        Self {
            time,
            name: name.into(),
        }
    }
}

/// A fired event, carrying contextual information about when and where it originated.
#[derive(Debug, Clone)]
pub struct FiredEvent {
    /// The event identifier from the clip.
    pub name: String,
    /// The clip name that produced this event.
    pub clip_name: String,
}

/// Collects events fired during a single frame of animation evaluation.
///
/// Events are detected by checking whether the playback time range `[t_prev, t_curr]`
/// crosses any event timestamp. Handles forward playback, loops, and wrapping.
pub(crate) fn collect_events(
    events: &[AnimationEvent],
    t_prev: f32,
    t_curr: f32,
    duration: f32,
    is_forward: bool,
    clip_name: &str,
    out: &mut Vec<FiredEvent>,
) {
    if events.is_empty() || duration <= 0.0 || (t_prev - t_curr).abs() < f32::EPSILON {
        return;
    }

    for ev in events {
        let fires;

        if is_forward {
            // Forward playback: trigger if we cross the event time in the forward direction
            if t_curr >= t_prev {
                fires = ev.time > t_prev && ev.time <= t_curr;
            } else {
                // Looping forward: trigger if we cross the event time either before or after the loop point
                fires = (ev.time > t_prev && ev.time <= duration)
                    || (ev.time >= 0.0 && ev.time <= t_curr);
            }
        } else {
            if t_curr <= t_prev {
                // Reverse playback: trigger if we cross the event time in the backward direction
                fires = ev.time >= t_curr && ev.time < t_prev;
            } else {
                // Looping backward: trigger if we cross the event time either before or after the loop point
                fires = (ev.time >= 0.0 && ev.time < t_prev)
                    || (ev.time >= t_curr && ev.time <= duration);
            }
        }

        if fires {
            out.push(FiredEvent {
                name: ev.name.clone(),
                clip_name: clip_name.to_string(),
            });
        }
    }
}
