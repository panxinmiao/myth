use std::sync::Arc;

use crate::animation::{
    MorphWeightData,
    binding::PropertyBinding,
    clip::{AnimationClip, TrackData},
    tracks::KeyframeCursor,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoopMode {
    Once,
    Loop,
    PingPong,
}

#[derive(Debug, Clone)]
pub struct AnimationAction {
    clip: Arc<AnimationClip>,

    pub time: f32,
    pub time_scale: f32,
    pub weight: f32,
    pub loop_mode: LoopMode,
    pub paused: bool,
    pub enabled: bool,

    pub bindings: Vec<PropertyBinding>,

    pub(crate) track_cursors: Vec<KeyframeCursor>,
}

impl AnimationAction {
    #[must_use]
    pub fn new(clip: Arc<AnimationClip>) -> Self {
        let track_count = clip.tracks.len();
        Self {
            clip,
            time: 0.0,
            time_scale: 1.0,
            weight: 1.0,
            loop_mode: LoopMode::Loop,
            paused: false,
            enabled: true,
            // Initialize corresponding number of cursors
            bindings: Vec::new(),
            track_cursors: vec![KeyframeCursor::default(); track_count],
        }
    }

    #[must_use]
    pub fn clip(&self) -> &Arc<AnimationClip> {
        &self.clip
    }

    /// Core logic: advance time.
    pub fn update(&mut self, dt: f32) {
        if self.paused || !self.enabled {
            return;
        }

        let duration = self.clip.duration;
        if duration <= 0.0 {
            return;
        }

        // 1. Accumulate time
        self.time += dt * self.time_scale;

        // 2. Handle loop mode
        match self.loop_mode {
            LoopMode::Once => {
                // Play once, stop at end or start
                if self.time >= duration {
                    self.time = duration;
                    self.paused = true; // Auto-pause
                } else if self.time < 0.0 {
                    self.time = 0.0;
                    self.paused = true;
                }
            }
            LoopMode::Loop => {
                // Standard loop: modulo
                if self.time >= duration {
                    self.time %= duration;
                } else if self.time < 0.0 {
                    // Handle reverse playback loop
                    self.time = duration + (self.time % duration);
                }
            }
            LoopMode::PingPong => {
                let double_duration = duration * 2.0;
                // Normalize time into [0, 2*duration) cycle
                let mut t = self.time % double_duration;
                if t < 0.0 {
                    t += double_duration;
                }
                // In the second half of the cycle, reverse direction
                if t > duration {
                    t = double_duration - t;
                }
                self.time = t;
            }
        }
    }

    /// Gets the value of the specified track at the current time.
    pub fn sample_track(&mut self, track_index: usize) -> Option<TrackValue> {
        let track = self.clip.tracks.get(track_index)?;
        let cursor = self.track_cursors.get_mut(track_index)?;

        Some(match &track.data {
            TrackData::Vector3(t) => TrackValue::Vector3(t.sample_with_cursor(self.time, cursor)),
            TrackData::Quaternion(t) => {
                TrackValue::Quaternion(t.sample_with_cursor(self.time, cursor))
            }
            TrackData::Scalar(t) => TrackValue::Scalar(t.sample_with_cursor(self.time, cursor)),
            TrackData::MorphWeights(t) => {
                TrackValue::MorphWeight(Box::new(t.sample_with_cursor(self.time, cursor)))
            }
        })
    }
}

pub enum TrackValue {
    Vector3(glam::Vec3),
    Quaternion(glam::Quat),
    Scalar(f32),
    MorphWeight(Box<MorphWeightData>),
}
