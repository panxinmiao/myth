use std::sync::Arc;

use crate::{
    MorphWeightData,
    binding::ClipBinding,
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
    pub(crate) clip: Arc<AnimationClip>,

    pub time: f32,
    pub time_scale: f32,
    pub weight: f32,
    pub loop_mode: LoopMode,
    pub paused: bool,
    pub enabled: bool,

    /// Precomputed track-to-bone mapping (set once during binding).
    pub clip_binding: ClipBinding,

    pub(crate) track_cursors: Vec<KeyframeCursor>,

    pub(crate) ping_pong_reverse: bool,
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
            clip_binding: ClipBinding {
                bindings: Vec::new(),
            },
            track_cursors: vec![KeyframeCursor::default(); track_count],
            ping_pong_reverse: false,
        }
    }

    #[must_use]
    pub fn clip(&self) -> &Arc<AnimationClip> {
        &self.clip
    }

    pub fn reset(&mut self) {
        self.time = 0.0;
        self.paused = false;
        self.enabled = true;
        self.ping_pong_reverse = false;
        // Reset all track cursors to initial state
        for cursor in &mut self.track_cursors {
            *cursor = KeyframeCursor::default();
        }
    }

    #[inline]
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.time >= self.clip.duration
    }

    #[inline]
    #[must_use]
    pub fn is_at_start(&self) -> bool {
        self.time <= 0.0
    }

    #[inline]
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.enabled && !self.paused && self.clip.duration > 0.0
    }

    #[inline]
    pub fn stop(&mut self) {
        self.reset();
        self.enabled = false;
    }

    #[inline]
    pub fn pause(&mut self) {
        self.paused = true;
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

        let mut current_dt = dt * self.time_scale;
        if self.loop_mode == LoopMode::PingPong && self.ping_pong_reverse {
            current_dt = -current_dt;
        }

        // 1. Accumulate time
        self.time += current_dt;

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
                    self.time = self.time.rem_euclid(duration);
                }
            }
            LoopMode::PingPong => {
                if self.time >= duration {
                    self.time = duration - (self.time - duration);
                    self.ping_pong_reverse = true;
                } else if self.time <= 0.0 {
                    self.time = -self.time;
                    self.ping_pong_reverse = false;
                }
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
