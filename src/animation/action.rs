use std::sync::Arc;

use crate::animation::{binding::PropertyBinding, clip::{AnimationClip, TrackData}, tracks::KeyframeCursor};


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
            // 初始化对应数量的游标
            bindings: Vec::new(),
            track_cursors: vec![KeyframeCursor::default(); track_count],
        }
    }

    pub fn clip(&self) -> &Arc<AnimationClip> {
        &self.clip
    }

    /// 核心逻辑：推进时间
    pub fn update(&mut self, dt: f32) {
        if self.paused || !self.enabled {
            return;
        }

        let duration = self.clip.duration;
        if duration <= 0.0 {
            return;
        }

        // 1. 累加时间
        self.time += dt * self.time_scale;

        // 2. 处理循环模式
        match self.loop_mode {
            LoopMode::Once => {
                // 播放一次，停在终点或起点
                if self.time >= duration {
                    self.time = duration;
                    self.paused = true; // 自动暂停
                } else if self.time < 0.0 {
                    self.time = 0.0;
                    self.paused = true;
                }
            }
            LoopMode::Loop => {
                // 标准循环：取模
                if self.time >= duration {
                    self.time %= duration;
                } else if self.time < 0.0 {
                    // 处理倒放循环
                    self.time = duration + (self.time % duration);
                }
            }
            LoopMode::PingPong => {
                // 往复模式逻辑相对复杂，暂略，需要记录方向
                // 简单实现：使用 PingPong 数学公式
                // time = ping_pong(time, duration)
                // 这里先简化为 Loop
                if self.time >= duration {
                    self.time %= duration;
                }
            }
        }
    }

    /// 获取指定轨道在当前时间的值
    pub fn sample_track(&mut self, track_index: usize) -> Option<TrackValue> {
        let track = self.clip.tracks.get(track_index)?;
        let cursor = self.track_cursors.get_mut(track_index)?;
        
        Some(match &track.data {
            TrackData::Vector3(t) => TrackValue::Vector3(t.sample_with_cursor(self.time, cursor)),
            TrackData::Quaternion(t) => TrackValue::Quaternion(t.sample_with_cursor(self.time, cursor)),
            TrackData::Scalar(t) => TrackValue::Scalar(t.sample_with_cursor(self.time, cursor)),
            TrackData::MorphWeights(_) => return None,
        })
    }
}

pub enum TrackValue {
    Vector3(glam::Vec3),
    Quaternion(glam::Quat),
    Scalar(f32),
}