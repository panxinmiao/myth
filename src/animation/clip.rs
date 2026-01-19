
use glam::{Vec3, Quat};

use crate::animation::tracks::KeyframeTrack;
use crate::animation::binding::TargetPath;

/// 轨道元数据：记录轨道绑定到哪个节点的哪个属性
#[derive(Debug, Clone)]
pub struct TrackMeta {
    pub node_name: String,
    pub target: TargetPath,
}

/// Morph Target 权重轨道
/// 每个关键帧包含多个权重值（对应多个 morph target）
#[derive(Debug, Clone)]
pub struct MorphWeightsTrack {
    pub times: Vec<f32>,
    pub weights_per_frame: usize,
    pub values: Vec<f32>,
}

impl MorphWeightsTrack {
    pub fn new(times: Vec<f32>, values: Vec<f32>, weights_per_frame: usize) -> Self {
        Self { times, values, weights_per_frame }
    }
    
    /// 采样指定时间的所有权重值
    pub fn sample(&self, time: f32, output: &mut [f32]) {
        if self.times.is_empty() || self.weights_per_frame == 0 {
            return;
        }
        
        let next_idx = self.times.partition_point(|&t| t <= time);
        let idx = if next_idx > 0 { next_idx - 1 } else { 0 };
        
        if next_idx >= self.times.len() {
            // 超过最后一帧，使用最后一帧的值
            let start = idx * self.weights_per_frame;
            let len = output.len().min(self.weights_per_frame);
            output[..len].copy_from_slice(&self.values[start..start + len]);
            return;
        }
        
        let t0 = self.times[idx];
        let t1 = self.times[next_idx];
        let dt = t1 - t0;
        let t = if dt > 1e-6 { ((time - t0) / dt).clamp(0.0, 1.0) } else { 0.0 };
        
        let start0 = idx * self.weights_per_frame;
        let start1 = next_idx * self.weights_per_frame;
        
        let len = output.len().min(self.weights_per_frame);
        for i in 0..len {
            let w0 = self.values[start0 + i];
            let w1 = self.values[start1 + i];
            output[i] = w0 + (w1 - w0) * t;
        }
    }
}

/// 统一封装不同类型的轨道
#[derive(Debug, Clone)]
pub enum TrackData {
    Vector3(KeyframeTrack<Vec3>),
    Quaternion(KeyframeTrack<Quat>),
    Scalar(KeyframeTrack<f32>),
    MorphWeights(MorphWeightsTrack),
}

/// 完整的轨道定义：包含元数据和关键帧数据
#[derive(Debug, Clone)]
pub struct Track {
    pub meta: TrackMeta,
    pub data: TrackData,
}

#[derive(Debug, Clone)]
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    pub tracks: Vec<Track>,
}

impl AnimationClip {
    pub fn new(name: String, tracks: Vec<Track>) -> Self {
        let duration = tracks.iter()
            .map(|t| match &t.data {
                TrackData::Vector3(track) => track.times.last().copied().unwrap_or(0.0),
                TrackData::Quaternion(track) => track.times.last().copied().unwrap_or(0.0),
                TrackData::Scalar(track) => track.times.last().copied().unwrap_or(0.0),
                TrackData::MorphWeights(track) => track.times.last().copied().unwrap_or(0.0),
            })
            .fold(0.0_f32, f32::max);

        Self {
            name,
            duration,
            tracks,
        }
    }
}