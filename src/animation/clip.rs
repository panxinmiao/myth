
use glam::{Vec3, Quat};

use crate::animation::tracks::KeyframeTrack;
use crate::animation::binding::TargetPath;

/// 轨道元数据：记录轨道绑定到哪个节点的哪个属性
#[derive(Debug, Clone)]
pub struct TrackMeta {
    pub node_name: String,
    pub target: TargetPath,
}

/// 统一封装不同类型的轨道
/// 对应 glTF 的 target.path
#[derive(Debug, Clone)]
pub enum TrackData {
    Vector3(KeyframeTrack<Vec3>),
    Quaternion(KeyframeTrack<Quat>),
    Scalar(KeyframeTrack<f32>),
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
        // 自动计算 duration: 所有轨道中最晚的那个时间点
        let duration = tracks.iter()
            .map(|t| match &t.data {
                TrackData::Vector3(track) => track.times.last().copied().unwrap_or(0.0),
                TrackData::Quaternion(track) => track.times.last().copied().unwrap_or(0.0),
                TrackData::Scalar(track) => track.times.last().copied().unwrap_or(0.0),
            })
            .fold(0.0_f32, f32::max);

        Self {
            name,
            duration,
            tracks,
        }
    }
}