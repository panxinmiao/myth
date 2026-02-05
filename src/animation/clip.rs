use glam::{Quat, Vec3};

use crate::animation::binding::TargetPath;
use crate::animation::tracks::KeyframeTrack;
use crate::animation::values::MorphWeightData;

#[derive(Debug, Clone)]
pub struct TrackMeta {
    pub node_name: String,
    pub target: TargetPath,
}

#[derive(Debug, Clone)]
pub enum TrackData {
    Vector3(KeyframeTrack<Vec3>),
    Quaternion(KeyframeTrack<Quat>),
    Scalar(KeyframeTrack<f32>),
    MorphWeights(KeyframeTrack<MorphWeightData>),
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
        let duration = tracks
            .iter()
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
