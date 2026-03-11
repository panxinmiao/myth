use glam::{Quat, Vec3};

use crate::animation::binding::TargetPath;
use crate::animation::events::AnimationEvent;
use crate::animation::tracks::KeyframeTrack;
use crate::animation::values::MorphWeightData;

/// Metadata identifying which scene node and property a track targets.
#[derive(Debug, Clone)]
pub struct TrackMeta {
    /// Name of the target node in the scene hierarchy.
    pub node_name: String,
    /// Which property of the node this track animates.
    pub target: TargetPath,
}

/// Type-erased keyframe data for a single animation track.
#[derive(Debug, Clone)]
pub enum TrackData {
    Vector3(KeyframeTrack<Vec3>),
    Quaternion(KeyframeTrack<Quat>),
    Scalar(KeyframeTrack<f32>),
    MorphWeights(KeyframeTrack<MorphWeightData>),
}

/// A complete track definition pairing metadata with keyframe data.
#[derive(Debug, Clone)]
pub struct Track {
    pub meta: TrackMeta,
    pub data: TrackData,
}

/// A named collection of animation tracks with a computed duration.
///
/// Clips are immutable animation data that can be shared (via `Arc`) across
/// multiple [`AnimationAction`](super::AnimationAction) instances.
#[derive(Debug, Clone)]
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    pub tracks: Vec<Track>,
    /// Frame-synchronized events that fire during playback.
    pub events: Vec<AnimationEvent>,
}

impl AnimationClip {
    /// Creates a new clip, automatically computing duration from the last keyframe.
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
            events: Vec::new(),
        }
    }
}
