use glam::{Quat, Vec3};

use crate::binding::TargetPath;
use crate::events::AnimationEvent;
use crate::tracks::KeyframeTrack;
use crate::values::MorphWeightData;

/// Metadata identifying which scene node and property a track targets.
///
/// Each track targets a node identified by a hierarchical path relative
/// to the animation root (e.g. `["Spine", "Arm_L", "Hand_L"]`). A
/// single-element path targets a direct child or the root itself.
#[derive(Debug, Clone)]
pub struct TrackMeta {
    /// Hierarchical path segments relative to the animation root node.
    /// Example: `["Spine", "Arm_L", "Hand_L"]` for a deeply nested bone.
    pub path: Vec<String>,
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
