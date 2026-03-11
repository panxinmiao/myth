//! Animation System
//!
//! This module provides a comprehensive animation system supporting:
//!
//! - **Skeletal animation**: Bone-based character animation
//! - **Morph target animation**: Blend shape/shape key animations
//! - **Property animation**: Generic interpolation of any property
//! - **Animation blending**: Weight-based accumulator for combining multiple actions
//! - **Animation events**: Frame-synchronized event triggering during playback
//! - **Rest pose restoration**: Automatic reset when animations stop or weight < 1.0
//! - **Animation retargeting**: Share clips across models with similar bone topology
//!
//! # Architecture
//!
//! - [`AnimationClip`]: Immutable animation data (tracks with hierarchical path metadata)
//! - [`AnimationAction`]: Controls playback of a clip (play, pause, loop, weight)
//! - [`AnimationMixer`]: Manages actions & blending; owns a [`Rig`] for O(1) bone lookup
//! - [`Rig`]: Logical skeleton mapping bone indices to scene node handles
//! - [`ClipBinding`]: Precomputed track → bone-index mapping (built once, reusable)
//! - [`Binder`]: Constructs [`Rig`] and [`ClipBinding`] from scene hierarchy
//! - [`AnimationSystem`]: Updates all mixers in a scene each frame
//!
//! # Two-Phase Binding
//!
//! 1. **Clip → Rig**: [`Binder::build_clip_binding`] matches track paths against rig
//!    bone paths, producing a [`ClipBinding`] (run once at init).
//! 2. **Rig → NodeHandle**: At runtime the mixer resolves `bone_index` to
//!    `rig.bones[bone_index]` in O(1), eliminating per-frame string comparisons.
//!
//! # Rest Pose
//!
//! The first time a node receives animation, its transform is lazily recorded
//! in [`Scene::rest_transforms`]. When animation influence is lost the mixer
//! restores the rest pose, fixing the "stop = pause" feedback bug.
//!
//! # Blending
//!
//! The mixer uses a per-frame accumulator to blend contributions from all
//! active actions. Translation, scale, and morph weights use weighted lerp;
//! rotations use NLerp with sign correction. When total weight is below 1.0,
//! the rest pose fills the remainder.
//!
//! # Track Types
//!
//! - `Vector3`: Position, scale animations
//! - `Quaternion`: Rotation animations
//! - `Scalar`: Single value animations
//! - `MorphWeights`: Blend shape weight animations
//!
//! # Interpolation Modes
//!
//! - `Linear`: Linear interpolation between keyframes
//! - `Step`: Instant jump to next keyframe
//! - `CubicSpline`: Smooth cubic spline interpolation
//!
//! # Example
//!
//! ```rust,ignore
//! // Create an action from a clip
//! let action = AnimationAction::new(clip.clone());
//!
//! // Add to mixer and play
//! let handle = mixer.add_action(action);
//! mixer.action("Walk")?.play();
//!
//! // After update, consume events
//! for event in mixer.drain_events() {
//!     println!("Event: {}", event.name);
//! }
//! ```

pub mod action;
pub mod binder;
pub mod binding;
pub(crate) mod blending;
pub mod clip;
pub mod events;
pub mod mixer;
pub mod system;
pub mod tracks;
pub mod values;

pub use action::{AnimationAction, LoopMode};
pub use binder::Binder;
pub use binding::{ClipBinding, Rig, TargetPath, TrackBinding};
pub use clip::{AnimationClip, Track, TrackData, TrackMeta};
pub use events::{AnimationEvent, FiredEvent};
pub use mixer::AnimationMixer;
pub use system::AnimationSystem;
pub use tracks::{InterpolationMode, KeyframeTrack};
pub use values::{Interpolatable, MorphWeightData};
