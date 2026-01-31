//! Animation System
//!
//! This module provides a comprehensive animation system supporting:
//!
//! - **Skeletal animation**: Bone-based character animation
//! - **Morph target animation**: Blend shape/shape key animations
//! - **Property animation**: Generic interpolation of any property
//!
//! # Architecture
//!
//! - [`AnimationClip`]: Contains animation data (tracks, duration)
//! - [`AnimationAction`]: Controls playback of a clip (play, pause, loop)
//! - [`AnimationMixer`]: Manages multiple actions for blending
//! - [`AnimationSystem`]: Updates all animations in a scene
//! - [`Binder`]: Resolves animation targets to scene nodes
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
//! ```

pub mod values;
pub mod tracks;
pub mod clip;
pub mod action;
pub mod binding;
pub mod binder;
pub mod mixer;
pub mod system;

pub use clip::{AnimationClip, Track, TrackData, TrackMeta};
pub use action::{AnimationAction, LoopMode};
pub use mixer::AnimationMixer;
pub use binder::Binder;
pub use binding::{PropertyBinding, TargetPath};
pub use tracks::{KeyframeTrack, InterpolationMode};
pub use values::{MorphWeightData, Interpolatable};
pub use system::AnimationSystem;