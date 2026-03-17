//! # Myth Animation
//!
//! Animation system for the Myth engine. Provides keyframe-based animation
//! with support for skeletal animation, morph targets, and blending.
//!
//! The [`AnimationTarget`] trait abstracts the scene graph operations needed
//! by the animation system, allowing it to work without depending on the
//! scene crate directly.

pub mod action;
pub mod binder;
pub mod binding;
pub(crate) mod blending;
pub mod clip;
pub mod events;
pub mod mixer;
pub mod system;
pub mod target;
pub mod tracks;
pub mod values;

pub use action::{AnimationAction, LoopMode};
pub use binder::Binder;
pub use binding::{ClipBinding, Rig, TargetPath, TrackBinding};
pub use clip::{AnimationClip, Track, TrackData, TrackMeta};
pub use events::{AnimationEvent, FiredEvent};
pub use mixer::{ActionHandle, AnimationMixer};
pub use system::AnimationSystem;
pub use target::AnimationTarget;
pub use tracks::{InterpolationMode, KeyframeTrack};
pub use values::{Interpolatable, MorphWeightData};

