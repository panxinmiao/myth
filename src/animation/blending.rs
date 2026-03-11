//! Animation Blending System
//!
//! Provides accumulator-based blending for combining multiple animation actions.
//! Instead of directly overwriting scene node transforms, sampled values are
//! accumulated into per-node blend buffers and applied once per frame.
//!
//! # Blending Algorithm
//!
//! For each animated property, the accumulator performs weighted blending:
//!
//! - **Translation / Scale / Morph Weights**: Weighted linear interpolation (lerp)
//! - **Rotation**: Weighted spherical interpolation (nlerp with sign correction)
//!
//! When total accumulated weight is less than 1.0, the remainder is filled
//! by the node's original (rest pose) value, preserving correct behavior
//! for partially-weighted animations.

use glam::{Quat, Vec3};
use rustc_hash::FxHashMap;

use crate::animation::binding::TargetPath;
use crate::animation::values::MorphWeightData;
use crate::scene::NodeHandle;

/// Per-property blend entry storing the accumulated value and total weight.
#[derive(Clone)]
pub(crate) enum BlendEntry {
    Translation { value: Vec3, weight: f32 },
    Rotation { value: Quat, weight: f32 },
    Scale { value: Vec3, weight: f32 },
    MorphWeights { weights: Vec<f32>, total_weight: f32 },
}

/// Blend state for the current frame, mapping each (node, property) pair
/// to its accumulated blend entry.
pub(crate) struct FrameBlendState {
    /// Outer map: NodeHandle -> inner map of TargetPath -> BlendEntry
    entries: FxHashMap<NodeHandle, FxHashMap<TargetPath, BlendEntry>>,
}

impl FrameBlendState {
    pub fn new() -> Self {
        Self {
            entries: FxHashMap::default(),
        }
    }

    /// Clears all accumulated data for the next frame. Retains allocated memory.
    pub fn clear(&mut self) {
        for props in self.entries.values_mut() {
            props.clear();
        }
    }

    /// Accumulates a translation value with the given weight.
    pub fn accumulate_translation(&mut self, node: NodeHandle, value: Vec3, weight: f32) {
        let props = self.entries.entry(node).or_default();
        match props.get_mut(&TargetPath::Translation) {
            Some(BlendEntry::Translation {
                value: acc,
                weight: w,
            }) => {
                let new_total = *w + weight;
                let mix = weight / new_total;
                *acc = acc.lerp(value, mix);
                *w = new_total;
            }
            _ => {
                props.insert(
                    TargetPath::Translation,
                    BlendEntry::Translation { value, weight },
                );
            }
        }
    }

    /// Accumulates a rotation value with the given weight.
    ///
    /// Uses normalized linear interpolation (NLerp) with sign correction
    /// to ensure shortest-path blending between quaternions.
    pub fn accumulate_rotation(&mut self, node: NodeHandle, value: Quat, weight: f32) {
        let props = self.entries.entry(node).or_default();
        match props.get_mut(&TargetPath::Rotation) {
            Some(BlendEntry::Rotation {
                value: acc,
                weight: w,
            }) => {
                let new_total = *w + weight;
                let mix = weight / new_total;
                // Sign correction: ensure shortest path
                let corrected = if acc.dot(value) < 0.0 {
                    -value
                } else {
                    value
                };
                *acc = acc.lerp(corrected, mix).normalize();
                *w = new_total;
            }
            _ => {
                props.insert(
                    TargetPath::Rotation,
                    BlendEntry::Rotation { value, weight },
                );
            }
        }
    }

    /// Accumulates a scale value with the given weight.
    pub fn accumulate_scale(&mut self, node: NodeHandle, value: Vec3, weight: f32) {
        let props = self.entries.entry(node).or_default();
        match props.get_mut(&TargetPath::Scale) {
            Some(BlendEntry::Scale {
                value: acc,
                weight: w,
            }) => {
                let new_total = *w + weight;
                let mix = weight / new_total;
                *acc = acc.lerp(value, mix);
                *w = new_total;
            }
            _ => {
                props.insert(TargetPath::Scale, BlendEntry::Scale { value, weight });
            }
        }
    }

    /// Accumulates morph target weights with the given animation weight.
    pub fn accumulate_morph_weights(
        &mut self,
        node: NodeHandle,
        data: &MorphWeightData,
        weight: f32,
    ) {
        let props = self.entries.entry(node).or_default();
        match props.get_mut(&TargetPath::Weights) {
            Some(BlendEntry::MorphWeights {
                weights: acc,
                total_weight: w,
            }) => {
                let new_total = *w + weight;
                let mix = weight / new_total;
                let len = acc.len().max(data.weights.len());
                acc.resize(len, 0.0);
                for i in 0..data.weights.len() {
                    acc[i] = acc[i] * (1.0 - mix) + data.weights[i] * mix;
                }
                *w = new_total;
            }
            _ => {
                props.insert(
                    TargetPath::Weights,
                    BlendEntry::MorphWeights {
                        weights: data.weights.to_vec(),
                        total_weight: weight,
                    },
                );
            }
        }
    }

    /// Returns an iterator over all nodes that have accumulated blend data.
    pub fn iter_nodes(&self) -> impl Iterator<Item = (&NodeHandle, &FxHashMap<TargetPath, BlendEntry>)> {
        self.entries.iter()
    }
}
