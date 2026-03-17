use crate::mixer::AnimationMixer;
use crate::target::AnimationTarget;
use slotmap::SlotMap;

/// Animation system.
///
/// Drives updates for all `AnimationMixer` components.
pub struct AnimationSystem;

impl AnimationSystem {
    /// Updates all animation mixers against the given animation target.
    ///
    /// The caller is responsible for extracting mixers from their storage
    /// before calling this method (to avoid borrow conflicts when the
    /// target and mixer storage share the same owner).
    ///
    /// # Arguments
    /// * `mixers` - All animation mixers to update
    /// * `target` - The animation target (e.g. scene graph) to apply results to
    /// * `dt` - Delta time per frame (in seconds)
    #[inline]
    pub fn update<K: slotmap::Key>(
        mixers: &mut SlotMap<K, AnimationMixer>,
        target: &mut dyn AnimationTarget,
        dt: f32,
    ) {
        for (_handle, mixer) in mixers {
            mixer.update(dt, target);
        }
    }
}
