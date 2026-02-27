use crate::scene::Scene;

/// Animation system.
///
/// Drives updates for all `AnimationMixer` components.
/// Uses the `std::mem::take` technique to avoid borrow conflicts.
pub struct AnimationSystem;

impl AnimationSystem {
    /// Updates all animation mixers.
    ///
    /// # Arguments
    /// * `scene` - Scene reference
    /// * `dt` - Delta time per frame (in seconds)
    #[inline]
    pub fn update(scene: &mut Scene, dt: f32) {
        // Temporarily take all mixers out to avoid borrow conflicts
        let mut mixers = std::mem::take(&mut scene.animation_mixers);

        for (_handle, mixer) in &mut mixers {
            mixer.update(dt, scene);
        }

        // Return mixers after update
        scene.animation_mixers = mixers;
    }
}
