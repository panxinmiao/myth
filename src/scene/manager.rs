use crate::Scene;
use slotmap::{SlotMap, new_key_type};

new_key_type! {
    pub struct SceneHandle;
}

/// Subsystem responsible for managing scene lifecycle
pub struct SceneManager {
    scenes: SlotMap<SceneHandle, Scene>,
    active_scene: Option<SceneHandle>,
}

impl SceneManager {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scenes: SlotMap::with_key(),
            active_scene: None,
        }
    }

    /// Creates a new scene and returns its handle
    pub fn create_scene(&mut self) -> SceneHandle {
        self.scenes.insert(Scene::new())
    }

    /// Removes a scene (with safety checks)
    pub fn remove_scene(&mut self, handle: SceneHandle) {
        if self.active_scene == Some(handle) {
            self.active_scene = None;
            log::warn!("Active scene was removed! Screen will be empty.");
        }
        self.scenes.remove(handle);
    }

    /// Sets the currently active scene
    pub fn set_active(&mut self, handle: SceneHandle) {
        if self.scenes.contains_key(handle) {
            self.active_scene = Some(handle);
        } else {
            log::error!("Attempted to set invalid SceneHandle as active.");
        }
    }

    /// Creates and sets a new active scene, returning its mutable reference
    pub fn create_active(&mut self) -> &mut Scene {
        let handle = self.create_scene();
        self.set_active(handle);
        self.get_scene_mut(handle).unwrap()
    }

    /// Gets the handle of the currently active scene
    #[must_use]
    pub fn active_handle(&self) -> Option<SceneHandle> {
        self.active_scene
    }

    /// Gets a reference to any scene
    #[must_use]
    pub fn get_scene(&self, handle: SceneHandle) -> Option<&Scene> {
        self.scenes.get(handle)
    }

    /// Gets a mutable reference to any scene
    pub fn get_scene_mut(&mut self, handle: SceneHandle) -> Option<&mut Scene> {
        self.scenes.get_mut(handle)
    }

    /// Gets the currently active scene
    #[must_use]
    pub fn active_scene(&self) -> Option<&Scene> {
        self.active_scene.and_then(|h| self.scenes.get(h))
    }

    /// Gets the currently active scene (mutable reference)
    pub fn active_scene_mut(&mut self) -> Option<&mut Scene> {
        self.active_scene.and_then(|h| self.scenes.get_mut(h))
    }
}

impl Default for SceneManager {
    fn default() -> Self {
        Self::new()
    }
}
