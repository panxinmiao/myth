use slotmap::{SlotMap, new_key_type};
use crate::Scene;

new_key_type! {
    pub struct SceneHandle;
}

/// 负责管理场景生命周期的子系统
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

    /// 创建一个新场景，返回其句柄
    pub fn create_scene(&mut self) -> SceneHandle {
        self.scenes.insert(Scene::new())
    }

    /// 删除场景（带安全检查）
    pub fn remove_scene(&mut self, handle: SceneHandle) {
        if self.active_scene == Some(handle) {
            self.active_scene = None;
            log::warn!("Active scene was removed! Screen will be empty.");
        }
        self.scenes.remove(handle);
    }

    /// 设置当前激活场景
    pub fn set_active(&mut self, handle: SceneHandle) {
        if self.scenes.contains_key(handle) {
            self.active_scene = Some(handle);
        } else {
            log::error!("Attempted to set invalid SceneHandle as active.");
        }
    }

    /// 创建并设置一个新的激活场景，返回其可变引用
    pub fn create_active(&mut self) -> &mut Scene {
        let handle = self.create_scene();
        self.set_active(handle);
        self.get_scene_mut(handle).unwrap()
    }

    /// 获取当前激活场景的句柄
    #[must_use]
    pub fn active_handle(&self) -> Option<SceneHandle> {
        self.active_scene
    }

    /// 获取任意场景的引用
    #[must_use]
    pub fn get_scene(&self, handle: SceneHandle) -> Option<&Scene> {
        self.scenes.get(handle)
    }

    /// 获取任意场景的可变引用
    pub fn get_scene_mut(&mut self, handle: SceneHandle) -> Option<&mut Scene> {
        self.scenes.get_mut(handle)
    }

    /// 获取当前激活场景
    #[must_use]
    pub fn active_scene(&self) -> Option<&Scene> {
        self.active_scene.and_then(|h| self.scenes.get(h))
    }

    /// 获取当前激活的场景（可变引用）
    pub fn active_scene_mut(&mut self) -> Option<&mut Scene> {
        self.active_scene.and_then(|h| self.scenes.get_mut(h))
    }
}

impl Default for SceneManager {
    fn default() -> Self {
        Self::new()
    }
}