use crate::scene::Scene;

/// 动画系统
///
/// 负责驱动所有 `AnimationMixer` 组件的更新。
/// 使用 `std::mem::take` 技巧避免借用冲突。
pub struct AnimationSystem;

impl AnimationSystem {
    /// 更新所有动画混合器
    ///
    /// # Arguments
    /// * `scene` - 场景引用
    /// * `dt` - 帧间隔时间（秒）
    #[inline]
    pub fn update(scene: &mut Scene, dt: f32) {
        // 将所有 mixer 暂时取出，避免借用冲突
        let mut mixers = std::mem::take(&mut scene.animation_mixers);

        for (_handle, mixer) in &mut mixers {
            mixer.update(dt, scene);
        }

        // 更新完成后归还
        scene.animation_mixers = mixers;
    }
}
