//! 资源句柄系统 (Asset Handle System)
//!
//! 提供引用计数的资源句柄，防止资源在使用时被意外卸载。
//!
//! # 设计原则
//! - 使用 Arc 进行引用计数，自动追踪资源使用情况
//! - 强句柄 (StrongHandle) 保持资源存活
//! - 弱句柄 (WeakHandle) 不阻止资源释放，用于缓存等场景
//! - 与现有的 SlotMap 句柄系统兼容

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// 资源状态追踪器
/// 
/// 用于追踪资源的引用计数和生命周期状态
#[derive(Debug)]
pub struct AssetTracker {
    /// 强引用计数
    strong_count: AtomicU32,
    /// 弱引用计数
    weak_count: AtomicU32,
    /// 资源是否已被标记为待删除
    marked_for_deletion: AtomicU32,
}

impl AssetTracker {
    /// 创建新的追踪器
    pub fn new() -> Self {
        Self {
            strong_count: AtomicU32::new(1), // 初始引用
            weak_count: AtomicU32::new(0),
            marked_for_deletion: AtomicU32::new(0),
        }
    }

    /// 增加强引用
    #[inline]
    pub fn add_strong(&self) -> u32 {
        self.strong_count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// 减少强引用，返回剩余计数
    #[inline]
    pub fn release_strong(&self) -> u32 {
        let prev = self.strong_count.fetch_sub(1, Ordering::Release);
        if prev == 1 {
            // 确保之前的写入对其他线程可见
            std::sync::atomic::fence(Ordering::Acquire);
        }
        prev - 1
    }

    /// 获取当前强引用计数
    #[inline]
    pub fn strong_count(&self) -> u32 {
        self.strong_count.load(Ordering::Relaxed)
    }

    /// 增加弱引用
    #[inline]
    pub fn add_weak(&self) -> u32 {
        self.weak_count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// 减少弱引用
    #[inline]
    pub fn release_weak(&self) -> u32 {
        self.weak_count.fetch_sub(1, Ordering::Release).saturating_sub(1)
    }

    /// 获取当前弱引用计数
    #[inline]
    pub fn weak_count(&self) -> u32 {
        self.weak_count.load(Ordering::Relaxed)
    }

    /// 标记资源待删除
    pub fn mark_for_deletion(&self) {
        self.marked_for_deletion.store(1, Ordering::Release);
    }

    /// 检查资源是否被标记待删除
    #[inline]
    pub fn is_marked_for_deletion(&self) -> bool {
        self.marked_for_deletion.load(Ordering::Acquire) != 0
    }

    /// 检查资源是否可以被安全释放
    /// 只有当强引用为0且被标记删除时才能释放
    #[inline]
    pub fn can_be_released(&self) -> bool {
        self.strong_count() == 0 && self.is_marked_for_deletion()
    }
}

impl Default for AssetTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// 强资源句柄
/// 
/// 持有此句柄会阻止资源被释放。
/// 当所有强句柄都被丢弃后，资源才会被标记为可释放。
pub struct StrongHandle<K: Copy> {
    key: K,
    tracker: Arc<AssetTracker>,
}

impl<K: Copy> StrongHandle<K> {
    /// 创建新的强句柄
    pub fn new(key: K, tracker: Arc<AssetTracker>) -> Self {
        tracker.add_strong();
        Self { key, tracker }
    }

    /// 获取底层的键
    #[inline]
    pub fn key(&self) -> K {
        self.key
    }

    /// 获取当前引用计数
    #[inline]
    pub fn ref_count(&self) -> u32 {
        self.tracker.strong_count()
    }

    /// 降级为弱句柄
    pub fn downgrade(&self) -> WeakHandle<K> {
        self.tracker.add_weak();
        WeakHandle {
            key: self.key,
            tracker: Arc::clone(&self.tracker),
        }
    }
}

impl<K: Copy> Clone for StrongHandle<K> {
    fn clone(&self) -> Self {
        self.tracker.add_strong();
        Self {
            key: self.key,
            tracker: Arc::clone(&self.tracker),
        }
    }
}

impl<K: Copy> Drop for StrongHandle<K> {
    fn drop(&mut self) {
        self.tracker.release_strong();
    }
}

/// 弱资源句柄
/// 
/// 不阻止资源被释放，适用于缓存等场景。
/// 使用前需要尝试升级为强句柄。
pub struct WeakHandle<K: Copy> {
    key: K,
    tracker: Arc<AssetTracker>,
}

impl<K: Copy> WeakHandle<K> {
    /// 获取底层的键
    #[inline]
    pub fn key(&self) -> K {
        self.key
    }

    /// 尝试升级为强句柄
    /// 
    /// 如果资源已被标记删除或强引用为0，则返回 None
    pub fn upgrade(&self) -> Option<StrongHandle<K>> {
        // 检查是否已被标记删除
        if self.tracker.is_marked_for_deletion() {
            return None;
        }

        // 尝试原子地增加强引用
        let mut current = self.tracker.strong_count.load(Ordering::Relaxed);
        loop {
            if current == 0 {
                return None;
            }
            match self.tracker.strong_count.compare_exchange_weak(
                current,
                current + 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Some(StrongHandle {
                        key: self.key,
                        tracker: Arc::clone(&self.tracker),
                    });
                }
                Err(new) => current = new,
            }
        }
    }

    /// 检查资源是否仍然有效
    #[inline]
    pub fn is_valid(&self) -> bool {
        !self.tracker.is_marked_for_deletion() && self.tracker.strong_count() > 0
    }
}

impl<K: Copy> Clone for WeakHandle<K> {
    fn clone(&self) -> Self {
        self.tracker.add_weak();
        Self {
            key: self.key,
            tracker: Arc::clone(&self.tracker),
        }
    }
}

impl<K: Copy> Drop for WeakHandle<K> {
    fn drop(&mut self) {
        self.tracker.release_weak();
    }
}

/// 资源条目
/// 
/// 将资源与其追踪器关联
pub struct TrackedAsset<T> {
    pub asset: T,
    pub tracker: Arc<AssetTracker>,
}

impl<T> TrackedAsset<T> {
    pub fn new(asset: T) -> Self {
        Self {
            asset,
            tracker: Arc::new(AssetTracker::new()),
        }
    }

    /// 创建对此资源的强句柄
    pub fn create_handle<K: Copy>(&self, key: K) -> StrongHandle<K> {
        StrongHandle::new(key, Arc::clone(&self.tracker))
    }

    /// 获取引用计数
    #[inline]
    pub fn ref_count(&self) -> u32 {
        self.tracker.strong_count()
    }

    /// 标记资源待删除
    pub fn mark_for_deletion(&self) {
        self.tracker.mark_for_deletion();
    }

    /// 检查是否可以安全释放
    #[inline]
    pub fn can_be_released(&self) -> bool {
        self.tracker.can_be_released()
    }
}

impl<T> std::ops::Deref for TrackedAsset<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.asset
    }
}

impl<T> std::ops::DerefMut for TrackedAsset<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.asset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strong_handle_ref_count() {
        let tracker = Arc::new(AssetTracker::new());
        assert_eq!(tracker.strong_count(), 1);

        let handle1: StrongHandle<u32> = StrongHandle::new(42, Arc::clone(&tracker));
        assert_eq!(tracker.strong_count(), 2);

        let handle2 = handle1.clone();
        assert_eq!(tracker.strong_count(), 3);

        drop(handle1);
        assert_eq!(tracker.strong_count(), 2);

        drop(handle2);
        assert_eq!(tracker.strong_count(), 1);
    }

    #[test]
    fn test_weak_handle_upgrade() {
        let tracker = Arc::new(AssetTracker::new());
        let strong: StrongHandle<u32> = StrongHandle::new(42, Arc::clone(&tracker));
        
        let weak = strong.downgrade();
        assert!(weak.is_valid());
        
        // 升级应该成功
        let upgraded = weak.upgrade();
        assert!(upgraded.is_some());
        
        drop(strong);
        drop(upgraded.unwrap());
        
        // 现在只剩初始引用，标记删除后升级应该失败
        tracker.mark_for_deletion();
        assert!(!weak.is_valid());
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn test_tracked_asset() {
        let tracked = TrackedAsset::new("Hello".to_string());
        assert_eq!(tracked.ref_count(), 1);
        assert_eq!(*tracked, "Hello");
        
        let handle = tracked.create_handle(0u32);
        assert_eq!(tracked.ref_count(), 2);
        
        drop(handle);
        assert_eq!(tracked.ref_count(), 1);
    }
}
