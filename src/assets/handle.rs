//! Asset Handle System
//!
//! Provides reference-counted asset handles to prevent assets from being accidentally unloaded while in use.
//!
//! # Design Principles
//! - Uses Arc for reference counting to automatically track asset usage
//! - Strong handles (StrongHandle) keep assets alive
//! - Weak handles (WeakHandle) don't prevent asset release, suitable for caching scenarios
//! - Compatible with existing SlotMap handle system

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// Asset state tracker
/// 
/// Tracks reference counts and lifecycle state of assets
#[derive(Debug)]
pub struct AssetTracker {
    /// Strong reference count
    strong_count: AtomicU32,
    /// Weak reference count
    weak_count: AtomicU32,
    /// Whether the asset has been marked for deletion
    marked_for_deletion: AtomicU32,
}

impl AssetTracker {
    /// Creates a new tracker
    pub fn new() -> Self {
        Self {
            strong_count: AtomicU32::new(1), // Initial reference
            weak_count: AtomicU32::new(0),
            marked_for_deletion: AtomicU32::new(0),
        }
    }

    /// Increments strong reference count
    #[inline]
    pub fn add_strong(&self) -> u32 {
        self.strong_count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Decrements strong reference count, returns remaining count
    #[inline]
    pub fn release_strong(&self) -> u32 {
        let prev = self.strong_count.fetch_sub(1, Ordering::Release);
        if prev == 1 {
            // Ensure previous writes are visible to other threads
            std::sync::atomic::fence(Ordering::Acquire);
        }
        prev - 1
    }

    /// Gets current strong reference count
    #[inline]
    pub fn strong_count(&self) -> u32 {
        self.strong_count.load(Ordering::Relaxed)
    }

    /// Increments weak reference count
    #[inline]
    pub fn add_weak(&self) -> u32 {
        self.weak_count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Decrements weak reference count
    #[inline]
    pub fn release_weak(&self) -> u32 {
        self.weak_count.fetch_sub(1, Ordering::Release).saturating_sub(1)
    }

    /// Gets current weak reference count
    #[inline]
    pub fn weak_count(&self) -> u32 {
        self.weak_count.load(Ordering::Relaxed)
    }

    /// Marks the asset for deletion
    pub fn mark_for_deletion(&self) {
        self.marked_for_deletion.store(1, Ordering::Release);
    }

    /// Checks whether the asset is marked for deletion
    #[inline]
    pub fn is_marked_for_deletion(&self) -> bool {
        self.marked_for_deletion.load(Ordering::Acquire) != 0
    }

    /// Checks whether the asset can be safely released
    /// Can only be released when strong reference count is 0 and marked for deletion
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

/// Strong asset handle
/// 
/// Holding this handle prevents the asset from being released.
/// The asset will only be marked as releasable after all strong handles are dropped.
pub struct StrongHandle<K: Copy> {
    key: K,
    tracker: Arc<AssetTracker>,
}

impl<K: Copy> StrongHandle<K> {
    /// Creates a new strong handle
    pub fn new(key: K, tracker: Arc<AssetTracker>) -> Self {
        tracker.add_strong();
        Self { key, tracker }
    }

    /// Gets the underlying key
    #[inline]
    pub fn key(&self) -> K {
        self.key
    }

    /// Gets the current reference count
    #[inline]
    pub fn ref_count(&self) -> u32 {
        self.tracker.strong_count()
    }

    /// Downgrades to a weak handle
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

/// Weak asset handle
/// 
/// Does not prevent the asset from being released, suitable for caching scenarios.
/// Must attempt to upgrade to a strong handle before use.
pub struct WeakHandle<K: Copy> {
    key: K,
    tracker: Arc<AssetTracker>,
}

impl<K: Copy> WeakHandle<K> {
    /// Gets the underlying key
    #[inline]
    pub fn key(&self) -> K {
        self.key
    }

    /// Attempts to upgrade to a strong handle
    /// 
    /// Returns None if the asset is marked for deletion or strong reference count is 0
    pub fn upgrade(&self) -> Option<StrongHandle<K>> {
        // Check if already marked for deletion
        if self.tracker.is_marked_for_deletion() {
            return None;
        }

        // Try to atomically increment strong reference count
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

    /// Checks whether the asset is still valid
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

/// Asset entry
/// 
/// Associates an asset with its tracker
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

    /// Creates a strong handle for this asset
    pub fn create_handle<K: Copy>(&self, key: K) -> StrongHandle<K> {
        StrongHandle::new(key, Arc::clone(&self.tracker))
    }

    /// Gets the reference count
    #[inline]
    pub fn ref_count(&self) -> u32 {
        self.tracker.strong_count()
    }

    /// Marks the asset for deletion
    pub fn mark_for_deletion(&self) {
        self.tracker.mark_for_deletion();
    }

    /// Checks whether it can be safely released
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
        
        // Upgrade should succeed
        let upgraded = weak.upgrade();
        assert!(upgraded.is_some());
        
        drop(strong);
        drop(upgraded.unwrap());
        
        // Now only initial reference remains, upgrade should fail after marking for deletion
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
