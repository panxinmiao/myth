use parking_lot::{RwLock, RwLockReadGuard};
use rustc_hash::FxHashMap;
use slotmap::{Key, SlotMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use uuid::Uuid;

/// Versioned wrapper around an asset stored in [`AssetStorage`].
///
/// The `version` counter is bumped every time the asset data is replaced
/// or mutated in place. The render backend compares this against its own
/// synced version to decide whether a GPU re-upload is needed.
///
/// Implements `Deref<Target = T>` for ergonomic access to the inner asset
/// through the `Arc`.
pub struct AssetEntry<T> {
    pub asset: Arc<T>,
    /// Monotonically increasing counter. Starts at 1 on first insert.
    pub version: u32,
}

impl<T> std::ops::Deref for AssetEntry<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.asset
    }
}

/// Internal data structure, protected by a lock.
pub struct StorageInner<H: Key, T> {
    pub map: SlotMap<H, AssetEntry<T>>,
    pub lookup: FxHashMap<Uuid, H>,
}

impl<H: Key, T> Default for StorageInner<H, T> {
    fn default() -> Self {
        Self {
            map: SlotMap::default(),
            lookup: FxHashMap::default(),
        }
    }
}

/// Thread-safe, version-tracked asset container.
///
/// Each stored asset is wrapped in [`AssetEntry`] carrying a monotonically
/// increasing version counter. The render backend compares this version
/// against its last-synced snapshot to detect stale GPU resources.
pub struct AssetStorage<H: Key, T> {
    inner: RwLock<StorageInner<H, T>>,
    /// Global mutation epoch — bumped on every write, enabling O(1) "anything
    /// changed?" checks by the render loop without iterating entries.
    global_version: AtomicU32,
}

impl<H: Key, T> Default for AssetStorage<H, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<H: Key, T> AssetStorage<H, T> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: RwLock::default(),
            global_version: AtomicU32::new(0),
        }
    }

    /// Returns the global mutation epoch.
    #[inline]
    pub fn global_version(&self) -> u32 {
        self.global_version.load(Ordering::Relaxed)
    }

    /// \[Write\] Adds a resource and returns a Handle.
    pub fn add(&self, asset: impl Into<T>) -> H {
        let mut guard = self.inner.write();
        let entry = AssetEntry {
            asset: Arc::new(asset.into()),
            version: 1,
        };
        self.global_version.fetch_add(1, Ordering::Relaxed);
        guard.map.insert(entry)
    }

    /// \[Write\] Adds a resource with a UUID (used for file-load deduplication).
    pub fn add_with_uuid(&self, uuid: Uuid, asset: impl Into<T>) -> H {
        let mut guard = self.inner.write();
        if let Some(&handle) = guard.lookup.get(&uuid) {
            return handle;
        }
        let entry = AssetEntry {
            asset: Arc::new(asset.into()),
            version: 1,
        };
        let handle = guard.map.insert(entry);
        guard.lookup.insert(uuid, handle);
        self.global_version.fetch_add(1, Ordering::Relaxed);
        handle
    }

    /// \[Write\] Replaces the asset data at `handle`, incrementing its version.
    ///
    /// Returns the new version, or `None` if the handle is invalid.
    pub fn update(&self, handle: H, asset: impl Into<T>) -> Option<u32> {
        let mut guard = self.inner.write();
        if let Some(entry) = guard.map.get_mut(handle) {
            entry.asset = Arc::new(asset.into());
            entry.version += 1;
            self.global_version.fetch_add(1, Ordering::Relaxed);
            Some(entry.version)
        } else {
            None
        }
    }

    /// \[Read\] Gets a single resource.
    /// Returns `Arc<T>` with minimal overhead.
    pub fn get(&self, handle: H) -> Option<Arc<T>> {
        let guard = self.inner.read();
        guard.map.get(handle).map(|e| e.asset.clone())
    }

    /// \[Read\] Gets the full versioned entry for a resource.
    pub fn get_entry(&self, handle: H) -> Option<(Arc<T>, u32)> {
        let guard = self.inner.read();
        guard
            .map
            .get(handle)
            .map(|e| (e.asset.clone(), e.version))
    }

    /// \[Read\] Gets just the version of a resource.
    pub fn get_version(&self, handle: H) -> Option<u32> {
        let guard = self.inner.read();
        guard.map.get(handle).map(|e| e.version)
    }

    pub fn get_by_uuid(&self, uuid: &Uuid) -> Option<Arc<T>> {
        let guard = self.inner.read();
        let handle = guard.lookup.get(uuid)?;
        guard.map.get(*handle).map(|e| e.asset.clone())
    }

    /// Gets a Handle by UUID (when only the UUID is known).
    pub fn get_handle_by_uuid(&self, uuid: &Uuid) -> Option<H> {
        let guard = self.inner.read();
        guard.lookup.get(uuid).copied()
    }

    /// \[Read - Advanced\] Acquires a read-lock guard.
    /// Used for batch access in the render loop to avoid acquiring the lock multiple times.
    pub fn read_lock(&self) -> RwLockReadGuard<'_, StorageInner<H, T>> {
        self.inner.read()
    }
}
