use parking_lot::{RwLock, RwLockReadGuard};
use rustc_hash::FxHashMap;
use slotmap::{Key, SlotMap};
use std::sync::Arc;
use uuid::Uuid;

// Internal data structure, protected by a lock.
pub struct StorageInner<H: Key, T> {
    pub map: SlotMap<H, Arc<T>>,
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

// Thread-safe container exposed to external consumers.
pub struct AssetStorage<H: Key, T> {
    inner: RwLock<StorageInner<H, T>>,
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
        }
    }

    /// [Write] Adds a resource and returns a Handle.
    /// Note: `&mut self` is no longer required.
    pub fn add(&self, asset: impl Into<T>) -> H {
        let mut guard = self.inner.write();
        guard.map.insert(Arc::new(asset.into()))
    }

    /// [Write] Adds a resource with a UUID (used for file-load deduplication).
    pub fn add_with_uuid(&self, uuid: Uuid, asset: impl Into<T>) -> H {
        let mut guard = self.inner.write();
        if let Some(&handle) = guard.lookup.get(&uuid) {
            return handle;
        }
        let handle = guard.map.insert(Arc::new(asset.into()));
        guard.lookup.insert(uuid, handle);
        handle
    }

    /// [Read] Gets a single resource.
    /// Returns `Arc<T>` with minimal overhead.
    pub fn get(&self, handle: H) -> Option<Arc<T>> {
        let guard = self.inner.read();
        guard.map.get(handle).cloned()
    }

    pub fn get_by_uuid(&self, uuid: &Uuid) -> Option<Arc<T>> {
        let guard = self.inner.read();
        let handle = guard.lookup.get(uuid)?;
        guard.map.get(*handle).cloned()
    }

    // Gets a Handle by UUID (when only the UUID is known).
    pub fn get_handle_by_uuid(&self, uuid: &Uuid) -> Option<H> {
        let guard = self.inner.read();
        guard.lookup.get(uuid).copied()
    }

    /// [Read - Advanced] Acquires a read-lock guard.
    /// Used for batch access in the render loop to avoid acquiring the lock multiple times.
    pub fn read_lock(&self) -> RwLockReadGuard<'_, StorageInner<H, T>> {
        self.inner.read()
    }
}
