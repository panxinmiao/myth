use parking_lot::{RwLock, RwLockReadGuard};
use rustc_hash::FxHashMap;
use slotmap::{Key, SlotMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use uuid::Uuid;

/// Versioned wrapper around a loaded asset in [`AssetStorage`].
///
/// The `version` counter is bumped every time the asset data is replaced
/// or mutated in place. The render backend compares this against its own
/// synced version to decide whether a GPU re-upload is needed.
///
/// Implements `Deref<Target = T>` for ergonomic access to the inner asset
/// through the `Arc`.
#[derive(Debug)]
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

/// Lifecycle state of an asset slot in [`AssetStorage`].
///
/// Binds resource data and its lifecycle tag together in a single enum,
/// making it impossible to observe a `Loaded` state without the
/// corresponding data, or vice-versa.
#[derive(Debug)]
pub enum AssetSlot<T> {
    /// Handle has been allocated; a background task is producing the data.
    Loading,
    /// Data is fully available and ready for use.
    Loaded(AssetEntry<T>),
    /// The loading attempt failed. The message is kept for diagnostics.
    Failed(String),
}

impl<T> AssetSlot<T> {
    /// Returns `true` if the slot contains loaded data.
    #[inline]
    pub fn is_loaded(&self) -> bool {
        matches!(self, Self::Loaded(_))
    }

    /// Returns `true` if the slot is still waiting for data.
    #[inline]
    pub fn is_loading(&self) -> bool {
        matches!(self, Self::Loading)
    }

    /// Extracts a reference to the loaded entry, if available.
    #[inline]
    pub fn as_loaded(&self) -> Option<&AssetEntry<T>> {
        match self {
            Self::Loaded(entry) => Some(entry),
            _ => None,
        }
    }

    /// Extracts a mutable reference to the loaded entry, if available.
    #[inline]
    pub fn as_loaded_mut(&mut self) -> Option<&mut AssetEntry<T>> {
        match self {
            Self::Loaded(entry) => Some(entry),
            _ => None,
        }
    }
}

/// Internal data structure, protected by a lock.
pub struct StorageInner<H: Key, T> {
    pub(crate) map: SlotMap<H, AssetSlot<T>>,
    pub lookup: FxHashMap<Uuid, H>,
}

impl<H: Key, T> StorageInner<H, T> {
    /// Gets a reference to the loaded [`AssetEntry`] at `handle`.
    ///
    /// Returns `None` if the handle is invalid or the asset is not yet loaded.
    #[inline]
    pub fn get_loaded(&self, handle: H) -> Option<&AssetEntry<T>> {
        self.map.get(handle).and_then(AssetSlot::as_loaded)
    }
}

impl<H: Key, T> Default for StorageInner<H, T> {
    fn default() -> Self {
        Self {
            map: SlotMap::default(),
            lookup: FxHashMap::default(),
        }
    }
}

/// Thread-safe, version-tracked asset container with lifecycle awareness.
///
/// Each slot in the storage is an [`AssetSlot`] — either `Loading`, `Loaded`,
/// or `Failed` — making it impossible to access data that hasn't arrived yet.
/// Loaded assets carry a monotonically increasing version counter that the
/// render backend compares against its last-synced snapshot to detect stale
/// GPU resources.
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

    // ── Immediate insertion (asset data available now) ───────────────────

    /// Inserts a fully-loaded resource and returns its handle.
    pub fn add(&self, asset: impl Into<T>) -> H {
        let mut guard = self.inner.write();
        let slot = AssetSlot::Loaded(AssetEntry {
            asset: Arc::new(asset.into()),
            version: 1,
        });
        self.global_version.fetch_add(1, Ordering::Relaxed);
        guard.map.insert(slot)
    }

    /// Inserts a fully-loaded resource keyed by UUID for deduplication.
    ///
    /// If an entry with the same UUID already exists, its handle is returned
    /// without inserting a duplicate.
    pub fn add_with_uuid(&self, uuid: Uuid, asset: impl Into<T>) -> H {
        let mut guard = self.inner.write();
        if let Some(&handle) = guard.lookup.get(&uuid) {
            return handle;
        }
        let slot = AssetSlot::Loaded(AssetEntry {
            asset: Arc::new(asset.into()),
            version: 1,
        });
        let handle = guard.map.insert(slot);
        guard.lookup.insert(uuid, handle);
        self.global_version.fetch_add(1, Ordering::Relaxed);
        handle
    }

    // ── Deferred insertion (reserve now, fill later) ────────────────────

    /// Pre-allocates a handle in `Loading` state.
    ///
    /// The returned handle is immediately usable as a placeholder (e.g. to
    /// bind into a material). When the background task finishes, call
    /// [`insert_ready`](Self::insert_ready) or [`mark_failed`](Self::mark_failed).
    pub fn reserve(&self) -> H {
        let mut guard = self.inner.write();
        guard.map.insert(AssetSlot::Loading)
    }

    /// Fills a previously-reserved handle with loaded data.
    ///
    /// This is an atomic state + data transition: the slot moves from
    /// `Loading` to `Loaded` in a single write, so no observer can ever
    /// see a half-initialised entry.
    pub fn insert_ready(&self, handle: H, asset: impl Into<T>) {
        let mut guard = self.inner.write();
        if let Some(slot) = guard.map.get_mut(handle) {
            *slot = AssetSlot::Loaded(AssetEntry {
                asset: Arc::new(asset.into()),
                version: 1,
            });
            self.global_version.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Marks a previously-reserved handle as failed.
    pub fn mark_failed(&self, handle: H, error: String) {
        let mut guard = self.inner.write();
        if let Some(slot) = guard.map.get_mut(handle) {
            *slot = AssetSlot::Failed(error);
        }
    }

    // ── Mutation ────────────────────────────────────────────────────────

    /// Replaces the asset data at `handle`, incrementing its version.
    ///
    /// Returns the new version, or `None` if the handle is invalid or not
    /// in the `Loaded` state.
    pub fn update(&self, handle: H, asset: impl Into<T>) -> Option<u32> {
        let mut guard = self.inner.write();
        if let Some(AssetSlot::Loaded(entry)) = guard.map.get_mut(handle) {
            entry.asset = Arc::new(asset.into());
            entry.version += 1;
            self.global_version.fetch_add(1, Ordering::Relaxed);
            Some(entry.version)
        } else {
            None
        }
    }

    /// Removes the slot (any state) and returns the previous value.
    pub fn remove(&self, handle: H) -> Option<AssetSlot<T>> {
        let mut guard = self.inner.write();
        guard.map.remove(handle)
    }

    // ── Read accessors ─────────────────────────────────────────────────

    /// Gets the loaded resource data, or `None` if the handle is invalid,
    /// still loading, or failed.
    pub fn get(&self, handle: H) -> Option<Arc<T>> {
        let guard = self.inner.read();
        match guard.map.get(handle) {
            Some(AssetSlot::Loaded(entry)) => Some(entry.asset.clone()),
            _ => None,
        }
    }

    /// Gets the loaded entry with its version, or `None` if unavailable.
    pub fn get_entry(&self, handle: H) -> Option<(Arc<T>, u32)> {
        let guard = self.inner.read();
        match guard.map.get(handle) {
            Some(AssetSlot::Loaded(entry)) => Some((entry.asset.clone(), entry.version)),
            _ => None,
        }
    }

    /// Gets just the version of a loaded resource.
    pub fn get_version(&self, handle: H) -> Option<u32> {
        let guard = self.inner.read();
        match guard.map.get(handle) {
            Some(AssetSlot::Loaded(entry)) => Some(entry.version),
            _ => None,
        }
    }

    /// Queries the lifecycle state of a slot.
    pub fn get_state(&self, handle: H) -> Option<&'static str> {
        let guard = self.inner.read();
        guard.map.get(handle).map(|slot| match slot {
            AssetSlot::Loading => "loading",
            AssetSlot::Loaded(_) => "loaded",
            AssetSlot::Failed(_) => "failed",
        })
    }

    /// Returns `true` if the handle points to a `Loaded` slot.
    pub fn is_loaded(&self, handle: H) -> bool {
        let guard = self.inner.read();
        matches!(guard.map.get(handle), Some(AssetSlot::Loaded(_)))
    }

    pub fn get_by_uuid(&self, uuid: &Uuid) -> Option<Arc<T>> {
        let guard = self.inner.read();
        let handle = guard.lookup.get(uuid)?;
        match guard.map.get(*handle) {
            Some(AssetSlot::Loaded(entry)) => Some(entry.asset.clone()),
            _ => None,
        }
    }

    /// Gets a Handle by UUID (when only the UUID is known).
    pub fn get_handle_by_uuid(&self, uuid: &Uuid) -> Option<H> {
        let guard = self.inner.read();
        guard.lookup.get(uuid).copied()
    }

    /// Acquires a read-lock guard for batch access.
    ///
    /// Use [`StorageInner::get_loaded`] on the returned guard to access
    /// individual entries without re-acquiring the lock on each lookup.
    pub fn read_lock(&self) -> RwLockReadGuard<'_, StorageInner<H, T>> {
        self.inner.read()
    }
}
