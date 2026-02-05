use bytemuck::Pod;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
#[cfg(debug_assertions)]
use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

pub trait GpuData {
    fn as_bytes(&self) -> &[u8];
    fn byte_size(&self) -> usize;
}

impl<T: Pod> GpuData for Vec<T> {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self)
    }

    fn byte_size(&self) -> usize {
        std::mem::size_of::<T>() * self.len()
    }
}

/// `BufferRef`: lightweight handle (id, label, usage, size). No CPU data owned.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferRef {
    pub id: u64,
    #[cfg(debug_assertions)]
    label: Cow<'static, str>,
    pub usage: wgpu::BufferUsages,
    pub size: usize,  // Recorded size, used as reference during creation
    pub version: u64, // Version number, used to track content changes
}

impl BufferRef {
    pub fn new(size: usize, usage: wgpu::BufferUsages, _label: Option<&str>) -> Self {
        Self {
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            #[cfg(debug_assertions)]
            label: _label.map_or(Cow::Borrowed("Unnamed Buffer"), |s| {
                Cow::Owned(s.to_string())
            }),
            usage,
            size,
            version: 0,
        }
    }

    /// Helper constructor: create handle based on data length
    pub fn from_data<T: Pod>(data: &[T], usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        let size = std::mem::size_of_val(data);
        Self::new(size, usage, label)
    }

    /// Helper constructor: create handle from byte array
    #[must_use]
    pub fn from_bytes(data: &[u8], usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        Self::new(data.len(), usage, label)
    }

    /// Create an empty handle
    #[must_use]
    pub fn empty(usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        Self::new(0, usage, label)
    }

    /// Create a handle with specified capacity
    #[must_use]
    pub fn with_capacity(capacity: usize, usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        Self::new(capacity, usage, label)
    }

    /// Create a handle with a specified ID (for global resource management)
    ///
    /// Note: This method allows specifying a fixed ID for global resources that need stable references.
    /// For normal use cases, use `new()` to let the system automatically allocate an ID.
    #[must_use]
    pub fn with_fixed_id(
        id: u64,
        size: usize,
        usage: wgpu::BufferUsages,
        version: u64,
        _label: Option<&str>,
    ) -> Self {
        Self {
            id,
            #[cfg(debug_assertions)]
            label: _label.map_or(Cow::Borrowed("Unnamed Buffer"), |s| {
                Cow::Owned(s.to_string())
            }),
            usage,
            size,
            version,
        }
    }

    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }
    #[must_use]
    pub fn usage(&self) -> wgpu::BufferUsages {
        self.usage
    }
    #[must_use]
    pub fn label(&self) -> Option<&str> {
        #[cfg(debug_assertions)]
        {
            Some(&self.label)
        }
        #[cfg(not(debug_assertions))]
        {
            None
        }
    }
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }
}

pub struct BufferGuard<'a, T: GpuData> {
    guard: RwLockWriteGuard<'a, CpuBufferState<T>>,
    pub(crate) changed: bool,
}

impl<T: GpuData> BufferGuard<'_, T> {
    // Allow user to manually skip version update
    pub fn skip_sync(&mut self) {
        self.changed = false;
    }
}

impl<T: GpuData> std::ops::Deref for BufferGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.guard.data
    }
}

impl<T: GpuData> std::ops::DerefMut for BufferGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard.data
    }
}

impl<T: GpuData> Drop for BufferGuard<'_, T> {
    fn drop(&mut self) {
        if self.changed {
            self.guard.version = self.guard.version.wrapping_add(1);
            self.guard.size = self.guard.data.byte_size();
        }
    }
}

/// Internal mutable state: only these data need lock protection
#[derive(Debug)]
struct CpuBufferState<T: GpuData> {
    data: T,
    version: u64,
    size: usize,
}

#[derive(Debug)]
pub struct CpuBuffer<T: GpuData> {
    // 1. Immutable Metadata - placed outside the lock, supports lock-free access
    id: u64,
    usage: wgpu::BufferUsages,
    #[cfg(debug_assertions)]
    label: Cow<'static, str>,

    // 2. Mutable State - placed inside the lock
    inner: RwLock<CpuBufferState<T>>,
}

impl<T: GpuData + Clone> Clone for CpuBuffer<T> {
    fn clone(&self) -> Self {
        // 1. Acquire read lock to get reference to current data
        let guard = self.inner.read();
        // 2. Construct new CpuBuffer with cloned data
        Self::new(
            guard.data.clone(),
            self.usage,
            self.label(), // Reuse Label
        )
    }
}

impl<T: GpuData> CpuBuffer<T> {
    pub fn new(data: T, usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        let size = data.byte_size();

        // Create BufferRef first mainly to reuse its ID generation logic
        let base_ref = BufferRef::new(size, usage, label);

        Self {
            // Extract immutable metadata to outer layer
            id: base_ref.id,
            usage: base_ref.usage,
            #[cfg(debug_assertions)]
            label: base_ref.label,

            // Initialize internal mutable state
            inner: RwLock::new(CpuBufferState {
                data,
                version: 0,
                size,
            }),
        }
    }

    #[must_use]
    pub fn default() -> Self
    where
        T: Default,
    {
        Self::new(
            T::default(),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            None,
        )
    }

    #[must_use]
    pub fn new_uniform(label: Option<&str>) -> Self
    where
        T: Default,
    {
        Self::new(
            T::default(),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            label,
        )
    }

    #[must_use]
    pub fn new_storage(label: Option<&str>) -> Self
    where
        T: Default,
    {
        Self::new(
            T::default(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            label,
        )
    }

    // === Lock-free operations ===
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn usage(&self) -> wgpu::BufferUsages {
        self.usage
    }

    pub fn label(&self) -> Option<&str> {
        #[cfg(debug_assertions)]
        {
            Some(&self.label)
        }
        #[cfg(not(debug_assertions))]
        {
            None
        }
    }

    // === Operations requiring lock ===

    pub fn size(&self) -> usize {
        self.inner.read().size
    }

    pub fn version(&self) -> u64 {
        self.inner.read().version
    }

    /// Acquire data read lock
    pub fn read(&self) -> BufferReadGuard<'_, T> {
        BufferReadGuard {
            guard: self.inner.read(),
        }
    }
    /// Acquire data write lock
    pub fn write(&self) -> BufferGuard<'_, T> {
        BufferGuard {
            guard: self.inner.write(),
            changed: true,
        }
    }

    /// Get a snapshot of the current `BufferRef`
    /// Since version number is inside the lock, we must construct a new `BufferRef` on the spot
    pub fn handle(&self) -> BufferRef {
        let state = self.inner.read();
        BufferRef {
            id: self.id,
            #[cfg(debug_assertions)]
            label: self.label.clone(),
            usage: self.usage,
            size: state.size,
            version: state.version,
        }
    }
}

pub struct BufferReadGuard<'a, T: GpuData> {
    guard: RwLockReadGuard<'a, CpuBufferState<T>>,
}

impl<T: GpuData> std::ops::Deref for BufferReadGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.guard.data
    }
}
