use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};

/// Global unique ID generator
static NEXT_RESOURCE_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_RESOURCE_ID.fetch_add(1, Ordering::Relaxed)
}

/// Resource wrapper with a unique ID
/// Used to assign identity to `FrameResources` or temporary resources
#[derive(Debug, Clone)]
pub struct Tracked<T> {
    inner: T,
    id: u64,
}

impl<T> Tracked<T> {
    /// Wrap a resource and allocate a new ID
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            id: next_id(),
        }
    }

    /// Get the unique ID (used as a key for `BindGroup` cache)
    #[inline]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Unwrap to get the inner resource
    pub fn into_inner(self) -> T {
        self.inner
    }
}

// Convenience Deref for directly accessing inner methods (e.g. texture_view.format())
impl<T> Deref for Tracked<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
