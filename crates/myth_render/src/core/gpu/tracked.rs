use std::ops::Deref;

use crate::core::gpu::generate_gpu_resource_id;

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
            id: generate_gpu_resource_id(),
        }
    }

    pub fn with_id(inner: T, id: u64) -> Self {
        Self { inner, id }
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
