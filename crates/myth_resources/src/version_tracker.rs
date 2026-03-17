/// Version tracker - used to mark resource changes
#[derive(Debug, Clone, Copy, Default)]
pub struct ChangeTracker {
    version: u64,
}

impl ChangeTracker {
    #[must_use]
    pub fn new() -> Self {
        Self { version: 0 }
    }

    /// Marks as modified, increments version by 1
    pub fn changed(&mut self) {
        self.version = self.version.wrapping_add(1);
    }

    /// Gets the current version number
    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }
}

/// Mutable guard - automatically updates version when scope ends
pub struct MutGuard<'a, T> {
    data: &'a mut T,
    version: &'a mut u64,
}

impl<'a, T> MutGuard<'a, T> {
    pub fn new(data: &'a mut T, version: &'a mut u64) -> Self {
        Self { data, version }
    }
}

impl<T> std::ops::Deref for MutGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<T> std::ops::DerefMut for MutGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
    }
}

// Key: when Guard is dropped, automatically increment version number
impl<T> Drop for MutGuard<'_, T> {
    fn drop(&mut self) {
        *self.version = self.version.wrapping_add(1);
    }
}
