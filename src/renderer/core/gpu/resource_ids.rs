//! Resource ID tracking module
//!
//! Provides a lightweight resource ID tracking mechanism for detecting GPU resource changes.
//!
//! # Design
//!
//! 1. **`EnsureResult`**: Return value of ensure operations, containing the physical resource ID
//! 2. **`ResourceIdSet`**: A set of resource IDs, supporting efficient comparison
//! 3. **`BindGroupFingerprint`**: Complete fingerprint of a `BindGroup`, containing all dependent resource IDs

use rustc_hash::FxHasher;
use smallvec::SmallVec;
use std::hash::{Hash, Hasher};

/// Unique identifier for a GPU resource
///
/// When a GPU resource is rebuilt (e.g. Buffer expansion, Texture recreation), the ID changes
pub type ResourceId = u64;
const INVALID_RESOURCE_ID: u64 = u64::MAX;
/// Result of an ensure operation
///
/// Contains the physical ID of the resource; callers can use it to detect resource changes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnsureResult {
    /// Physical ID of the GPU resource
    pub resource_id: ResourceId,
    /// Whether the resource was just created or rebuilt
    pub was_recreated: bool,
}

impl EnsureResult {
    #[inline]
    #[must_use]
    pub fn new(resource_id: ResourceId, was_recreated: bool) -> Self {
        Self {
            resource_id,
            was_recreated,
        }
    }

    #[inline]
    #[must_use]
    pub fn existing(resource_id: ResourceId) -> Self {
        Self {
            resource_id,
            was_recreated: false,
        }
    }

    #[inline]
    #[must_use]
    pub fn created(resource_id: ResourceId) -> Self {
        Self {
            resource_id,
            was_recreated: true,
        }
    }
}

/// Resource ID set
///
/// Used to track a set of resource physical IDs, supporting fast change comparison
#[derive(Debug, Clone, Default)]
pub struct ResourceIdSet {
    /// Resource IDs stored in insertion order
    ids: SmallVec<[ResourceId; 16]>,
    /// Pre-computed hash value (for fast comparison)
    cached_hash: u64,
    /// Flag indicating whether the hash needs recomputation
    hash_dirty: bool,
}

impl ResourceIdSet {
    #[must_use]
    pub fn new() -> Self {
        Self {
            ids: SmallVec::new(),
            cached_hash: 0,
            hash_dirty: true,
        }
    }

    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            ids: SmallVec::with_capacity(capacity),
            cached_hash: 0,
            hash_dirty: true,
        }
    }

    /// Add a resource ID
    #[inline]
    pub fn push(&mut self, id: ResourceId) {
        self.ids.push(id);
        self.hash_dirty = true;
    }

    /// Add an optional resource ID
    #[inline]
    pub fn push_optional(&mut self, id: Option<ResourceId>) {
        // Use a special value to represent None
        self.ids.push(id.unwrap_or(INVALID_RESOURCE_ID));
        self.hash_dirty = true;
    }

    /// Clear the set
    #[inline]
    pub fn clear(&mut self) {
        self.ids.clear();
        self.hash_dirty = true;
    }

    /// Get the number of resource IDs
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Check if the set is empty
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Get a slice of all IDs
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[ResourceId] {
        &self.ids
    }

    #[inline]
    #[must_use]
    pub fn matches_slice(&self, other_ids: &[ResourceId]) -> bool {
        self.ids.as_slice() == other_ids
    }

    /// Compute and cache the hash value
    fn compute_hash(&mut self) {
        if !self.hash_dirty {
            return;
        }

        let mut hasher = FxHasher::default();
        self.ids.len().hash(&mut hasher);
        for id in &self.ids {
            id.hash(&mut hasher);
        }
        self.cached_hash = hasher.finish();
        self.hash_dirty = false;
    }

    /// Get the hash value (for fast comparison)
    #[inline]
    pub fn hash_value(&mut self) -> u64 {
        self.compute_hash();
        self.cached_hash
    }

    /// Compare whether two sets are identical
    pub fn matches(&mut self, other: &mut ResourceIdSet) -> bool {
        if self.ids.len() != other.ids.len() {
            return false;
        }

        // Compare hashes first (fast path)
        if self.hash_value() != other.hash_value() {
            return false;
        }

        // If hashes match, compare element by element (handle collisions)
        self.ids == other.ids
    }
}

impl PartialEq for ResourceIdSet {
    fn eq(&self, other: &Self) -> bool {
        self.ids == other.ids
    }
}

impl Eq for ResourceIdSet {}

impl Hash for ResourceIdSet {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ids.hash(state);
    }
}

/// Complete fingerprint of a `BindGroup`
///
/// Contains all physical resource IDs the `BindGroup` depends on, used to determine if a rebuild is needed
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BindGroupFingerprint {
    /// IDs of all dependent resources (in binding order)
    pub resource_ids: Vec<ResourceId>,
    /// Hash of layout entries
    pub layout_hash: u64,
}

impl BindGroupFingerprint {
    #[must_use]
    pub fn new(resource_ids: Vec<ResourceId>, layout_hash: u64) -> Self {
        Self {
            resource_ids,
            layout_hash,
        }
    }

    /// Check whether resource IDs have changed
    #[must_use]
    pub fn resources_changed(&self, new_ids: &[ResourceId]) -> bool {
        self.resource_ids != new_ids
    }

    /// Check whether the Layout has changed
    #[must_use]
    pub fn layout_changed(&self, new_layout_hash: u64) -> bool {
        self.layout_hash != new_layout_hash
    }
}

/// Compute the hash of a `BindGroupLayoutEntry` list
///
/// Leverages the Hash trait implemented by `wgpu::BindGroupLayoutEntry`
#[must_use]
pub fn hash_layout_entries(entries: &[wgpu::BindGroupLayoutEntry]) -> u64 {
    let mut hasher = FxHasher::default();
    entries.len().hash(&mut hasher);
    for entry in entries {
        entry.hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_id_set_equality() {
        let mut set1 = ResourceIdSet::new();
        set1.push(1);
        set1.push(2);
        set1.push(3);

        let mut set2 = ResourceIdSet::new();
        set2.push(1);
        set2.push(2);
        set2.push(3);

        assert!(set1.matches(&mut set2));
    }

    #[test]
    fn test_resource_id_set_difference() {
        let mut set1 = ResourceIdSet::new();
        set1.push(1);
        set1.push(2);

        let mut set2 = ResourceIdSet::new();
        set2.push(1);
        set2.push(3); // different

        assert!(!set1.matches(&mut set2));
    }
}
