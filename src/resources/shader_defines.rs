//! Shader Macro Definition System
//!
//! Provides a unified, high-performance shader macro management system.
//! Uses string interning and hash caching for O(1) macro set comparison.
//!
//! # Architecture
//!
//! The system uses [`Symbol`] (interned strings) for both keys and values,
//! providing:
//!
//! - **Memory efficiency**: Duplicate strings share storage
//! - **Fast comparison**: Symbol comparison is just integer comparison
//! - **Consistent hashing**: Same macro sets always produce same hashes
//!
//! # Usage
//!
//! ```rust,ignore
//! use myth::resources::ShaderDefines;
//!
//! let mut defines = ShaderDefines::new();
//! defines.set("HAS_NORMAL_MAP", "1");
//! defines.set("MAX_LIGHTS", "8");
//!
//! // Fast hash for pipeline cache lookup
//! let hash = defines.compilation_hash();
//! ```
//!
//! # Integration with Materials
//!
//! Materials implement [`RenderableMaterialTrait::shader_defines`] to declare
//! their required shader macros based on current state (e.g., enabled textures).

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use crate::utils::interner::{self, Symbol};

/// A collection of shader macro definitions.
///
/// Internally uses an ordered `Vec<(Symbol, Symbol)>` to store definitions,
/// ensuring that identical macro sets produce identical hash values.
///
/// # Performance
///
/// - Insertion/lookup: O(log n) due to binary search
/// - Hash computation: O(n) but cached
/// - Comparison: O(1) when using cached hashes
#[derive(Debug, Clone, Default)]
pub struct ShaderDefines {
    defines: Vec<(Symbol, Symbol)>,
}

impl ShaderDefines {
    /// Create empty shader defines collection
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            defines: Vec::new(),
        }
    }

    /// Create shader defines collection with pre-allocated capacity
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            defines: Vec::with_capacity(capacity),
        }
    }

    /// Set shader define (maintains sorted order)
    ///
    /// If key exists, updates its value; otherwise inserts new entry.
    pub fn set(&mut self, key: &str, value: &str) {
        let key_sym = interner::intern(key);
        let value_sym = interner::intern(value);
        self.set_symbol(key_sym, value_sym);
    }

    /// Set shader define using Symbol (internal method, more efficient)
    #[inline]
    pub fn set_symbol(&mut self, key: Symbol, value: Symbol) {
        match self.defines.binary_search_by_key(&key, |&(k, _)| k) {
            Ok(idx) => {
                self.defines[idx].1 = value;
            }
            Err(idx) => {
                self.defines.insert(idx, (key, value));
            }
        }
    }

    /// Remove shader define
    pub fn remove(&mut self, key: &str) -> bool {
        if let Some(key_sym) = interner::get(key) {
            self.remove_symbol(key_sym)
        } else {
            false
        }
    }

    /// Remove shader define using Symbol
    #[inline]
    pub fn remove_symbol(&mut self, key: Symbol) -> bool {
        if let Ok(idx) = self.defines.binary_search_by_key(&key, |&(k, _)| k) {
            self.defines.remove(idx);
            true
        } else {
            false
        }
    }

    /// Check if contains a shader define
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        interner::get(key).is_some_and(|key_sym| self.contains_symbol(key_sym))
    }

    /// Check if contains a shader define using Symbol
    #[inline]
    #[must_use]
    pub fn contains_symbol(&self, key: Symbol) -> bool {
        self.defines.binary_search_by_key(&key, |&(k, _)| k).is_ok()
    }

    /// Get shader define value
    #[must_use]
    pub fn get(&self, key: &str) -> Option<String> {
        interner::get(key).and_then(|key_sym| self.get_symbol(key_sym).map(|s| s.to_string()))
    }

    /// Get shader define value using Symbol
    #[inline]
    #[must_use]
    pub fn get_symbol(&self, key: Symbol) -> Option<std::borrow::Cow<'static, str>> {
        self.defines
            .binary_search_by_key(&key, |&(k, _)| k)
            .ok()
            .map(|idx| interner::resolve(self.defines[idx].1))
    }

    /// Clear all shader defines
    #[inline]
    pub fn clear(&mut self) {
        self.defines.clear();
    }

    /// Get shader defines count
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.defines.len()
    }

    /// Check if empty
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.defines.is_empty()
    }

    /// Iterate all shader defines (as Symbols)
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &(Symbol, Symbol)> {
        self.defines.iter()
    }

    /// Iterate all shader defines (as strings)
    #[inline]
    pub fn iter_strings(&self) -> impl Iterator<Item = (String, String)> + '_ {
        self.defines.iter().map(|&(k, v)| {
            (
                interner::resolve(k).to_string(),
                interner::resolve(v).to_string(),
            )
        })
    }

    /// Convert to `BTreeMap` (for template rendering)
    #[must_use]
    pub fn to_map(&self) -> BTreeMap<String, String> {
        self.defines
            .iter()
            .map(|&(k, v)| {
                (
                    interner::resolve(k).to_string(),
                    interner::resolve(v).to_string(),
                )
            })
            .collect()
    }

    /// Merge shader defines from another `ShaderDefines`
    ///
    /// If there are conflicts, values from other will override values in self.
    pub fn merge(&mut self, other: &ShaderDefines) {
        for &(key, value) in &other.defines {
            self.set_symbol(key, value);
        }
    }

    /// Create a new merged `ShaderDefines`
    #[must_use]
    pub fn merged_with(&self, other: &ShaderDefines) -> ShaderDefines {
        let mut result = self.clone();
        result.merge(other);
        result
    }

    /// Compute content hash (for caching)
    #[must_use]
    pub fn compute_hash(&self) -> u64 {
        use std::hash::BuildHasher;

        rustc_hash::FxBuildHasher.hash_one(self)
    }

    /// Get internal defines reference (for direct access)
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[(Symbol, Symbol)] {
        &self.defines
    }
}

impl Hash for ShaderDefines {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.defines.hash(state);
    }
}

impl PartialEq for ShaderDefines {
    fn eq(&self, other: &Self) -> bool {
        self.defines == other.defines
    }
}

impl Eq for ShaderDefines {}

/// Create `ShaderDefines` from list of macro definitions
impl From<&[(&str, &str)]> for ShaderDefines {
    fn from(defines: &[(&str, &str)]) -> Self {
        let mut result = Self::with_capacity(defines.len());
        for (k, v) in defines {
            result.set(k, v);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_and_get() {
        let mut defines = ShaderDefines::new();
        defines.set("USE_MAP", "1");
        defines.set("USE_NORMAL_MAP", "1");

        assert!(defines.contains("USE_MAP"));
        assert!(defines.contains("USE_NORMAL_MAP"));
        assert!(!defines.contains("USE_AO_MAP"));

        assert_eq!(defines.get("USE_MAP"), Some("1".to_string()));
    }

    #[test]
    fn test_ordering() {
        let mut defines = ShaderDefines::new();
        defines.set("B", "1");
        defines.set("A", "1");
        defines.set("C", "1");

        // Verify internal sorting by Symbol (integer ID), ensuring hash consistency
        // Note: Symbol order depends on intern order, not string lexicographic order
        let symbols: Vec<_> = defines.iter().map(|(k, _)| k).collect();
        assert!(
            symbols.windows(2).all(|w| w[0] < w[1]),
            "Symbols should be sorted by numeric value"
        );

        // Verify all keys exist
        assert!(defines.contains("A"));
        assert!(defines.contains("B"));
        assert!(defines.contains("C"));
    }

    #[test]
    fn test_merge() {
        let mut d1 = ShaderDefines::new();
        d1.set("A", "1");
        d1.set("B", "2");

        let mut d2 = ShaderDefines::new();
        d2.set("B", "3");
        d2.set("C", "4");

        d1.merge(&d2);

        assert_eq!(d1.get("A"), Some("1".to_string()));
        assert_eq!(d1.get("B"), Some("3".to_string())); // Overwritten
        assert_eq!(d1.get("C"), Some("4".to_string()));
    }

    #[test]
    fn test_hash_consistency() {
        let mut d1 = ShaderDefines::new();
        d1.set("A", "1");
        d1.set("B", "2");

        let mut d2 = ShaderDefines::new();
        d2.set("B", "2");
        d2.set("A", "1");

        assert_eq!(d1.compute_hash(), d2.compute_hash());
    }
}
